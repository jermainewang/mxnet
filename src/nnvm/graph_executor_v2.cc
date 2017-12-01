/*!
 * Copyright (c) 2016 by Contributors
 * \file graph_ptr_executor_v2.cc
 * \brief Executor to execute the computation graph.
 */
#include "./graph_executor_v2.h"

using namespace std;
using namespace nnvm;

namespace mxnet {
namespace op {
const OperatorProperty* OpPropGetOpProperty(const NodeAttrs& attrs);
}  // namespace op

namespace exec {

using StorageRef = pass::plan_memory::StorageRef;

struct Closure {
  // The name of the operator
  std::string opr_name;
  // the context of the node
  Context ctx;
  // Whether is in training mode.
  bool is_train{false};
  // Input and output arrays.
  std::vector<NDArray> in_array, out_array;
  // Requested resources.
  std::vector<Resource> requested;
  // Execution function of this operator.
  std::shared_ptr<OpExecutorV2> op_exec;
  // Whether the closure needs to be recreated.
  bool dirty{true};
};

// forward executor
class ForwardOpExecutorV2 : public OpExecutorV2 {
 public:
  void Run(const OpContext& op_ctx) const override {
    auto* opr = state_.get_state<op::OperatorState>().opr();
    opr->Forward(op_ctx, in_data_, req, out_data_, aux_data_);
#if MKL_EXPERIMENTAL == 1
    mkl_tblobs_prv_to_cpu(in_data_);
    mkl_tblobs_prv_to_cpu(out_data_);
    mkl_tblobs_prv_to_cpu(aux_data_);
#endif
  }
  ExecType exec_type() const override {
    auto* opr = state_.get_state<op::OperatorState>().opr();
    return opr->exec_type();
  }
  OpStatePtr state() const {
    return state_;
  }
  explicit ForwardOpExecutorV2(OpStatePtr state,
                               const OperatorProperty* prop,
                               vector<uint32_t> aux_index)
      : state_(state) {
    const size_t num_inputs = prop->ListArguments().size() + aux_index.size();
    const size_t num_outputs = prop->NumOutputs();
    this->Reset(num_inputs, num_outputs);
    out_data_.resize(prop->NumOutputs());
    in_data_.resize(prop->ListArguments().size());
    aux_data_.resize(aux_index.size());
    // Setup in tblob pointer.
    std::sort(aux_index.begin(), aux_index.end());
    size_t nml_top = 0, aux_top = 0;
    for (size_t i = 0; i < num_inputs; ++i) {
      if (!std::binary_search(aux_index.begin(), aux_index.end(), i)) {
        CHECK_GT(in_data_.size(), nml_top);
        in_tblob_ptr_[i] = &in_data_[nml_top++];
      } else {
        CHECK_GT(aux_data_.size(), aux_top);
        in_tblob_ptr_[i] = &aux_data_[aux_top++];
      }
    }
    // Setup out tblob pointer.
    for (size_t i = 0; i < num_outputs; ++i) {
      out_tblob_ptr_[i] = &out_data_[i];
    }
  }

 private:
  OpStatePtr state_;
  vector<TBlob> in_data_, out_data_, aux_data_;
};

// backward executor
class BackwardOpExecutorV2 : public OpExecutorV2 {
 public:
  void Run(const OpContext& op_ctx) const override {
    auto* opr = state_.get_state<op::OperatorState>().opr();
    opr->Backward(op_ctx, out_grad_, in_data_, out_data_,
                  req, in_grad_, aux_data_);
#if MKL_EXPERIMENTAL == 1
    mkl_tblobs_prv_to_cpu(out_grad_);
    mkl_tblobs_prv_to_cpu(in_data_);
    mkl_tblobs_prv_to_cpu(out_data_);
    mkl_tblobs_prv_to_cpu(in_grad_);
    mkl_tblobs_prv_to_cpu(aux_data_);
#endif
  }
  ExecType exec_type() const override {
    auto* opr = state_.get_state<op::OperatorState>().opr();
    return opr->exec_type();
  }
  explicit BackwardOpExecutorV2(OpStatePtr state,
                                const OperatorProperty* prop,
                                vector<uint32_t> aux_index)
      : state_(state) {
    out_grad_.resize(prop->NumVisibleOutputs());
    in_data_.resize(prop->ListArguments().size());
    in_grad_.resize(in_data_.size());
    out_data_.resize(prop->NumOutputs());
    aux_data_.resize(aux_index.size());
    // Compute backward dependencies.
    vector<TBlob*> out_grad_ptr(out_grad_.size());
    for (size_t i = 0; i < out_grad_.size(); ++i) {
      out_grad_ptr[i] = &out_grad_[i];
    }
    vector<TBlob*> in_data_ptr(in_data_.size());
    for (size_t i = 0; i < in_data_.size(); ++i) {
      in_data_ptr[i] = &in_data_[i];
    }
    vector<TBlob*> out_data_ptr(out_data_.size());
    for (size_t i = 0; i < out_data_.size(); ++i) {
      out_data_ptr[i] = &out_data_[i];
    }
    vector<TBlob*> bwd_in_ptr = prop->BackwardInputs(
        out_grad_ptr, in_data_ptr, out_data_ptr);
    const size_t num_inputs = bwd_in_ptr.size() + aux_index.size();
    const size_t num_outputs = in_data_.size();
    this->Reset(num_inputs, num_outputs);
    // Setup input tblob pointers.
    std::sort(aux_index.begin(), aux_index.end());
    size_t nml_top = 0, aux_top = 0;
    for (size_t i = 0; i < num_inputs; ++i) {
      if (!std::binary_search(aux_index.begin(), aux_index.end(), i)) {
        CHECK_GT(bwd_in_ptr.size(), nml_top);
        in_tblob_ptr_[i] = bwd_in_ptr[nml_top++];
      } else {
        CHECK_GT(aux_data_.size(), aux_top);
        in_tblob_ptr_[i] = &aux_data_[aux_top++];
      }
    }
    // Setup output tblob pointers.
    for (size_t i = 0; i < out_tblob_ptr_.size(); ++i) {
      out_tblob_ptr_[i] = &in_grad_[i];
    }
  }

 private:
  OpStatePtr state_;
  vector<TBlob> out_grad_, in_data_, out_data_, aux_data_;
  vector<TBlob> in_grad_;
};

// fcompute executor executor
class FComputeExecutorV2 : public OpExecutorV2 {
 public:
  void Run(const OpContext& op_ctx) const override {
    fcompute_(attrs_, op_ctx, in_data_, req, out_data_);
#if MKL_EXPERIMENTAL == 1
    mkl_tblobs_prv_to_cpu(in_data_);
    mkl_tblobs_prv_to_cpu(out_data_);
#endif
  }
  ExecType exec_type() const override {
    return ExecType::kSync;
  }
  OpStatePtr state() const {
    return OpStatePtr();
  }
  explicit FComputeExecutorV2(FCompute fcompute,
                              const NodeAttrs& attrs,
                              size_t num_inputs,
                              size_t num_outputs)
      : fcompute_(fcompute), attrs_(attrs) {
    in_data_.resize(num_inputs);
    out_data_.resize(num_outputs);
    this->Reset(num_inputs, num_outputs);
    // Setup input tblob pointers.
    for (size_t i = 0; i < num_inputs; ++i) {
      in_tblob_ptr_[i] = &in_data_[i];
    }
    // Setup output tblob pointers.
    for (size_t i = 0; i < num_outputs; ++i) {
      out_tblob_ptr_[i] = &out_data_[i];
    }
  }

 private:
  FCompute fcompute_;
  NodeAttrs attrs_;
  vector<TBlob> in_data_, out_data_;
};

namespace {
vector<string> _InitRequiredGraphAttrs() {
  using namespace pass;
  return {shape::key,
          dtype::key,
          ctx::device_key,
          ctx::ctx_key,
          plan_memory::ref_key,
          plan_memory::storage_key,
          mutate::key};
}
void _CheckAllAttrsExist(const Graph& graph, const vector<string>& required) {
  for (const auto& attr : required) {
    if (!graph.global_attrs.count(attr)
        && !graph.entry_attrs.count(attr)
        && !graph.node_attrs.count(attr)) {
      LOG(FATAL) << "Evaluating a graph requires attribute \""
        << attr << "\" to be specialized beforehand but is currently missing.";
    }
  }
}

// 1. Create input/output arrays with proper shapes and point them to the
//    correct data entry.
// 2. Allocate temporary resources.
// 3. Create operator handler for executing using engine.
void SetupClosure(const Graph& graph,
                  uint32_t nid,
                  const Column<TShape>* shapes,
                  const Column<int>* dtypes,
                  const Column<StorageRef>* mem_plan,
                  const vector<NDArray>& data_pool,
                  const unordered_map<uint32_t, NDArray>& external_pool,
                  Closure* cl) {
  using pass::plan_memory::kExternalStorageID;
  using pass::plan_memory::kNull;

  const auto& idx = graph.indexed_graph();
  const Node* node = idx[nid].source;

  cl->dirty = false;

  CHECK_EQ(node->inputs.size(), cl->in_array.size());
  CHECK_EQ(node->num_outputs(), cl->out_array.size());

  // Input arrays.
  for (size_t i = 0; i < node->inputs.size(); ++i) {
    const uint32_t eid = idx.entry_id(node->inputs[i]);
    const StorageRef& store_ref = mem_plan->value[eid];
    const int storageid = mem_plan->value[eid].storage_id;
    if (external_pool.count(eid)) {  // Check external pool first.
      cl->in_array[i] = external_pool.at(eid);
      //const auto& array = cl->in_array[i];
      //CHECK_EQ(shapes->value[eid], array.shape())
        //<< "Shape mismatch: expect " << shapes->value[eid]
        //<< " provided " << array.shape();
      //CHECK_EQ(dtypes->value[eid], array.dtype())
        //<< "DType mismatch: expect " << dtypes->value[eid]
        //<< " provided " << array.dtype();
    } else if (storageid >= 0) {
      cl->in_array[i] = data_pool[storageid].AsArray(
          shapes->value[eid], dtypes->value[eid]);
    } else {
      if (storageid == kExternalStorageID) {
        LOG(FATAL) << "Entry#" << eid << " requires external data"
          << " but is not provided.";
      } else {
        LOG(FATAL) << "Cannot create ndarray for entry#" << eid
          << " with provided shape=" << shapes->value[eid]
          << " & dtype=" << dtypes->value[eid];
      }
    }
  }
  // Output arrays.
  for (uint32_t i = 0; i < node->num_outputs(); ++i) {
    const uint32_t eid = idx.entry_id(nid, i);
    const int storageid = mem_plan->value[eid].storage_id;
    if (external_pool.count(eid)) {  // Check external pool first.
      cl->out_array[i] = external_pool.at(eid);
      if (cl->op_exec->req[i] == kWriteInplace) {
        // Inplace write is not available is the output buffer
        // is provided externally.
        cl->op_exec->req[i] = kWriteTo;
      }
      //const auto& array = cl->out_array[i];
      //CHECK_EQ(shapes->value[eid], array.shape())
        //<< "Shape mismatch: expect " << shapes->value[eid]
        //<< " provided " << array.shape();
      //CHECK_EQ(dtypes->value[eid], array.dtype())
        //<< "DType mismatch: expect " << dtypes->value[eid]
        //<< " provided " << array.dtype();
    } else if (storageid >= 0) {
      cl->out_array[i] = data_pool[storageid].AsArray(
          shapes->value[eid], dtypes->value[eid]);
    } else if (storageid == kNull) {
      // The output entry will never be used by any other nodes.
      // However, we still need to create the memory so that the node that
      // generates this output can be correctly executed.
      // TODO(minjie): How to set context for null output?
      cl->out_array[i] = NDArray(shapes->value[eid],
          cl->ctx, true, dtypes->value[eid]);
    } else {
      if (storageid == kExternalStorageID) {
        LOG(FATAL) << "Entry#" << eid << " requires external data"
          << " but is not provided.";
      } else {
        LOG(FATAL) << "Cannot create ndarray for entry#" << eid
          << " with provided shape=" << shapes->value[eid]
          << " & dtype=" << dtypes->value[eid];
      }
    }
  }
  // Requested resources.
  static auto& fresource = Op::GetAttr<FResourceRequest>("FResourceRequest");
  if (fresource.count(node->op())) {
    // Allocate op resources.
    auto reqs = fresource[node->op()](node->attrs);
    auto& requested = cl->requested;
    // Get the resource of temporal space.
    for (const ResourceRequest& req : reqs) {
      requested.push_back(ResourceManager::Get()->Request(cl->ctx, req));
    }
  }
}

inline void EngineAsyncFn(
    const Closure& cl,
    RunContext ctx,
    Engine::CallbackOnComplete on_complete) {
  auto exec = cl.op_exec;
  const bool is_async = exec->exec_type() == ExecType::kAsync;
  const bool is_gpu = cl.ctx.dev_mask() == gpu::kDevMask;
  OpContext op_ctx{cl.is_train, ctx, on_complete, cl.requested};
  exec->Setup(cl.in_array, cl.out_array);
  exec->Run(op_ctx);
  // call on complete only if it is async op
  if (!is_async) {
    if (is_gpu) {
#if MXNET_USE_CUDA
      // Wait GPU kernel to finish.
      ctx.get_stream<gpu>()->Wait();
#else
      LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
    }
    on_complete();
  }
}

inline Engine::AsyncFn CreateEngineAsyncFn(
    const Closure& cl) {
  // Execution functor.
  return [cl] (RunContext ctx, Engine::CallbackOnComplete on_complete) {
    EngineAsyncFn(cl, ctx, on_complete);
  };
}

void CreateReadWriteVar(const Closure& cl, 
                        vector<Engine::VarHandle>* use_vars,
                        vector<Engine::VarHandle>* mutate_vars) {
  use_vars->clear();
  mutate_vars->clear();
  for (size_t i = 0; i < cl.in_array.size(); ++i) {
    use_vars->push_back(cl.in_array[i].var());
  }
  for (auto& r : cl.requested) {
    mutate_vars->push_back(r.var);
  }
  for (size_t i = 0; i < cl.out_array.size(); ++i) {
    if (!cl.out_array[i].is_none()) {
      mutate_vars->push_back(cl.out_array[i].var());
    }
  }
  // Remove dupliated vars.
  Engine::Get()->DeduplicateVarHandle(use_vars, mutate_vars);
}

inline Engine::OprHandle CreateCachedOpr(
    const Closure& cl) {
  // Setup variables
  vector<Engine::VarHandle> use_vars, mutate_vars;
  CreateReadWriteVar(cl, &use_vars, &mutate_vars);
  return Engine::Get()->NewOperator(
      CreateEngineAsyncFn(cl), use_vars, mutate_vars, FnProperty::kNormal,
      PROFILER_MESSAGE(closures_[nid].opr_name));
}

void AttachOpClosuresRec(const Graph& graph,
                         const Column<int>* vdevice,
                         const Column<StorageRef>* mem_plan,
                         const Column<vector<uint32_t>>* mutate_index,
                         const Column<FunctorInfo>* infos,
                         const vector<Context>& context,
                         Column<Closure>* closures) {
  using pass::plan_memory::StorageRef;
  using pass::plan_memory::kNull;
  const auto& idx = graph.indexed_graph();
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    Closure& cl = closures->value[nid];
    const Node* node = idx[nid].source;
    if (node->is_variable()) {
      continue;
    }
    if (node->is_graph()) {
      // XXX(minjie): Note that subgraph operator closures are currently never reused.
      // To reuse them, one must make sure the OpExecutorV2 can be reused.
      // See attach_op_exec_pass_v2.cc
      auto subgraph = node->graph();
      AttachOpClosuresRec(*subgraph,
                          vdevice->children[nid].get(),
                          mem_plan->children[nid].get(),
                          mutate_index->children[nid].get(),
                          infos->children[nid].get(),
                          context,
                          closures->children[nid].CopyOnWrite());
    } else {
      cl.opr_name = node->op()->name;
      switch (infos->value[nid].type) {
      case FunctorType::kFCompute:
        {
          FCompute fcompute = GetFCompute(node->op(), context.at(vdevice->value[nid]));
          CHECK_NOTNULL(fcompute);
          cl.op_exec = std::make_shared<FComputeExecutorV2>(
              fcompute, node->attrs, node->inputs.size(), node->num_outputs());
          break;
        }
      case FunctorType::kForward:
        {
          cl.op_exec = std::make_shared<ForwardOpExecutorV2>(
              infos->value[nid].state,
              mxnet::op::OpPropGetOpProperty(node->attrs),
              mutate_index->value[nid]);
          break;
        }
      case FunctorType::kBackward:
        {
          cl.op_exec = std::make_shared<BackwardOpExecutorV2>(
              infos->value[nid].state,
              mxnet::op::OpPropGetOpProperty(node->attrs),
              mutate_index->value[nid]);
          break;
        }
      case FunctorType::kUndefined:
        LOG(FATAL) << "No functor registered for operator \"" << node->op()->name << "\".";
      }
      // Setup output requests.
      for (size_t i = 0; i < node->num_outputs(); ++i) {
        const uint32_t eid = idx.entry_id(nid, i);
        const StorageRef& store_ref = mem_plan->value[eid];
        const int storageid = mem_plan->value[eid].storage_id;
        // Output request.
        // TODO(minjie): handle external request type.
        if (false) {
          // TODO(minjie): addto inplace optimization.
        } else if (store_ref.inplace_index >= 0) {
          cl.op_exec->req.push_back(kWriteInplace);
        } else if (storageid == kNull) {
          // TODO(minjie): need double-check.
          cl.op_exec->req.push_back(kNullOp);
        } else {
          cl.op_exec->req.push_back(kWriteTo);
        }
      }
    }
    cl.in_array.resize(node->inputs.size());
    cl.out_array.resize(node->num_outputs());
    cl.ctx = context.at(vdevice->value[nid]);
  }
}

// Check all operator closures and re-make those dirty ones.
void SetupClosureRec(const Graph& graph,
                     const vector<NDArray>& data_pool,
                     const unordered_map<uint32_t, NDArray>& external_pool,
                     const GraphExecutorV2::RunOption& option,
                     const Column<TShape>* shapes,
                     const Column<int>* dtypes,
                     const Column<StorageRef>* mem_plan,
                     Column<Closure>* closures) {
  const auto& idx = graph.indexed_graph();
  // Check all operator closures and re-make those dirty ones.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    if (node->is_variable()) {
      continue;
    } else if (node->is_graph()) {
      // XXX(minjie): Note that subgraph operator closures are currently never reused.
      // To reuse them, one must make sure the OpExecutorV2 can be reused.
      // See attach_op_exec_pass_v2.cc
      const auto& subidx = node->graph()->indexed_graph();
      unordered_map<uint32_t, NDArray> new_ext_pool;
      for (uint32_t i = 0; i < node->inputs.size(); ++i) {
        const uint32_t eid = idx.entry_id(node->inputs[i]);
        if (external_pool.count(eid)) {
          const uint32_t subeid = subidx.entry_id(subidx.input_nodes()[i], 0);
          new_ext_pool[subeid] = external_pool.at(eid);
        }
      }
      for (uint32_t i = 0; i < node->num_outputs(); ++i) {
        const uint32_t eid = idx.entry_id(nid, i);
        if (external_pool.count(eid)) {
          const uint32_t subeid = subidx.entry_id(node->graph()->outputs[i]);
          new_ext_pool[subeid] = external_pool.at(eid);
        }
      }
      SetupClosureRec(*node->graph(),
                      data_pool,
                      new_ext_pool,
                      option,
                      shapes->children[nid].get(),
                      dtypes->children[nid].get(),
                      mem_plan->children[nid].get(),
                      closures->children[nid].CopyOnWrite());
    } else {
      Closure& cl = closures->value[nid];
      cl.is_train = option.is_train;
      if (cl.dirty) {
        // Create operator closure for dirty op.
        DLOG(INFO) << "Setup closure for node#" << nid << ": " << node->attrs.name;
        SetupClosure(graph, nid, shapes, dtypes, mem_plan,
                     data_pool, external_pool, &cl);
      }
    }
  }
}

void ResetClosure(const Node* node, Closure* cl) {
  cl->in_array.clear();
  cl->in_array.resize(node->inputs.size());
  cl->out_array.clear();
  cl->out_array.resize(node->num_outputs());
  cl->requested.clear();  // TODO(minjie): how to reclaim the requested resources?
  cl->dirty = true;
}

void ExecBulkRec(const Graph& graph,
                 const GraphExecutorV2::Config& cfg,
                 OpContext op_ctx,
                 Column<Closure>* closures) {
  const auto& idx = graph.indexed_graph();
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    if (node->is_variable()) {
      continue;
    } else if (node->is_graph()) {
      ExecBulkRec(*node->graph(),
                  cfg,
                  op_ctx,
                  closures->children[nid].CopyOnWrite());
    } else {
      Closure& cl = closures->value[nid];
      auto exec = cl.op_exec;
      exec->Setup(cl.in_array, cl.out_array);
      op_ctx.requested = cl.requested;
      exec->Run(op_ctx);
      if (cfg.dynamic_allocation) {
        ResetClosure(node, &cl);
      }
    }
  }
}
}  // namespace

GraphExecutorV2::GraphExecutorV2(nnvm::GraphPtr graph,
                                 const GraphExecutorV2::ExecState& fwd_states,
                                 const GraphExecutorV2::Config& config)
  : graph_ptr_(graph), config_(config), fwd_states_(fwd_states),
    required_graph_ptr_attrs_(_InitRequiredGraphAttrs()) {
  _CheckAllAttrsExist(*graph_ptr_, required_graph_ptr_attrs_);
  SetupResources();
  //if (config_.bulk_execution) {
    //CheckAllowBulkExec();
  //}
}

GraphExecutorV2::~GraphExecutorV2() {
}

void GraphExecutorV2::CheckAllowBulkExec() const {
  using namespace pass;
  const auto& idx = graph_ptr_->indexed_graph();
  const auto* vdevice = graph_ptr_->node_attrs.GetColumn<int>(ctx::device_key).get();
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    if (node->is_variable()) {
      continue;
    } else if (node->is_graph()) {
      continue;
    } else {
      CHECK_EQ(vdevice->value[nid], vdevice->value[0])
        << "Bulk execution is NOT allowed when some nodes in the graph are"
        << " assigned to different devices.";
      CHECK(closures_->value[nid].op_exec->exec_type() != ExecType::kCrossDeviceCopy
          && closures_->value[nid].op_exec->exec_type() != ExecType::kAsync)
        << "Bulk execution is NOT allowed when some nodes in the graph are"
        << " asynchronous.";
    }
  }
}

void GraphExecutorV2::AttachOps() {
  using namespace pass;
  const auto* shapes = graph_ptr_->entry_attrs.GetColumn<TShape>(shape::key).get();
  const auto* dtypes = graph_ptr_->entry_attrs.GetColumn<int>(dtype::key).get();
  const auto* mem_plan = graph_ptr_->entry_attrs.GetColumn<StorageRef>(plan_memory::ref_key).get();
  const auto* vdevice = graph_ptr_->node_attrs.GetColumn<int>(ctx::device_key).get();
  const auto* mutate = graph_ptr_->node_attrs.GetColumn<vector<uint32_t>>(mutate::key).get();
  const auto& context = graph_ptr_->GetGlobalAttr<vector<Context>>(pass::ctx::ctx_key);
  states_ = CreateNodeColumn<FunctorInfo>(*graph_ptr_);
  AttachFunctorInfoRec(*graph_ptr_,
                       shapes,
                       dtypes,
                       vdevice,
                       fwd_states_.get(),
                       context,
                       states_.CopyOnWrite());
  closures_ = CreateNodeColumn<Closure>(*graph_ptr_);
  AttachOpClosuresRec(*graph_ptr_,
                      vdevice,
                      mem_plan,
                      mutate,
                      states_.get(),
                      context,
                      closures_.CopyOnWrite());
}

void GraphExecutorV2::Run(const vector<NDArray>& arguments,
                          const std::vector<OpReqType>& result_req,
                          vector<NDArray>* results,
                          const RunOption& option) {
  // Sanity checks.
  const auto& idx = graph_ptr_->indexed_graph();
  CHECK_EQ(arguments.size(), idx.input_nodes().size());
  CHECK_NOTNULL(results);
  CHECK(results->empty() || results->size() == graph_ptr_->outputs.size());

  // TODO(minjie): currently use new operators for each run.
  AttachOps();

  unordered_map<uint32_t, NDArray> external_pool;
  for (size_t i = 0; i < arguments.size(); ++i) {
    const uint32_t eid = idx.entry_id(idx.input_nodes()[i], 0);
    external_pool[eid] = arguments[i];
  }
  if (!results->empty()) {
    for (size_t i = 0; i < results->size(); ++i) {
      const uint32_t eid = idx.entry_id(graph_ptr_->outputs[i]);
      external_pool[eid] = (*results)[i];
    }
  }

  const auto* shapes = graph_ptr_->entry_attrs.GetColumn<TShape>(pass::shape::key).get();
  const auto* dtypes = graph_ptr_->entry_attrs.GetColumn<int>(pass::dtype::key).get();
  const auto* mem_plan = graph_ptr_->entry_attrs.GetColumn<StorageRef>(
      pass::plan_memory::ref_key).get();
  SetupClosureRec(*graph_ptr_,
                  data_entries_,
                  external_pool,
                  option,
                  shapes,
                  dtypes,
                  mem_plan,
                  closures_.CopyOnWrite());
  
  if (results->empty()) {
    DLOG(INFO) << "Fetching result ndarrays.";
    // Result array is not provided. Fetch the output arrays
    // of the graph as the result array.
    for (size_t i = 0; i < graph_ptr_->outputs.size(); ++i) {
      results->push_back(FetchRstArray(i));
    }
  }

  if (config_.dynamic_allocation) {
    ResetDataEntries();
  }

  if (config_.bulk_execution) {
    RunOpsInBulk(arguments, *results);
  } else {
    RunOps();
  }
}

void GraphExecutorV2::RunOps() {
  const auto& idx = graph_ptr_->indexed_graph();
  vector<Engine::VarHandle> use_vars, mutate_vars;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    if (node->is_variable()) {
      continue;
    } else if (node->is_graph()) {
      // TODO(minjie): subgraph node.
      LOG(FATAL) << "Not implemented.";
    } else {
      Closure& cl = closures_.CopyOnWrite()->value[nid];
      if (cl.op_exec->exec_type() == ExecType::kCrossDeviceCopy) {
        CHECK_EQ(node->inputs.size(), 1U);
        CHECK_EQ(cl.in_array.size(), 1U);
        CHECK_EQ(cl.out_array.size(), 1U);
        CopyFromTo(cl.in_array[0], &(cl.out_array[0]));
      } else {
        Engine::AsyncFn fn = CreateEngineAsyncFn(cl);
        CreateReadWriteVar(cl, &use_vars, &mutate_vars);
        Engine::Get()->PushAsync(fn, cl.ctx, use_vars, mutate_vars, FnProperty::kNormal, 0,
            PROFILER_MESSAGE(cl.opr_name));
      }
      /* TODO(minjie): cached opr mode
        else if (cl.cached_opr != nullptr) {
#if MXNET_USE_PROFILER
        bool profiling = engine::Profiler::Get()->GetState() == engine::Profiler::kRunning;
#else
        bool profiling = false;
#endif
        DLOG(INFO) << "Push Node#" << nid << " " << node->attrs.name;
        Engine::Get()->Push(cl.cached_opr, cl.ctx, 0, profiling);
      } else {
        LOG(FATAL) << "Cannot execute Node#" << nid << " " << node->attrs.name;
      }
      */
      if (config_.dynamic_allocation) {
        ResetClosure(node, &cl);
      }
    }
  }
}

void GraphExecutorV2::RunOpsInBulk(const vector<NDArray>& arguments,
                                   const vector<NDArray>& results) {
  const auto& idx = graph_ptr_->indexed_graph();
  uint32_t first_non_var_nid = 0;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    if (!node->is_variable()) {
      first_non_var_nid = nid;
      break;
    }
  }
  const Context& bulkctx = closures_->value[first_non_var_nid].ctx;
  const bool is_train = closures_->value[first_non_var_nid].is_train;
  const bool is_gpu = bulkctx.dev_mask() == gpu::kDevMask;
  std::vector<Engine::VarHandle> use_vars, mutate_vars;
  for (const auto& nd : arguments) {
    use_vars.push_back(nd.var());
  }
  for (const auto& nd : results) {
    mutate_vars.push_back(nd.var());
  }
  Engine::Get()->DeduplicateVarHandle(&use_vars, &mutate_vars);
  // TODO(minjie): how about temporary resources?
  shared_ptr<const Graph> graph = graph_ptr_;
  const Config& cfg = config_;
  // Note: use move to clear the reference of the closures and op_execs.
  ColumnRef<Closure> closures = std::move(closures_);
  auto fn = [graph, closures, cfg, is_gpu, is_train] 
    (RunContext ctx, Engine::CallbackOnComplete on_complete) mutable {
    const auto& idx = graph->indexed_graph();
    OpContext op_ctx{is_train, ctx, on_complete, {}};
    ExecBulkRec(*graph, cfg, op_ctx,
                closures.CopyOnWrite());
    if (is_gpu) {
#if MXNET_USE_CUDA
      // Wait GPU kernel to finish.
      ctx.get_stream<gpu>()->Wait();
#else
      LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
    }
    on_complete();
  };
  Engine::Get()->PushAsync(fn, bulkctx, use_vars, mutate_vars, FnProperty::kNormal, 0,
      PROFILER_MESSAGE("bulk-execution"));
}
  
const NDArray& GraphExecutorV2::FetchRstArray(size_t i) {
  const auto& idx = graph_ptr_->indexed_graph();
  const NodeEntry& outent = graph_ptr_->outputs[i];
  const uint32_t nid = idx.node_id(outent.node.get());
  return closures_->value[nid].out_array[outent.index];
}

void GraphExecutorV2::SetupResources() {
  SetupDataEntries();
}

const vector<string>& GraphExecutorV2::RequiredGraphAttrs() const {
  return required_graph_ptr_attrs_;
}

void GraphExecutorV2::ResetDataEntries() {
  DLOG(INFO) << "Reset all data entries to allow dynamic allocation.";
  for (size_t i = 0; i < data_entries_.size(); ++i) {
    const NDArray& old = data_entries_[i];
    data_entries_[i] = NDArray(old.shape(), old.ctx(), true, old.dtype());
  }
}

void GraphExecutorV2::SetupDataEntries() {
  using pass::plan_memory::Storage;
  using pass::plan_memory::StorageRef;
  using pass::plan_memory::storage_key;
  using pass::ctx::ctx_key;
  const auto& storage = graph_ptr_->GetGlobalAttr<vector<Storage>>(storage_key);
  const auto& context = graph_ptr_->GetGlobalAttr<vector<Context>>(ctx_key);
  const auto* shapes = graph_ptr_->entry_attrs.GetColumn<TShape>(pass::shape::key).get();
  const auto* dtypes = graph_ptr_->entry_attrs.GetColumn<int>(pass::dtype::key).get();
  const auto* mem_plan = graph_ptr_->entry_attrs.GetColumn<StorageRef>(
      pass::plan_memory::ref_key).get();
  const bool delay_alloc = !config_.dynamic_allocation;
  data_entries_.reserve(storage.size());
  for (const Storage& st : storage) {
    // XXX(minjie): Use float array for all dtype allocation. Should change
    // to other dtype in the future.
    const size_t nword = (st.max_bytes + 3) / 4;
    CHECK_LE(nword, std::numeric_limits<dim_t>::max());
    TShape shape{static_cast<dim_t>(nword)};
    DLOG(INFO) << "Allocate data entry#" << data_entries_.size()
               << " size=" << nword << " floats.";
    data_entries_.push_back(NDArray(shape, context.at(st.device_id), delay_alloc));
  }
}

}  // namespace exec
}  // namespace mxnet
