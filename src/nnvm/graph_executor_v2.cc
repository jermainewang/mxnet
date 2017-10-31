/*!
 * Copyright (c) 2016 by Contributors
 * \file graph_ptr_executor_v2.cc
 * \brief Executor to execute the computation graph.
 */
#include "./graph_executor_v2.h"
#include <mxnet/imperative.h>

using namespace std;
using namespace nnvm;

namespace mxnet {
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
  // Whether the closure needs to be recreated.
  bool dirty{true};
};

// Internal data structure used for creating
// engine operators.
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
    if (storageid >= 0) {
      cl->in_array[i] = data_pool[storageid].AsArray(
          shapes->value[eid], dtypes->value[eid]);
    } else if (storageid == kExternalStorageID) {
      CHECK(!cl->in_array[i].is_none());
      //const auto& array = cl->in_array[i];
      //CHECK_EQ(shapes->value[eid], array.shape())
        //<< "Shape mismatch: expect " << shapes->value[eid]
        //<< " provided " << array.shape();
      //CHECK_EQ(dtypes->value[eid], array.dtype())
        //<< "DType mismatch: expect " << dtypes->value[eid]
        //<< " provided " << array.dtype();
    } else {
      LOG(FATAL) << "Cannot create ndarray for entry#" << eid
        << " with provided shape=" << shapes->value[eid]
        << " & dtype=" << dtypes->value[eid];
    }
  }
  // Output arrays.
  for (uint32_t i = 0; i < node->num_outputs(); ++i) {
    const uint32_t eid = idx.entry_id(nid, i);
    const int storageid = mem_plan->value[eid].storage_id;
    if (storageid >= 0) {
      cl->out_array[i] = data_pool[storageid].AsArray(
          shapes->value[eid], dtypes->value[eid]);
    } else if (storageid == kExternalStorageID) {
      CHECK(!cl->out_array[i].is_none());
      //const auto& array = cl->out_array[i];
      //CHECK_EQ(shapes->value[eid], array.shape())
        //<< "Shape mismatch: expect " << shapes->value[eid]
        //<< " provided " << array.shape();
      //CHECK_EQ(dtypes->value[eid], array.dtype())
        //<< "DType mismatch: expect " << dtypes->value[eid]
        //<< " provided " << array.dtype();
    } else if (storageid == kNull) {
      // TODO(minjie): Context for null output?
      cl->out_array[i] = NDArray(shapes->value[eid],
          cl->ctx, true, dtypes->value[eid]);
    } else {
      LOG(FATAL) << "Cannot create ndarray for entry#" << eid
        << " with provided shape=" << shapes->value[eid]
        << " & dtype=" << dtypes->value[eid];
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
    shared_ptr<OpExecutorV2> exec,
    RunContext ctx,
    Engine::CallbackOnComplete on_complete) {
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
    const Closure& cl, shared_ptr<OpExecutorV2> exec) {
  // Execution functor.
  return [cl, exec] (RunContext ctx, Engine::CallbackOnComplete on_complete) {
    EngineAsyncFn(cl, exec, ctx, on_complete);
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
    const Closure& cl, shared_ptr<OpExecutorV2> exec) {
  // Setup variables
  vector<Engine::VarHandle> use_vars, mutate_vars;
  CreateReadWriteVar(cl, &use_vars, &mutate_vars);
  return Engine::Get()->NewOperator(
      CreateEngineAsyncFn(cl, exec), use_vars, mutate_vars, FnProperty::kNormal,
      PROFILER_MESSAGE(closures_[nid].opr_name));
}

void AttachOpClosuresRec(const Graph& graph,
                         const Column<shared_ptr<OpExecutorV2>>* op_execs,
                         const Column<int>* vdevice,
                         const Column<StorageRef>* mem_plan,
                         Column<Closure>* closures) {
  const auto& idx = graph.indexed_graph();
  const auto& context = graph.GetGlobalAttr<vector<Context>>(pass::ctx::ctx_key);
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    Closure& cl = closures->value[nid];
    const Node* node = idx[nid].source;
    if (node->is_variable()) {
      continue;
    }
    cl.opr_name = node->op()->name;
    if (node->is_graph()) {
      // XXX(minjie): Note that subgraph operator closures are currently never reused.
      // To reuse them, one must make sure the OpExecutorV2 can be reused.
      // See attach_op_exec_pass_v2.cc
      auto subgraph = node->graph();
      auto subref = subgraph->CreateNodeColumn<Closure>();
      AttachOpClosuresRec(*subgraph,
                          op_execs->children[nid].get(),
                          vdevice->children[nid].get(),
                          mem_plan->children[nid].get(),
                          subref.CopyOnWrite());
      closures->children[nid] = subref;
    } else {
      cl.in_array.resize(node->inputs.size());
      cl.out_array.resize(node->num_outputs());
      cl.ctx = context.at(vdevice->value[nid]);
    }
  }
}

// Check all operator closures and re-make those dirty ones.
void SetupClosureRec(const Graph& graph,
                     const vector<NDArray>& data_pool,
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
      SetupClosureRec(*node->graph(),
                      data_pool,
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
        SetupClosure(graph, nid, shapes, dtypes, mem_plan, data_pool, &cl);
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
}  // namespace

GraphExecutorV2::GraphExecutorV2(shared_ptr<const Graph> graph,
                                 const GraphExecutorV2::ExecState& fwd_state,
                                 const GraphExecutorV2::Config& config)
  : graph_ptr_(graph), config_(config), fwd_execs_(fwd_state),
    required_graph_ptr_attrs_(_InitRequiredGraphAttrs()) {
  _CheckAllAttrsExist(*graph_ptr_, required_graph_ptr_attrs_);
  //AttachOps();
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
      CHECK(op_execs_->value[nid]->exec_type() != ExecType::kCrossDeviceCopy
          && op_execs_->value[nid]->exec_type() != ExecType::kAsync)
        << "Bulk execution is NOT allowed when some nodes in the graph are"
        << " asynchronous.";
    }
  }
}

void GraphExecutorV2::AttachOps() {
  using namespace pass;
  op_execs_ = graph_ptr_->CreateNodeColumn<shared_ptr<OpExecutorV2>>();
  const auto* shapes = graph_ptr_->entry_attrs.GetColumn<TShape>(shape::key).get();
  const auto* dtypes = graph_ptr_->entry_attrs.GetColumn<int>(dtype::key).get();
  const auto* mem_plan = graph_ptr_->entry_attrs.GetColumn<StorageRef>(plan_memory::ref_key).get();
  const auto* vdevice = graph_ptr_->node_attrs.GetColumn<int>(ctx::device_key).get();
  const auto* mutate = graph_ptr_->node_attrs.GetColumn<vector<uint32_t>>(mutate::key).get();
  AttachOpExecsRec(*graph_ptr_,
                   shapes,
                   dtypes,
                   mem_plan,
                   vdevice,
                   mutate,
                   fwd_execs_.get(),
                   op_execs_.CopyOnWrite());
  closures_ = graph_ptr_->CreateNodeColumn<Closure>();
  AttachOpClosuresRec(*graph_ptr_,
                      op_execs_.get(),
                      vdevice,
                      mem_plan,
                      closures_.CopyOnWrite());
}

void GraphExecutorV2::Run(const vector<NDArray>& arguments,
                          vector<NDArray>* results,
                          const RunOption& option) {
  // TODO(minjie): currently use new operators for each run.
  AttachOps();

  const auto& idx = graph_ptr_->indexed_graph();
  DLOG(INFO) << "Graph execution starts.";
  // Feed arguments.
  DLOG(INFO) << "Feeding argument ndarrays.";
  CHECK_EQ(arguments.size(), idx.input_nodes().size());
  for (size_t i = 0; i < arguments.size(); ++i) {
    FeedArgArray(arguments[i], i);
  }
  // Feed results.
  CHECK_NOTNULL(results);
  if (!results->empty()) {
    DLOG(INFO) << "Feeding result ndarrays.";
    // Result storage is provided. Feed the result array
    // as the output of the related operator.
    CHECK_EQ(results->size(), graph_ptr_->outputs.size());
    for (size_t i = 0; i < results->size(); ++i) {
      FeedRstArray((*results)[i], i);
    }
  }

  const auto* shapes = graph_ptr_->entry_attrs.GetColumn<TShape>(pass::shape::key).get();
  const auto* dtypes = graph_ptr_->entry_attrs.GetColumn<int>(pass::dtype::key).get();
  const auto* mem_plan = graph_ptr_->entry_attrs.GetColumn<StorageRef>(
      pass::plan_memory::ref_key).get();
  SetupClosureRec(*graph_ptr_,
                  data_entries_,
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
      //LOG(INFO) << "Fetch rst#" << i << " name="
        //<< graph_ptr_.outputs[i].node->attrs.name << "_output"
        //<< graph_ptr_.outputs[i].index;
      results->push_back(FetchRstArray(i));
    }
  }

  if (config_.dynamic_allocation) {
    ResetDataEntries();
  }

  if (config_.bulk_execution) {
    RunOpsInBulk();
  } else {
    RunOps();
  }

  if (Imperative::Get()->is_recording()) {
    //TODO(state)
    //exec->GetState();
    NodeAttrs attrs;
    //Imperative::Get()->RecordOp(std::move(attrs), arguments, results);
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
      auto exec = op_execs_.CopyOnWrite()->value[nid];
      if (exec->exec_type() == ExecType::kCrossDeviceCopy) {
        CHECK_EQ(node->inputs.size(), 1U);
        CHECK_EQ(cl.in_array.size(), 1U);
        CHECK_EQ(cl.out_array.size(), 1U);
        CopyFromTo(cl.in_array[0], &(cl.out_array[0]));
      } else {
        Engine::AsyncFn fn = CreateEngineAsyncFn(cl, exec);
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

void GraphExecutorV2::RunOpsInBulk() {
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
  for (const auto& op_inent : arg_to_op_input_) {
    const uint32_t nid = op_inent[0].first;
    const size_t index = op_inent[0].second;
    const NDArray& nd = closures_->value[nid].in_array[index];
    use_vars.push_back(nd.var());
  }
  for (const auto& ent : idx.outputs()) {
    const NDArray& nd = closures_->value[ent.node_id].out_array[ent.index];
    mutate_vars.push_back(nd.var());
  }
  Engine::Get()->DeduplicateVarHandle(&use_vars, &mutate_vars);
  // TODO(minjie): how about temporary resources?
  shared_ptr<const Graph> graph = graph_ptr_;
  const Config& cfg = config_;
  // Note: use move to clear the reference of the closures and op_execs.
  ColumnRef<Closure> closures = std::move(closures_);
  ExecState op_execs = std::move(op_execs_);
  auto fn = [graph, op_execs, closures, cfg, is_gpu, is_train] 
    (RunContext ctx, Engine::CallbackOnComplete on_complete) mutable {
    const auto& idx = graph->indexed_graph();
    OpContext op_ctx{is_train, ctx, on_complete, {}};
    for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
      const Node* node = idx[nid].source;
      if (node->is_variable()) {
        continue;
      } else if (node->is_graph()) {
        // TODO(minjie): subgraph node.
        LOG(FATAL) << "Not implemented.";
      } else {
        Closure& cl = closures.CopyOnWrite()->value[nid];
        auto exec = op_execs.CopyOnWrite()->value[nid];
        exec->Setup(cl.in_array, cl.out_array);
        op_ctx.requested = cl.requested;
        exec->Run(op_ctx);
        if (cfg.dynamic_allocation) {
          ResetClosure(node, &cl);
        }
      }
    }
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
  
void GraphExecutorV2::FeedArgArray(const NDArray& array, size_t i) {
  const auto* shapes = graph_ptr_->entry_attrs.GetColumn<TShape>(pass::shape::key).get();
  const auto* dtypes = graph_ptr_->entry_attrs.GetColumn<int>(pass::dtype::key).get();
  const auto& idx = graph_ptr_->indexed_graph();
  for (const OpInputEntry& ent : arg_to_op_input_[i]) {
    const uint32_t nid = ent.first;
    const size_t i = ent.second;
    closures_.CopyOnWrite()->value[nid].in_array[i] = array;
    closures_.CopyOnWrite()->value[nid].dirty = true;
  }
}

void GraphExecutorV2::FeedRstArray(const NDArray& array, size_t i) {
  const auto& idx = graph_ptr_->indexed_graph();
  const NodeEntry& outent = graph_ptr_->outputs[i];
  const uint32_t nid = idx.node_id(outent.node.get());
  closures_.CopyOnWrite()->value[nid].out_array[outent.index] = array;
  closures_.CopyOnWrite()->value[nid].dirty = true;
}

const NDArray& GraphExecutorV2::FetchRstArray(size_t i) {
  const auto& idx = graph_ptr_->indexed_graph();
  const NodeEntry& outent = graph_ptr_->outputs[i];
  const uint32_t nid = idx.node_id(outent.node.get());
  return closures_->value[nid].out_array[outent.index];
}

void GraphExecutorV2::SetupResources() {
  SetupDataEntries();
  SetupOpResources();
}

const vector<string>& GraphExecutorV2::RequiredGraphAttrs() const {
  return required_graph_ptr_attrs_;
}

void GraphExecutorV2::SetupOpResources() {
  // Save mapping from arguments to operator inputs.
  const auto& idx = graph_ptr_->indexed_graph();
  const auto& input_nids = idx.input_nodes();
  arg_to_op_input_.resize(input_nids.size());
  unordered_map<uint32_t, size_t> argnid2idx;
  for (size_t i = 0; i < input_nids.size(); ++i) {
    argnid2idx[input_nids[i]] = i;
  }
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    for (size_t i = 0; i < node->inputs.size(); ++i) {
      const Node* innode = node->inputs[i].node.get();
      const uint32_t innid = idx.node_id(innode);
      if (innode->is_variable()) {
        arg_to_op_input_[argnid2idx[innid]].push_back(
            std::make_pair(nid, i));
      }
    }
  }
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
