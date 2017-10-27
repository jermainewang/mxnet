/*!
 * Copyright (c) 2016 by Contributors
 * \file graph_ptr_executor_v2.cc
 * \brief Executor to execute the computation graph.
 */
#include "./graph_executor_v2.h"

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
  // The executor
  std::shared_ptr<OpExecutorV2> exec;
  // skip the execution of this node
  bool skip_exec_node{false};
  // Whether is in training mode.
  bool is_train{false};
  // Input and output arrays.
  std::vector<NDArray> in_array, out_array;
  // Requested resources.
  std::vector<Resource> requested;
  // Engine operator handler.
  Engine::OprHandle cached_opr{nullptr};
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
  auto exec = cl->exec;

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
      const auto& array = cl->in_array[i];
      CHECK_EQ(shapes->value[eid], array.shape())
        << "Shape mismatch: expect " << shapes->value[eid]
        << " provided " << array.shape();
      CHECK_EQ(dtypes->value[eid], array.dtype())
        << "DType mismatch: expect " << dtypes->value[eid]
        << " provided " << array.dtype();
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
      const auto& array = cl->out_array[i];
      CHECK_EQ(shapes->value[eid], array.shape())
        << "Shape mismatch: expect " << shapes->value[eid]
        << " provided " << array.shape();
      CHECK_EQ(dtypes->value[eid], array.dtype())
        << "DType mismatch: expect " << dtypes->value[eid]
        << " provided " << array.dtype();
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
    requested.clear();
    // Get the resource of temporal space.
    for (const ResourceRequest& req : reqs) {
      requested.push_back(ResourceManager::Get()->Request(cl->ctx, req));
    }
  }

  // Execution functor.
  const Closure& capture = *cl;
  auto exec_fun = [capture] (
      RunContext ctx, Engine::CallbackOnComplete on_complete) {
    auto exec = capture.exec;
    const bool is_async = exec->exec_type() == ExecType::kAsync;
    const bool is_gpu = capture.ctx.dev_mask() == gpu::kDevMask;
    OpContext op_ctx;
    if (is_async) {
      op_ctx.async_on_complete = on_complete;
    }
    op_ctx.is_train = capture.is_train;
    op_ctx.run_ctx = ctx;
    op_ctx.requested = std::move(capture.requested);
    exec->Setup(capture.in_array, capture.out_array);
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
  };
  // Setup variables
  std::vector<Engine::VarHandle> use_vars, mutate_vars;
  for (size_t i = 0; i < cl->in_array.size(); ++i) {
    use_vars.push_back(cl->in_array[i].var());
  }
  for (auto& r : cl->requested) {
    mutate_vars.push_back(r.var);
  }
  for (size_t i = 0; i < cl->out_array.size(); ++i) {
    if (!cl->out_array[i].is_none()) {
      mutate_vars.push_back(cl->out_array[i].var());
    }
  }
  /*ostringstream oss;
  oss << "Read: [";
  for (const auto& v : use_vars) {
    oss << v << " ";
  }
  oss << "] Write: [";
  for (const auto& v : mutate_vars) {
    oss << v << " ";
  }
  oss << "]";
  LOG(INFO) << oss.str();*/
  // Remove dupliated vars.
  Engine::Get()->DeduplicateVarHandle(&use_vars, &mutate_vars);
  // Save everything.
  cl->cached_opr = Engine::Get()->NewOperator(
      exec_fun, use_vars, mutate_vars, FnProperty::kNormal,
      PROFILER_MESSAGE(closures_[nid].opr_name));
  cl->dirty = false;
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
      cl.exec = op_execs->value[nid];
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
}  // namespace

GraphExecutorV2::GraphExecutorV2(shared_ptr<const Graph> graph,
                                 const GraphExecutorV2::ExecState& fwd_state,
                                 const GraphExecutorV2::Config& config)
  :graph_ptr_(graph), config_(config),
   required_graph_ptr_attrs_(_InitRequiredGraphAttrs()) {
  _CheckAllAttrsExist(*graph_ptr_, required_graph_ptr_attrs_);
  AttachOps(fwd_state);
  SetupResources();
}

GraphExecutorV2::~GraphExecutorV2() {
  const auto& idx = graph_ptr_->indexed_graph();
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    Closure& cl = closures_.CopyOnWrite()->value[nid];
    if (cl.cached_opr) {
      Engine::Get()->DeleteOperator(cl.cached_opr);
    }
  }
}

void GraphExecutorV2::AttachOps(const GraphExecutorV2::ExecState& fwd_state) {
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
                   fwd_state.get(),
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

  // Schedule everything to run.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    // Normal mode
    const Node* node = idx[nid].source;
    if (node->is_variable()) {
      continue;
    } else if (node->is_graph()) {
      // TODO(minjie): subgraph node.
      LOG(FATAL) << "Not implemented.";
    } else {
      Closure& cl = closures_.CopyOnWrite()->value[nid];
      if (cl.skip_exec_node) continue;
      if (cl.exec->exec_type() == ExecType::kCrossDeviceCopy) {
        CHECK_EQ(node->inputs.size(), 1U);
        CHECK_EQ(cl.in_array.size(), 1U);
        CHECK_EQ(cl.out_array.size(), 1U);
        CopyFromTo(cl.in_array[0], &(cl.out_array[0]));
      } else if (cl.cached_opr != nullptr) {
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
      if (config_.dynamic_allocation) {
        ResetClosure(nid);
      }
    }
    // TODO(minjie): Monitor callbacks
    //if (monitor_callback_) {
      //ExecuteMonCallback(nid);
    //}
  }
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
void GraphExecutorV2::ResetClosure(uint32_t nid) {
  const auto& idx = graph_ptr_->indexed_graph();
  const Node* node = idx[nid].source;
  Closure& cl = closures_.CopyOnWrite()->value[nid];
  cl.in_array.clear();
  cl.in_array.resize(node->inputs.size());
  cl.out_array.clear();
  cl.out_array.resize(node->num_outputs());
  cl.requested.clear();  // TODO(minjie): how to reclaim the requested resources?
  cl.dirty = true;
  Engine::Get()->DeleteOperator(cl.cached_opr);
  cl.cached_opr = nullptr;
}
}  // namespace exec
}  // namespace mxnet
