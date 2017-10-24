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

using OpExecutor = pass::attach_op::OpExecutor;
using StorageRef = pass::plan_memory::StorageRef;

// Internal data structure used for creating
// engine operators.
struct OpNode {
  // The name of the operator
  std::string opr_name;
  // the context of the node
  Context ctx;
  // The executor
  // TODO(minjie): change this to shared_ptr<const OpExecutor>
  shared_ptr<OpExecutor> exec;
  // skip the execution of this node
  bool skip_exec_node{false};
  // cached operator handle
  Engine::OprHandle cached_opr{nullptr};

  vector<NDArray> in_array, out_array;
  // cached const vars, used for seg ops creation
  //vector<Engine::VarHandle> use_vars;
  // cached mutate vars, used for seg ops creation
  //vector<Engine::VarHandle> mutate_vars;

  bool dirty{true};
};

namespace {
vector<string> _InitRequiredGraphAttrs() {
  return {pass::shape::key,
          pass::dtype::key,
          pass::ctx::device_key,
          pass::ctx::ctx_key,
          pass::plan_memory::ref_key,
          pass::plan_memory::storage_key,
          pass::attach_op::key};
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

void SetupOpResourcesRec(const Graph& graph,
                            const Column<shared_ptr<OpExecutor>>* op_execs,
                            const Column<int>* vdevice,
                            map<Context, Resource>* cached_resources) {
  // TODO(minjie): delay alloc resources when config.dynamic_allocation is true.
  static auto& fresource = Op::GetAttr<FResourceRequest>("FResourceRequest");
  const auto& idx = graph.indexed_graph();
  const auto& context = graph.GetGlobalAttr<vector<Context>>(pass::ctx::ctx_key);
  // Resource allocation
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    if (node->is_variable()) {
      continue;
    }
    DLOG(INFO) << "Allocating operator resources for node#" << nid;
    if (node->is_graph()) {
      // TODO
    } else {
      if (fresource.count(node->op())) {
        // Allocate op resources.
        auto reqs = fresource[node->op()](node->attrs);
        auto& requested = op_execs->value[nid]->op_ctx.requested;
        requested.clear();
        // Get the resource of temporal space.
        for (const ResourceRequest& req : reqs) {
          const Context &ctx = context.at(vdevice->value[nid]);
          if (req.type == ResourceRequest::kTempSpace) {
            if (cached_resources->count(ctx) != 0) {
              requested.push_back(cached_resources->at(ctx));
            } else {
              Resource r = ResourceManager::Get()->Request(ctx, req);
              requested.push_back(r);
              cached_resources->insert({ctx,  r});
            }
          } else if (req.type == ResourceRequest::kRandom) {
            requested.push_back(ResourceManager::Get()->Request(ctx, req));
          } else {
            LOG(FATAL) << "resource type not yet supported";
          }
        }
      }
    }
  }
}

void SetupClosure(const Graph& graph,
                  uint32_t nid,
                  const Column<TShape>* shapes,
                  const Column<int>* dtypes,
                  const Column<StorageRef>* mem_plan,
                  const vector<NDArray>& data_pool,
                  OpNode* opnode) {
  using pass::plan_memory::kExternalStorageID;
  using pass::plan_memory::kNull;

  const auto& idx = graph.indexed_graph();
  const Node* node = idx[nid].source;
  auto exec = opnode->exec;

  CHECK_EQ(node->inputs.size(), opnode->in_array.size());
  CHECK_EQ(node->num_outputs(), opnode->out_array.size());

  // Input arrays.
  for (size_t i = 0; i < node->inputs.size(); ++i) {
    const uint32_t eid = idx.entry_id(node->inputs[i]);
    const StorageRef& store_ref = mem_plan->value[eid];
    const int storageid = mem_plan->value[eid].storage_id;
    if (storageid >= 0) {
      opnode->in_array[i] = data_pool[storageid].AsArray(
          shapes->value[eid], dtypes->value[eid]);
    } else if (storageid == kExternalStorageID) {
      CHECK(!opnode->in_array[i].is_none());
      const auto& array = opnode->in_array[i];
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
      opnode->out_array[i] = data_pool[storageid].AsArray(
          shapes->value[eid], dtypes->value[eid]);
    } else if (storageid == kExternalStorageID) {
      CHECK(!opnode->out_array[i].is_none());
      const auto& array = opnode->out_array[i];
      CHECK_EQ(shapes->value[eid], array.shape())
        << "Shape mismatch: expect " << shapes->value[eid]
        << " provided " << array.shape();
      CHECK_EQ(dtypes->value[eid], array.dtype())
        << "DType mismatch: expect " << dtypes->value[eid]
        << " provided " << array.dtype();
    } else if (storageid == kNull) {
      // TODO(minjie): Context for null output?
      opnode->out_array[i] = NDArray(shapes->value[eid],
          opnode->ctx, true, dtypes->value[eid]);
    } else {
      LOG(FATAL) << "Cannot create ndarray for entry#" << eid
        << " with provided shape=" << shapes->value[eid]
        << " & dtype=" << dtypes->value[eid];
    }
  }

  // Execution functor.
  const OpNode& capture = *opnode;
  auto exec_fun = [capture] (
      RunContext ctx, Engine::CallbackOnComplete on_complete) {
    auto exec = capture.exec;
    const bool is_async = exec->exec_type() == ExecType::kAsync;
    const bool is_gpu = capture.ctx.dev_mask() == gpu::kDevMask;
    if (is_async) {
      exec->op_ctx.async_on_complete = on_complete;
    }
    exec->Setup(capture.in_array, capture.out_array);
    exec->Run(ctx);
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
  for (size_t i = 0; i < opnode->in_array.size(); ++i) {
    use_vars.push_back(opnode->in_array[i].var());
  }
  for (auto& r : exec->op_ctx.requested) {
    mutate_vars.push_back(r.var);
  }
  for (size_t i = 0; i < opnode->out_array.size(); ++i) {
    if (!opnode->out_array[i].is_none()) {
      mutate_vars.push_back(opnode->out_array[i].var());
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
  opnode->cached_opr = Engine::Get()->NewOperator(
      exec_fun, use_vars, mutate_vars, FnProperty::kNormal,
      PROFILER_MESSAGE(op_nodes_[nid].opr_name));
  opnode->dirty = false;
}

void InitOpClosureRec(const Graph& graph,
                      const Column<shared_ptr<OpExecutor>>* op_execs,
                      const Column<int>* vdevice,
                      const Column<StorageRef>* mem_plan,
                      Column<OpNode>* op_nodes) {
  using pass::plan_memory::kExternalStorageID;
  using pass::plan_memory::kNull;

  const auto& idx = graph.indexed_graph();
  const auto& context = graph.GetGlobalAttr<vector<Context>>(pass::ctx::ctx_key);
  // setup the array and requirements.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    OpNode& opnode = op_nodes->value[nid];
    const Node* node = idx[nid].source;
    if (node->is_variable()) {
      continue;
    }
    opnode.opr_name = node->op()->name;
    if (node->is_graph()) {
      // TODO
    } else {
      opnode.in_array.resize(node->inputs.size());
      opnode.out_array.resize(node->num_outputs());
      opnode.exec = op_execs->value[nid];
      opnode.ctx = context.at(vdevice->value[nid]);
      // TODO(minjie): move below to attach op exec pass.
      for (size_t i = 0; i < node->num_outputs(); ++i) {
        const uint32_t eid = idx.entry_id(nid, i);
        const StorageRef& store_ref = mem_plan->value[eid];
        const int storageid = mem_plan->value[eid].storage_id;
        // Output request.
        if (false) {
          // TODO(minjie): addto inplace optimization.
        } else if (store_ref.inplace_index >= 0) {
          opnode.exec->req.push_back(kWriteInplace);
        } else if (storageid == kNull) {
          // TODO(minjie): need double-check.
          opnode.exec->req.push_back(kNullOp);
        } else {
          opnode.exec->req.push_back(kWriteTo);
        }
      }
    }
  }
}
}  // namespace

GraphExecutorV2::GraphExecutorV2(shared_ptr<const Graph> graph,
                                 const GraphExecutorV2::Config& config)
  :graph_ptr_(graph), config_(config),
   required_graph_ptr_attrs_(_InitRequiredGraphAttrs()) {
  _CheckAllAttrsExist(*graph_ptr_, required_graph_ptr_attrs_);
  SetupResources();
}

GraphExecutorV2::~GraphExecutorV2() {
  const auto& idx = graph_ptr_->indexed_graph();
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    OpNode& opnode = op_nodes_.CopyOnWrite()->value[nid];
    if (opnode.cached_opr) {
      Engine::Get()->DeleteOperator(opnode.cached_opr);
    }
  }
}

void GraphExecutorV2::Run(const vector<NDArray>& arguments,
                          vector<NDArray>* results,
                          const RunOption& option) {
  const auto& idx = graph_ptr_->indexed_graph();
  const auto& op_execs = graph_ptr_->node_attrs.GetColumn<shared_ptr<OpExecutor>>(
      pass::attach_op::key);
  const auto& device = graph_ptr_->node_attrs.GetColumn<int>(pass::ctx::device_key);
  const auto& context = graph_ptr_->GetGlobalAttr<vector<Context>>(pass::ctx::ctx_key);
  const auto* shapes = graph_ptr_->entry_attrs.GetColumn<TShape>(pass::shape::key).get();
  const auto* dtypes = graph_ptr_->entry_attrs.GetColumn<int>(pass::dtype::key).get();
  const auto* mem_plan = graph_ptr_->entry_attrs.GetColumn<StorageRef>(
      pass::plan_memory::ref_key).get();

  LOG(INFO) << "Graph execution starts.";

  // Feed arguments.
  LOG(INFO) << "Feeding argument ndarrays.";
  CHECK_EQ(arguments.size(), idx.input_nodes().size());
  for (size_t i = 0; i < arguments.size(); ++i) {
    FeedArgArray(arguments[i], i);
  }
  // Feed results.
  CHECK_NOTNULL(results);
  if (!results->empty()) {
    LOG(INFO) << "Feeding result ndarrays.";
    // Result storage is provided. Feed the result array
    // as the output of the related operator.
    CHECK_EQ(results->size(), graph_ptr_->outputs.size());
    for (size_t i = 0; i < results->size(); ++i) {
      FeedRstArray((*results)[i], i);
    }
  }

  // Check all operator closures and re-make those dirty ones.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    if (node->is_variable()) {
      continue;
    } else if (node->is_graph()) {
      // TODO(minjie): subgraph node.
    } else {
      OpNode& opnode = op_nodes_.CopyOnWrite()->value[nid];
      opnode.exec->op_ctx.is_train = option.is_train;
      if (opnode.dirty) {
        // Create operator closure for dirty op.
        DLOG(INFO) << "Setup closure for node#" << nid << ": " << node->attrs.name;
        const auto& op_exec = op_execs->value[nid];
        const Context& ctx = context.at(device->value[nid]);
        SetupClosure(*graph_ptr_, nid, shapes, dtypes, mem_plan, data_entries_,
                     &opnode);
      }
    }
  }

  if (results->empty()) {
    LOG(INFO) << "Fetching result ndarrays.";
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
    } else {
      OpNode& opnode = op_nodes_.CopyOnWrite()->value[nid];
      if (opnode.skip_exec_node) continue;
      if (opnode.exec->exec_type() == ExecType::kCrossDeviceCopy) {
        CHECK_EQ(node->inputs.size(), 1U);
        CHECK_EQ(opnode.in_array.size(), 1U);
        CHECK_EQ(opnode.out_array.size(), 1U);
        CopyFromTo(opnode.in_array[0], &(opnode.out_array[0]));
      } else if (opnode.cached_opr != nullptr) {
#if MXNET_USE_PROFILER
        bool profiling = engine::Profiler::Get()->GetState() == engine::Profiler::kRunning;
#else
        bool profiling = false;
#endif
        DLOG(INFO) << "Push Node#" << nid << " " << node->attrs.name;
        Engine::Get()->Push(opnode.cached_opr, opnode.ctx, 0, profiling);
      } else {
        LOG(FATAL) << "Cannot execute Node#" << nid << " " << node->attrs.name;
      }
      if (config_.dynamic_allocation) {
        ResetOpNode(nid);
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
    op_nodes_.CopyOnWrite()->value[nid].in_array[i] = array;
    op_nodes_.CopyOnWrite()->value[nid].dirty = true;
  }
}

void GraphExecutorV2::FeedRstArray(const NDArray& array, size_t i) {
  const auto& idx = graph_ptr_->indexed_graph();
  const auto& op_execs = graph_ptr_->node_attrs.GetColumn<shared_ptr<OpExecutor>>(
      pass::attach_op::key);
  const NodeEntry& outent = graph_ptr_->outputs[i];
  const uint32_t nid = idx.node_id(outent.node.get());
  op_nodes_.CopyOnWrite()->value[nid].out_array[outent.index] = array;
  op_nodes_.CopyOnWrite()->value[nid].dirty = true;
}

const NDArray& GraphExecutorV2::FetchRstArray(size_t i) {
  const auto& idx = graph_ptr_->indexed_graph();
  const NodeEntry& outent = graph_ptr_->outputs[i];
  const uint32_t nid = idx.node_id(outent.node.get());
  return op_nodes_->value[nid].out_array[outent.index];
}

void GraphExecutorV2::SetupResources() {
  LOG(INFO) << "Pre-allocating all data entries.";
  SetupDataEntries();
  LOG(INFO) << "Pre-allocating all operator resources.";
  SetupOpResources();
}

const vector<string>& GraphExecutorV2::RequiredGraphAttrs() const {
  return required_graph_ptr_attrs_;
}

void GraphExecutorV2::SetupOpResources() {
  using pass::plan_memory::StorageRef;
  // Use global resource pool for each executor for now.
  std::map<Context, Resource> cached_temp;
  const auto* op_execs = graph_ptr_->node_attrs.GetColumn<shared_ptr<OpExecutor>>(
      pass::attach_op::key).get();
  const Column<int>* device = graph_ptr_->node_attrs.GetColumn<int>(pass::ctx::device_key).get();
  const auto* shapes = graph_ptr_->entry_attrs.GetColumn<TShape>(pass::shape::key).get();
  const auto* dtypes = graph_ptr_->entry_attrs.GetColumn<int>(pass::dtype::key).get();
  const auto* mem_plan = graph_ptr_->entry_attrs.GetColumn<StorageRef>(
      pass::plan_memory::ref_key).get();
  SetupOpResourcesRec(*graph_ptr_, op_execs, device, &cached_temp);
  op_nodes_ = graph_ptr_->CreateNodeColumn<OpNode>();
  InitOpClosureRec(*graph_ptr_, op_execs, device, mem_plan, op_nodes_.CopyOnWrite());
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
  LOG(INFO) << "Reset all data entries to allow dynamic allocation.";
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

void GraphExecutorV2::ResetOpNode(uint32_t nid) {
  const auto& idx = graph_ptr_->indexed_graph();
  const Node* node = idx[nid].source;
  OpNode& opnode = op_nodes_.CopyOnWrite()->value[nid];
  opnode.in_array.clear();
  opnode.in_array.resize(node->inputs.size());
  opnode.out_array.clear();
  opnode.out_array.resize(node->num_outputs());
  opnode.dirty = true;
  Engine::Get()->DeleteOperator(opnode.cached_opr);
  opnode.cached_opr = nullptr;
}

}  // namespace exec
}  // namespace mxnet
