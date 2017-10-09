/*!
 * Copyright (c) 2016 by Contributors
 * \file graph_executor_v2.cc
 * \brief Executor to execute the computation graph.
 */
#include "./graph_executor_v2.h"

using namespace std;
using namespace nnvm;

namespace mxnet {
namespace exec {
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
}  // namespace

using OpExecutor = pass::attach_op::OpExecutor;
using StorageRef = pass::plan_memory::StorageRef;

// Information about operational node
struct OpNode {
  // The name of the operator
  const char* opr_name = "";
  // the context of the node
  Context ctx;
  // The executor
  shared_ptr<OpExecutor> exec;
  // skip the execution of this node
  bool skip_exec_node{false};
  // cached operator handle
  Engine::OprHandle cached_opr{nullptr};
  // cached const vars, used for seg ops creation
  vector<Engine::VarHandle> use_vars;
  // cached mutate vars, used for seg ops creation
  vector<Engine::VarHandle> mutate_vars;
};

struct OpEntry {
  TShape shape;
  int dtype;
  pass::plan_memory::StorageRef storage_ref;
};

GraphExecutorV2::GraphExecutorV2(const Graph& graph,
                                 const GraphExecutorV2::Config& config)
  :graph_(graph), config_(config),
   required_graph_attrs_(_InitRequiredGraphAttrs()) {
  _CheckAllAttrsExist(graph_, required_graph_attrs_);
  AllocateResources();
}

void GraphExecutorV2::Run(const vector<NDArray>& inputs,
                          const vector<NDArray>& outputs,
                          const RunOption& option) {
  // TODO
}

void GraphExecutorV2::AllocateResources() {
  LOG(INFO) << "Pre-allocating all data entries.";
  vector<NDArray> data_entries;
  AllocateDataEntries(&data_entries);
  LOG(INFO) << "Pre-allocating all operator resources.";
  AllocateOpResources(data_entries);
}

void GraphExecutorV2::ReleaseResources() {
  ReleaseDataEntries();
  ReleaseOpResources();
}

const vector<string>& GraphExecutorV2::RequiredGraphAttrs() const {
  return required_graph_attrs_;
}

void AllocateOpResourcesRec(const Graph& graph,
                            const Column<shared_ptr<OpExecutor>>* op_execs,
                            const Column<int>* vdevice,
                            map<Context, Resource>* cached_resources) {
  // TODO(minjie): delay alloc resources when config.dynamic_allocation is true.
  static auto& fresource = Op::GetAttr<FResourceRequest>("FResourceRequest");
  const auto& idx = graph.indexed_graph();
  const auto& context = graph.GetGlobalAttr<vector<Context>>(pass::ctx::ctx_key);
  // Resource allocation
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    LOG(INFO) << "Allocating operator resources for node#" << nid;
    const Node* node = idx[nid].source;
    if (node->is_variable()) {
      continue;
    }
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

void CreateOpNode(const Graph& graph,
                  uint32_t nid,
                  shared_ptr<OpExecutor> exec,
                  const Context& ctx,
                  const vector<OpEntry>& in_entries,
                  const vector<OpEntry>& out_entries,
                  const vector<NDArray>& data_pool,
                  OpNode* opnode) {
  using pass::plan_memory::kNull;
  const auto& idx = graph.indexed_graph();
  const Node* node = idx[nid].source;

  CHECK_EQ(node->inputs.size(), in_entries.size());
  CHECK_EQ(node->num_outputs(), out_entries.size());

  const bool is_async = exec->exec_type() == Operator::kAsync;
  const bool is_gpu = ctx.dev_mask() == gpu::kDevMask;

  // Attach input data entries (ndarrays).
  for (size_t i = 0; i < node->inputs.size(); ++i) {
    const OpEntry& ent = in_entries[i];
    const auto& nd = data_pool[ent.storage_ref.storage_id].AsArray(ent.shape, ent.dtype);
    exec->SetInput(nd, i);
  }
  // Attach output data entries (ndarrays).
  for (uint32_t i = 0; i < node->num_outputs(); ++i) {
    const OpEntry& ent = out_entries[i];
    const auto& nd = data_pool[ent.storage_ref.storage_id].AsArray(ent.shape, ent.dtype);
    exec->SetOutput(nd, i);
    if (false) {
      // TODO(minjie): addto inplace optimization.
    } else if (ent.storage_ref.inplace_index >= 0) {
      exec->req.push_back(kWriteInplace);
    } else if (ent.storage_ref.storage_id == kNull) {
      // TODO(minjie): need double-check.
      exec->req.push_back(kNullOp);
    } else {
      exec->req.push_back(kWriteTo);
    }
  }
  // Execution functor.
  auto exec_fun = [exec, is_async, is_gpu] (
      RunContext ctx, Engine::CallbackOnComplete on_complete) {
    if (is_async) {
      exec->op_ctx.async_on_complete = on_complete;
    }
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
  for (size_t i = 0; i < node->inputs.size(); ++i) {
    use_vars.push_back(exec->GetInput(i).var());
  }
  for (auto& r : exec->op_ctx.requested) {
    mutate_vars.push_back(r.var);
  }
  for (size_t i = 0; i < node->num_outputs(); ++i) {
    mutate_vars.push_back(exec->GetOutput(i).var());
  }
  // Remove dupliated vars.
  Engine::Get()->DeduplicateVarHandle(&use_vars, &mutate_vars);
  LOG(INFO) << "Create OpNode for node#" << nid << ": " << node->attrs.name;
  // Save everything.
  opnode->exec = exec;
  opnode->ctx = ctx;
  opnode->cached_opr = Engine::Get()->NewOperator(
      exec_fun, use_vars, mutate_vars, FnProperty::kNormal,
      PROFILER_MESSAGE(op_nodes_[nid].opr_name));
  opnode->mutate_vars = mutate_vars;
  opnode->use_vars = use_vars;
}


void CreateCachedOpsRec(const Graph& graph,
                        const Column<shared_ptr<OpExecutor>>* op_execs,
                        const Column<TShape>* shapes,
                        const Column<int>* dtypes,
                        const Column<int>* vdevice,
                        const Column<StorageRef>* mem_plan,
                        const vector<NDArray>& data_pool,
                        Column<OpNode>* op_nodes) {
  const auto& idx = graph.indexed_graph();
  const auto& context = graph.GetGlobalAttr<vector<Context>>(pass::ctx::ctx_key);
  static auto f_is_valid_ent = [] (const OpEntry& ent) {
      return ent.shape.ndim() != 0 && ent.dtype != -1 && ent.storage_ref.storage_id >= 0;
    };
  // setup the array and requirements.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    if (node->is_variable()) {
      continue;
    }
#if MXNET_USE_PROFILER
    op_nodes->value[nid].opr_name = node->op()->name.c_str();
#else
    op_nodes->value[nid].opr_name = nullptr;
#endif
    if (node->is_graph()) {
      // TODO
    } else {
      vector<OpEntry> in_entries, out_entries;
      for (size_t i = 0; i < node->inputs.size(); ++i) {
        const uint32_t eid = idx.entry_id(node->inputs[i]);
        in_entries.emplace_back(
            OpEntry{shapes->value[eid], dtypes->value[eid], mem_plan->value[eid]});
      }
      for (size_t i = 0; i < node->num_outputs(); ++i) {
        const uint32_t eid = idx.entry_id(nid, i);
        out_entries.emplace_back(
            OpEntry{shapes->value[eid], dtypes->value[eid], mem_plan->value[eid]});
      }
      if (std::all_of(in_entries.begin(), in_entries.end(), f_is_valid_ent)
          && std::all_of(out_entries.begin(), out_entries.end(), f_is_valid_ent)) {
        CreateOpNode(graph, nid, 
                     op_execs->value[nid],
                     context.at(vdevice->value[nid]),
                     in_entries, out_entries, data_pool, &op_nodes->value[nid]);
      }
    }
  }
}

void GraphExecutorV2::AllocateOpResources(const vector<NDArray>& data_entries) {
  using pass::plan_memory::StorageRef;
  // Use global resource pool for each executor for now.
  std::map<Context, Resource> cached_temp;
  const auto* op_execs = graph_.node_attrs.GetColumn<shared_ptr<OpExecutor>>(
      pass::attach_op::key).get();
  const Column<int>* device = graph_.node_attrs.GetColumn<int>(pass::ctx::device_key).get();
  const auto* shapes = graph_.entry_attrs.GetColumn<TShape>(pass::shape::key).get();
  const auto* dtypes = graph_.entry_attrs.GetColumn<int>(pass::dtype::key).get();
  const auto* mem_plan = graph_.entry_attrs.GetColumn<StorageRef>(pass::plan_memory::ref_key).get();
  AllocateOpResourcesRec(graph_, op_execs, device, &cached_temp);
  op_nodes_ = graph_.CreateNodeColumn<OpNode>();
  LOG(INFO) << "Creating cached operator for execution.";
  CreateCachedOpsRec(graph_, op_execs, shapes, dtypes, device, mem_plan,
                     data_entries, op_nodes_.CopyOnWrite());
}

void GraphExecutorV2::AllocateDataEntries(vector<NDArray>* data_entries) {
  using pass::plan_memory::Storage;
  using pass::plan_memory::StorageRef;
  using pass::plan_memory::storage_key;
  using pass::ctx::ctx_key;
  const auto& storage = graph_.GetGlobalAttr<vector<Storage>>(storage_key);
  const auto& context = graph_.GetGlobalAttr<vector<Context>>(ctx_key);
  const bool delay_alloc = !config_.dynamic_allocation;
  data_entries->reserve(storage.size());
  for (const Storage& st : storage) {
    // XXX(minjie): Use float array for all dtype allocation. Should change
    // to other dtype in the future.
    const size_t nword = (st.max_bytes + 3) / 4;
    CHECK_LE(nword, std::numeric_limits<nnvm::dim_t>::max());
    TShape shape{static_cast<nnvm::dim_t>(nword)};
    LOG(INFO) << "Allocate data entry#" << data_entries->size()
              << " size=" << nword << " floats.";
    data_entries->push_back(NDArray(shape, context.at(st.device_id), delay_alloc));
  }
}

void GraphExecutorV2::ReleaseOpResources() {
  // TODO
  LOG(FATAL) << "Not implemented.";
}

void GraphExecutorV2::ReleaseDataEntries() {
  // TODO
  LOG(FATAL) << "Not implemented.";
}

}  // namespace exec
}  // namespace mxnet
