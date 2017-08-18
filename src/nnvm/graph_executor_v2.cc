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
          //pass::ctx::key,
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
  const char* opr_name;
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

GraphExecutorV2::GraphExecutorV2(const Graph& graph, const Context& default_ctx)
  :graph_(graph), default_context_(default_ctx),
   required_graph_attrs_(_InitRequiredGraphAttrs()) {
  _CheckAllAttrsExist(graph_, required_graph_attrs_);
}

void GraphExecutorV2::Eval(const vector<NDArray>& inputs,
                           const vector<NDArray>& outputs,
                           bool is_train) {
  CHECK(resource_allocated_)
    << "Resources must be allocated before evaluating the graph.";
}

void GraphExecutorV2::AllocateResources() {
  if (resource_allocated_) {
    return;
  }
  resource_allocated_ = true;
  AllocateOpResources();
  AllocateDataEntries();
}

void GraphExecutorV2::ReleaseResources() {
  if (!resource_allocated_) {
    return;
  }
  ReleaseOpResources();
  ReleaseDataEntries();
  resource_allocated_ = false;
}

const vector<string>& GraphExecutorV2::RequiredGraphAttrs() const {
  return required_graph_attrs_;
}

void AllocateOpResourcesRec(const Graph& graph,
                            const Column<shared_ptr<OpExecutor>>* op_execs,
                            const Column<Context>* vctx,
                            const Context& default_ctx,
                            map<Context, Resource>* cached_resources) {
  static auto& fresource = Op::GetAttr<FResourceRequest>("FResourceRequest");
  const auto& idx = graph.indexed_graph();
  // Resource allocation
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
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
          const Context &ctx = (vctx != nullptr)? vctx->value[nid] : default_ctx;
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
                  const Column<StorageRef>* mem_plan,
                  const Column<NDArray>* data_entry,
                  OpNode* opnode) {
  using pass::plan_memory::kNull;
  const auto& idx = graph.indexed_graph();
  const Node* node = idx[nid].source;

  const bool is_async = exec->exec_type() == Operator::kAsync;
  const bool is_gpu = ctx.dev_mask() == gpu::kDevMask;
  CHECK_EQ(exec->in_array.size(), 0U);
  CHECK_EQ(exec->out_array.size(), 0U);
  // Attach input data entries (ndarrays).
  for (const auto& e : node->inputs) {
    const uint32_t eid = idx.entry_id(e);
    exec->in_array.push_back(data_entry->value[eid]);
  }
  // Attach output data entries (ndarrays).
  for (uint32_t i = 0; i < node->num_outputs(); ++i) {
    const uint32_t eid = idx.entry_id(nid, i);
    exec->out_array.push_back(data_entry->value[eid]);
    if (false) {
      // TODO(minjie): addto inplace optimization.
    } else if (mem_plan->value[eid].inplace_index >= 0) {
      exec->req.push_back(kWriteInplace);
    } else if (mem_plan->value[eid].storage_id == kNull) {
      // TODO(minjie): need double-check.
      exec->req.push_back(kNullOp);
    } else {
      exec->req.push_back(kWriteTo);
    }
  }
  // Setup variables
  std::vector<Engine::VarHandle> use_vars, mutate_vars;
  for (size_t i = 0; i < exec->in_array.size(); ++i) {
    auto& nd = exec->in_array[i];
    use_vars.push_back(nd.var());
  }
  for (auto& r : exec->op_ctx.requested) {
    mutate_vars.push_back(r.var);
  }
  for (auto& nd : exec->out_array) {
    mutate_vars.push_back(nd.var());
  }
  // Remove dupliated vars.
  Engine::Get()->DeduplicateVarHandle(&use_vars, &mutate_vars);
  // all vars include both mutate vars and use vars
  std::vector<Engine::VarHandle> all_vars(use_vars);
  std::copy(mutate_vars.begin(), mutate_vars.end(),
      std::inserter(all_vars, all_vars.end()));
  // setup exec vars
  Engine::Get()->PushSync([exec](RunContext rctx) {
      exec->Setup();
      }, Context::CPU(), {}, all_vars, FnProperty::kNormal, 0,
      PROFILER_MESSAGE("SetupExec"));
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
                        const Column<Context>* vctx,
                        const Column<StorageRef>* mem_plan,
                        const Column<NDArray>* data_entry,
                        const Context& default_ctx,
                        Column<OpNode>* op_nodes) {
  const auto& idx = graph.indexed_graph();
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
      // TODO(minjie): all operators that use inputs and generate outputs should
      // not be created since the engine variables are unknown at the moment.
      CreateOpNode(graph, nid, 
                   op_execs->value[nid],
                   (vctx != nullptr)? vctx->value[nid] : default_ctx,
                   mem_plan, data_entry, &op_nodes->value[nid]);
    }
  }
}

void GraphExecutorV2::AllocateOpResources() {
  // TODO
  // Use global resource pool for each executor for now.
}

void GraphExecutorV2::AllocateDataEntries() {
  using pass::plan_memory::Storage;
  using pass::plan_memory::StorageRef;
  using pass::plan_memory::storage_key;
  const auto& storage = graph_.GetGlobalAttr<vector<Storage>>(storage_key);
  data_pool_.reserve(storage.size());
  for (const Storage& st : storage) {
    // XXX(minjie): Use float array for all dtype allocation. Should change
    // to other dtype in the future.
    const size_t nword = (st.max_bytes + 3) / 4;
    CHECK_LE(nword, std::numeric_limits<nnvm::dim_t>::max());
    TShape shape{static_cast<nnvm::dim_t>(nword)};
    // TODO(minjie): only use default ctx.
    data_pool_.push_back(NDArray(shape, default_context_));
  }
}

void GraphExecutorV2::ReleaseOpResources() {
  // TODO
}

void GraphExecutorV2::ReleaseDataEntries() {
  // TODO
}

}  // namespace exec
}  // namespace mxnet
