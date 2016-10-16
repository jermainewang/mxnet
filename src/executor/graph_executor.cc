/*!
 *  Copyright (c) 2015 by Contributors
 * \file graph_executor.cc
 * \brief graph executor
 */
#include <mxnet/base.h>
#include <nnvm/graph.h>
#include <nnvm/pass_functions.h>
#include <vector>
#include <algorithm>
#include <sys/time.h>
#include <fstream>

#include "./exec_pass.h"
#include "./graph_executor.h"

namespace mxnet {
namespace exec {
namespace {
void SortAndUnique(std::vector<Engine::VarHandle>* vars) {
  std::sort(vars->begin(), vars->end());
  vars->resize(std::unique(vars->begin(), vars->end()) - vars->begin());
};
void EnableP2P(const std::vector<Context>& devs) {
#if MXNET_USE_CUDA
  std::vector<int> gpus;
  for (const auto& d : devs) {
    if (d.dev_mask() == gpu::kDevMask) {
      gpus.push_back(d.dev_id);
    }
  }
  int n = static_cast<int>(gpus.size());
  if (n <= 8) {
    int enabled = 0;
    std::vector<int> p2p(n*n);
    for (int i = 0; i < n; ++i) {
      cudaSetDevice(gpus[i]);
      for (int j = 0; j < n; j++) {
        int access;
        cudaDeviceCanAccessPeer(&access, gpus[i], gpus[j]);
        if (access) {
          cudaError_t e = cudaDeviceEnablePeerAccess(gpus[j], 0);
          if (e == cudaSuccess) {
            ++enabled;
            p2p[i*n+j] = 1;
          }
        }
      }
    }
    if (enabled != n*(n-1)) {
      // print warning info if not fully enabled
      LOG(WARNING) << "only " << enabled <<  " out of "
                   << n*(n-1) << " GPU pairs are enabled direct access. "
                   << "It may affect the performance. "
                   << "You can set MXNET_ENABLE_GPU_P2P=0 to turn it off";
    }
  } else {
    CHECK(n == 16);
    int succ = 0;
    for (int i = 0; i < 8; ++i) {
      // Group 1:
      cudaSetDevice(i);
      for (int j = 0; j < 8; ++j) {
        if (i == j) {
          continue;
        }
        cudaDeviceCanAccessPeer(&succ, i, j);
        CHECK(succ);
        cudaDeviceEnablePeerAccess(i, j);
      }
      // Jump link
      cudaDeviceCanAccessPeer(&succ, i, i + 8);
      CHECK(succ);
      cudaDeviceEnablePeerAccess(i, i + 8);
      // Group 2:
      cudaSetDevice(i + 8);
      for (int j = 0; j < 8; ++j) {
        if (i == j) {
          continue;
        }
        cudaDeviceCanAccessPeer(&succ, i + 8, j + 8);
        CHECK(succ);
        cudaDeviceEnablePeerAccess(i + 8, j + 8);
      }
      cudaDeviceCanAccessPeer(&succ, i + 8, i);
      CHECK(succ);
      cudaDeviceEnablePeerAccess(i + 8, i);
    }
  }
  std::string access(n, '.');
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      int succ;
      cudaDeviceCanAccessPeer(&succ, gpus[i], gpus[j]);
      access[j] = succ ? 'v' : '.';
    }
    LOG(INFO) << access;
  }
#endif
}
}  // namespace
GraphExecutor::~GraphExecutor() {
  for (auto& n : op_nodes_) {
    if (n.cached_opr != nullptr) {
      Engine::Get()->DeleteOperator(n.cached_opr);
    }
  }
}

void GraphExecutor::Forward(bool is_train) {
//  static int calls = 0;
//  std::ofstream fout("trace_" + std::to_string(calls) + ".txt");
//  for (const auto& rec : *trace_records_) {
//    fout << std::get<0>(rec) << " " << std::get<1>(rec)
//      << " " << std::get<2>(rec) << std::endl;
//  }
//  ++calls;
//  trace_records_->clear();
  RunOps(is_train, 0, num_forward_nodes_);
}

void GraphExecutor::PartialForward(bool is_train, int step, int *step_left) {
  size_t sstep = static_cast<size_t>(step);
  if (sstep >= num_forward_nodes_) {
    *step_left = 0; return;
  }
  RunOps(is_train, sstep, sstep + 1);
  *step_left = static_cast<int>(num_forward_nodes_ - sstep - 1);
}

void GraphExecutor::Backward(const std::vector<NDArray>& head_grads) {
  const auto& idx = graph_.indexed_graph();
  if (num_forward_inputs_ != idx.input_nodes().size()) {
    for (size_t i = 0; i < head_grad_array_.size(); ++i) {
      if (!head_grad_array_[i].is_none()) {
        CHECK(i < head_grads.size() && !head_grads[i].is_none())
            << "Because the last operator is not Loss function, "
            << "head_gradient is required in calling backward.";
        CopyFromTo(head_grads[i], &(head_grad_array_[i]));
      }
    }
  }
  RunOps(true, num_forward_nodes_, idx.num_nodes());
}

void GraphExecutor::Print(std::ostream &os) const {  // NOLINT(*)
  nnvm::Symbol s; s.outputs = graph_.outputs;
  s.Print(os);
  // message to be backward compatible with the memonger
  size_t total_bytes = graph_.GetAttr<size_t>("storage_allocated_bytes");
  os << "Total " << (total_bytes >> 20UL) <<" MB allocated\n";
  os << "Total " << 11 << " TempSpace resource requested\n";
}

void GraphExecutor::SetMonitorCallback(const MonitorCallback& callback) {
  CHECK(callback) << "invalid callback";
  monitor_callback_ = callback;
}

const std::vector<NDArray>& GraphExecutor::outputs() const {
  return output_arrays_;
}

nnvm::NodeEntry AggregateGradient(std::vector<nnvm::NodeEntry>&& v) {
  using nnvm::Op;
  using nnvm::Node;
  if (v.size() == 1) {
    return std::move(v[0]);
  } else if (v.size() == 0) {
    // TODO(tqchen) should be zero node
    nnvm::NodePtr ng = Node::Create();
    ng->attrs.op = Op::Get("_NoGradient");
    ng->attrs.name = "NoGradient";
    return nnvm::NodeEntry{ng, 0, 0};
  } else {
    nnvm::NodePtr sum_node = Node::Create();
    sum_node->attrs.op = Op::Get("ElementWiseSum");
    sum_node->attrs.name = "sum_grad";
    sum_node->attrs.dict["num_args"] = std::to_string(v.size());
    sum_node->attrs.op->attr_parser(&(sum_node->attrs));
    sum_node->inputs = std::move(v);
    return nnvm::NodeEntry{sum_node, 0, 0};
  }
}

template<typename ValueType>
inline ValueType get_node_attr(
    const nnvm::Node& node,
    const std::string& key, ValueType default_value) {
  auto it = node.attrs.dict.find(key);
  if (it == node.attrs.dict.end()) {
    return default_value;
  } else {
    ValueType ret;
    dmlc::parameter::FieldEntry<ValueType> e;
    e.Init(key, &ret, ret);
    e.Set(&ret, it->second);
    return ret;
  }
}

nnvm::Graph GraphExecutor::InitFullGraph(
    nnvm::Symbol symbol,
    const std::vector<OpReqType>& grad_req_type,
    const std::vector<NDArray>& arg_grad_store) {
  using nnvm::NodePtr;
  using nnvm::NodeEntry;
  // initial information
  num_forward_outputs_ = symbol.outputs.size();
  num_forward_inputs_ = symbol.ListInputs(nnvm::Symbol::kAll).size();

  nnvm::Graph g;
  g.outputs = symbol.outputs;

  bool need_grad = false;
  for (OpReqType req : grad_req_type) {
    if (req != kNullOp) {
      need_grad = true;
      break;
    }
  }
  if (!need_grad) {
    // No gradient is needed, no need to generate backward graph.
    return g;
  }
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    NodePtr node = nnvm::Node::Create();
    node->attrs.name = "__head_grad_" + std::to_string(i);
    head_grad_entry_.emplace_back(NodeEntry{node, 0, 0});
    head_grad_map_[node.get()] = i;
  }
  std::vector<NodePtr> args = symbol.ListInputs(nnvm::Symbol::kReadOnlyArgs);
  std::vector<NodeEntry> xs;
  for (size_t i = 0; i < grad_req_type.size(); ++i) {
    if (grad_req_type[i] != kNullOp) {
      grad_store_.emplace_back(grad_req_type[i], arg_grad_store[i]);
      xs.emplace_back(NodeEntry{args[i], 0, 0});
    }
  }

  int do_mirror = dmlc::GetEnv("MXNET_BACKWARD_DO_MIRROR", 0);
  auto need_mirror = [do_mirror](const nnvm::Node& node) -> int {
    if (node.is_variable()) return 0;
    const std::string& type = node.attrs.op->name;
    if (type == "Dropout") return false;
    if (get_node_attr(node, "force_mirroring", false)) return true;
    if (do_mirror == 0) return false;
    if (type == "Convolution") return false;
    if (type == "FullyConnected") return false;
    if (type == "Concat") return false;
    if (type == "SoftmaxOutput") return false;
    if (type == "CuDNNBatchNorm") return false;
    return true;
  };

  nnvm::Graph g_grad = nnvm::pass::Gradient(
      g, symbol.outputs, xs, head_grad_entry_,
      AggregateGradient, need_mirror);
  return g_grad;
}

// pass to assign context to the graph
Graph AssignContext(Graph g,
                    const Context& default_ctx,
                    const std::map<std::string, Context>& ctx_map,
                    const std::vector<NDArray>& in_args,
                    const std::vector<std::pair<OpReqType, NDArray> >& grad_store,
                    const std::vector<NDArray>& aux_states,
                    size_t num_forward_inputs,
                    size_t num_forward_outputs) {
  const auto& idx = g.indexed_graph();
  const auto& mutable_nodes = idx.mutable_input_nodes();
  // If no context other than default context is given, then all the nodes
  // are placed on the default context.
  if (ctx_map.size() == 0) {
    g.attrs["context"] = std::make_shared<nnvm::any>(
        ContextVector(idx.num_nodes(), default_ctx));
    return g;
  }

  // Map from context object to a unique id.
  std::map<Context, int> ctx2id;
  // Map from the unique id to the context object.
  std::vector<Context> ctx_list;
  // Map from each node id to its assigned device id.
  nnvm::DeviceVector device(idx.num_nodes(), -1);
  // Map from group name to device id.
  nnvm::DeviceAssignMap device_map;

  for (auto &kv : ctx_map) {
    if (ctx2id.count(kv.second) == 0) {
      ctx2id[kv.second] = static_cast<int>(ctx_list.size());
      ctx_list.push_back(kv.second);
    }
    device_map[kv.first] = ctx2id.at(kv.second);
  }
  
  // Enable P2P connection.
  EnableP2P(ctx_list);

  // Place input and output entries of the graph manually on the device
  // specified by users during creation.
  size_t arg_top = 0, aux_top = 0;
  for (size_t i = 0; i < num_forward_inputs; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    Context ctx;
    if (mutable_nodes.count(nid)) {
      CHECK_LT(aux_top, aux_states.size());
      ctx = aux_states[aux_top].ctx();
      ++aux_top;
    } else {
      CHECK_LT(arg_top, in_args.size());
      ctx = in_args[arg_top].ctx();
      ++arg_top;
    }
    if (ctx2id.count(ctx) == 0) {
      ctx2id[ctx] = static_cast<int>(ctx_list.size());
      ctx_list.push_back(ctx);
    }
    device[nid] = ctx2id.at(ctx);
  }
  for (size_t i = num_forward_outputs; i < g.outputs.size(); ++i) {
    const uint32_t nid = idx.outputs()[i].node_id;
    Context ctx = grad_store[i - num_forward_outputs].second.ctx();
    if (ctx2id.count(ctx) == 0) {
      ctx2id[ctx] = static_cast<int>(ctx_list.size());
      ctx_list.push_back(ctx);
    }
    int devid = ctx2id.at(ctx);
    if (device[nid] != -1) {
      CHECK_EQ(device[nid], devid) << "device of same output not equal to each other";
    } else {
      device[nid] = devid;
    }
  }
  
  // Call the PlaceDevice pass.
  g.attrs["device"] = std::make_shared<dmlc::any>(std::move(device));
  g = nnvm::pass::PlaceDevice(g, "ctx_group", device_map, "_CrossDeviceCopy");
  const auto& assigned_device = g.GetAttr<nnvm::DeviceVector>("device");

  // Convert the device id back to context objects.
  ContextVector vcontext;
  for (size_t i = 0; i < assigned_device.size(); ++i) {
    if (assigned_device[i] == -1) {
      vcontext.push_back(default_ctx);
    } else {
      vcontext.push_back(ctx_list[assigned_device[i]]);
    }
  }
  g.attrs["context"] = std::make_shared<nnvm::any>(std::move(vcontext));
  return g;
}

void GraphExecutor::Init(nnvm::Symbol symbol,
                         const Context& default_ctx,
                         const std::map<std::string, Context>& ctx_map,
                         const std::vector<NDArray>& in_args,
                         const std::vector<NDArray>& arg_grad_store,
                         const std::vector<OpReqType>& grad_req_type,
                         const std::vector<NDArray>& aux_states,
                         Executor* shared_exec) {
  nnvm::Graph g = InitGraph(symbol, default_ctx,
                            ctx_map, in_args, arg_grad_store,
                            grad_req_type, aux_states);
  g = AttachOpExecs(g);
  LOG(INFO) << "Finished attach operator executors.";
  g = AttachOpResources(g);
  LOG(INFO) << "Finished attach operator resources.";
  graph_ = std::move(g);
  if (shared_exec != nullptr) {
    this->InitDataEntryMemory(dynamic_cast<GraphExecutor*>(shared_exec)->data_pool_);
  } else {
    this->InitDataEntryMemory({});
  }
  LOG(INFO) << "Finished initialize all memory.";
  {
    // initialize output arrays
    auto& idx = graph_.indexed_graph();
    for (size_t i = 0; i < num_forward_outputs_; ++i) {
      auto& e = idx.outputs()[i];
      output_arrays_.push_back(data_entry_[idx.entry_id(e)]);
    }
    // initialize head gradient array
    if (num_forward_inputs_ != idx.input_nodes().size()) {
      // TODO(minjie): this ugly check is required since the graph may be changed
      // after several passes and the head_grad_map_ may be invalid. This is an
      // example of external reference to internal graph structure.
      for (const auto& kv : head_grad_map_) {
        const nnvm::Node* head_grad_node = kv.first;
        const uint32_t nid = idx.node_id(head_grad_node);
        head_grad_array_.push_back(data_entry_[idx.entry_id(nid, 0)]);
      }
    }
  }
  LOG(INFO) << "Finished initialize output and head gradient arrays.";
  InitCachedOps();
  LOG(INFO) << "Finished initialize engine operators for execution.";
}

Graph GraphExecutor::InferShapeType(
    Graph g,
    const std::vector<NDArray>& in_args,
    const std::vector<NDArray>& aux_states) {
  const auto& idx = g.indexed_graph();
  // Setup argument shape and type.
  const std::unordered_set<uint32_t>& mutable_nodes = idx.mutable_input_nodes();
  nnvm::ShapeVector arg_shapes;
  nnvm::DTypeVector arg_types;
  size_t arg_top = 0, aux_top = 0;
  for (size_t i = 0; i < num_forward_inputs_; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    if (mutable_nodes.count(nid)) {
      CHECK_LT(aux_top, aux_states.size());
      arg_shapes.push_back(aux_states[aux_top].shape());
      arg_types.push_back(aux_states[aux_top].dtype());
      ++aux_top;
    } else {
      CHECK_LT(arg_top, in_args.size());
      arg_shapes.push_back(in_args[arg_top].shape());
      arg_types.push_back(in_args[arg_top].dtype());
      ++arg_top;
    }
  }
  arg_shapes.resize(idx.input_nodes().size(), TShape());
  arg_types.resize(idx.input_nodes().size(), 0);
  g = nnvm::pass::InferShape(g, arg_shapes, "__shape__");
  g = nnvm::pass::InferType(g, arg_types);
  return g;
}



Graph GraphExecutor::InitGraph(nnvm::Symbol symbol,
                               const Context& default_ctx,
                               const std::map<std::string, Context>& ctx_map,
                               const std::vector<NDArray>& in_args,
                               const std::vector<NDArray>& arg_grad_store,
                               const std::vector<OpReqType>& grad_req_type,
                               const std::vector<NDArray>& aux_states) {
  // setup gradient
  /*nnvm::Graph g = InitFullGraph(symbol, grad_req_type, arg_grad_store);
  g = AssignContext(g, default_ctx, ctx_map,
                    in_args,
                    grad_store_,
                    aux_states,
                    num_forward_inputs_,
                    num_forward_outputs_);
  const auto& idx = g.indexed_graph();
  // get number of nodes used in forward pass
  num_forward_nodes_ = 0;
  for (size_t i = 0; i < num_forward_outputs_; ++i) {
    num_forward_nodes_ = std::max(
        num_forward_nodes_, static_cast<size_t>(idx.outputs()[i].node_id + 1));
  }
  // Setup data entry, shape and type.
  data_entry_.resize(idx.num_node_entries());
  const std::unordered_set<uint32_t>& mutable_nodes = idx.mutable_input_nodes();
  nnvm::ShapeVector arg_shapes;
  nnvm::DTypeVector arg_types;
  size_t arg_top = 0, aux_top = 0;
  for (size_t i = 0; i < num_forward_inputs_; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    if (mutable_nodes.count(nid)) {
      CHECK_LT(aux_top, aux_states.size());
      data_entry_[idx.entry_id(nid, 0)] = aux_states[aux_top];
      arg_shapes.push_back(aux_states[aux_top].shape());
      arg_types.push_back(aux_states[aux_top].dtype());
      ++aux_top;
    } else {
      CHECK_LT(arg_top, in_args.size());
      data_entry_[idx.entry_id(nid, 0)] = in_args[arg_top];
      arg_shapes.push_back(in_args[arg_top].shape());
      arg_types.push_back(in_args[arg_top].dtype());
      ++arg_top;
    }
  }
  for (size_t j = num_forward_outputs_; j < idx.outputs().size(); ++j) {
    data_entry_[idx.entry_id(idx.outputs()[j])]
        = grad_store_[j - num_forward_outputs_].second;
  }
  arg_shapes.resize(idx.input_nodes().size(), TShape());
  arg_types.resize(idx.input_nodes().size(), 0);
  // other initializations
  g = nnvm::pass::InferShape(g, arg_shapes, "__shape__");
  g = nnvm::pass::InferType(g, arg_types);
  g = nnvm::ApplyPass(g, "PlanMemory");
  return g;*/

  // setup gradient
  nnvm::Graph g = InitFullGraph(symbol, grad_req_type, arg_grad_store);
  g = InferShapeType(g, in_args, aux_states);
  int num_gpus = 1;
  std::string num_gpus_attrs;
  if (symbol.GetAttr("num_gpus", &num_gpus_attrs)) {
    num_gpus = std::stoi(num_gpus_attrs);
  }
  LOG(INFO) << "Num GPUs: " << num_gpus;
  std::map<std::string, Context> group_contexts;
  if (num_gpus > 1) {
    g.attrs["num_devices"] = std::make_shared<nnvm::any>(num_gpus);
    g = nnvm::ApplyPass(g, "PartitionPass");
    //group_contexts["group:0"] = Context::CPU();
    //group_contexts["group:1"] = Context::GPU(0);
    for (int i = 0; i < num_gpus; ++i) {
      group_contexts["group:" + std::to_string(i)] = Context::GPU(i);
    }
  } else {
    group_contexts = ctx_map;
  }
  g = AssignContext(g, default_ctx, group_contexts,
                    in_args,
                    grad_store_,
                    aux_states,
                    num_forward_inputs_,
                    num_forward_outputs_);
  const nnvm::IndexedGraph& idx = g.indexed_graph();
  const std::unordered_set<uint32_t>& mutable_nodes = idx.mutable_input_nodes();
  // Setup data entry and point input/output to proper arguments.
  data_entry_.resize(idx.num_node_entries());
  size_t arg_top = 0, aux_top = 0;
  for (size_t i = 0; i < num_forward_inputs_; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    if (mutable_nodes.count(nid)) {
      data_entry_[idx.entry_id(nid, 0)] = aux_states[aux_top];
      ++aux_top;
    } else {
      data_entry_[idx.entry_id(nid, 0)] = in_args[arg_top];
      ++arg_top;
    }
  }
  for (size_t j = num_forward_outputs_; j < idx.outputs().size(); ++j) {
      data_entry_[idx.entry_id(idx.outputs()[j])]
        = grad_store_[j - num_forward_outputs_].second;
  }
  // get number of nodes used in forward pass
  num_forward_nodes_ = 0;
  for (size_t i = 0; i < num_forward_outputs_; ++i) {
    num_forward_nodes_ = std::max(
        num_forward_nodes_, static_cast<size_t>(idx.outputs()[i].node_id + 1));
  }
  LOG(INFO) << "Num forward nodes: " << num_forward_nodes_;
  g = nnvm::ApplyPass(g, "PlanMemory");
  LOG(INFO) << "Successfully applied all NNVM passes.";
  return g;
}

// initialize the memory of each entries
void GraphExecutor::InitDataEntryMemory(const std::vector<NDArray>& shared_pool) {
  using nnvm::DTypeVector;
  using nnvm::ShapeVector;
  using nnvm::StorageVector;
  // get the graph
  const auto& idx = graph_.indexed_graph();
  // get the storage
  const DTypeVector& vdtype = graph_.GetAttr<DTypeVector>("dtype");
  const ShapeVector& vshape = graph_.GetAttr<ShapeVector>("shape");
  const StorageVector& vstorage = graph_.GetAttr<StorageVector>("storage_id");
  const ContextVector& vctx = graph_.GetAttr<ContextVector>("context");
  CHECK_EQ(idx.num_node_entries(), vshape.size());
  CHECK_EQ(idx.num_node_entries(), vdtype.size());
  CHECK_EQ(idx.num_node_entries(), vstorage.size());
  CHECK_EQ(data_entry_.size(), vshape.size());
  std::vector<Context> data_context(idx.num_node_entries());
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    for (uint32_t i = 0; i < idx[nid].source->num_outputs(); ++i) {
      data_context[idx.entry_id(nid, i)] = vctx[nid];
    }
  }

  // information about the pool
  using PoolEntry = std::pair<Context, size_t>;
  std::vector<PoolEntry> pool_info;

  // assign array to head gradient
  if (num_forward_inputs_ != idx.input_nodes().size()) {
    // TODO(minjie): this ugly check is required since the graph may be changed
    // after several passes and the head_grad_map_ may be invalid. This is an
    // example of external reference to internal graph structure.
    LOG(INFO) << "Point head gradient entry to given NDArray";
    for (const auto& kv : head_grad_map_) {
      const nnvm::Node* head_grad_node = kv.first;
      const size_t output_idx = kv.second;
      const uint32_t nid = idx.node_id(head_grad_node);
      const uint32_t eid = idx.entry_id(idx.outputs()[output_idx]);
      CHECK_NE(vshape[eid].ndim(), 0);
      CHECK_NE(vdtype[eid], -1);
      data_entry_[idx.entry_id(nid, 0)] =
          NDArray(vshape[eid], data_context[eid], vdtype[eid]);
    }
  }
  // get maximum bytes in each pool
  LOG(INFO) << "Compute maximum bytes of each memory pool.";
  for (size_t i = 0; i < vshape.size(); ++i) {
    size_t bytes = vshape[i].Size() * mshadow::mshadow_sizeof(vdtype[i]);
    int storage_id = vstorage[i];
    if (storage_id < 0) continue;
    if (!data_entry_[i].is_none()) continue;
    size_t sid = static_cast<size_t>(storage_id);
    if (sid >= pool_info.size()) {
      pool_info.resize(sid + 1, PoolEntry{Context::CPU(), size_t(0)});
    }
    PoolEntry& info = pool_info[sid];
    if (info.second == 0) {
      info = PoolEntry{data_context[i], bytes};
    } else {
      info.second = std::max(info.second, bytes);
    }
  }
  // construct the re-use pool, if needed
  std::multimap<size_t, NDArray> free_pool;
  for (const NDArray& nd : shared_pool) {
    size_t bytes = nd.shape().Size() * mshadow::mshadow_sizeof(nd.dtype());
    free_pool.insert(std::make_pair(bytes, nd));
  }
  // remake the data pool
  data_pool_.clear();
  for (size_t i = 0; i < pool_info.size(); ++i) {
    const Context& ctx = pool_info[i].first;
    size_t bytes = pool_info[i].second;
    bool allocated = false;
    for (auto it = free_pool.lower_bound(bytes); it != free_pool.end(); ++it) {
      if (it->second.ctx() == ctx && it->first >= bytes) {
        data_pool_.push_back(it->second);
        free_pool.erase(it);
        allocated = true;
        break;
      }
    }
    if (!allocated) {
      size_t nword = (bytes + 3) / 4;
      CHECK_LE(nword, std::numeric_limits<index_t>::max());
      // allocate float arrays
      TShape shape{index_t(nword)};
      data_pool_.emplace_back(NDArray(shape, ctx));
    }
  }
  CHECK_EQ(data_pool_.size(), pool_info.size());
  // assign the data entries
  for (size_t i = 0; i < data_entry_.size(); ++i) {
    // avoid pre-allocated arrays
    if (!data_entry_[i].is_none()) continue;
    // assign allocated array by storage id
    int storage_id = vstorage[i];
    CHECK_GE(storage_id, 0)
      << "No storage assigned to entry#" << i
      << " storage_code=" << storage_id;
    const NDArray& src = data_pool_.at(storage_id);
    data_entry_[i] = src.AsArray(vshape[i], vdtype[i]);
  }
}


void GraphExecutor::InitCachedOps() {
  // get the graph
  const auto& idx = graph_.indexed_graph();
  const auto& vstorage_inplace =
      graph_.GetAttr<std::vector<int> >("storage_inplace_index");
  const auto& op_execs =
      graph_.GetAttr<OpExecVector>("op_execs");
  const auto& vctx = graph_.GetAttr<ContextVector>("context");

  op_nodes_.resize(idx.num_nodes());
  // setup the array and requirements.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) {
      // Nothing to be done for variable.
      continue;
    }
    op_nodes_[nid].exec = op_execs[nid];
    op_nodes_[nid].ctx = vctx[nid];
    CHECK_NOTNULL(op_nodes_[nid].exec);
    auto& exec = op_nodes_[nid].exec;
    CHECK_EQ(exec->in_array.size(), 0);
    CHECK_EQ(exec->out_array.size(), 0);
    for (const auto& e : inode.inputs) {
      exec->in_array.push_back(data_entry_[idx.entry_id(e)]);
    }
    // detect inplace requirement
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
      uint32_t eid = idx.entry_id(nid, index);
      exec->out_array.push_back(data_entry_[eid]);
      if (vstorage_inplace[eid] >= 0) {
        exec->req.push_back(kWriteInplace);
      } else if (vstorage_inplace[eid] == -2) {
        // -2 indicate that the entry is never referenced.
        exec->req.push_back(kNullOp);
      } else {
        exec->req.push_back(kWriteTo);
      }
    }
  }
  for (size_t j = num_forward_outputs_; j < idx.outputs().size(); ++j) {
    auto& e = idx.outputs()[j];
    op_nodes_[e.node_id].exec->req[e.index] =
        grad_store_[j - num_forward_outputs_].first;
  }

  // A vector that contains special variables used to indicate the accomplishment
  // of each operator. This variable will be used as control dependency.
  std::vector<Engine::VarHandle> finish_vars(idx.num_nodes());
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    const std::string& node_name = inode.source->attrs.name;
    // Create finish variable.
    finish_vars[nid] = Engine::Get()->NewVariable();
    if (inode.source->is_variable()) {
      // Nothing more to do with variable node.
      continue;
    }

    auto& exec = op_nodes_[nid].exec;
    std::vector<uint32_t> inplace_inputs;
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
      const uint32_t eid = idx.entry_id(nid, index);
      if (vstorage_inplace[eid] >= 0 && exec->req[index] == kWriteInplace) {
        inplace_inputs.push_back(vstorage_inplace[eid]);
      }
    }
    std::sort(inplace_inputs.begin(), inplace_inputs.end());

    const bool is_async = op_nodes_[nid].exec->exec_type() == Operator::kAsync;
    const bool is_gpu = op_nodes_[nid].ctx.dev_mask() == gpu::kDevMask;

    std::vector<Engine::VarHandle> use_vars, mutate_vars;
    // Use variables include:
    // - Variables of input arrays. Note that the input array that has the same
    //   storage id (could be computed inplacely) will not be put in the use_vars.
    // - Finish varaibles of control dependencies.
    for (size_t i = 0; i < exec->in_array.size(); ++i) {
      if (!std::binary_search(inplace_inputs.begin(), inplace_inputs.end(), i)) {
        auto& nd = exec->in_array[i];
        use_vars.push_back(nd.var());
      }
    }
    // Handle control dependencies.
    for (nnvm::NodePtr depend_node : inode.source->control_deps) {
      // Put the first output of the depend_node in use_vars.
      const uint32_t depend_nid = idx.node_id(depend_node.get());
      CHECK_LT(depend_nid, nid);
      use_vars.push_back(finish_vars[depend_nid]);
    }
    // Mutate variables include:
    // - Auxiliary states used by the operator.
    // - Output arrays.
    // - Finish variable.
    for (auto& r : exec->op_ctx.requested) {
      mutate_vars.push_back(r.var);
    }
    for (size_t idx = 0; idx < inode.source->num_outputs(); ++idx) {
      if (exec->req[idx] != kNullOp) {
        mutate_vars.push_back(exec->out_array[idx].var());
      }
    }
    mutate_vars.push_back(finish_vars[nid]);

    // Sort and make the vars unique.
    std::vector<Engine::VarHandle> all_vars = use_vars;
    all_vars.insert(all_vars.end(), mutate_vars.begin(), mutate_vars.end());
    SortAndUnique(&use_vars);
    SortAndUnique(&mutate_vars);
    SortAndUnique(&all_vars);
    CHECK_EQ(use_vars.size() + mutate_vars.size(), all_vars.size())
      << "Variable should not both used and mutated in one operator."
      << " This will cause deadlock during execution.";

    Engine::Get()->PushSync([exec](RunContext rctx) {
        exec->Setup();
      }, Context::CPU(), {}, all_vars);

    trace_records_ = std::shared_ptr<std::vector<TraceRecord>>(new std::vector<TraceRecord>());

    auto exec_fun = [exec, is_async, is_gpu, nid, node_name, this](
        RunContext ctx, Engine::CallbackOnComplete on_complete) {
      if (is_async) {
        exec->op_ctx.async_on_complete = on_complete;
      }

      //struct timeval st, ed;
      //gettimeofday(&st, nullptr);

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
      } else {
        LOG(INFO) << "Async !!!!";
      }

      //gettimeofday(&ed, nullptr);
      //TraceRecord rec = std::make_tuple(node_name, st.tv_usec / 1000.0, ed.tv_usec / 1000.0);
      //{
        //std::lock_guard<std::mutex> guard(trace_mutex_);
        //trace_records_->push_back(rec);
      //}
    };
    // setup the vars
    op_nodes_[nid].cached_opr = Engine::Get()->NewOperator(
        exec_fun, use_vars, mutate_vars, FnProperty::kNormal);
  }
}

void GraphExecutor::RunOps(bool is_train, size_t topo_start, size_t topo_end) {
  static const auto& flist_outputs =
      nnvm::Op::GetAttr<nnvm::FListOutputNames>("FListOutputNames");
  const auto& idx = graph_.indexed_graph();
  for (size_t nid = topo_start; nid < topo_end; ++nid) {
    const auto& inode = idx[nid];
    //LOG(INFO) << "Push node #" << nid << " " << inode.source->attrs.name;
    if (inode.source->is_variable()) continue;
    OpNode& opnode = op_nodes_[nid];
    opnode.exec->op_ctx.is_train = is_train;
    if (opnode.exec->exec_type() == Operator::kCrossDeviceCopy) {
      CHECK_EQ(inode.inputs.size(), 1);
      CHECK_EQ(opnode.exec->in_array.size(), 1);
      CHECK_EQ(opnode.exec->out_array.size(), 1);
      CopyFromTo(opnode.exec->in_array[0], &(opnode.exec->out_array[0]));
    } else if (opnode.cached_opr != nullptr) {
      Engine::Get()->Push(opnode.cached_opr, opnode.ctx);
    } else {
      LOG(FATAL) << "Not accessed";
    }

    if (monitor_callback_) {
      std::vector<std::string> output_names;
      const auto& node = idx[nid].source;
      if (flist_outputs.count(node->op())) {
        output_names = flist_outputs[node->op()](node->attrs);
      } else {
        for (size_t i = 0; i < node->num_outputs(); ++i) {
          output_names.emplace_back(std::to_string(i));
        }
      }
      for (index_t i = 0; i < opnode.exec->out_array.size(); ++i) {
        NDArray *cpy = new NDArray(opnode.exec->out_array[i]);
        std::string name = inode.source->attrs.name + "_" + output_names[i];
        this->monitor_callback_(name.c_str(), reinterpret_cast<void*>(cpy));
      }
    }
  }
}

}  // namespace exec

Executor *Executor::Bind(nnvm::Symbol symbol,
                         const Context& default_ctx,
                         const std::map<std::string, Context>& group2ctx,
                         const std::vector<NDArray> &in_args,
                         const std::vector<NDArray> &arg_grad_store,
                         const std::vector<OpReqType> &grad_req_type,
                         const std::vector<NDArray> &aux_states,
                         Executor* shared_exec) {
  auto exec = new exec::GraphExecutor();
  exec->Init(symbol, default_ctx, group2ctx,
             in_args, arg_grad_store, grad_req_type, aux_states,
             reinterpret_cast<Executor*>(shared_exec));
  return exec;
}
}  // namespace mxnet
