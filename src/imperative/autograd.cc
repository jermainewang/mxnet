#include <mxnet/base.h>
#include <mxnet/op_attr_types.h>

#include "./autograd.h"
#include "../nnvm/mx_passes.h"

using namespace std;
using namespace nnvm;

namespace mxnet {
namespace ag {

void AutogradTape::SaveInfo(const NDArray* nd, bool save_value) {
  const tape::Tape& tape = tape::Tape::Get(tape::kGradTape);
  uint32_t tapeid = 0, pos = 0, index = 0, sid = 0;
  tie(tapeid, pos, index, sid) = tape::ParseTapeEntryId(nd->tape_entry_id());
  if (pos >= saved_info_.size()) {
    saved_info_.resize(pos + 1);
  }
  if (saved_info_[pos].empty()) {
    saved_info_[pos].resize(tape[pos].node->num_outputs());
  }
  auto& aginfo = saved_info_[pos][index];
  if (!aginfo.saved) {
    aginfo.shape = nd->shape();
    aginfo.dtype = nd->dtype();
    
    if (nd->HasGradAttached()) {
      tie(aginfo.req_type, aginfo.grad_buffer) = nd->GetAttachedGrad();
      grad_attached_entries_.push_back(nd->tape_entry_id());
    }
    aginfo.saved = true;
  }
  if (save_value && aginfo.value.is_none()) {
    aginfo.value = *nd;
  }
}

void AutogradTape::Record(
    const nnvm::NodeAttrs& attrs,
    const vector<NDArray*>& ndinputs,
    const vector<NDArray*>& ndoutputs) {
  // TODO(minjie): grad_tape should record operators whose parents are
  // 1. grad-attached; or 2. already on the tape.
  tape::Tape& tape = tape::Tape::Get(tape::kGradTape);
  const uint32_t pos = tape.Record(attrs, ndinputs, ndoutputs);
  const auto& node = tape[pos].node;
  // Compute values to be saved.
  vector<bool> save_inputs, save_outputs;
  static auto& fbwddep_map = Op::GetAttr<FBackwardDependency>("FBackwardDependency");
  if (node->is_graph() && node->graph()->global_attrs.count("FBackwardDependency")) {
    //CHECK(node->graph()->global_attrs.count("FBackwardDependency"))
      //<< "Cannot differentiate subgraph node \"" << node->attrs.name << "\""
      //<< ": its graph has not been specialized for gradient computation.";
    auto fbwddep = node->graph()->GetGlobalAttr<FBackwardDependency>("FBackwardDependency");
    fbwddep(node.get(), &save_inputs, &save_outputs);
  } else if (!node->is_graph() && fbwddep_map.count(node->op())) {
    //CHECK(fbwddep_map.count(node->op()))
      //<< "Cannot differentiate node \"" << node->attrs.name << "\""
      //<< ": operator \"" << node->op()->name
      //<< "\" does not have gradient function registered.";
    fbwddep_map[node->op()](node.get(), &save_inputs, &save_outputs);
  } else {
    save_inputs.resize(node->inputs.size(), false);
    save_outputs.resize(node->num_outputs(), false);
  }
  CHECK_EQ(save_inputs.size(), ndinputs.size());
  CHECK_EQ(save_outputs.size(), ndoutputs.size());
  for (size_t i = 0; i < ndinputs.size(); ++i) {
    SaveInfo(ndinputs[i], save_inputs[i]);
  }
  for (size_t i = 0; i < ndoutputs.size(); ++i) {
    SaveInfo(ndoutputs[i], save_outputs[i]);
  }
}

void AutogradTape::NewSession() {
  tape::Tape::Get(tape::kGradTape).NewSession();
  grad_attached_entries_.clear();
  saved_info_.clear();
}

void AutogradTape::AttachGrad(tape::TapeEntryId teid,
                              OpReqType req_type,
                              const NDArray& grad_buf) {
  tape::Tape& grad_tape = tape::Tape::Get(tape::kGradTape);
  if (grad_tape.HasTaped(teid)) {
    uint32_t tapeid = 0, pos = 0, index = 0, sid = 0;
    std::tie(tapeid, pos, index, sid) = tape::ParseTapeEntryId(teid);
    auto& aginfo = saved_info_[pos][index];
    aginfo.req_type = req_type;
    aginfo.grad_buffer = grad_buf;
    grad_attached_entries_.push_back(teid);
  }
}

Graph AutogradTape::GetForwardGraph(const vector<const NDArray*>& ys,
                                    const vector<const NDArray*>& xs) {
  using pass::grad::MXGradientArgs;
  using pass::MXEntryArg;

  tape::Tape& grad_tape = tape::Tape::Get(tape::kGradTape);
  Graph graph = grad_tape.GetGraph(ys);
  unordered_map<string, shared_ptr<any>> kwargs_any;
  graph = nnvm::Transform(graph, {"MXExposeInvisibleOutputs"}, kwargs_any);

  uint32_t tapeid = 0, pos = 0, index = 0, sid = 0;

  std::vector<tape::TapeEntryId> xs_teid;
  if (!xs.empty()) {
    // Compute gradients for all provided xs.
    // TODO(minjie): undefined behaviors when variables are provided
    // to compute gradient on. This may conflict the attached grad entries.
    LOG(FATAL) << "#provided variables=" << xs.size();
    for (const NDArray* x : xs) {
      xs_teid.push_back(x->tape_entry_id());
    }
  } else {
    // Compute gradients for all grad-attached variables.
    // Note: Other grad-attached entries will not become graph outputs.
    //   Their gradient values will be stored in the provided buffers.
    for (auto teid : grad_attached_entries_) {
      std::tie(tapeid, pos, index, sid) = tape::ParseTapeEntryId(teid);
      if (saved_info_[pos][index].req_type != kNullOp) {
        xs_teid.push_back(teid);
      }
    }
  }

  const auto& idx = graph.indexed_graph();
  MXGradientArgs args;
  for (auto teid : xs_teid) {
    std::tie(tapeid, pos, index, sid) = tape::ParseTapeEntryId(teid);
    const uint32_t nid = idx.node_id(grad_tape[pos].node.get());
    MXEntryArg entarg;
    entarg.node = nid;
    entarg.index = index;
    args.xs.emplace_back(std::move(entarg));
  }

  kwargs_any["mx_gradient_args"] = std::make_shared<any>(std::move(args));
  graph = nnvm::Transform(graph, {"MXGradient"}, kwargs_any);
  const auto& graph_idx = graph.indexed_graph();
  // Create column attributes for the forward graph.
  auto values_col = graph.CreateEntryColumn<NDArray>();
  auto* shapes = graph.CreateOrWriteEntryColumn<TShape>(pass::shape::key);
  auto* dtypes = graph.CreateOrWriteEntryColumn<int>(pass::dtype::key, -1);
  auto* values = graph.CreateOrWriteEntryColumn<NDArray>("value");
  auto* grad_buffers = graph.CreateOrWriteEntryColumn<NDArray>("grad_buffer");
  auto* req_type = graph.CreateOrWriteEntryColumn<OpReqType>("op_req");
  for (uint32_t pos = 0; pos < grad_tape.size(); ++pos) {
    const Node* node = grad_tape[pos].node.get();
    if (!graph_idx.exist(node)) {
      continue;
    }
    const uint32_t nid = graph_idx.node_id(node);
    for (uint32_t i = 0; i < node->num_outputs(); ++i) {
      const uint32_t eid = graph_idx.entry_id(nid, i);
      const auto& info = saved_info_[pos][i];
      shapes->value[eid] = info.shape;
      dtypes->value[eid] = info.dtype;
      values->value[eid] = info.value;
      grad_buffers->value[eid] = info.grad_buffer;
      req_type->value[eid] = info.req_type;
    }
  }
  saved_info_.clear();
  return graph;
}


Graph AutogradTape::GetSpecializedBackwardGraph(
    const vector<const NDArray*>& ys,
    const vector<const NDArray*>& xs,
    const vector<const NDArray*>& ys_grad) {
  using pass::shape::MXInferShapeArgs;
  using pass::dtype::MXInferTypeArgs;
  using pass::plan_memory::MXPlanMemoryArgs;
  using pass::MXEntryArg;
  Graph fwd_graph = GetForwardGraph(ys, xs);
  GraphPtr bwd_graph = fwd_graph.GetGlobalAttr<GraphPtr>("gradient_graph");
  const auto& fwd_graph_idx = fwd_graph.indexed_graph();
  const auto& bwd_graph_idx = bwd_graph->indexed_graph();
  LOG(INFO) << "#Fwd nodes: " << fwd_graph_idx.num_nodes();
  LOG(INFO) << "#Bwd nodes: " << bwd_graph_idx.num_nodes();
  LOG(INFO) << "#Fwd outputs: " << fwd_graph.outputs.size();
  LOG(INFO) << "#Bwd outputs: " << bwd_graph->outputs.size();
  LOG(INFO) << "#Bwd inputs: " << bwd_graph_idx.input_nodes().size();
  LOG(INFO) << "#Fwd visible outputs: " << fwd_graph.GetGlobalAttr<size_t>("num_visible_outputs");
  /*for (uint32_t nid = 0; nid < bwd_graph_idx.num_nodes(); ++nid) {
    const auto* node = bwd_graph_idx[nid].source;
    LOG(INFO) << "#" << nid << " " << node->attrs.name << " " << (node->is_variable()? "var" : node->op()->name);
    for (const auto& in : node->inputs) {
      LOG(INFO) << "\t<-" << in.node->attrs.name;
    }
    for (const auto& n : node->control_deps) {
      LOG(INFO) << "\t<c-" << n->attrs.name;
    }
  }*/

  tape::Tape& tape = tape::Tape::Get(tape::kGradTape);
  // Create shape/type/value columns of the forward graph.
  MXInferShapeArgs shape_args;
  MXInferTypeArgs dtype_args;
  shape_args.forward_shapes = fwd_graph.entry_attrs.GetColumn<TShape>(pass::shape::key);
  dtype_args.forward_dtypes = fwd_graph.entry_attrs.GetColumn<int>(pass::dtype::key);
  // TODO(minjie): shape_inputs & dtype_inputs.
  // TODO(minjie): ctx.
  std::vector<Context> ctx = {Context::CPU(0)};
  // Mark grad-attached entries with external storage so memory planning will
  // not share their spaces with others.
  // TODO(minjie): This is not necessary at the moment since all grad-attached
  // entries will be included in the outputs of the gradient graph. And outputs will
  // never share memory with others.
  MXPlanMemoryArgs pm_args;
  /*const auto& fwdent2bwdent = bwd_graph->GetGlobalAttr<vector<uint32_t>>("fwdent2bwdent");
  for (auto teid : grad_attached_entries_) {
    uint32_t tapeid = 0, pos = 0, index = 0, sid = 0;
    std::tie(tapeid, pos, index, sid) = tape::ParseTapeEntryId(teid);
    if (saved_info_[pos][index].req_type == kNullOp) {
      continue;
    }
    const uint32_t fwd_nid = fwd_graph_idx.node_id(tape[pos].node.get());
    const uint32_t fwd_eid = fwd_graph_idx.entry_id(fwd_nid, index);
    const uint32_t bwd_eid = fwdent2bwdent[fwd_eid];
    CHECK_LT(bwd_eid, bwd_graph_idx.num_node_entries());
    pm_args.external_entry_ids.push_back(bwd_eid);
  }*/

  // Specialization.
  std::unordered_map<std::string, std::shared_ptr<any>> kwargs_any;
  kwargs_any[pass::shape::arg_name] = std::make_shared<any>(std::move(shape_args));
  kwargs_any[pass::dtype::arg_name] = std::make_shared<any>(std::move(dtype_args));
  kwargs_any[pass::ctx::ctx_key] = std::make_shared<any>(std::move(ctx));
  kwargs_any[pass::plan_memory::arg_name] = std::make_shared<any>(std::move(pm_args));
  kwargs_any["graph_frozen"] = std::make_shared<any>(1);
  nnvm::Specialize(bwd_graph.get(), kwargs_any);
  /*const auto& bshapes = bwd_graph->entry_attrs.GetColumn<TShape>("shape");
  for (uint32_t nid = 0; nid < bwd_graph_idx.num_nodes(); ++nid) {
    const Node* node = bwd_graph_idx[nid].source;
    for (uint32_t i = 0; i < node->num_outputs(); ++i) {
      const uint32_t eid = bwd_graph_idx.entry_id(nid, i);
      LOG(INFO) << "Node#" << nid << "#" << i << ": "
        << "shape=" << bshapes->value[eid];
    }
  }*/

  auto values = fwd_graph.entry_attrs.MoveColumn<NDArray>("value");
  auto grad_buffers = fwd_graph.entry_attrs.MoveColumn<NDArray>("grad_buffer");
  auto req_type = fwd_graph.entry_attrs.MoveColumn<OpReqType>("op_req");
  // Pack all inputs.
  using pass::grad::GradNodeInInfo;
  const auto& in_info = bwd_graph->GetGlobalAttr<vector<GradNodeInInfo>>("grad_node_in_info");
  vector<NDArray> arguments;
  for (const auto& info : in_info) {
    switch (info.type) {
    case GradNodeInInfo::kFromGradOut:
      if (ys_grad[info.index] == nullptr) {
        arguments.emplace_back(ys[info.index]->shape(),
                               ys[info.index]->ctx(),
                               true,
                               ys[info.index]->dtype());
        arguments.back() = static_cast<real_t>(1.0);
      } else {
        arguments.push_back(*ys_grad[info.index]);
      }
      break;
    case GradNodeInInfo::kFromFwdOut:
      {
      const uint32_t eid = fwd_graph_idx.entry_id(fwd_graph.outputs[info.index]);
      CHECK(!values->value[eid].is_none());
      arguments.push_back(values->value[eid]);
      }
      break;
    case GradNodeInInfo::kFromFwdIn:
      {
      const uint32_t nid = fwd_graph_idx.input_nodes()[info.index];
      CHECK(fwd_graph_idx[nid].source->is_variable());
      const uint32_t eid = fwd_graph_idx.entry_id(nid, 0);
      CHECK(!values->value[eid].is_none());
      arguments.push_back(values->value[eid]);
      }
      break;
    default:
      LOG(FATAL) << "Backward error.";
    }
  }
  // Pack all outputs.

  return *bwd_graph;
}

AutogradTape& AutogradTape::Get() {
  static AutogradTape tape;
  return tape;
}

}  // namespace ag
}  // namespace mxnet
