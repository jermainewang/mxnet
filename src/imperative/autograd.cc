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
  if (!saved_info_[pos][index].saved) {
    auto& aginfo = saved_info_[pos][index];
    aginfo.shape = nd->shape();
    aginfo.dtype = nd->dtype();
    if (save_value) {
      aginfo.value = *nd;
    }
    if (nd->HasGradAttached()) {
      tie(aginfo.req_type, aginfo.grad_buffer) = nd->GetAttachedGrad();
      grad_attached_entries_.push_back(nd->tape_entry_id());
    }
    aginfo.saved = true;
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
  using pass::grad::MXEntryArg;

  tape::Tape& grad_tape = tape::Tape::Get(tape::kGradTape);
  Graph graph = grad_tape.GetGraph(ys);
  unordered_map<string, shared_ptr<any>> kwargs_any;
  graph = nnvm::Transform(graph, {"MXExposeInvisibleOutputs"}, kwargs_any);

  uint32_t tapeid = 0, pos = 0, index = 0, sid = 0;

  std::vector<tape::TapeEntryId> xs_teid;
  if (!xs.empty()) {
    // Compute gradients for all provided xs.
    LOG(INFO) << "#provided variables=" << xs.size();
    for (const NDArray* x : xs) {
      xs_teid.push_back(x->tape_entry_id());
    }
  } else {
    // Compute gradients for all grad-attached variables.
    // Note: Other grad-attached entries will not become graph outputs.
    //   Their gradient values will be stored in the provided buffers.
    for (auto teid : grad_attached_entries_) {
      std::tie(tapeid, pos, index, sid) = tape::ParseTapeEntryId(teid);
      xs_teid.push_back(teid);
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
  return nnvm::Transform(graph, {"MXGradient"}, kwargs_any);
}


Graph AutogradTape::GetSpecializedBackwardGraph(
    const vector<const NDArray*>& ys,
    const vector<const NDArray*>& xs,
    const vector<const NDArray*>& ys_grad) {
  Graph fwd_graph = GetForwardGraph(ys, xs);
  GraphPtr bwd_graph = fwd_graph.GetGlobalAttr<GraphPtr>("gradient_graph");
  const auto& fwd_graph_idx = fwd_graph.indexed_graph();
  const auto& bwd_graph_idx = bwd_graph->indexed_graph();
  LOG(INFO) << "#Fwd nodes: " << fwd_graph_idx.num_nodes();
  LOG(INFO) << "#Bwd nodes: " << bwd_graph_idx.num_nodes();
  /*for (uint32_t nid = 0; nid < ig.num_nodes(); ++nid) {
    const auto* node = ig[nid].source;
    LOG(INFO) << "Node#" << nid << ": " << node->attrs.name << " op="
      << (node->is_variable()? "var" : (node->is_graph()? "graph" : node->op()->name));
  }*/
  LOG(INFO) << "#Fwd outputs: " << fwd_graph.outputs.size();
  LOG(INFO) << "#Bwd outputs: " << bwd_graph->outputs.size();
  LOG(INFO) << "#Fwd visible outputs: " << fwd_graph.GetGlobalAttr<size_t>("num_visible_outputs");

  // Create shape/type/value columns of the forward graph.

  //nnvm::MoveEntryAttrsRowToColumn<TShape>(g, "shape");
  //nnvm::MoveEntryAttrsRowToColumn<int>(g, "dtype");
  //const auto& shapes = g.entry_attrs.GetColumn<TShape>("shape");
  //const auto& dtypes = g.entry_attrs.GetColumn<int>("dtype");
  /*for (uint32_t nid = 0; nid < ig.num_nodes(); ++nid) {
    const auto* node = ig[nid].source;
    for (uint32_t i = 0; i < node->num_outputs(); ++i) {
      const uint32_t eid = ig.entry_id(nid, i);
      LOG(INFO) << "Node#" << nid << "#" << i << ": "
        << "shape=" << shapes->value[eid] << " dtype=" << dtypes->value[eid];
    }
  }*/
  return *bwd_graph;
}

AutogradTape& AutogradTape::Get() {
  static AutogradTape tape;
  return tape;
}

}  // namespace ag
}  // namespace mxnet
