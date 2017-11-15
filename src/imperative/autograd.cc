#include <mxnet/base.h>
#include <mxnet/op_attr_types.h>

#include "./autograd.h"
#include "../nnvm/mx_passes.h"

using namespace std;

namespace mxnet {
namespace ag {

void RecordGradientInfo(const nnvm::NodeAttrs& attrs,
                        const vector<NDArray*>& ndinputs,
                        const vector<NDArray*>& ndoutputs,
                        tape::Tape* tape) {
  const uint32_t pos = tape->Record(attrs, ndinputs, ndoutputs);
  auto& node = (*tape)[pos].node;
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
  CHECK_EQ(save_inputs.size(), node->inputs.size());
  CHECK_EQ(save_outputs.size(), node->num_outputs());
  for (size_t i = 0; i < save_inputs.size(); ++i) {
    const auto& ent = node->inputs[i];
    auto& values = ent.node->CreateOrGetEntryAttrs<NDArray>("value");
    auto& shapes = ent.node->CreateOrGetEntryAttrs<TShape>("shape");
    auto& dtypes = ent.node->CreateOrGetEntryAttrs<int>("dtype");
    if (save_inputs[i]) {
      values.value[ent.index] = *ndinputs[i];
    }
    shapes.value[ent.index] = ndinputs[i]->shape();
    dtypes.value[ent.index] = ndinputs[i]->dtype();
  }
  auto& values = node->CreateOrGetEntryAttrs<NDArray>("value");
  auto& shapes = node->CreateOrGetEntryAttrs<TShape>("shape");
  auto& dtypes = node->CreateOrGetEntryAttrs<int>("dtype");
  for (size_t i = 0; i < save_outputs.size(); ++i) {
    if (save_outputs[i]) {
      values.value[i] = *ndoutputs[i];
    }
    shapes.value[i] = ndoutputs[i]->shape();
    dtypes.value[i] = ndoutputs[i]->dtype();
  }
}

void GenerateBackwardGraph(const tape::Tape& tape,
                           const std::vector<const NDArray*>& ys,
                           const std::vector<const NDArray*>& xs) {
  using pass::grad::MXGradientArgs;
  using pass::grad::MXEntryArg;
  using nnvm::any;
  nnvm::Graph g = tape.GetGraph(ys);
  const auto& ig = g.indexed_graph();
  LOG(INFO) << ig.num_nodes();
  /*for (uint32_t nid = 0; nid < ig.num_nodes(); ++nid) {
    const auto* node = ig[nid].source;
    LOG(INFO) << "Node#" << nid << ": " << node->attrs.name << " op="
      << (node->is_variable()? "var" : (node->is_graph()? "graph" : node->op()->name));
  }*/
  MXGradientArgs args;
  uint32_t tapeid = 0, pos = 0, index = 0, sid = 0;
  LOG(INFO) << "#variables=" << xs.size();
  for (const NDArray* x : xs) {
    const uint32_t teid = x->tape_entry_id();
    std::tie(tapeid, pos, index, sid) = tape::ParseTapeEntryId(teid);
    const uint32_t nid = ig.node_id(tape[pos].node.get());
    MXEntryArg entarg;
    entarg.node = nid;
    entarg.index = index;
    args.xs.emplace_back(std::move(entarg));
  }
  unordered_map<string, shared_ptr<any>> kwargs_any;
  kwargs_any["mx_gradient_args"] = std::make_shared<any>(std::move(args));
  g = nnvm::Transform(g, {"MXGradientFull"}, kwargs_any);
  LOG(INFO) << g.outputs.size();
  LOG(INFO) << g.GetGlobalAttr<size_t>("num_visible_outputs");
  LOG(INFO) << g.indexed_graph().num_nodes();

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
}

}  // namespace ag
}  // namespace mxnet
