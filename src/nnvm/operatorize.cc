#include "./mx_passes.h"

using namespace nnvm;
using namespace std;

namespace mxnet {
namespace pass {

Graph ExposeInvisibleOutputsRec(const Graph& graph) {
  static auto& fnum_vis_output = Op::GetAttr<FNumVisibleOutputs>("FNumVisibleOutputs");
  const auto& idx = graph.indexed_graph();
  std::vector<NodeEntry> invisible_outputs;
  DFSVisit(graph.outputs, [&] (const NodePtr& node) {
      if (node->is_variable()) {
        return;
      } else if (node->is_graph() && node->graph()->global_attrs.count("num_visible_outputs")) {
        const size_t sub_nvis = node->graph()->GetGlobalAttr<size_t>("num_visible_outputs");
        CHECK_LE(sub_nvis, node->num_outputs());
        for (uint32_t i = sub_nvis; i < node->num_outputs(); ++i) {
          invisible_outputs.emplace_back(NodeEntry{node, i, 0});
        }
      } else if (fnum_vis_output.count(node->op())) {
        const size_t nvis = fnum_vis_output[node->op()](node->attrs);
        CHECK_LE(nvis, node->num_outputs());
        for (uint32_t i = nvis; i < node->num_outputs(); ++i) {
          invisible_outputs.emplace_back(NodeEntry{node, i, 0});
        }
      }
    });
  Graph new_graph;
  new_graph.outputs = graph.outputs;
  new_graph.global_attrs["num_visible_outputs"] =
    std::make_shared<any>(graph.outputs.size());
  new_graph.outputs.insert(new_graph.outputs.end(),
                           invisible_outputs.begin(),
                           invisible_outputs.end());
  return new_graph;
}

Graph MXExposeInvisibleOutputs(const Graph& graph) {
  if (graph.global_attrs.count("num_visible_outputs")) {
    return graph;
  }
  return ExposeInvisibleOutputsRec(graph);
}

NNVM_REGISTER_PASS(MXExposeInvisibleOutputs)
.describe("Transform the graph to have invisible output entries.")
.set_body(MXExposeInvisibleOutputs)
.set_change_graph(true)
.provide_global_attr("num_visible_outputs");

}  // namespace pass
}  // namespace mxnet
