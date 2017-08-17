/*!
 *  Copyright (c) 2016 by Contributors
 * \file compute_ignored_inputs.cc
 * \brief Compute which inputs are ignored for each node in the graph.
 */
#include "./mx_passes.h"

using namespace std;
using namespace nnvm;

namespace mxnet {
namespace pass {

static const string kAttrIgnoredInputs = "ignored_inputs";

void ComputeIgnoredInputsRec(
    const Graph& graph,
    Column<vector<uint32_t>>* ignored_inputs) {
  const IndexedGraph& idx = graph.indexed_graph();
  static auto& fignore_inputs = Op::GetAttr<FIgnoreInputs>("FIgnoreInputs");
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    if (node->is_variable()) {
      continue;
    }
    if (node->is_graph()) {
      auto subgraph = node->graph();
      if (subgraph->node_attrs.count(kAttrIgnoredInputs) == 0) {
        auto subref = subgraph->CreateNodeColumn<vector<uint32_t>>();
        ComputeIgnoredInputsRec(*subgraph, subref.CopyOnWrite());
        ignored_inputs->children[nid] = subref;
      } else {
        ignored_inputs->children[nid] =
          subgraph->node_attrs.GetColumn<vector<uint32_t>>(kAttrIgnoredInputs);
      }
      // Summarize.
      const auto& subidx = subgraph->indexed_graph();
      const auto* sub_ignore = ignored_inputs->children[nid].get();
      vector<size_t> refcounts(subidx.num_nodes(), 0);
      for (uint32_t subnid = 0; subnid < subidx.num_nodes(); ++subnid) {
        for (const auto& ent : subidx[subnid].inputs) {
          ++refcounts[ent.node_id];
        }
      }
      for (uint32_t subnid = 0; subnid < subidx.num_nodes(); ++subnid) {
        for (const uint32_t ignored : sub_ignore->value[subnid]) {
          --refcounts[subidx[subnid].inputs[ignored].node_id];
        }
      }
      for (size_t i = 0; i < subidx.input_nodes().size(); ++i) {
        const uint32_t sub_innid = subidx.input_nodes()[i];
        if (refcounts[sub_innid] == 0) {
          ignored_inputs->value[nid].push_back(i);
        }
      }
    } else if (fignore_inputs.count(node->op()) != 0) {
      ignored_inputs->value[nid] = fignore_inputs[node->op()](node->attrs);
    }
  }
}

Graph MXComputeIgnoredInputs(Graph &&graph) {
  if (graph.node_attrs.count(kAttrIgnoredInputs) == 0) {
    auto ref = graph.CreateNodeColumn<vector<uint32_t>>();
    ComputeIgnoredInputsRec(graph, ref.CopyOnWrite());
    graph.node_attrs.SetColumn(kAttrIgnoredInputs, ref);
  }
  return graph;
}

NNVM_REGISTER_PASS(MXComputeIgnoredInputs)
.describe("Compute which inputs are ignored for each node.")
.set_body(MXComputeIgnoredInputs)
.set_change_graph(false)
.depend_global_attr("graph_frozen")
.provide_node_attr(kAttrIgnoredInputs);

}  // namespace pass
}  // namespace mxnet
