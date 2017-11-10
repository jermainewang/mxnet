/*!
 *  Copyright (c) 2016 by Contributors
 * \file compute_mutate_index.cc
 * \brief Compute input index that will be mutated for each node in the graph.
 */
#include "./mx_passes.h"

using namespace std;
using namespace nnvm;

namespace mxnet {
namespace pass {

void ComputeMutateIndexRec(
    const Graph& graph,
    Column<vector<uint32_t>>* mutate_index) {
  const IndexedGraph& idx = graph.indexed_graph();
  static auto& fmutate_inputs = Op::GetAttr<FMutateInputs>("FMutateInputs");
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    if (node->is_variable()) {
      continue;
    }
    if (node->is_graph()) {
      auto subgraph = node->graph();
      if (subgraph->node_attrs.count(mutate::key) == 0) {
        auto subref = subgraph->CreateNodeColumn<vector<uint32_t>>();
        ComputeMutateIndexRec(*subgraph, subref.CopyOnWrite());
        mutate_index->children[nid] = subref;
      } else {
        mutate_index->children[nid] =
          subgraph->node_attrs.GetColumn<vector<uint32_t>>(mutate::key);
      }
      // Summarize. If any input entry of the subgraph is mutated, put that
      // in the mutate index.
      const auto& subidx = subgraph->indexed_graph();
      const auto* sub_mutate = mutate_index->children[nid].get();
      vector<bool> is_mutated(subidx.num_nodes(), false);
      for (uint32_t subnid = 0; subnid < subidx.num_nodes(); ++subnid) {
        for (const uint32_t mutate : sub_mutate->value[subnid]) {
          is_mutated[subidx[subnid].inputs[mutate].node_id] = true;
        }
      }
      for (size_t i = 0; i < subidx.input_nodes().size(); ++i) {
        const uint32_t sub_innid = subidx.input_nodes()[i];
        if (is_mutated[sub_innid] == 0) {
          mutate_index->value[nid].push_back(i);
        }
      }
    } else if (fmutate_inputs.count(node->op()) != 0) {
      mutate_index->value[nid] = fmutate_inputs[node->op()](node->attrs);
    }
  }
}

Graph MXComputeMutateIndex(Graph &&graph) {
  if (graph.node_attrs.count(mutate::key) == 0) {
    auto ref = graph.CreateNodeColumn<vector<uint32_t>>();
    ComputeMutateIndexRec(graph, ref.CopyOnWrite());
    graph.node_attrs.SetColumn(mutate::key, ref);
  }
  return graph;
}

NNVM_REGISTER_PASS(MXComputeMutateIndex)
.describe("Compute input index that will be mutated for each node in the graph.")
.set_body(MXComputeMutateIndex)
.set_change_graph(false)
.set_argument("graph_frozen")
.provide_node_attr(mutate::key);

}  // namespace pass
}  // namespace mxnet
