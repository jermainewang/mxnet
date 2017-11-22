/*!
 *  Copyright (c) 2016 by Contributors
 * \file compute_inplace.cc
 * \brief Compute the inplace option for each node in the graph.
 */
#include "./mx_passes.h"

using namespace std;
using namespace nnvm;

namespace mxnet {
namespace pass {

// Summarize. There are two cases.
// Case #1:
//  We traverse the graph in topological order. If an inplace pair is found,
//  we mark the output entry and keep going. After the traversal, we check
//  whether the output entries of the graph can be inplaced with input entries.
// Case #2:
//  If an output entry size is smaller than an input entry and they are connected
//  by some path, then these two can be an inplace pair.
// TODO
// For inplace_identity, we only have case #1.
vector<inplace::InplaceOption> Summarize(
    const Graph& graph,
    const Column<TShape>* shapes,
    const Column<vector<inplace::InplaceOption>>* options) {
  using inplace::InplaceOption;
  vector<inplace::InplaceOption> ret;
  return ret;
}

void ComputeInplaceOptionRec(
    const Graph& graph,
    const Column<TShape>* shapes,
    Column<vector<inplace::InplaceOption>>* options) {
  using inplace::InplaceOption;
  const IndexedGraph& idx = graph.indexed_graph();
  static auto& finplace_option = Op::GetAttr<FInplaceOption>("FInplaceOption");
  static auto& finplace_identity = Op::GetAttr<FInplaceIdentity>("FInplaceIdentity");
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    if (node->is_variable()) {
      continue;
    }
    if (node->is_graph()) {
      auto subgraph = node->graph();
      if (subgraph->node_attrs.count(inplace::key) == 0) {
        auto subref = subgraph->CreateNodeColumn<vector<InplaceOption>>();
        ComputeInplaceOptionRec(*subgraph,
                                shapes->children[nid].get(),
                                subref.CopyOnWrite());
        options->children[nid] = subref;
      } else {
        options->children[nid] =
          subgraph->node_attrs.GetColumn<vector<InplaceOption>>(inplace::key);
      }
      options->value[nid] = Summarize(*subgraph,
                                      shapes->children[nid].get(),
                                      options->children[nid].get());
    } else if (finplace_option.count(node->op()) != 0) {
      const auto& inp_opt = finplace_option[node->op()](node->attrs);
      const auto& inp_idt = finplace_identity.count(node->op()) ?
        finplace_identity[node->op()](node->attrs) :
        vector<bool>(inp_opt.size(), false);
      CHECK_EQ(inp_opt.size(), inp_idt.size())
        << "FInplaceOption and FInplaceIdentity returned vectors of different "
        << "size for operator " << node->op()->name;
      options->value[nid].reserve(inp_opt.size());
      for (size_t i = 0; i < inp_opt.size(); ++i) {
        options->value[nid].emplace_back(InplaceOption{inp_opt[i], inp_idt[i]});
      }
    }
  }
}

Graph MXComputeInplaceOption(Graph &&graph) {
  using inplace::InplaceOption;
  using inplace::key;
  if (graph.node_attrs.count(key) == 0) {
    auto ref = graph.CreateNodeColumn<vector<InplaceOption>>();
    const auto* shapes = graph.entry_attrs.GetColumn<TShape>("shape").get();
    ComputeInplaceOptionRec(graph, shapes, ref.CopyOnWrite());
    graph.node_attrs.SetColumn(key, ref);
  }
  return graph;
}

NNVM_REGISTER_PASS(MXComputeInplaceOption)
.describe("Compute which inputs are ignored for each node.")
.set_body(MXComputeInplaceOption)
.set_change_graph(false)
.depend_entry_attr(shape::key)
.provide_node_attr(inplace::key);

}  // namespace pass
}  // namespace mxnet
