#include "./mx_passes.h"

using namespace std;
using namespace nnvm;

namespace mxnet {
namespace pass {

void PlaceDeviceDefaultRec(const Graph& graph,
                           const Context& default_ctx,
                           Column<Context>* ctx) {
  const auto& idx = graph.indexed_graph();
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    if (node->is_graph()) {
      auto subgraph = node->graph();
      ColumnRef<Context> subctx = subgraph->CreateNodeColumn<Context>(default_ctx);
      PlaceDeviceDefaultRec(*subgraph, default_ctx, subctx.CopyOnWrite());
      ctx->children[nid] = subctx;
    } else {
      // Do nothing.
    }
  }
}

Graph MXPlaceDevice(Graph&& graph) {
  const Context& default_ctx = graph.GetGlobalAttr<Context>("default_ctx");
  ColumnRef<Context> ctxref = graph.CreateNodeColumn<Context>(default_ctx);
  PlaceDeviceDefaultRec(graph, default_ctx, ctxref.CopyOnWrite());
  graph.node_attrs.SetColumn(ctx::key, ctxref);
  return graph;
}

NNVM_REGISTER_PASS(MXPlaceDevice)
.describe("Assign each node to different devices.")
.set_body(MXPlaceDevice)
.set_change_graph(false)
.depend_global_attr("default_ctx")
.provide_node_attr(ctx::key);

}  // namespace pass
}  // namespace mxnet
