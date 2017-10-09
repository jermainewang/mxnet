#include "./mx_passes.h"

using namespace std;
using namespace nnvm;

namespace mxnet {
namespace pass {

void PlaceDeviceDefaultRec(const Graph& graph,
                           Column<int>* device) {
  const auto& idx = graph.indexed_graph();
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    if (node->is_graph()) {
      auto subgraph = node->graph();
      ColumnRef<int> subdev = subgraph->CreateNodeColumn<int>(0);
      PlaceDeviceDefaultRec(*subgraph, subdev.CopyOnWrite());
      device->children[nid] = subdev;
    } else {
      // Do nothing.
    }
  }
}

Graph MXPlaceDefaultDevice(Graph&& graph) {
  ColumnRef<int> devref = graph.CreateNodeColumn<int>(0);
  PlaceDeviceDefaultRec(graph, devref.CopyOnWrite());
  graph.node_attrs.SetColumn(ctx::device_key, devref);
  return graph;
}

NNVM_REGISTER_PASS(MXPlaceDefaultDevice)
.describe("Assign all nodes to the default device.")
.set_body(MXPlaceDefaultDevice)
.set_change_graph(false)
.depend_global_attr(ctx::ctx_key)
.provide_node_attr(ctx::device_key);

}  // namespace pass
}  // namespace mxnet
