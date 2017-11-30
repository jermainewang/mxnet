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
      PlaceDeviceDefaultRec(*subgraph, device->children[nid].CopyOnWrite());
    } else {
      device->value[nid] = 0;
    }
  }
}

Graph MXPlaceDefaultDevice(Graph&& graph) {
  PlaceDeviceDefaultRec(graph,
      graph.CreateOrWriteNodeColumn<int>(ctx::device_key, 0));
  const auto& ctx = GetPassArgument<vector<Context>>(graph, ctx::arg_name);
  graph.global_attrs[ctx::ctx_key] = std::make_shared<any>(ctx);
  return graph;
}

NNVM_REGISTER_PASS(MXPlaceDefaultDevice)
.describe("Assign all nodes to the default device.")
.set_body(MXPlaceDefaultDevice)
.set_change_graph(false)
.set_argument(ctx::arg_name)
.provide_global_attr(ctx::ctx_key)
.provide_node_attr(ctx::device_key);

}  // namespace pass
}  // namespace mxnet
