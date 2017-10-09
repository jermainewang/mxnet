/*!
 * Copyright (c) 2016 by Contributors
 * \file graph_executor_v2.h
 * \brief Executor to execute the computation graph.
 */
#ifndef MXNET_EXECUTOR_GRAPH_EXECUTOR_V2_H_
#define MXNET_EXECUTOR_GRAPH_EXECUTOR_V2_H_

#include <mxnet/base.h>
#include "./mx_passes.h"

namespace mxnet {
namespace exec {

struct OpNode;
struct OpEntry;

// TODO(minjie):
// * Bulk Execution
// * DetectInplaceAddTo
// * Multi-devices, Data Parallelism
class GraphExecutorV2 {
 public:
  struct Config {
    bool dynamic_allocation{false};
    bool zero_copy{false};
  };
  struct RunOption {
    bool is_train{false};
  };

  GraphExecutorV2(const nnvm::Graph& graph,
                  const Config& config);

  void Run(const std::vector<NDArray>& inputs,
           const std::vector<NDArray>& outputs,
           const RunOption& option);

  const std::vector<std::string>& RequiredGraphAttrs() const;

 private:
  void AllocateResources();

  void ReleaseResources();

  void AllocateOpResources(const std::vector<NDArray>& data_entries);

  void AllocateDataEntries(std::vector<NDArray>* data_entries);

  void ReleaseOpResources();

  void ReleaseDataEntries();

 private:
  // The graph to be evaluated.
  const nnvm::Graph& graph_;
  const Config config_;
  // Attributes required for graph evaluation.
  const std::vector<std::string> required_graph_attrs_;
  // Operator nodes.
  nnvm::ColumnRef<OpNode> op_nodes_;
  // Internal data entry of each node.
  // Note that the NDArray here shares the memory pointer with the NDArray in
  // data_pool_. The reason why we use NDArray rather than NDArray pointer is
  // because different entries may have different shapes.
  // nnvm::ColumnRef<NDArray> data_entry_;
  // Internal data pool of allocated entries. Note that all NDArrays are 1D
  // arrays to represent memory buffers.
  //std::vector<NDArray> data_pool_;
};

}  // namespace exec
}  // namespace mxnet

#endif  // MXNET_EXECUTOR_GRAPH_EXECUTOR_V2_H_
