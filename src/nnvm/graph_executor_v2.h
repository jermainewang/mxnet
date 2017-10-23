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

// TODO(minjie):
// * Bulk Execution
// * DetectInplaceAddTo
// * Multi-devices, Data Parallelism
class GraphExecutorV2 {
 public:
  struct Config {
    bool dynamic_allocation{true};
    bool zero_copy{false};
    Config() {}
  };
  struct RunOption {
    bool is_train{false};
    RunOption() {}
  };

  GraphExecutorV2(const nnvm::Graph& graph,
                  const Config& config = Config());

  void Run(const std::vector<NDArray>& arguments,
           std::vector<NDArray>* results,
           const RunOption& option = RunOption());

  const std::vector<std::string>& RequiredGraphAttrs() const;

 private:
  void SetupResources();

  void SetupOpResources();

  void SetupDataEntries();

  void ReleaseResources();

  void ReleaseOpResources();

  void ReleaseDataEntries();

  void FeedArgArray(const NDArray& array, size_t i);
  void FeedRstArray(const NDArray& array, size_t i);
  const NDArray& FetchRstArray(size_t i);

  void ResetDataEntries();

 private:
  // The graph to be evaluated.
  const nnvm::Graph& graph_;
  const Config config_;
  // Attributes required for graph evaluation.
  const std::vector<std::string> required_graph_attrs_;

  // Data entries.
  std::vector<NDArray> data_entries_;
  // Operator nodes.
  nnvm::ColumnRef<OpNode> op_nodes_;
  // Data structure used to feed argument to the operator.
  typedef std::pair<uint32_t, size_t> OpInputEntry;
  std::vector<std::vector<OpInputEntry>> arg_to_op_input_;
};

}  // namespace exec
}  // namespace mxnet

#endif  // MXNET_EXECUTOR_GRAPH_EXECUTOR_V2_H_
