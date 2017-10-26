/*!
 * Copyright (c) 2016 by Contributors
 * \file graph_ptr_executor_v2.h
 * \brief Executor to execute the computation graph.
 */
#ifndef MXNET_EXECUTOR_GRAPH_EXECUTOR_V2_H_
#define MXNET_EXECUTOR_GRAPH_EXECUTOR_V2_H_

#include <mxnet/base.h>
#include "./mx_passes.h"

namespace mxnet {
namespace exec {

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

  GraphExecutorV2(std::shared_ptr<const nnvm::Graph> graph,
                  const Config& config = Config());

  ~GraphExecutorV2();

  void Run(const std::vector<NDArray>& arguments,
           std::vector<NDArray>* results,
           const RunOption& option = RunOption());

  const std::vector<std::string>& RequiredGraphAttrs() const;

  const nnvm::Graph& graph() const { return *graph_ptr_; }

 private:
  void SetupResources();
  void SetupOpResources();
  void SetupDataEntries();

  void FeedArgArray(const NDArray& array, size_t i);
  void FeedRstArray(const NDArray& array, size_t i);
  const NDArray& FetchRstArray(size_t i);

  void ResetDataEntries();
  void ResetClosure(uint32_t nid);

 private:
  // The graph to be evaluated.
  std::shared_ptr<const nnvm::Graph> graph_ptr_;
  const Config config_;
  // Attributes required for graph evaluation.
  const std::vector<std::string> required_graph_ptr_attrs_;
  // Operator nodes.
  nnvm::Column<pass::cl::Closure>* closures_;

  // Data entries.
  std::vector<NDArray> data_entries_;

  // Data structure used to feed argument to the operator.
  typedef std::pair<uint32_t, size_t> OpInputEntry;
  std::vector<std::vector<OpInputEntry>> arg_to_op_input_;
};

}  // namespace exec
}  // namespace mxnet

#endif  // MXNET_EXECUTOR_GRAPH_EXECUTOR_V2_H_
