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

struct Closure;

// TODO(minjie):
// * DetectInplaceAddTo
// * Multi-devices, Data Parallelism
class GraphExecutorV2 {
 public:
  typedef nnvm::ColumnRef<std::shared_ptr<OpExecutorV2>> ExecState;
  struct Config {
    bool dynamic_allocation{true};
    bool zero_copy{true};
    bool bulk_execution{true};
    Config() {}
  };
  struct RunOption {
    bool is_train{false};
    RunOption() {}
  };

  GraphExecutorV2(std::shared_ptr<const nnvm::Graph> graph,
                  const ExecState& fwd_state = ExecState(),
                  const Config& config = Config());

  ~GraphExecutorV2();

  void Run(const std::vector<NDArray>& arguments,
           std::vector<NDArray>* results,
           const RunOption& option = RunOption());

  const std::vector<std::string>& RequiredGraphAttrs() const;

  // XXX(minjie): Note that once this is called, one must make sure that
  // the GraphExecutor will not be used any more. Otherwise,
  // the returned state may be polluted.
  const ExecState& GetState() const { return op_execs_; }

  const nnvm::Graph& graph() const { return *graph_ptr_; }

 private:
  void AttachOps();
  void SetupResources();
  void SetupOpResources();
  void SetupDataEntries();
  void CheckAllowBulkExec() const;

  void FeedArgArray(const NDArray& array, size_t i);
  void FeedRstArray(const NDArray& array, size_t i);
  const NDArray& FetchRstArray(size_t i);

  void RunOps();
  void RunOpsInBulk();

  void ResetDataEntries();

 private:
  // The graph to be evaluated.
  std::shared_ptr<const nnvm::Graph> graph_ptr_;
  // Configurations of this executor.
  const Config config_;
  // OpExecutors of forward graph.
  const ExecState fwd_execs_;
  // Attributes required for graph evaluation.
  const std::vector<std::string> required_graph_ptr_attrs_;

  // Internal (stateful) operator executors.
  ExecState op_execs_;
  // Internal data structure for executing each node.
  nnvm::ColumnRef<Closure> closures_;

  // Data entries.
  std::vector<NDArray> data_entries_;

  // Data structure used to feed argument to the operator.
  typedef std::pair<uint32_t, size_t> OpInputEntry;
  std::vector<std::vector<OpInputEntry>> arg_to_op_input_;
};

}  // namespace exec
}  // namespace mxnet

#endif  // MXNET_EXECUTOR_GRAPH_EXECUTOR_V2_H_
