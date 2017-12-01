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
  typedef nnvm::ColumnRef<FunctorInfo> ExecState;
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

  GraphExecutorV2(nnvm::GraphPtr graph,
                  const ExecState& fwd_states = ExecState(),
                  const Config& config = Config());

  ~GraphExecutorV2();

  void Run(const std::vector<NDArray>& arguments,
           const std::vector<OpReqType>& result_req,
           std::vector<NDArray>* results,
           const RunOption& option = RunOption());

  const std::vector<std::string>& RequiredGraphAttrs() const;

  // XXX(minjie): Note that once this is called, one must make sure that
  // the GraphExecutor will not be used any more. Otherwise,
  // the returned state may be polluted.
  const ExecState& states() const { return states_; }

  nnvm::GraphPtr graph() const { return graph_ptr_; }

 private:
  void AttachOps();
  void SetupResources();
  void SetupDataEntries();
  void CheckAllowBulkExec() const;

  const NDArray& FetchRstArray(size_t i);

  void RunOps();
  void RunOpsInBulk(const std::vector<NDArray>& arguments,
                    const std::vector<NDArray>& results);

  void ResetDataEntries();

 private:
  // The graph to be evaluated.
  nnvm::GraphPtr graph_ptr_;
  // Configurations of this executor.
  const Config config_;
  // OpExecutors of forward graph.
  const ExecState fwd_states_;
  // Attributes required for graph evaluation.
  const std::vector<std::string> required_graph_ptr_attrs_;

  // Executor states.
  ExecState states_;
  // Internal operator executors.
  nnvm::ColumnRef<std::shared_ptr<OpExecutorV2>> op_execs_;
  // Internal data structure for executing each node.
  nnvm::ColumnRef<Closure> closures_;

  // Data entries.
  std::vector<NDArray> data_entries_;
};

}  // namespace exec
}  // namespace mxnet

#endif  // MXNET_EXECUTOR_GRAPH_EXECUTOR_V2_H_
