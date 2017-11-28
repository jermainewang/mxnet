#ifndef MXNET_IMPERATIVE_AUTOGRAD_H_
#define MXNET_IMPERATIVE_AUTOGRAD_H_

#include <mxnet/op_attr_types.h>
#include "./taping.h"

namespace mxnet {
namespace ag {

struct AGInfo2 {
  TShape shape;
  int dtype = -1;
  NDArray value;
  OpReqType req_type = kNullOp;
  NDArray grad_buffer;
  bool saved = false;
};

class AutogradTape {
 public:
  uint32_t Record(const nnvm::NodeAttrs& attrs,
                  const std::vector<NDArray*>& ndinputs,
                  const std::vector<NDArray*>& ndoutputs,
                  OpStatePtr state);

  uint32_t Record(const nnvm::NodeAttrs& attrs,
                  const std::vector<NDArray*>& ndinputs,
                  const std::vector<NDArray*>& ndoutputs,
                  nnvm::ColumnRef<OpStatePtr> graph_state);

  void AttachGrad(tape::TapeEntryId teid, OpReqType req_type, const NDArray& grad_buf);

  // ys cannot be empty. If xs is empty, the returned graph will compute
  // gradients for all grad-attached entries.
  nnvm::GraphPtr GetSpecializedBackwardGraph(
      const std::vector<const NDArray*>& ys,
      const std::vector<const NDArray*>& xs,
      const std::vector<const NDArray*>& ys_grad);

  std::vector<tape::TapeEntryId> GetGradAttachedTapeEntryIds() const {
    return grad_attached_entries_;
  }

  void NewSession();

  static AutogradTape& Get();

 private:
  uint32_t Record(const nnvm::NodeAttrs& attrs,
                  const std::vector<NDArray*>& ndinputs,
                  const std::vector<NDArray*>& ndoutputs);

  std::vector<nnvm::NodeEntry> GetGradTargets(
      const std::vector<const NDArray*>& xs) const;

  nnvm::Graph SpecializeForwardGraph(
      nnvm::Graph graph,
      const std::vector<nnvm::NodeEntry>& xs);

  void SaveInfo(const NDArray* nd, bool save_value);

  std::vector<tape::TapeEntryId> grad_attached_entries_;
  std::vector<std::vector<AGInfo2>> saved_info_;
  std::vector<OpStatePtr> saved_states_;
  std::vector<nnvm::ColumnRef<OpStatePtr>> saved_graph_states_;
};

}  // namespace ag
}  // namespace mxnet

#endif  // MXNET_IMPERATIVE_AUTOGRAD_H_
