#ifndef MXNET_IMPERATIVE_AUTOGRAD_H_
#define MXNET_IMPERATIVE_AUTOGRAD_H_

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
  void Record(const nnvm::NodeAttrs& attrs,
              const std::vector<NDArray*>& ndinputs,
              const std::vector<NDArray*>& ndoutputs);

  void AttachGrad(tape::TapeEntryId teid, OpReqType req_type, const NDArray& grad_buf);

  // ys cannot be empty. If xs is empty, the returned graph will compute
  // gradients for all grad-attached entries.
  nnvm::Graph GetSpecializedBackwardGraph(
      const std::vector<const NDArray*>& ys,
      const std::vector<const NDArray*>& xs,
      const std::vector<const NDArray*>& ys_grad);

  std::vector<tape::TapeEntryId> GetGradAttachedTapeEntryIds() const {
    return grad_attached_entries_;
  }

  void NewSession();

  static AutogradTape& Get();

 private:
  std::vector<nnvm::NodeEntry> GetGradTargets(
      const std::vector<const NDArray*>& xs) const;

  nnvm::Graph SpecializeForwardGraph(
      nnvm::Graph graph,
      const std::vector<nnvm::NodeEntry>& xs);

  void SaveInfo(const NDArray* nd, bool save_value);

  std::vector<tape::TapeEntryId> grad_attached_entries_;
  std::vector<std::vector<AGInfo2>> saved_info_;
};

}  // namespace ag
}  // namespace mxnet

#endif  // MXNET_IMPERATIVE_AUTOGRAD_H_
