#ifndef MXNET_IMPERATIVE_AUTOGRAD_H_
#define MXNET_IMPERATIVE_AUTOGRAD_H_

#include "./taping.h"

namespace mxnet {
namespace ag {

void RecordGradientInfo(const nnvm::NodeAttrs& attrs,
                        const std::vector<NDArray*>& ndinputs,
                        const std::vector<NDArray*>& ndoutputs,
                        tape::Tape* tape);

void GenerateBackwardGraph(const tape::Tape& tape,
                           const std::vector<const NDArray*>& ys,
                           const std::vector<const NDArray*>& xs);

}  // namespace ag
}  // namespace mxnet

#endif  // MXNET_IMPERATIVE_AUTOGRAD_H_
