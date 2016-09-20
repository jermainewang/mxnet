/*!
 * Copyright (c) 2015 by Contributors
 * \file activation.cc
 * \brief activation op
 * \author Bing Xu
*/
#include "./activation-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(ActivationParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    switch (param.act_type) {
      case activation::kReLU:
        op = new ActivationOp<cpu, mshadow_op::relu, mshadow_op::relu_grad, DType>();
        break;
      case activation::kSigmoid:
        op = new ActivationOp<cpu, mshadow_op::sigmoid, mshadow_op::sigmoid_grad, DType>();
        break;
      case activation::kTanh:
        op = new ActivationOp<cpu, mshadow_op::tanh, mshadow_op::tanh_grad, DType>();
        break;
      case activation::kSoftReLU:
        op = new ActivationOp<cpu, mshadow_op::softrelu, mshadow_op::softrelu_grad, DType>();
        break;
      default:
        LOG(FATAL) << "unknown activation type";
    }
  })
  return op;
}

ForwardSchemeRequests
ActivationProp::ForwardAlignedSchemes(
    const std::vector<TShape>& in_data_shapes,
    const std::vector<TShape>& out_data_shapes) const {
  using nnvm::Scheme;
  size_t ndim = in_data_shapes[activation::kData].ndim();
  ForwardSchemeRequests reqs;
  for (size_t i = 0; i < ndim; ++i) {
    ForwardSchemeRequest req;
    req.in_data_schemes.push_back(Scheme::Cut(i));
    req.out_data_schemes.push_back(Scheme::Cut(i));
    reqs.push_back(req);
  }
  return reqs;
}

BackwardSchemeRequests
ActivationProp::BackwardAlignedSchemes(
    const std::vector<TShape>& out_grad_shapes,
    const std::vector<TShape>& in_data_shapes,
    const std::vector<TShape>& out_data_shapes,
    const std::vector<TShape>& in_grad_shapes) const {
  using nnvm::Scheme;
  size_t ndim = out_grad_shapes[activation::kOut].ndim();
  BackwardSchemeRequests reqs;
  for (size_t i = 0; i < ndim; ++i) {
    BackwardSchemeRequest req;
#if MXNET_USE_CUDNN == 1
    req.in_data_schemes.push_back(Scheme::Cut(i));
    req.out_data_schemes.push_back(Scheme::Cut(i));
    req.out_grad_schemes.push_back(Scheme::Cut(i));
#else
    req.out_data_schemes.push_back(Scheme::Cut(i));
    req.out_grad_schemes.push_back(Scheme::Cut(i));
#endif  // MXNET_USE_CUDNN
    req.in_grad_schemes.push_back(Scheme::Cut(i));
    reqs.push_back(req);
  }
  return reqs;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *ActivationProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(ActivationParam);

MXNET_REGISTER_OP_PROPERTY(Activation, ActivationProp)
.describe("Apply activation function to input."
          "Softmax Activation is only available with CUDNN on GPU"
          "and will be computed at each location across channel if input is 4D.")
.add_argument("data", "Symbol", "Input data to activation function.")
.add_arguments(ActivationParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

