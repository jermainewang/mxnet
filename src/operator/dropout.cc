/*!
 * Copyright (c) 2015 by Contributors
 * \file dropout.cc
 * \brief
 * \author Bing Xu
*/

#include "./dropout-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(DropoutParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new DropoutOp<cpu, DType>(param);
  });
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *DropoutProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                              std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

ForwardSchemeRequests
DropoutProp::ForwardAlignedSchemes(
    const std::vector<TShape>& in_data_shapes,
    const std::vector<TShape>& out_data_shapes) const {
  using nnvm::Scheme;
  size_t ndim = in_data_shapes[dropout::kData].ndim();
  ForwardSchemeRequests reqs;
  for (size_t i = 0; i < ndim; ++i) {
    ForwardSchemeRequest req;
    for (size_t j = 0; j < in_data_shapes.size(); ++j) {
      req.in_data_schemes.push_back(Scheme::Cut(i));
    }
    for (size_t j = 0; j < out_data_shapes.size(); ++j) {
      req.out_data_schemes.push_back(Scheme::Cut(i));
    }
    reqs.push_back(req);
  }
  return reqs;
}

BackwardSchemeRequests
DropoutProp::BackwardAlignedSchemes(
    const std::vector<TShape>& out_grad_shapes,
    const std::vector<TShape>& in_data_shapes,
    const std::vector<TShape>& out_data_shapes,
    const std::vector<TShape>& in_grad_shapes) const {
  using nnvm::Scheme;
  size_t ndim = out_grad_shapes[dropout::kOut].ndim();
  BackwardSchemeRequests reqs;
  for (size_t i = 0; i < ndim; ++i) {
    BackwardSchemeRequest req;
    for (size_t j = 0; j < out_grad_shapes.size(); ++j) {
      req.out_grad_schemes.push_back(Scheme::Cut(i));
    }
    for (size_t j = 0; j < in_data_shapes.size(); ++j) {
      req.in_data_schemes.push_back(Scheme::Cut(i));
    }
    for (size_t j = 0; j < out_data_shapes.size(); ++j) {
      req.out_data_schemes.push_back(Scheme::Cut(i));
    }
    for (size_t j = 0; j < in_grad_shapes.size(); ++j) {
      req.in_grad_schemes.push_back(Scheme::Cut(i));
    }
    reqs.push_back(req);
  }
  return reqs;
}

DMLC_REGISTER_PARAMETER(DropoutParam);

MXNET_REGISTER_OP_PROPERTY(Dropout, DropoutProp)
.describe("Apply dropout to input")
.add_argument("data", "Symbol", "Input data to dropout.")
.add_arguments(DropoutParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet


