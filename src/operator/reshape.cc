/*!
 * Copyright (c) 2015 by Contributors
 * \file flatten.cc
 * \brief
 * \author Bing Xu
*/

#include "./reshape-inl.h"


namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(ReshapeParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ReshapeOp<cpu, DType>(param);
  });
  return op;
}

Operator* ReshapeProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                        std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK(InferType(in_type, &out_type, &aux_type));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

ForwardSchemeRequests
FlattenProp::ForwardAlignedSchemes(
    const std::vector<TShape>& in_data_shapes,
    const std::vector<TShape>& out_data_shapes) const {
  using nnvm::Scheme;
  CHECK_GT(in_data_shapes.size(), 0);
  ForwardSchemeRequests reqs;
  // Flatten only support partition on the first and second dimension.
  for (size_t i = 0; i < 2; ++i) {
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
FlattenProp::BackwardAlignedSchemes(
    const std::vector<TShape>& out_grad_shapes,
    const std::vector<TShape>& in_data_shapes,
    const std::vector<TShape>& out_data_shapes,
    const std::vector<TShape>& in_grad_shapes) const {
  using nnvm::Scheme;
  CHECK_GT(out_grad_shapes.size(), 0);
  BackwardSchemeRequests reqs;
  // Flatten only support partition on the first and second dimension.
  for (size_t i = 0; i < 2; ++i) {
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

DMLC_REGISTER_PARAMETER(ReshapeParam);

MXNET_REGISTER_OP_PROPERTY(Reshape, ReshapeProp)
.describe("Reshape input to target shape")
.add_argument("data", "Symbol", "Input data to reshape.")
.add_arguments(ReshapeParam::__FIELDS__());

MXNET_REGISTER_OP_PROPERTY(Flatten, FlattenProp)
.describe("Flatten input")
.add_argument("data", "Symbol", "Input data to flatten.");
}  // namespace op
}  // namespace mxnet
