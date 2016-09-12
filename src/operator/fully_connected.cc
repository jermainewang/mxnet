/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connected.cc
 * \brief fully connect operator
*/
#include "./fully_connected-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(FullyConnectedParam param, int dtype) {
  Operator *op = NULL;
  switch (dtype) {
  case mshadow::kFloat32:
    op = new FullyConnectedOp<cpu, float>(param);
    break;
  case mshadow::kFloat64:
    op = new FullyConnectedOp<cpu, double>(param);
    break;
  case mshadow::kFloat16:
    LOG(FATAL) << "float16 fully connected layer is currently"
                  "only supported by CuDNN version.";
    break;
  default:
    LOG(FATAL) << "Unsupported type " << dtype;
  }
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *FullyConnectedProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

ForwardSchemeRequests
FullyConnectedProp::ForwardAlignedSchemes(
    const std::vector<TShape>& in_data_shapes,
    const std::vector<TShape>& out_data_shapes) const {
  using nnvm::Scheme;
  // TODO(minjie): only support no bias right now.
  CHECK(param_.no_bias);
  CHECK_EQ(in_data_shapes[fullc::kData].ndim(), 2);
  CHECK_EQ(in_data_shapes[fullc::kWeight].ndim(), 2);
  CHECK_EQ(out_data_shapes[fullc::kOut].ndim(), 2);
  ForwardSchemeRequest req1, req2, req3;
  // One matmult in the forward propagation:
  //   - y = dot(x, w.T)
  // Therefore, there are following aligned schemes:
  //   - x: R, w: r, y: R
  //   - x: r, w: R, y: C
  //   - x: C, w: C, y: red
  req1.in_data_schemes.resize(2);
  req1.out_data_schemes.resize(1);
  req1.in_data_schemes[fullc::kData] = Scheme::Cut(0);  // x: R
  req1.in_data_schemes[fullc::kWeight] = Scheme::Rep(); // w: r
  req1.out_data_schemes[fullc::kOut] = Scheme::Cut(0);  // y: R

  req2.in_data_schemes.resize(2);
  req2.out_data_schemes.resize(1);
  req2.in_data_schemes[fullc::kData] = Scheme::Rep(); // x: r
  req2.in_data_schemes[fullc::kWeight] = Scheme::Cut(0);  // w: R
  req2.out_data_schemes[fullc::kOut] = Scheme::Cut(1);  // y: C

  req3.in_data_schemes.resize(2);
  req3.out_data_schemes.resize(1);
  req3.in_data_schemes[fullc::kData] = Scheme::Cut(1); // x: C
  req3.in_data_schemes[fullc::kWeight] = Scheme::Cut(1); // w: C
  req3.out_data_schemes[fullc::kOut] = Scheme::Red(); // y: red

  return {req1, req2, req3};
}

BackwardSchemeRequests
FullyConnectedProp::BackwardAlignedSchemes(
    const std::vector<TShape>& out_grad_shapes,
    const std::vector<TShape>& in_data_shapes,
    const std::vector<TShape>& out_data_shapes,
    const std::vector<TShape>& in_grad_shapes) const {
  using nnvm::Scheme;
  // TODO(minjie): only support no bias right now.
  CHECK(param_.no_bias);
  BackwardSchemeRequest req1, req2, req3;

  // Two matmults in the backward propagation:
  //   - dw = dot(dy.T, x)
  //   - dx = dot(dy, w)
  // Therefore, there are following aligned schemes:
  //   - dy: C, x: r, dw: R, w: R, dx: red
  //   - dy: r, x: C, dw: C, w: C, dx, C
  //   - dy: R, x: R, dw: red, w: r, dx: R
  req1.in_data_schemes.resize(2);
  req1.in_grad_schemes.resize(2);
  req1.out_grad_schemes.resize(1);
  req1.out_grad_schemes[fullc::kOut] = Scheme::Cut(1); // dy: C
  req1.in_data_schemes[fullc::kData] = Scheme::Rep(); // x: r
  req1.in_grad_schemes[fullc::kWeight] = Scheme::Cut(0); // dw: R
  req1.in_data_schemes[fullc::kWeight] = Scheme::Cut(0); // w: R
  req1.in_grad_schemes[fullc::kData] = Scheme::Red(); // dx: red

  req2.in_data_schemes.resize(2);
  req2.in_grad_schemes.resize(2);
  req2.out_grad_schemes.resize(1);
  req2.out_grad_schemes[fullc::kOut] = Scheme::Rep(); // dy: r
  req2.in_data_schemes[fullc::kData] = Scheme::Cut(1); // x: C
  req2.in_grad_schemes[fullc::kWeight] = Scheme::Cut(1); // dw: C
  req2.in_data_schemes[fullc::kWeight] = Scheme::Cut(1); // w: C
  req2.in_grad_schemes[fullc::kData] = Scheme::Cut(1); // dx: C

  req3.in_data_schemes.resize(2);
  req3.in_grad_schemes.resize(2);
  req3.out_grad_schemes.resize(1);
  req3.out_grad_schemes[fullc::kOut] = Scheme::Cut(0); // dy: R
  req3.in_data_schemes[fullc::kData] = Scheme::Cut(0); // x: R
  req3.in_grad_schemes[fullc::kWeight] = Scheme::Red(); // dw: red
  req3.in_data_schemes[fullc::kWeight] = Scheme::Rep(); // w: r
  req3.in_grad_schemes[fullc::kData] = Scheme::Cut(0); // dx: R

  return {req1, req2, req3};
}

DMLC_REGISTER_PARAMETER(FullyConnectedParam);

MXNET_REGISTER_OP_PROPERTY(FullyConnected, FullyConnectedProp)
.describe("Apply matrix multiplication to input then add a bias.")
.add_argument("data", "Symbol", "Input data to the FullyConnectedOp.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(FullyConnectedParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
