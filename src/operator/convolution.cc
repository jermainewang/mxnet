/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution.cc
 * \brief
 * \author Bing Xu
*/

#include "./convolution-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(ConvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ConvolutionOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *ConvolutionProp::CreateOperatorEx(Context ctx,
                                            std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}

ForwardSchemeRequests
ConvolutionProp::ForwardAlignedSchemes(
    const std::vector<TShape>& in_data_shapes,
    const std::vector<TShape>& out_data_shapes) const {
  using nnvm::Scheme;
  CHECK_EQ(param_.num_group, 1);
  ForwardSchemeRequest req1, req2, req3;
  // Data format NCHW
  // Filter format CoCiHW
  // Two operations in ConvolutionOp:
  //   - y = ConvForward(x, w)
  //   - y = AddTensor(y, b)
  // Therefore, there are following aligned schemes:
  //   - x: R, w: r, y: R, b: r
  //   - x: r, w: R, y: C, b: R
  //   - x: C, w: C, y: red, b: r
  req1.in_data_schemes.resize(param_.no_bias? 2: 3);
  req1.out_data_schemes.resize(1);
  req1.in_data_schemes[conv::kData] = Scheme::Cut(0);  // x: R
  req1.in_data_schemes[conv::kWeight] = Scheme::Rep(); // w: r
  req1.out_data_schemes[conv::kOut] = Scheme::Cut(0);  // y: R
  if (!param_.no_bias) {
    req1.in_data_schemes[conv::kBias] = Scheme::Rep(); // b: r
  }

  req2.in_data_schemes.resize(param_.no_bias? 2: 3);
  req2.out_data_schemes.resize(1);
  req2.in_data_schemes[conv::kData] = Scheme::Rep();    // x: r
  req2.in_data_schemes[conv::kWeight] = Scheme::Cut(0); // w: R
  req2.out_data_schemes[conv::kOut] = Scheme::Cut(1);   // y: C
  if (!param_.no_bias) {
    req2.in_data_schemes[conv::kBias] = Scheme::Cut(0); // b: R
  }

  req3.in_data_schemes.resize(param_.no_bias? 2: 3);
  req3.out_data_schemes.resize(1);
  req3.in_data_schemes[conv::kData] = Scheme::Cut(1);   // x: C
  req3.in_data_schemes[conv::kWeight] = Scheme::Cut(1); // w: C
  req3.out_data_schemes[conv::kOut] = Scheme::Red();    // y: red
  if (!param_.no_bias) {
    req3.in_data_schemes[conv::kBias] = Scheme::Rep();  // b: r
  }

  return {req1, req2, req3};
}

BackwardSchemeRequests
ConvolutionProp::BackwardAlignedSchemes(
    const std::vector<TShape>& out_grad_shapes,
    const std::vector<TShape>& in_data_shapes,
    const std::vector<TShape>& out_data_shapes,
    const std::vector<TShape>& in_grad_shapes) const {
  using nnvm::Scheme;
  CHECK_EQ(param_.num_group, 1);
  BackwardSchemeRequest req1, req2, req3;
  // Data format NCHW
  // Filter format CoCiHW
  // Three operations in the backward propagation:
  //   - dw = ConvBackwardFilter(dy, x)
  //   - dx = ConvBackwardData(dy, w)
  //   - db = ConvBackwardBias(dy)
  // Therefore, there are following aligned schemes:
  //   - dy: C, x: r, dw: R, w: R, dx: red, db: R
  //   - dy: r, x: C, dw: C, w: C, dx, C, db: r
  //   - dy: R, x: R, dw: red, w: r, dx: R, db: red
  req1.in_data_schemes.resize(2);
  req1.in_grad_schemes.resize(param_.no_bias? 2: 3);
  req1.out_grad_schemes.resize(1);
  req1.out_grad_schemes[conv::kOut] = Scheme::Cut(1);   // dy: C
  req1.in_data_schemes[conv::kData] = Scheme::Rep();    // x: r
  req1.in_grad_schemes[conv::kWeight] = Scheme::Cut(0); // dw: R
  req1.in_data_schemes[conv::kWeight] = Scheme::Cut(0); // w: R
  req1.in_grad_schemes[conv::kData] = Scheme::Red();    // dx: red
  if (!param_.no_bias) {
    req1.in_grad_schemes[conv::kBias] = Scheme::Cut(0); // db: R
  }

  req2.in_data_schemes.resize(2);
  req2.in_grad_schemes.resize(param_.no_bias? 2: 3);
  req2.out_grad_schemes.resize(1);
  req2.out_grad_schemes[conv::kOut] = Scheme::Rep();    // dy: r
  req2.in_data_schemes[conv::kData] = Scheme::Cut(1);   // x: C
  req2.in_grad_schemes[conv::kWeight] = Scheme::Cut(1); // dw: C
  req2.in_data_schemes[conv::kWeight] = Scheme::Cut(1); // w: C
  req2.in_grad_schemes[conv::kData] = Scheme::Cut(1);   // dx: C
  if (!param_.no_bias) {
    req2.in_grad_schemes[conv::kBias] = Scheme::Rep();  // db: r
  }

  req3.in_data_schemes.resize(2);
  req3.in_grad_schemes.resize(param_.no_bias? 2: 3);
  req3.out_grad_schemes.resize(1);
  req3.out_grad_schemes[conv::kOut] = Scheme::Cut(0);  // dy: R
  req3.in_data_schemes[conv::kData] = Scheme::Cut(0);  // x: R
  req3.in_grad_schemes[conv::kWeight] = Scheme::Red(); // dw: red
  req3.in_data_schemes[conv::kWeight] = Scheme::Rep(); // w: r
  req3.in_grad_schemes[conv::kData] = Scheme::Cut(0);  // dx: R
  if (!param_.no_bias) {
    req3.in_grad_schemes[conv::kBias] = Scheme::Red(); // db: red
  }

  return {req1, req2, req3};
}

DMLC_REGISTER_PARAMETER(ConvolutionParam);

MXNET_REGISTER_OP_PROPERTY(Convolution, ConvolutionProp)
.add_argument("data", "Symbol", "Input data to the ConvolutionOp.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(ConvolutionParam::__FIELDS__())
.describe("Apply convolution to input then add a bias.");

}  // namespace op
}  // namespace mxnet

