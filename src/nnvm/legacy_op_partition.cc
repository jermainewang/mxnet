#include "./legacy_op_partition.h"

#include <dmlc/base.h>
#include <mxnet/base.h>
#include <nnvm/scheme.h>
#include "../operator/convolution-inl.h"
#include "../operator/fully_connected-inl.h"
#include "./legacy_op_util.h"

using namespace std;
using nnvm::NodeAttrs;
using nnvm::pass::Scheme;
using nnvm::pass::SchemeRequest;

namespace mxnet {
namespace op {

struct ForwardSchemeRequest {
  std::vector<Scheme> in_data_schemes;
  std::vector<Scheme> out_data_schemes;
};

typedef std::vector<ForwardSchemeRequest> ForwardSchemeRequests;

struct BackwardSchemeRequest {
  std::vector<Scheme> out_grad_schemes;
  std::vector<Scheme> in_data_schemes;
  std::vector<Scheme> out_data_schemes;
  std::vector<Scheme> in_grad_schemes;
};

typedef std::vector<BackwardSchemeRequest> BackwardSchemeRequests;

template<typename ParamType>
ForwardSchemeRequests ForwardAlignedSchemes(
    const ParamType& param,
    const std::vector<TShape>& in_data_shapes,
    const std::vector<TShape>& out_data_shapes) {
  LOG(FATAL) << "Not implemented";
  return ForwardSchemeRequests();
}

template<typename ParamType>
BackwardSchemeRequests BackwardAlignedSchemes(
    const ParamType& param,
    const std::vector<TShape>& out_grad_shapes,
    const std::vector<TShape>& in_data_shapes,
    const std::vector<TShape>& out_data_shapes,
    const std::vector<TShape>& in_grad_shapes) {
  LOG(FATAL) << "Not implemented";
  return BackwardSchemeRequests();
}

///////////////////////////////////////////////////////////////////////////

// Partitioner that will only partition on the first dimension (batch dimension).
// It will generate N independent operators that works on different cuts of data.
ForwardSchemeRequests BatchForwardSchemes(
    const std::vector<TShape>& in_data_shapes,
    const std::vector<TShape>& out_data_shapes) {
  ForwardSchemeRequest req;
  Scheme batch_cut = Scheme::Cut(0);
  req.in_data_schemes.resize(in_data_shapes.size(), batch_cut);
  req.out_data_schemes.resize(out_data_shapes.size(), batch_cut);
  return {req};
}

// Partitioner that will only partition on the first dimension (batch dimension).
// It will generate N independent operators that works on different cuts of data.
BackwardSchemeRequests BatchBackwardSchemes(
    const std::vector<TShape>& out_grad_shapes,
    const std::vector<TShape>& in_data_shapes,
    const std::vector<TShape>& out_data_shapes,
    const std::vector<TShape>& in_grad_shapes) {
  BackwardSchemeRequest req;
  Scheme batch_cut = Scheme::Cut(0);
  req.out_grad_schemes.resize(out_grad_shapes.size(), batch_cut);
  req.in_data_schemes.resize(in_data_shapes.size(), batch_cut);
  req.out_data_schemes.resize(out_data_shapes.size(), batch_cut);
  req.in_grad_schemes.resize(in_grad_shapes.size(), batch_cut);
  return {req};
}

template<size_t K>
vector<SchemeRequest> CutFirstKDimsSchemes(
    const NodeAttrs& attrs,
    const std::vector<TShape>& input_shapes,
    const std::vector<TShape>& output_shapes) {
  vector<SchemeRequest> reqs;
  for (size_t i = 0; i < K; ++i) {
    SchemeRequest req;
    for (size_t j = 0; j < input_shapes.size(); ++j) {
      CHECK_LT(i, input_shapes[j].ndim());
      req.input_schemes.push_back(Scheme::Cut(i));
    }
    for (size_t j = 0; j < output_shapes.size(); ++j) {
      CHECK_LT(i, output_shapes[j].ndim());
      req.output_schemes.push_back(Scheme::Cut(i));
    }
    reqs.push_back(req);
  }
  return reqs;
}

///////////////////////////////////////////////////////////////////////////////////
template<>
ForwardSchemeRequests ForwardAlignedSchemes(
    const FullyConnectedParam& param,
    const std::vector<TShape>& in_data_shapes,
    const std::vector<TShape>& out_data_shapes) {
  CHECK_EQ(in_data_shapes[fullc::kData].ndim(), 2);
  CHECK_EQ(in_data_shapes[fullc::kWeight].ndim(), 2);
  CHECK_EQ(out_data_shapes[fullc::kOut].ndim(), 2);
  ForwardSchemeRequest req1, req2, req3;
  // One matmult in the forward propagation:
  //   - y = dot(x, w.T) + b
  // Therefore, there are following aligned schemes:
  //   - x: R, w: r, y: R, b: r
  //   - x: r, w: R, y: C, b: R
  //   - x: C, w: C, y: red, b: r
  req1.in_data_schemes.resize(param.no_bias? 2: 3);
  req1.out_data_schemes.resize(1);
  req1.in_data_schemes[fullc::kData] = Scheme::Cut(0);  // x: R
  req1.in_data_schemes[fullc::kWeight] = Scheme::Rep(); // w: r
  req1.out_data_schemes[fullc::kOut] = Scheme::Cut(0);  // y: R
  if (!param.no_bias) {
    req1.in_data_schemes[fullc::kBias] = Scheme::Rep(); // b: r
  }

  req2.in_data_schemes.resize(param.no_bias? 2: 3);
  req2.out_data_schemes.resize(1);
  req2.in_data_schemes[fullc::kData] = Scheme::Rep();    // x: r
  req2.in_data_schemes[fullc::kWeight] = Scheme::Cut(0); // w: R
  req2.out_data_schemes[fullc::kOut] = Scheme::Cut(1);   // y: C
  if (!param.no_bias) {
    req2.in_data_schemes[fullc::kBias] = Scheme::Cut(0); // b: R
  }

  req3.in_data_schemes.resize(param.no_bias? 2: 3);
  req3.out_data_schemes.resize(1);
  req3.in_data_schemes[fullc::kData] = Scheme::Cut(1);   // x: C
  req3.in_data_schemes[fullc::kWeight] = Scheme::Cut(1); // w: C
  req3.out_data_schemes[fullc::kOut] = Scheme::Red();    // y: red
  if (!param.no_bias) {
    req3.in_data_schemes[fullc::kBias] = Scheme::Rep();  // b: r
  }

  return {req1, req2, req3};
}

template<>
BackwardSchemeRequests BackwardAlignedSchemes(
    const FullyConnectedParam& param,
    const std::vector<TShape>& out_grad_shapes,
    const std::vector<TShape>& in_data_shapes,
    const std::vector<TShape>& out_data_shapes,
    const std::vector<TShape>& in_grad_shapes) {
  BackwardSchemeRequest req1, req2, req3;

  // Two matmults in the backward propagation:
  //   - dw = dot(dy.T, x)
  //   - dx = dot(dy, w)
  //   - db = reduce_sum(dy, 0)
  // Therefore, there are following aligned schemes:
  //   - dy: C, x: r, dw: R, w: R, dx: red, db: R
  //   - dy: r, x: C, dw: C, w: C, dx, C, db: r
  //   - dy: R, x: R, dw: red, w: r, dx: R, db: red
  req1.in_data_schemes.resize(2);
  req1.in_grad_schemes.resize(param.no_bias? 2: 3);
  req1.out_grad_schemes.resize(1);
  req1.out_grad_schemes[fullc::kOut] = Scheme::Cut(1);   // dy: C
  req1.in_data_schemes[fullc::kData] = Scheme::Rep();    // x: r
  req1.in_grad_schemes[fullc::kWeight] = Scheme::Cut(0); // dw: R
  req1.in_data_schemes[fullc::kWeight] = Scheme::Cut(0); // w: R
  req1.in_grad_schemes[fullc::kData] = Scheme::Red();    // dx: red
  if (!param.no_bias) {
    req1.in_grad_schemes[fullc::kBias] = Scheme::Cut(0); // db: R
  }

  req2.in_data_schemes.resize(2);
  req2.in_grad_schemes.resize(param.no_bias? 2: 3);
  req2.out_grad_schemes.resize(1);
  req2.out_grad_schemes[fullc::kOut] = Scheme::Rep();    // dy: r
  req2.in_data_schemes[fullc::kData] = Scheme::Cut(1);   // x: C
  req2.in_grad_schemes[fullc::kWeight] = Scheme::Cut(1); // dw: C
  req2.in_data_schemes[fullc::kWeight] = Scheme::Cut(1); // w: C
  req2.in_grad_schemes[fullc::kData] = Scheme::Cut(1);   // dx: C
  if (!param.no_bias) {
    req2.in_grad_schemes[fullc::kBias] = Scheme::Rep();  // db: r
  }

  req3.in_data_schemes.resize(2);
  req3.in_grad_schemes.resize(param.no_bias? 2: 3);
  req3.out_grad_schemes.resize(1);
  req3.out_grad_schemes[fullc::kOut] = Scheme::Cut(0);  // dy: R
  req3.in_data_schemes[fullc::kData] = Scheme::Cut(0);  // x: R
  req3.in_grad_schemes[fullc::kWeight] = Scheme::Red(); // dw: red
  req3.in_data_schemes[fullc::kWeight] = Scheme::Rep(); // w: r
  req3.in_grad_schemes[fullc::kData] = Scheme::Cut(0);  // dx: R
  if (!param.no_bias) {
    req3.in_grad_schemes[fullc::kBias] = Scheme::Red(); // db: red
  }

  return {req1, req2, req3};
}

template<>
ForwardSchemeRequests ForwardAlignedSchemes(
    const ConvolutionParam& param,
    const std::vector<TShape>& in_data_shapes,
    const std::vector<TShape>& out_data_shapes) {
  CHECK_EQ(param.num_group, 1);
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
  req1.in_data_schemes.resize(param.no_bias? 2: 3);
  req1.out_data_schemes.resize(1);
  req1.in_data_schemes[conv::kData] = Scheme::Cut(0);  // x: R
  req1.in_data_schemes[conv::kWeight] = Scheme::Rep(); // w: r
  req1.out_data_schemes[conv::kOut] = Scheme::Cut(0);  // y: R
  if (!param.no_bias) {
    req1.in_data_schemes[conv::kBias] = Scheme::Rep(); // b: r
  }

  req2.in_data_schemes.resize(param.no_bias? 2: 3);
  req2.out_data_schemes.resize(1);
  req2.in_data_schemes[conv::kData] = Scheme::Rep();    // x: r
  req2.in_data_schemes[conv::kWeight] = Scheme::Cut(0); // w: R
  req2.out_data_schemes[conv::kOut] = Scheme::Cut(1);   // y: C
  if (!param.no_bias) {
    req2.in_data_schemes[conv::kBias] = Scheme::Cut(0); // b: R
  }

  req3.in_data_schemes.resize(param.no_bias? 2: 3);
  req3.out_data_schemes.resize(1);
  req3.in_data_schemes[conv::kData] = Scheme::Cut(1);   // x: C
  req3.in_data_schemes[conv::kWeight] = Scheme::Cut(1); // w: C
  req3.out_data_schemes[conv::kOut] = Scheme::Red();    // y: red
  if (!param.no_bias) {
    req3.in_data_schemes[conv::kBias] = Scheme::Rep();  // b: r
  }

  return {req1, req2, req3};
}

template<>
BackwardSchemeRequests BackwardAlignedSchemes(
    const ConvolutionParam& param,
    const std::vector<TShape>& out_grad_shapes,
    const std::vector<TShape>& in_data_shapes,
    const std::vector<TShape>& out_data_shapes,
    const std::vector<TShape>& in_grad_shapes) {
  CHECK_EQ(param.num_group, 1);
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
  req1.in_grad_schemes.resize(param.no_bias? 2: 3);
  req1.out_grad_schemes.resize(1);
  req1.out_grad_schemes[conv::kOut] = Scheme::Cut(1);   // dy: C
  req1.in_data_schemes[conv::kData] = Scheme::Rep();    // x: r
  req1.in_grad_schemes[conv::kWeight] = Scheme::Cut(0); // dw: R
  req1.in_data_schemes[conv::kWeight] = Scheme::Cut(0); // w: R
  req1.in_grad_schemes[conv::kData] = Scheme::Red();    // dx: red
  if (!param.no_bias) {
    req1.in_grad_schemes[conv::kBias] = Scheme::Cut(0); // db: R
  }

  req2.in_data_schemes.resize(2);
  req2.in_grad_schemes.resize(param.no_bias? 2: 3);
  req2.out_grad_schemes.resize(1);
  req2.out_grad_schemes[conv::kOut] = Scheme::Rep();    // dy: r
  req2.in_data_schemes[conv::kData] = Scheme::Cut(1);   // x: C
  req2.in_grad_schemes[conv::kWeight] = Scheme::Cut(1); // dw: C
  req2.in_data_schemes[conv::kWeight] = Scheme::Cut(1); // w: C
  req2.in_grad_schemes[conv::kData] = Scheme::Cut(1);   // dx: C
  if (!param.no_bias) {
    req2.in_grad_schemes[conv::kBias] = Scheme::Rep();  // db: r
  }

  req3.in_data_schemes.resize(2);
  req3.in_grad_schemes.resize(param.no_bias? 2: 3);
  req3.out_grad_schemes.resize(1);
  req3.out_grad_schemes[conv::kOut] = Scheme::Cut(0);  // dy: R
  req3.in_data_schemes[conv::kData] = Scheme::Cut(0);  // x: R
  req3.in_grad_schemes[conv::kWeight] = Scheme::Red(); // dw: red
  req3.in_data_schemes[conv::kWeight] = Scheme::Rep(); // w: r
  req3.in_grad_schemes[conv::kData] = Scheme::Cut(0);  // dx: R
  if (!param.no_bias) {
    req3.in_grad_schemes[conv::kBias] = Scheme::Red(); // db: red
  }

  return {req1, req2, req3};
}


/////////////////////////////////////////////////////////////////
ForwardSchemeRequests ForwardAlignedSchemesCaller(
    const NodeAttrs& attrs,
    const std::vector<TShape>& input_shapes,
    const std::vector<TShape>& output_shapes) {
  std::vector<std::pair<std::string, std::string> > kwargs(
      attrs.dict.begin(), attrs.dict.end());
  if (attrs.op->name == "FullyConnected") {
    FullyConnectedParam param;
    param.Init(kwargs);
    return ForwardAlignedSchemes<FullyConnectedParam>(param, input_shapes, output_shapes);
  } else {
    LOG(FATAL) << "No aligned scheme defined for operator: " << attrs.op->name;
    return ForwardSchemeRequests();
  }
}

BackwardSchemeRequests BackwardAlignedSchemesCaller(
    const NodeAttrs& attrs,
    const std::vector<TShape>& out_grad_shapes,
    const std::vector<TShape>& in_data_shapes,
    const std::vector<TShape>& out_data_shapes,
    const std::vector<TShape>& in_grad_shapes) {
  std::vector<std::pair<std::string, std::string> > kwargs(
      attrs.dict.begin(), attrs.dict.end());
  if (attrs.op->name == "_backward_FullyConnected") {
    FullyConnectedParam param;
    param.Init(kwargs);
    return BackwardAlignedSchemes<FullyConnectedParam>(
        param, out_grad_shapes, in_data_shapes,
        out_data_shapes, in_grad_shapes);
  } else {
    LOG(FATAL) << "No aligned scheme defined for operator: " << attrs.op->name;
    return BackwardSchemeRequests();
  }
}

std::vector<SchemeRequest> OpForwardAlignedSchemes(
    const NodeAttrs& attrs,
    const std::vector<TShape>& input_shapes,
    const std::vector<TShape>& output_shapes) {
  const ForwardSchemeRequests& fwdreqs =
    ForwardAlignedSchemesCaller(attrs, input_shapes, output_shapes);
  std::vector<SchemeRequest> reqs;
  for (size_t i = 0; i < fwdreqs.size(); ++i) {
    reqs.emplace_back(fwdreqs[i].in_data_schemes,
                      fwdreqs[i].out_data_schemes);
  }
  return reqs;
}

std::vector<SchemeRequest> OpBackwardAlignedSchemes(
    const NodeAttrs& attrs,
    const std::vector<TShape>& input_shapes,
    const std::vector<TShape>& output_shapes) {
  const ParsedOpProp& prop = nnvm::get<ParsedOpProp>(attrs.parsed);
  // Split inputs into multiple vectors.
  std::vector<TShape> out_grad_shapes(prop.ptr->NumVisibleOutputs());
  std::vector<TShape> in_data_shapes(prop.ptr->ListArguments().size());
  std::vector<TShape> out_data_shapes(prop.ptr->NumOutputs());
  // Pointers to convert the one input array to multiple arrays with semantics.
  std::vector<TShape*> ogs_ptr(out_grad_shapes.size());
  for (size_t i = 0; i < out_grad_shapes.size(); ++i) {
    ogs_ptr[i] = &out_grad_shapes[i];
  }
  std::vector<TShape*> ids_ptr(in_data_shapes.size());
  for (size_t i = 0; i < in_data_shapes.size(); ++i) {
    ids_ptr[i] = &in_data_shapes[i];
  }
  std::vector<TShape*> ods_ptr(out_data_shapes.size());
  for (size_t i = 0; i < out_data_shapes.size(); ++i) {
    ods_ptr[i] = &out_data_shapes[i];
  }
  std::vector<TShape*> arg_ptr = prop.ptr->BackwardInputs(
      ogs_ptr, ids_ptr, ods_ptr);
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    *arg_ptr[i] = input_shapes[i];
  }
  const BackwardSchemeRequests& bwdreqs =
    BackwardAlignedSchemesCaller(
        attrs, out_grad_shapes, in_data_shapes,
        out_data_shapes, output_shapes);
  std::vector<SchemeRequest> reqs;
  for (const BackwardSchemeRequest& breq : bwdreqs) {
    // Convert back to one input array.
    const std::vector<Scheme>& input_schemes =
      prop.ptr->BackwardInputs(breq.out_grad_schemes,
                               breq.in_data_schemes,
                               breq.out_data_schemes);
    reqs.emplace_back(input_schemes, breq.in_grad_schemes);
  }
  return reqs;
}

void RegisterOpAlignedSchemes() {
  // TODO
  for (auto reg : dmlc::Registry<OperatorPropertyReg>::List()) {
    Op& op = ::dmlc::Registry<::nnvm::Op>::Get()->__REGISTER_OR_GET__(reg->name);
  }
    //op.set_attr<nnvm::FAlignedSchemes>("FAlignedSchemes", OpPropForwardAlignedSchemes);
    //back_op.set_attr<nnvm::FAlignedSchemes>("FAlignedSchemes", OpPropBackwardAlignedSchemes);
}

}  // namespace op
}  // namespace mxnet
