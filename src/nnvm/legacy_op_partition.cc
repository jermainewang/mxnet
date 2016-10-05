#include "./legacy_op_partition.h"

#include <sstream>

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
namespace {
// Split inputs into multiple vectors.
template<typename T>
void SplitBackwardInputs(const NodeAttrs& attrs,
                         const vector<T>& inputs,
                         vector<T>* out_grad,
                         vector<T>* in_data,
                         vector<T>* out_data) {
  const ParsedOpProp& prop = nnvm::get<ParsedOpProp>(attrs.parsed);
  out_grad->resize(prop.ptr->NumVisibleOutputs());
  in_data->resize(prop.ptr->ListArguments().size());
  out_data->resize(prop.ptr->NumOutputs());
  // Pointers to convert the one input array to multiple arrays with semantics.
  vector<T*> ogs_ptr(out_grad->size());
  for (size_t i = 0; i < out_grad->size(); ++i) {
    ogs_ptr[i] = &(*out_grad)[i];
  }
  vector<T*> ids_ptr(in_data->size());
  for (size_t i = 0; i < in_data->size(); ++i) {
    ids_ptr[i] = &(*in_data)[i];
  }
  vector<T*> ods_ptr(out_data->size());
  for (size_t i = 0; i < out_data->size(); ++i) {
    ods_ptr[i] = &(*out_data)[i];
  }
  vector<T*> arg_ptr = prop.ptr->BackwardInputs(
      ogs_ptr, ids_ptr, ods_ptr);
  CHECK_EQ(arg_ptr.size(), inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    *arg_ptr[i] = inputs[i];
  }
}
}  // namespace

class OpPartitioner {
 public:
  OpPartitioner(const NodeAttrs& attrs): attrs_(attrs) {}
  virtual void AttrsAlignedScheme(
      const vector<TShape>& input_shapes,
      const vector<TShape>& output_shapes,
      const vector<Scheme*>& input_schemes,
      const vector<Scheme*>& output_schemes) = 0;
  virtual void AttrsPartition(const NodeAttrs& attrs, size_t num_partitions) = 0;

  const NodeAttrs& attrs() const { return attrs_; }
 protected:
  NodeAttrs attrs_;
};

template<typename ParamType>
class ForwardOpPartitioner : public OpPartitioner {
 public:
  ForwardOpPartitioner(const NodeAttrs& attrs): OpPartitioner(attrs) {
    vector<pair<string, string> > kwargs(attrs_.dict.begin(), attrs_.dict.end());
    param_.Init(kwargs);
  }
  virtual void AlignedScheme(
      const ParamType& param,
      const vector<TShape>& in_data_shapes,
      const vector<TShape>& out_data_shapes,
      const vector<Scheme*>& in_data_schemes,
      const vector<Scheme*>& out_data_schemes) = 0;
  virtual void Partition(size_t num_partitions) = 0;
  void AttrsAlignedScheme(
      const vector<TShape>& input_shapes,
      const vector<TShape>& output_shapes,
      const vector<Scheme*>& input_schemes,
      const vector<Scheme*>& output_schemes) override {
    this->AlignedScheme(param_,
                        input_shapes,
                        output_shapes,
                        input_schemes,
                        output_schemes);
  }
  void AttrsPartition(const NodeAttrs& attrs, size_t num_partitions) override {
    attrs_ = attrs;
    CHECK_EQ(attrs_.scalars.size(), 0)
      << "Cannot have positional attributes";
    vector<pair<string, string> > kwargs(attrs_.dict.begin(), attrs_.dict.end());
    param_.Init(kwargs);
    this->Partition(num_partitions);
    // Update attributes.
    for (const auto kv : param_.__DICT__()) {
      attrs_.dict[kv.first] = kv.second;
    }
  }
 protected:
  ParamType param_;
};

template<typename ParamType>
class BackwardOpPartitioner : public OpPartitioner {
 public:
  BackwardOpPartitioner(const NodeAttrs& attrs): OpPartitioner(attrs) {
    vector<pair<string, string> > kwargs(attrs_.dict.begin(), attrs_.dict.end());
    param_.Init(kwargs);
  }
  virtual void AlignedScheme(
      const ParamType& param,
      const vector<TShape>& out_grad_shapes,
      const vector<TShape>& in_data_shapes,
      const vector<TShape>& out_data_shapes,
      const vector<TShape>& in_grad_shapes,
      const vector<Scheme*>& out_grad_schemes,
      const vector<Scheme*>& in_data_schemes,
      const vector<Scheme*>& out_data_schemes,
      const vector<Scheme*>& in_grad_schemes) = 0;
  virtual void Partition(size_t num_partitions) = 0;
  void AttrsAlignedScheme(
      const vector<TShape>& input_shapes,
      const vector<TShape>& output_shapes,
      const vector<Scheme*>& input_schemes,
      const vector<Scheme*>& output_schemes) override {
    vector<TShape> out_grad_shapes, in_data_shapes, out_data_shapes;
    vector<Scheme*> out_grad_schemes, in_data_schemes, out_data_schemes;
    SplitBackwardInputs(attrs_,
                        input_shapes,
                        &out_grad_shapes,
                        &in_data_shapes,
                        &out_data_shapes);
    SplitBackwardInputs(attrs_,
                        input_schemes,
                        &out_grad_schemes,
                        &in_data_schemes,
                        &out_data_schemes);
    this->AlignedScheme(param_,
                        out_grad_shapes,
                        in_data_shapes,
                        out_data_shapes,
                        output_shapes,
                        out_grad_schemes,
                        in_data_schemes,
                        out_data_schemes,
                        output_schemes);
  }
  void AttrsPartition(const NodeAttrs& attrs, size_t num_partitions) override {
    attrs_ = attrs;
    CHECK_EQ(attrs_.scalars.size(), 0)
      << "Cannot have positional attributes";
    vector<pair<string, string> > kwargs(attrs_.dict.begin(), attrs_.dict.end());
    param_.Init(kwargs);
    this->Partition(num_partitions);
    // Update attributes.
    for (const auto kv : param_.__DICT__()) {
      attrs_.dict[kv.first] = kv.second;
    }
  }
 protected:
  ParamType param_;
};

///////////////////////////////////////////////////////////////////////////
// Return node attributes that are exactly the same as the given one.
NodeAttrs IdenticalPartition(const NodeAttrs& attrs,
                            size_t num_partitions) {
  return attrs;
}

template<size_t K>
vector<SchemeRequest> CutFirstKDimsSchemes(
    const NodeAttrs&,
    const vector<TShape>& input_shapes,
    const vector<TShape>& output_shapes) {
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
    // Attribute partitioner.
    req.partitioner = &IdenticalPartition;
    reqs.push_back(req);
  }
  return reqs;
}

vector<SchemeRequest> CutAllDimsSchemes(
    const NodeAttrs& attrs,
    const vector<TShape>& input_shapes,
    const vector<TShape>& output_shapes) {
  vector<SchemeRequest> reqs;
  CHECK_GT(input_shapes.size(), 0);
  for (size_t i = 0; i < input_shapes[0].ndim(); ++i) {
    SchemeRequest req;
    for (size_t j = 0; j < input_shapes.size(); ++j) {
      CHECK_LT(i, input_shapes[j].ndim());
      req.input_schemes.push_back(Scheme::Cut(i));
    }
    for (size_t j = 0; j < output_shapes.size(); ++j) {
      CHECK_LT(i, output_shapes[j].ndim());
      req.output_schemes.push_back(Scheme::Cut(i));
    }
    // Attribute partitioner.
    req.partitioner = &IdenticalPartition;
    reqs.push_back(req);
  }
  return reqs;
}
///////////////////////////////////////////////////////////////////////////////////
// FullyConnectedOp
// One matmult in the forward propagation:
//   - y = dot(x, w.T) + b
// Therefore, there are following aligned schemes:
//   - x: R, w: r, y: R, b: r
//   - x: r, w: R, y: C, b: R
//   - x: C, w: C, y: red, b: r
class FCForwardPartitioner1 : public ForwardOpPartitioner<FullyConnectedParam> {
 public:
  FCForwardPartitioner1(const NodeAttrs& attrs):
    ForwardOpPartitioner<FullyConnectedParam>(attrs) {}
  void AlignedScheme(
      const FullyConnectedParam& param,
      const vector<TShape>& in_data_shapes,
      const vector<TShape>& out_data_shapes,
      const vector<Scheme*>& in_data_schemes,
      const vector<Scheme*>& out_data_schemes) override {
    CHECK_EQ(in_data_shapes[fullc::kData].ndim(), 2);
    CHECK_EQ(in_data_shapes[fullc::kWeight].ndim(), 2);
    CHECK_EQ(out_data_shapes[fullc::kOut].ndim(), 2);
    if (param.no_bias) {
      CHECK_EQ(in_data_shapes.size(), 2);
    } else {
      CHECK_EQ(in_data_shapes.size(), 3);
    }
    *in_data_schemes[fullc::kData] = Scheme::Cut(0);  // x: R
    *in_data_schemes[fullc::kWeight] = Scheme::Rep(); // w: r
    *out_data_schemes[fullc::kOut] = Scheme::Cut(0);  // y: R
    if (!param.no_bias) {
      *in_data_schemes[fullc::kBias] = Scheme::Rep(); // b: r
    }
  }
  void Partition(size_t num_partitions) override {
    // Do nothing.
  }
};
class FCForwardPartitioner2 : public ForwardOpPartitioner<FullyConnectedParam> {
 public:
  FCForwardPartitioner2(const NodeAttrs& attrs):
    ForwardOpPartitioner<FullyConnectedParam>(attrs) {}
  void AlignedScheme(
      const FullyConnectedParam& param,
      const vector<TShape>& in_data_shapes,
      const vector<TShape>& out_data_shapes,
      const vector<Scheme*>& in_data_schemes,
      const vector<Scheme*>& out_data_schemes) override {
    CHECK_EQ(in_data_shapes[fullc::kData].ndim(), 2);
    CHECK_EQ(in_data_shapes[fullc::kWeight].ndim(), 2);
    CHECK_EQ(out_data_shapes[fullc::kOut].ndim(), 2);
    if (param.no_bias) {
      CHECK_EQ(in_data_shapes.size(), 2);
    } else {
      CHECK_EQ(in_data_shapes.size(), 3);
    }
    *in_data_schemes[fullc::kData] = Scheme::Rep();    // x: r
    *in_data_schemes[fullc::kWeight] = Scheme::Cut(0); // w: R
    *out_data_schemes[fullc::kOut] = Scheme::Cut(1);   // y: C
    if (!param.no_bias) {
      *in_data_schemes[fullc::kBias] = Scheme::Cut(0); // b: R
    }
  }
  void Partition(size_t num_partitions) override {
    CHECK(param_.num_hidden % num_partitions == 0);
    param_.num_hidden /= num_partitions;
  }
};
class FCForwardPartitioner3 : public ForwardOpPartitioner<FullyConnectedParam> {
 public:
  FCForwardPartitioner3(const NodeAttrs& attrs):
    ForwardOpPartitioner<FullyConnectedParam>(attrs) {}
  void AlignedScheme(
      const FullyConnectedParam& param,
      const vector<TShape>& in_data_shapes,
      const vector<TShape>& out_data_shapes,
      const vector<Scheme*>& in_data_schemes,
      const vector<Scheme*>& out_data_schemes) override {
    CHECK_EQ(in_data_shapes[fullc::kData].ndim(), 2);
    CHECK_EQ(in_data_shapes[fullc::kWeight].ndim(), 2);
    CHECK_EQ(out_data_shapes[fullc::kOut].ndim(), 2);
    if (param.no_bias) {
      CHECK_EQ(in_data_shapes.size(), 2);
    } else {
      CHECK_EQ(in_data_shapes.size(), 3);
    }
    *in_data_schemes[fullc::kData] = Scheme::Cut(1);   // x: C
    *in_data_schemes[fullc::kWeight] = Scheme::Cut(1); // w: C
    *out_data_schemes[fullc::kOut] = Scheme::Red();    // y: red
    if (!param.no_bias) {
      *in_data_schemes[fullc::kBias] = Scheme::Rep();  // b: r
    }
  }
  void Partition(size_t num_partitions) override {
    // Do nothing.
  }
};
// Two matmults in the backward propagation:
//   - dw = dot(dy.T, x)
//   - dx = dot(dy, w)
//   - db = reduce_sum(dy, 0)
// Therefore, there are following aligned schemes:
//   - dy: C, x: r, dw: R, w: R, dx: red, db: R
//   - dy: r, x: C, dw: C, w: C, dx, C, db: r
//   - dy: R, x: R, dw: red, w: r, dx: R, db: red
class FCBackwardPartitioner1 : public BackwardOpPartitioner<FullyConnectedParam> {
 public:
  FCBackwardPartitioner1(const NodeAttrs& attrs):
    BackwardOpPartitioner<FullyConnectedParam>(attrs) {}
  void AlignedScheme(
      const FullyConnectedParam& param,
      const vector<TShape>& out_grad_shapes,
      const vector<TShape>& in_data_shapes,
      const vector<TShape>& out_data_shapes,
      const vector<TShape>& in_grad_shapes,
      const vector<Scheme*>& out_grad_schemes,
      const vector<Scheme*>& in_data_schemes,
      const vector<Scheme*>& out_data_schemes,
      const vector<Scheme*>& in_grad_schemes) {
    *out_grad_schemes[fullc::kOut] = Scheme::Cut(1);   // dy: C
    *in_data_schemes[fullc::kData] = Scheme::Rep();    // x: r
    *in_grad_schemes[fullc::kWeight] = Scheme::Cut(0); // dw: R
    *in_data_schemes[fullc::kWeight] = Scheme::Cut(0); // w: R
    *in_grad_schemes[fullc::kData] = Scheme::Red();    // dx: red
    if (!param.no_bias) {
      *in_grad_schemes[fullc::kBias] = Scheme::Cut(0); // db: R
    }
  }
  void Partition(size_t num_partitions) {
    CHECK(param_.num_hidden % num_partitions == 0);
    param_.num_hidden /= num_partitions;
  }
};
class FCBackwardPartitioner2 : public BackwardOpPartitioner<FullyConnectedParam> {
 public:
  FCBackwardPartitioner2(const NodeAttrs& attrs):
    BackwardOpPartitioner<FullyConnectedParam>(attrs) {}
  void AlignedScheme(
      const FullyConnectedParam& param,
      const vector<TShape>& out_grad_shapes,
      const vector<TShape>& in_data_shapes,
      const vector<TShape>& out_data_shapes,
      const vector<TShape>& in_grad_shapes,
      const vector<Scheme*>& out_grad_schemes,
      const vector<Scheme*>& in_data_schemes,
      const vector<Scheme*>& out_data_schemes,
      const vector<Scheme*>& in_grad_schemes) {
    *out_grad_schemes[fullc::kOut] = Scheme::Rep();    // dy: r
    *in_data_schemes[fullc::kData] = Scheme::Cut(1);   // x: C
    *in_grad_schemes[fullc::kWeight] = Scheme::Cut(1); // dw: C
    *in_data_schemes[fullc::kWeight] = Scheme::Cut(1); // w: C
    *in_grad_schemes[fullc::kData] = Scheme::Cut(1);   // dx: C
    if (!param.no_bias) {
      *in_grad_schemes[fullc::kBias] = Scheme::Rep();  // db: r
    }
  }
  void Partition(size_t num_partitions) {
    // Do nothing.
  }
};
class FCBackwardPartitioner3 : public BackwardOpPartitioner<FullyConnectedParam> {
 public:
  FCBackwardPartitioner3(const NodeAttrs& attrs):
    BackwardOpPartitioner<FullyConnectedParam>(attrs) {}
  void AlignedScheme(
      const FullyConnectedParam& param,
      const vector<TShape>& out_grad_shapes,
      const vector<TShape>& in_data_shapes,
      const vector<TShape>& out_data_shapes,
      const vector<TShape>& in_grad_shapes,
      const vector<Scheme*>& out_grad_schemes,
      const vector<Scheme*>& in_data_schemes,
      const vector<Scheme*>& out_data_schemes,
      const vector<Scheme*>& in_grad_schemes) {
    *out_grad_schemes[fullc::kOut] = Scheme::Cut(0);  // dy: R
    *in_data_schemes[fullc::kData] = Scheme::Cut(0);  // x: R
    *in_grad_schemes[fullc::kWeight] = Scheme::Red(); // dw: red
    *in_data_schemes[fullc::kWeight] = Scheme::Rep(); // w: r
    *in_grad_schemes[fullc::kData] = Scheme::Cut(0);  // dx: R
    if (!param.no_bias) {
      *in_grad_schemes[fullc::kBias] = Scheme::Red(); // db: red
    }
  }
  void Partition(size_t num_partitions) {
    // Do nothing.
  }
};
///////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////
// ConvolutionOp
// Data format NCHW
// Filter format CoCiHW
// Two operations in ConvolutionOp:
//   - y = ConvForward(x, w)
//   - y = AddTensor(y, b)
// Therefore, there are following aligned schemes:
//   - x: R, w: r, y: R, b: r
//   - x: r, w: R, y: C, b: R
//   - x: C, w: C, y: red, b: r
class ConvForwardPartitioner1 : public ForwardOpPartitioner<ConvolutionParam> {
 public:
  ConvForwardPartitioner1(const NodeAttrs& attrs):
    ForwardOpPartitioner<ConvolutionParam>(attrs) {}
  void AlignedScheme(
      const ConvolutionParam& param,
      const vector<TShape>& in_data_shapes,
      const vector<TShape>& out_data_shapes,
      const vector<Scheme*>& in_data_schemes,
      const vector<Scheme*>& out_data_schemes) override {
    *in_data_schemes[conv::kData] = Scheme::Cut(0);  // x: R
    *in_data_schemes[conv::kWeight] = Scheme::Rep(); // w: r
    *out_data_schemes[conv::kOut] = Scheme::Cut(0);  // y: R
    if (!param.no_bias) {
      *in_data_schemes[conv::kBias] = Scheme::Rep(); // b: r
    }
  }
  void Partition(size_t num_partitions) override {
    CHECK_EQ(param_.num_group, 1);
    // Do nothing.
  }
};
class ConvForwardPartitioner2 : public ForwardOpPartitioner<ConvolutionParam> {
 public:
  ConvForwardPartitioner2(const NodeAttrs& attrs):
    ForwardOpPartitioner<ConvolutionParam>(attrs) {}
  void AlignedScheme(
      const ConvolutionParam& param,
      const vector<TShape>& in_data_shapes,
      const vector<TShape>& out_data_shapes,
      const vector<Scheme*>& in_data_schemes,
      const vector<Scheme*>& out_data_schemes) override {
    *in_data_schemes[conv::kData] = Scheme::Rep();    // x: r
    *in_data_schemes[conv::kWeight] = Scheme::Cut(0); // w: R
    *out_data_schemes[conv::kOut] = Scheme::Cut(1);   // y: C
    if (!param.no_bias) {
      *in_data_schemes[conv::kBias] = Scheme::Cut(0); // b: R
    }
  }
  void Partition(size_t num_partitions) override {
    CHECK_EQ(param_.num_group, 1);
    CHECK_EQ(param_.num_filter % num_partitions, 0);
    param_.num_filter /= num_partitions;
  }
};
class ConvForwardPartitioner3 : public ForwardOpPartitioner<ConvolutionParam> {
 public:
  ConvForwardPartitioner3(const NodeAttrs& attrs):
    ForwardOpPartitioner<ConvolutionParam>(attrs) {}
  void AlignedScheme(
      const ConvolutionParam& param,
      const vector<TShape>& in_data_shapes,
      const vector<TShape>& out_data_shapes,
      const vector<Scheme*>& in_data_schemes,
      const vector<Scheme*>& out_data_schemes) override {
    *in_data_schemes[conv::kData] = Scheme::Cut(1);   // x: C
    *in_data_schemes[conv::kWeight] = Scheme::Cut(1); // w: C
    *out_data_schemes[conv::kOut] = Scheme::Red();    // y: red
    if (!param.no_bias) {
      *in_data_schemes[conv::kBias] = Scheme::Rep();  // b: r
    }
  }
  void Partition(size_t num_partitions) override {
    CHECK_EQ(param_.num_group, 1);
    // Do nothing.
  }
};
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
class ConvBackwardPartitioner1 : public BackwardOpPartitioner<ConvolutionParam> {
 public:
  ConvBackwardPartitioner1(const NodeAttrs& attrs):
    BackwardOpPartitioner<ConvolutionParam>(attrs) {}
  void AlignedScheme(
      const ConvolutionParam& param,
      const vector<TShape>& out_grad_shapes,
      const vector<TShape>& in_data_shapes,
      const vector<TShape>& out_data_shapes,
      const vector<TShape>& in_grad_shapes,
      const vector<Scheme*>& out_grad_schemes,
      const vector<Scheme*>& in_data_schemes,
      const vector<Scheme*>& out_data_schemes,
      const vector<Scheme*>& in_grad_schemes) {
    *out_grad_schemes[conv::kOut] = Scheme::Cut(1);   // dy: C
    *in_data_schemes[conv::kData] = Scheme::Rep();    // x: r
    *in_grad_schemes[conv::kWeight] = Scheme::Cut(0); // dw: R
    *in_data_schemes[conv::kWeight] = Scheme::Cut(0); // w: R
    *in_grad_schemes[conv::kData] = Scheme::Red();    // dx: red
    if (!param.no_bias) {
      *in_grad_schemes[conv::kBias] = Scheme::Cut(0); // db: R
    }
  }
  void Partition(size_t num_partitions) {
    CHECK_EQ(param_.num_group, 1);
    CHECK_EQ(param_.num_filter % num_partitions, 0);
    param_.num_filter /= num_partitions;
  }
};
class ConvBackwardPartitioner2 : public BackwardOpPartitioner<ConvolutionParam> {
 public:
  ConvBackwardPartitioner2(const NodeAttrs& attrs):
    BackwardOpPartitioner<ConvolutionParam>(attrs) {}
  void AlignedScheme(
      const ConvolutionParam& param,
      const vector<TShape>& out_grad_shapes,
      const vector<TShape>& in_data_shapes,
      const vector<TShape>& out_data_shapes,
      const vector<TShape>& in_grad_shapes,
      const vector<Scheme*>& out_grad_schemes,
      const vector<Scheme*>& in_data_schemes,
      const vector<Scheme*>& out_data_schemes,
      const vector<Scheme*>& in_grad_schemes) {
    *out_grad_schemes[conv::kOut] = Scheme::Rep();    // dy: r
    *in_data_schemes[conv::kData] = Scheme::Cut(1);   // x: C
    *in_grad_schemes[conv::kWeight] = Scheme::Cut(1); // dw: C
    *in_data_schemes[conv::kWeight] = Scheme::Cut(1); // w: C
    *in_grad_schemes[conv::kData] = Scheme::Cut(1);   // dx: C
    if (!param.no_bias) {
      *in_grad_schemes[conv::kBias] = Scheme::Rep();  // db: r
    }
  }
  void Partition(size_t num_partitions) {
    CHECK_EQ(param_.num_group, 1);
  }
};
class ConvBackwardPartitioner3 : public BackwardOpPartitioner<ConvolutionParam> {
 public:
  ConvBackwardPartitioner3(const NodeAttrs& attrs):
    BackwardOpPartitioner<ConvolutionParam>(attrs) {}
  void AlignedScheme(
      const ConvolutionParam& param,
      const vector<TShape>& out_grad_shapes,
      const vector<TShape>& in_data_shapes,
      const vector<TShape>& out_data_shapes,
      const vector<TShape>& in_grad_shapes,
      const vector<Scheme*>& out_grad_schemes,
      const vector<Scheme*>& in_data_schemes,
      const vector<Scheme*>& out_data_schemes,
      const vector<Scheme*>& in_grad_schemes) {
    *out_grad_schemes[conv::kOut] = Scheme::Cut(0);  // dy: R
    *in_data_schemes[conv::kData] = Scheme::Cut(0);  // x: R
    *in_grad_schemes[conv::kWeight] = Scheme::Red(); // dw: red
    *in_data_schemes[conv::kWeight] = Scheme::Rep(); // w: r
    *in_grad_schemes[conv::kData] = Scheme::Cut(0);  // dx: R
    if (!param.no_bias) {
      *in_grad_schemes[conv::kBias] = Scheme::Red(); // db: red
    }
  }
  void Partition(size_t num_partitions) {
    CHECK_EQ(param_.num_group, 1);
  }
};

////////////////////////////////////////////////////////////////////////////
vector<SchemeRequest> MakeSchemeRequest(
    const vector<TShape>& input_shapes,
    const vector<TShape>& output_shapes,
    const vector<shared_ptr<OpPartitioner>>& pttns) {
  vector<SchemeRequest> ret;
  for (auto pttn : pttns) {
    SchemeRequest req;
    req.input_schemes.resize(input_shapes.size());
    req.output_schemes.resize(output_shapes.size());
    vector<Scheme*> is_ptr(input_shapes.size()), os_ptr(output_shapes.size());
    for (size_t i = 0; i < input_shapes.size(); ++i) {
      is_ptr[i] = &req.input_schemes[i];
    }
    for (size_t i = 0; i < output_shapes.size(); ++i) {
      os_ptr[i] = &req.output_schemes[i];
    }
    pttn->AttrsAlignedScheme(input_shapes, output_shapes, is_ptr, os_ptr);
    req.partitioner = [pttn](const NodeAttrs& attrs, size_t n) {
      pttn->AttrsPartition(attrs, n);
      return pttn->attrs();
    };
    ret.push_back(std::move(req));
  }
  return ret;
}

vector<SchemeRequest> OpForwardAlignedSchemes(
    const NodeAttrs& attrs,
    const vector<TShape>& input_shapes,
    const vector<TShape>& output_shapes) {
  if (attrs.op->name == "FullyConnected") {
    shared_ptr<OpPartitioner> pttn1(new FCForwardPartitioner1(attrs));
    shared_ptr<OpPartitioner> pttn2(new FCForwardPartitioner2(attrs));
    shared_ptr<OpPartitioner> pttn3(new FCForwardPartitioner3(attrs));
    return MakeSchemeRequest(input_shapes, output_shapes, {pttn1, pttn2, pttn3});
  } else if (attrs.op->name == "Convolution") {
    shared_ptr<OpPartitioner> pttn1(new ConvForwardPartitioner1(attrs));
    shared_ptr<OpPartitioner> pttn2(new ConvForwardPartitioner2(attrs));
    shared_ptr<OpPartitioner> pttn3(new ConvForwardPartitioner3(attrs));
    return MakeSchemeRequest(input_shapes, output_shapes, {pttn1, pttn2, pttn3});
  } else {
    LOG(FATAL) << "No aligned scheme defined for operator: " << attrs.op->name;
  }
  return vector<SchemeRequest>();
}

vector<SchemeRequest> OpBackwardAlignedSchemes(
    const NodeAttrs& attrs,
    const vector<TShape>& input_shapes,
    const vector<TShape>& output_shapes) {
  if (attrs.op->name == "_backward_FullyConnected") {
    shared_ptr<OpPartitioner> pttn1(new FCBackwardPartitioner1(attrs));
    shared_ptr<OpPartitioner> pttn2(new FCBackwardPartitioner2(attrs));
    shared_ptr<OpPartitioner> pttn3(new FCBackwardPartitioner3(attrs));
    return MakeSchemeRequest(input_shapes, output_shapes, {pttn1, pttn2, pttn3});
  } else if (attrs.op->name == "_backward_Convolution") {
    shared_ptr<OpPartitioner> pttn1(new ConvBackwardPartitioner1(attrs));
    shared_ptr<OpPartitioner> pttn2(new ConvBackwardPartitioner2(attrs));
    shared_ptr<OpPartitioner> pttn3(new ConvBackwardPartitioner3(attrs));
    return MakeSchemeRequest(input_shapes, output_shapes, {pttn1, pttn2, pttn3});
  } else {
    LOG(FATAL) << "No aligned scheme defined for operator: " << attrs.op->name;
  }
  return vector<SchemeRequest>();
}

void RegisterOpAlignedSchemes() {
  using namespace nnvm::pass;
  const string kAttrName = "FAlignedSchemes";
  using AType = nnvm::pass::FAlignedSchemes;
  for (const string& name : dmlc::Registry<::nnvm::Op>::ListAllNames()) {
    Op& op = dmlc::Registry<::nnvm::Op>::Get()->__REGISTER_OR_GET__(name);
    if (Op::GetAttr<AType>(kAttrName).count(&op) > 0) {
      // Already registered.
      continue;
    }
    if (name == "FullyConnected" || name == "Convolution") {
      op.set_attr<AType>(kAttrName, OpForwardAlignedSchemes);
    } else if (name == "_backward_FullyConnected" || name == "_backward_Convolution") {
      op.set_attr<AType>(kAttrName, OpBackwardAlignedSchemes);
    } else if (name == "Activation" || name == "_backward_Activation"
        || name == "Dropout" || name == "_backward_Dropout"
        || name == "LeakyReLU" || name == "_backward_LeadkyReLU"
        || name == "ElementWiseSum"
        || name == "_plus" || name == "_plus_scalar"
        || name == "_backward_plus" || name == "_backward_plus_scalar"
        || name == "_minus" || name == "_minus_scalar"
        || name == "_backward_minus" || name == "_backward_minus_scalar"
        || name == "_mul" || name == "_mul_scalar"
        || name == "_backward_mul" || name == "_backward_mul_scalar"
        || name == "_div" || name == "_div_scalar"
        || name == "_backward_div" || name == "_backward_div_scalar"
        || name == "_maximum" || name == "_maximum_scalar"
        || name == "_backward_maximum" || name == "_backward_maximum_scalar"
        || name == "_minimum" || name == "_minimum_scalar"
        || name == "_backward_minimum" || name == "_backward_minimum_scalar"
        || name == "_power" || name == "_power_scalar"
        || name == "_backward_power" || name == "_backward_power_scalar") {
      op.set_attr<AType>(kAttrName, CutAllDimsSchemes);
    } else if (name == "Pooling" || name == "_backward_Pooling"
        || name == "Flatten" || name == "_backward_Flatten") {
      op.set_attr<AType>(kAttrName, CutFirstKDimsSchemes<2>);
    }
  }
}

}  // namespace op
}  // namespace mxnet
