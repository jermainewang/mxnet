/*!
 *  Copyright (c) 2016 by Contributors
 * \file mx_passes.h
 * \brief Header file for shared pass attributes.
 */
#ifndef MXNET_NNVM_MX_PASSES_H_
#define MXNET_NNVM_MX_PASSES_H_

#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>

namespace mxnet {
namespace pass {

namespace grad {
struct MXEntryArg {
  uint32_t node = 0;
  uint32_t index = 0;
  uint32_t version = 0;
  void Load(dmlc::JSONReader *reader) {
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("node", &node);
    helper.DeclareOptionalField("index", &index);
    helper.DeclareOptionalField("version", &version);
    helper.ReadAllFields(reader);
  }
};
struct MXGradientArgs {
  // The targets to be differentiated. If none is given,
  // all the input entries will be included.
  // Note: Only one of "xs" and "xs_blacklist" should be provided.
  // If none is provided, blacklist rule is applied.
  std::vector<MXEntryArg> xs;
  // Alternative way to specify the gradable targets. The list
  // specifies which input entries do NOT need to be differentiated.
  // Note: Only one of "xs" and "xs_blacklist" should be provided.
  // If none is provided, blacklist rule is applied.
  std::vector<MXEntryArg> xs_blacklist;

  // The objective entries to compute gradients from. If none is
  // given, all the output entries will be included.
  // Note: Only one of "ys" and "ys_blacklist" should be provided.
  // If none is provided, blacklist rule is applied.
  std::vector<MXEntryArg> ys;
  // Alternative way to specify the objective entries. The list
  // specifies which objective entries are NOT included in gradient
  // computation.
  // Note: Only one of "ys" and "ys_blacklist" should be provided.
  // If none is provided, blacklist rule is applied.
  std::vector<MXEntryArg> ys_blacklist;

  void Load(dmlc::JSONReader *reader) {
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareOptionalField("xs", &xs);
    helper.DeclareOptionalField("xs_blacklist", &xs_blacklist);
    helper.DeclareOptionalField("ys", &ys);
    helper.DeclareOptionalField("ys_blacklist", &ys_blacklist);
    helper.ReadAllFields(reader);
  }
};
struct GradNodeInInfo {
  enum Type {
    kFromGradOut = 0,
    kFromFwdOut,
    kFromFwdIn,
  };
  // Whether the input should be fetched from out_grads. If false, the input entry
  // should be fetched from forward outputs.
  int type = -1;
  // The index of the entry to fetch from. If `from_out_grads` is true, the input
  // should be fetched from out_grads[index]; otherwise, it should be fetched from
  // forward_outputs[index].
  size_t index = 0;
  
  inline static GradNodeInInfo CreateFromOutGrads(size_t idx) {
    GradNodeInInfo ret;
    ret.type = kFromGradOut;
    ret.index = idx;
    return ret;
  }

  inline static GradNodeInInfo CreateFromForwardOut(size_t idx) {
    GradNodeInInfo ret;
    ret.type = kFromFwdOut;
    ret.index = idx;
    return ret;
  }

  inline static GradNodeInInfo CreateFromForwardIn(size_t idx) {
    GradNodeInInfo ret;
    ret.type = kFromFwdIn;
    ret.index = idx;
    return ret;
  }
};
}  // namespace grad

namespace shape {
static const std::string key = "shape";
struct MXInferShapeArgs {
  // Shapes of the input entries.
  std::vector<TShape> shape_inputs;
  // Shapes of the forward graph. This is used when the backward
  // graph is generated separately.
  nnvm::ColumnRef<TShape> forward_shapes;

  void Load(dmlc::JSONReader *reader) {
    dmlc::JSONObjectReadHelper helper;
    std::vector<std::vector<int>> raw_shapes;
    helper.DeclareOptionalField("shape_inputs", &raw_shapes);
    helper.DeclareOptionalField("forward_shapes", &forward_shapes);
    helper.ReadAllFields(reader);
    for (const auto& rs : raw_shapes) {
      shape_inputs.emplace_back(rs.begin(), rs.end());
    }
  }
};
}  // namespace shape

namespace dtype {
static const std::string key = "dtype";
struct MXInferTypeArgs {
  // Types of the input entries.
  std::vector<int> dtype_inputs;
  // Types of the forward graph. This is used when the backward
  // graph is generated separately.
  nnvm::ColumnRef<int> forward_dtypes;

  void Load(dmlc::JSONReader *reader) {
    dmlc::JSONObjectReadHelper helper;
    std::vector<int> raw_dtypes;
    helper.DeclareOptionalField("dtype_inputs", &raw_dtypes);
    helper.DeclareOptionalField("forward_dtypes", &forward_dtypes);
    helper.ReadAllFields(reader);
    for (const auto& rt : raw_dtypes) {
      dtype_inputs.emplace_back(rt);
    }
  }
};
}  // namespace dtype

namespace ctx {
static const std::string key = "context";
}  // namespace ctx

namespace inplace {
static const std::string key = "inplace_option";
struct InplaceOption {
  // A map from input to output.
  std::pair<int, int> inplace_pair;
  bool is_identity;
};
}  // namespace inplace

namespace mutate {
static const std::string key = "mutate_index";
}  // namespace mutate

namespace plan_memory {
static const std::string ref_key = "storage_ref";
static const std::string storage_key = "storage";
// Special storage id when storage cannot be assigned due to many reasons
// (e.g. the shape or dtype is unknown).
static const int kBadStorageID = -1;
// Special storage id for external space.
static const int kExternalStorageID = -2;
// Special storage id for space that is not allocated.
static const int kNull = -3;
struct StorageRef {
  int storage_id;
  int inplace_index;
};
struct Storage {
  int id;
  int device_id;
  size_t max_bytes;
};
}  // namespace plan_memory

namespace attach_op {
static const std::string key = "op_execs";
/*!
 * \brief executor to execute an operator
 * This is a graph executor dependent interface
 * that unifies all the operator
 */
class OpExecutor {
 public:
  /*! \brief output requirement on each array */
  std::vector<OpReqType> req;
  /*! \brief runtime op context, contains allocated resources */
  OpContext op_ctx;
  /*! \brief virtual destructor */
  virtual ~OpExecutor() {}
  /*!
   * \brief Setup the executor for given NDArray member
   * this can be called multiple times if NDArray changed during reshape.
   *  It is safe to call it via asynchronize engine lambda
   */
  //virtual void Setup() = 0;
  void SetInput(const NDArray& nd, size_t idx) {
    CHECK_LT(idx, in_array_.size());
    in_array_[idx] = nd;
    *in_tblob_ptr_[idx] = nd.data();
  }
  void SetOutput(const NDArray& nd, size_t idx) {
    CHECK_LT(idx, out_array_.size());
    out_array_[idx] = nd;
    *out_tblob_ptr_[idx] = nd.data();
  }
  const NDArray& GetInput(size_t idx) {
    return in_array_[idx];
  }
  const NDArray& GetOutput(size_t idx) {
    return out_array_[idx];
  }
  /*!
   * \brief run the operator given runtime context on device.
   *  This function call do not synchronize the stream.
   * \param rctx The runtime context passed in by environment.
   */
  virtual void Run(RunContext rctx) = 0;
  /*! \return the execution type */
  virtual Operator::ExecType exec_type() const = 0;
 
 protected:
  void Reset(size_t num_inputs, size_t num_outputs) {
    in_array_.clear();
    out_array_.clear();
    in_tblob_ptr_.clear();
    out_tblob_ptr_.clear();
    in_array_.resize(num_inputs);
    in_tblob_ptr_.resize(num_inputs, nullptr);
    out_array_.resize(num_outputs);
    out_tblob_ptr_.resize(num_outputs, nullptr);
  }

  /*! \brief Input and output arrays. */
  std::vector<NDArray> in_array_, out_array_;
  /*! \brief Input and output tblob pointers. */
  std::vector<TBlob*> in_tblob_ptr_, out_tblob_ptr_;
};

}  // namespace attach_op

}  // namespace pass
}  // namespace mxnet
#endif  // MXNET_NNVM_MX_PASSES_H_
