/*!
 * Copyright (c) 2016 by Contributors
 * \file attach_op_execs_pass.cc
 * \brief Operator executor to execute each operator.
 */
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include "./mx_passes.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "../operator/mkl/mkl_memory-inl.h"
#include "../operator/mkl/mkl_util-inl.h"
#endif
namespace mxnet {

namespace op {
const OperatorProperty* OpPropGetOpProperty(const NodeAttrs& attrs);
}  // namespace op

namespace pass {

using namespace nnvm;
using attach_op::OpExecutor;

// forward executor
class ForwardOpExecutor : public OpExecutor {
 public:
  void Run(RunContext rctx) override {
    op_ctx.run_ctx = rctx;
    op_->Forward(op_ctx, in_data_, req, out_data_, aux_data_);
#if MKL_EXPERIMENTAL == 1
    mkl_tblobs_prv_to_cpu(in_data_);
    mkl_tblobs_prv_to_cpu(out_data_);
    mkl_tblobs_prv_to_cpu(aux_data_);
#endif
  }

  void Setup() override {
    in_data_.clear(); aux_data_.clear();
    for (size_t i = 0; i < in_array.size(); ++i) {
      if (!std::binary_search(aux_index_.begin(), aux_index_.end(), i)) {
        in_data_.push_back(in_array[i].data());
      } else {
        aux_data_.push_back(in_array[i].data());
      }
    }
    out_data_.resize(out_array.size());
    std::transform(out_array.begin(), out_array.end(), out_data_.begin(), [](const NDArray& nd) {
        return nd.data();
      });
  }
  Operator::ExecType exec_type() const override {
    return op_->exec_type();
  }

  std::shared_ptr<Operator> op() const {
    return op_;
  }

  explicit ForwardOpExecutor(std::shared_ptr<Operator> op,
      std::vector<uint32_t> aux_index)
      : op_(op), aux_index_(aux_index) {
    std::sort(aux_index_.begin(), aux_index_.end());
  }

 private:
  friend Graph MXAttachOpExecs(Graph g);
  std::shared_ptr<Operator> op_;
  std::vector<uint32_t> aux_index_;
  std::vector<TBlob> in_data_, out_data_, aux_data_;
};

// backward executor
class BackwardOpExecutor : public OpExecutor {
 public:
  void Run(RunContext rctx) override {
    op_ctx.run_ctx = rctx;
    op_->Backward(op_ctx, out_grad_, in_data_, out_data_,
                  req, in_grad_, aux_data_);
#if MKL_EXPERIMENTAL == 1
    mkl_tblobs_prv_to_cpu(out_grad_);
    mkl_tblobs_prv_to_cpu(in_data_);
    mkl_tblobs_prv_to_cpu(out_data_);
    mkl_tblobs_prv_to_cpu(in_grad_);
    mkl_tblobs_prv_to_cpu(aux_data_);
#endif
  }
  void Setup() override {
    size_t arg_top = 0, aux_top = 0;
    aux_data_.resize(aux_index_.size());
    for (size_t i = 0; i < in_array.size(); ++i) {
      if (!std::binary_search(aux_index_.begin(), aux_index_.end(), i)) {
        CHECK_GT(arg_data_ptr_.size(), arg_top);
        *arg_data_ptr_[arg_top++] = in_array[i].data();
      } else {
        aux_data_.at(aux_top++) = in_array[i].data();
      }
    }
    CHECK_EQ(out_array.size(), in_grad_.size());
    std::transform(out_array.begin(), out_array.end(),
                   in_grad_.begin(), [](const NDArray& nd) {
        return nd.data();
      });
  }
  Operator::ExecType exec_type() const override {
    return op_->exec_type();
  }
  explicit BackwardOpExecutor(std::shared_ptr<Operator> op,
                              const OperatorProperty* prop,
                              std::vector<uint32_t> aux_index)
      : op_(op), aux_index_(aux_index) {
    std::sort(aux_index_.begin(), aux_index_.end());
    out_grad_.resize(prop->NumVisibleOutputs());
    in_data_.resize(prop->ListArguments().size());
    in_grad_.resize(in_data_.size());
    out_data_.resize(prop->NumOutputs());

    std::vector<TBlob*> out_grad_ptr(out_grad_.size());
    for (size_t i = 0; i < out_grad_.size(); ++i) {
      out_grad_ptr[i] = &out_grad_[i];
    }
    std::vector<TBlob*> in_data_ptr(in_data_.size());
    for (size_t i = 0; i < in_data_.size(); ++i) {
      in_data_ptr[i] = &in_data_[i];
    }
    std::vector<TBlob*> out_data_ptr(out_data_.size());
    for (size_t i = 0; i < out_data_.size(); ++i) {
      out_data_ptr[i] = &out_data_[i];
    }
    arg_data_ptr_ = prop->BackwardInputs(
        out_grad_ptr, in_data_ptr, out_data_ptr);
  }

 private:
  std::shared_ptr<Operator> op_;
  std::vector<uint32_t> aux_index_;
  std::vector<TBlob> out_grad_, in_grad_, in_data_, out_data_, aux_data_;
  std::vector<TBlob*> arg_data_ptr_;
};

// fcompute executor executor
class FComputeExecutor : public OpExecutor {
 public:
  void Run(RunContext rctx) override {
    op_ctx.run_ctx = rctx;
    fcompute_(attrs_, op_ctx, in_data_, req, out_data_);
#if MKL_EXPERIMENTAL == 1
    mkl_tblobs_prv_to_cpu(in_data_);
    mkl_tblobs_prv_to_cpu(out_data_);
#endif
  }
  void Setup() override {
    in_data_.resize(in_array.size());
    out_data_.resize(out_array.size());
    auto get_blob =  [](const NDArray& nd) {
      return nd.data();
    };
    std::transform(in_array.begin(), in_array.end(), in_data_.begin(), get_blob);
    std::transform(out_array.begin(), out_array.end(), out_data_.begin(), get_blob);
  }
  Operator::ExecType exec_type() const override {
    return Operator::kSync;
  }
  explicit FComputeExecutor(FCompute fcompute, const NodeAttrs& attrs)
      : fcompute_(fcompute), attrs_(attrs) {
  }

  static FCompute GetFCompute(const Op* op, Context ctx) {
    static auto& fcompute_cpu = nnvm::Op::GetAttr<FCompute>("FCompute<cpu>");
    static auto& fcompute_gpu = nnvm::Op::GetAttr<FCompute>("FCompute<gpu>");
    if (ctx.dev_mask() == cpu::kDevMask) {
      return fcompute_cpu.get(op, nullptr);
    } else if (ctx.dev_mask() == gpu::kDevMask) {
      return fcompute_gpu.get(op, nullptr);
    } else {
      LOG(FATAL) << "Unknown device mask";
      return nullptr;
    }
  }

 private:
  FCompute fcompute_;
  NodeAttrs attrs_;
  std::vector<TBlob> in_data_, out_data_;
};

void AttachOpExecsRec(const Graph& g,
                      const Column<int>* vdtype,
                      const Column<TShape>* vshape,
                      const Column<Context>* vctx,
                      const Column<std::shared_ptr<OpExecutor>>* fwd_execs,
                      Column<std::shared_ptr<OpExecutor>>* execs) {
  static auto& fcreate_layer_op = nnvm::Op::GetAttr<FCreateLayerOp>("FCreateLayerOp");
  static auto& fmutate_inputs = nnvm::Op::GetAttr<FMutateInputs>("FMutateInputs");
  static auto& is_layer_backward = nnvm::Op::GetAttr<bool>("TIsLayerOpBackward");
  const auto& idx = g.indexed_graph();
  // TODO: set executor using fwd_execs.
  
  for (size_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    if (node->is_variable()) {
      continue;
    }
    if (execs->value[nid] != nullptr) {
      continue;
    }
    if (node->is_graph()) {
      // TODO
    } else {
      std::vector<uint32_t> mutate_index;
      if (fmutate_inputs.count(node->op())) {
        mutate_index = fmutate_inputs[node->op()](node->attrs);
      }
      FCompute fcompute = FComputeExecutor::GetFCompute(node->op(), vctx->value[nid]);
      if (fcreate_layer_op.count(node->op())) {
        std::vector<TShape> ishape;
        std::vector<int> itype;
        for (const auto& e : node->inputs) {
          ishape.emplace_back(vshape->value[idx.entry_id(e)]);
          itype.emplace_back(vdtype->value[idx.entry_id(e)]);
        }
        // TODO: find forward operator.
        /*std::shared_ptr<Operator> opr;
        if (saved_opr.count(inode.source)) {
          opr = saved_opr.at(inode.source);
        } else {
          opr.reset(fcreate_layer_op[node->op()](
                node->attrs, vctx->value[nid], ishape, itype));
        }*/
        std::shared_ptr<Operator> opr;
        opr.reset(fcreate_layer_op[node->op()](
              node->attrs, vctx->value[nid], ishape, itype));
        execs->value[nid] = std::make_shared<ForwardOpExecutor>(opr, mutate_index);
      } else if (is_layer_backward.get(node->op(), false)) {
        CHECK_GE(node->control_deps.size(), 1);
        const uint32_t fwd_nid = idx.node_id(node->control_deps[0].get());
        CHECK(vctx->value[fwd_nid] == vctx->value[nid]);
        std::shared_ptr<OpExecutor> fwd_opexec =
          CHECK_NOTNULL(execs->value[fwd_nid]);
        execs->value[nid] = std::make_shared<BackwardOpExecutor>(
            dynamic_cast<ForwardOpExecutor*>(fwd_opexec.get())->op(),
            mxnet::op::OpPropGetOpProperty(node->attrs),
            mutate_index);
      } else if (fcompute != nullptr) {
        execs->value[nid] = std::make_shared<FComputeExecutor>(fcompute, node->attrs);
      } else {
        LOG(INFO) << "FCompute not registered for operator " << node->op()->name;
      }
    }
  }
}

// pass to attach operator executors
Graph MXAttachOpExecs(Graph g) {
  using nnvm::DTypeVector;
  using nnvm::ShapeVector;
  using nnvm::FMutateInputs;

  auto& fcreate_layer_op = nnvm::Op::GetAttr<FCreateLayerOp>("FCreateLayerOp");
  auto& fmutate_inputs = nnvm::Op::GetAttr<FMutateInputs>("FMutateInputs");
  auto& is_layer_backward = nnvm::Op::GetAttr<bool>("TIsLayerOpBackward");

  /*const auto& vdtype = g.entry_attrs.GetColumn<int>("dtype");
  const auto& vshape = g.entry_attrs.GetColumn<TShape>("shape");
  const auto& vctx = g.GetAttr<ContextVector>("context");
  //const auto& saved_opr = g.GetAttr<
    //std::unordered_map<const nnvm::Node*, std::shared_ptr<Operator>>>("saved_opr");

  // Traverse the graph.
  const auto& idx = g.indexed_graph();
  std::vector<std::shared_ptr<OpExecutor> > ret(idx.num_nodes());

  // initialize the nodes
  for (size_t i = 0; i < idx.num_nodes(); ++i) {
    const auto& inode = idx[i];
    if (inode.source->is_variable()) continue;
    std::vector<uint32_t> mutate_index;
    if (fmutate_inputs.count(inode.source->op())) {
      mutate_index = fmutate_inputs[inode.source->op()](inode.source->attrs);
    }
    FCompute fcompute = FComputeExecutor::GetFCompute(inode.source->op(), vctx[i]);
    if (fcreate_layer_op.count(inode.source->op())) {
      std::vector<TShape> ishape;
      std::vector<int> itype;
      for (const auto& e : inode.inputs) {
        ishape.emplace_back(vshape[idx.entry_id(e)]);
        itype.emplace_back(vdtype[idx.entry_id(e)]);
      }
      std::shared_ptr<Operator> opr;
      if (saved_opr.count(inode.source)) {
        opr = saved_opr.at(inode.source);
      } else {
        opr.reset(fcreate_layer_op[inode.source->op()](
              inode.source->attrs, vctx[i], ishape, itype));
      }
      ret[i] = std::make_shared<ForwardOpExecutor>(opr, mutate_index);
    } else if (is_layer_backward.get(inode.source->op(), false)) {
      CHECK_GE(inode.control_deps.size(), 1);
      uint32_t fwd_id = inode.control_deps[0];
      CHECK(vctx[fwd_id] == vctx[i]);
      CHECK(ret[fwd_id] != nullptr);
      ret[i] = std::make_shared<BackwardOpExecutor>(
          dynamic_cast<ForwardOpExecutor*>(ret[fwd_id].get())->op_,
          mxnet::op::OpPropGetOpProperty(inode.source->attrs),
          mutate_index);
    } else if (fcompute != nullptr) {
      ret[i] = std::make_shared<FComputeExecutor>(fcompute, inode.source->attrs);
    } else {
      LOG(INFO) << "FCompute not registered " << inode.source->op()->name;
    }
  }
  g.attrs["op_execs"] = std::make_shared<nnvm::any>(ret);*/
  return g;
}

}  // namespace pass
}  // namespace mxnet
