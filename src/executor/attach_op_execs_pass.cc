/*!
 * Copyright (c) 2016 by Contributors
 * \file attach_op_execs_pass.cc
 * \brief Operator executor to execute each operator.
 */
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include "./exec_pass.h"

namespace mxnet {
namespace op {
const OperatorProperty* OpPropGetOpProperty(const NodeAttrs& attrs);
}  // namespace op

namespace exec {
namespace {
void DoNothingFCompute(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  // DO nothing.
}
}  // namespace

// forward executor
class ForwardOpExecutor : public OpExecutor {
 public:
  // Constructor.
  explicit ForwardOpExecutor(Operator* op, std::vector<uint32_t> aux_index)
      : op_(op), aux_index_(aux_index) {
    std::sort(aux_index_.begin(), aux_index_.end());
  }

  void Run(RunContext rctx) override {
    op_ctx.run_ctx = rctx;
    op_->Forward(op_ctx, in_data_, req, out_data_, aux_data_);
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
 private:
  friend Graph AttachOpExecs(Graph g);
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
  const NodeAttrs& attrs_;
  std::vector<TBlob> in_data_, out_data_;
};

// pass to attach operator executors
Graph AttachOpExecs(Graph g) {
  using nnvm::DTypeVector;
  using nnvm::ShapeVector;
  using nnvm::FMutateInputs;

  auto& fcreate_layer_op = nnvm::Op::GetAttr<FCreateLayerOp>("FCreateLayerOp");
  auto& fmutate_inputs = nnvm::Op::GetAttr<FMutateInputs>("FMutateInputs");
  auto& bwdop2fwdop = nnvm::Op::GetAttr<const Op*>("TIsLayerOpBackward");

  const auto& vdtype = g.GetAttr<DTypeVector>("dtype");
  const auto& vshape = g.GetAttr<ShapeVector>("shape");
  const auto& vctx = g.GetAttr<ContextVector>("context");

  // get the graph
  const auto& idx = g.indexed_graph();
  std::vector<std::shared_ptr<OpExecutor> > ret(idx.num_nodes());

  // initialize the nodes
  for (size_t nodeid = 0; nodeid < idx.num_nodes(); ++nodeid) {
    const auto& inode = idx[nodeid];
    //LOG(INFO) << "Initialize node #" << nodeid << ": " << inode.source->attrs.name;
    if (inode.source->is_variable()) {
      // Do nothing for variable node.
      continue;
    }
    const nnvm::Op* op = CHECK_NOTNULL(inode.source->op());
    std::vector<uint32_t> mutate_index;
    if (fmutate_inputs.count(op)) {
      mutate_index = fmutate_inputs[op](inode.source->attrs);
    }
    const FCompute fcompute = FComputeExecutor::GetFCompute(op, vctx[nodeid]);

    /*if (inode.source->attrs.dict.count("ctx_group") != 0) {
      if (inode.source->attrs.dict.at("ctx_group") != "group:0") {
        LOG(WARNING) << "--------->\""
          << inode.source->op()->name << "\". DoNothing.";
        ret[nodeid] = std::make_shared<FComputeExecutor>(DoNothingFCompute, inode.source->attrs);
        continue;
      }
    } else {
      LOG(WARNING) << "!!! No ctx assigned? " << inode.source->attrs.name;
    }*/

    /*if (op->name == "_CrossDeviceCopy"
        //|| op->name == "Concat"
        //|| op->name == "FullyConnected"
        //|| op->name == "ElementWiseSum" || op->name == "_backward_FullyConnected"
        ) {
      LOG(WARNING) << "==========>\""
        << inode.source->op()->name << "\". DoNothing.";
      ret[nodeid] = std::make_shared<FComputeExecutor>(DoNothingFCompute, inode.source->attrs);
      continue;
    }*/

    if (fcreate_layer_op.count(op)) {
      ShapeVector ishape;
      DTypeVector itype;
      for (const auto& e : inode.inputs) {
        ishape.emplace_back(vshape[idx.entry_id(e)]);
        itype.emplace_back(vdtype[idx.entry_id(e)]);
      }
      Operator* layer_op = fcreate_layer_op[op](
          inode.source->attrs, vctx[nodeid], ishape, itype);
      ret[nodeid] = std::make_shared<ForwardOpExecutor>(layer_op, mutate_index);
    } else if (bwdop2fwdop.count(op) != 0) {
      if (inode.control_deps.size() >= 1) {
        // A normal backward node with the first control dependency pointing to its
        // corresponding forward node.
        const uint32_t fwd_id = inode.control_deps[0];
        CHECK_EQ(vctx[fwd_id], vctx[nodeid]);
        CHECK(ret[fwd_id] != nullptr);
        ret[nodeid] = std::make_shared<BackwardOpExecutor>(
            dynamic_cast<ForwardOpExecutor*>(ret[fwd_id].get())->op_,
            mxnet::op::OpPropGetOpProperty(inode.source->attrs),
            mutate_index);
      } else {
        // Backward operator with no forward control dependency. This happens when
        // the backward operator is specially generated.
        // First create the layer operator. FIXME: The input shape and dtype are
        // inferred from the output of the backward op.
        /*const nnvm::Op* fwdop = bwdop2fwdop[op];
        CHECK_NE(fcreate_layer_op.count(fwdop), 0);
        ShapeVector ishape;
        DTypeVector itype;
        for (size_t outidx = 0; outidx < inode.source->num_outputs(); ++outidx) {
          const uint32_t out_entid = idx.entry_id(nodeid, outidx);
          ishape.push_back(vshape[out_entid]);
          itype.push_back(vdtype[out_entid]);
        }
        std::shared_ptr<Operator> layer_op(fcreate_layer_op[fwdop](
            inode.source->attrs, vctx[nodeid], ishape, itype));
        // Then create the BackwardOpExecutor.
        ret[nodeid] = std::make_shared<BackwardOpExecutor>(
            layer_op,
            mxnet::op::OpPropGetOpProperty(inode.source->attrs),
            mutate_index);*/
        // Create FCompute for this:
        //LOG(WARNING) << "Current workaround for node \""
          //<< inode.source->op()->name << "\". DoNothing.";
        ret[nodeid] = std::make_shared<FComputeExecutor>(DoNothingFCompute, inode.source->attrs);
      }
    } else if (fcompute != nullptr) {
      // Not forward layer op or backward layer op. Just use the registered compute
      // function as executor.
      ret[nodeid] = std::make_shared<FComputeExecutor>(fcompute, inode.source->attrs);
    } else {
      //LOG(WARNING) << "FCompute not registered for node \""
        //<< inode.source->op()->name << "\". DoNothingFCompute will be registered.";
      ret[nodeid] = std::make_shared<FComputeExecutor>(DoNothingFCompute, NodeAttrs());
    }
  }
  g.attrs["op_execs"] = std::make_shared<nnvm::any>(ret);
  return g;
}

}  // namespace exec
}  // namespace mxnet
