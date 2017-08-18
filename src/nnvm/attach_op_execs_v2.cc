/*!
 * Copyright (c) 2016 by Contributors
 * \file attach_op_execs_pass.cc
 * \brief Operator executor to execute each operator.
 */
#include "./mx_passes.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "../operator/mkl/mkl_memory-inl.h"
#include "../operator/mkl/mkl_util-inl.h"
#endif

using namespace std;
using namespace nnvm;

namespace mxnet {

namespace op {
const OperatorProperty* OpPropGetOpProperty(const NodeAttrs& attrs);
}  // namespace op

namespace pass {

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

  shared_ptr<Operator> op() const {
    return op_;
  }

  explicit ForwardOpExecutor(shared_ptr<Operator> op,
      vector<uint32_t> aux_index)
      : op_(op), aux_index_(aux_index) {
    std::sort(aux_index_.begin(), aux_index_.end());
  }

 private:
  shared_ptr<Operator> op_;
  vector<uint32_t> aux_index_;
  vector<TBlob> in_data_, out_data_, aux_data_;
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
  explicit BackwardOpExecutor(shared_ptr<Operator> op,
                              const OperatorProperty* prop,
                              vector<uint32_t> aux_index)
      : op_(op), aux_index_(aux_index) {
    std::sort(aux_index_.begin(), aux_index_.end());
    out_grad_.resize(prop->NumVisibleOutputs());
    in_data_.resize(prop->ListArguments().size());
    in_grad_.resize(in_data_.size());
    out_data_.resize(prop->NumOutputs());

    vector<TBlob*> out_grad_ptr(out_grad_.size());
    for (size_t i = 0; i < out_grad_.size(); ++i) {
      out_grad_ptr[i] = &out_grad_[i];
    }
    vector<TBlob*> in_data_ptr(in_data_.size());
    for (size_t i = 0; i < in_data_.size(); ++i) {
      in_data_ptr[i] = &in_data_[i];
    }
    vector<TBlob*> out_data_ptr(out_data_.size());
    for (size_t i = 0; i < out_data_.size(); ++i) {
      out_data_ptr[i] = &out_data_[i];
    }
    arg_data_ptr_ = prop->BackwardInputs(
        out_grad_ptr, in_data_ptr, out_data_ptr);
  }

 private:
  shared_ptr<Operator> op_;
  vector<uint32_t> aux_index_;
  vector<TBlob> out_grad_, in_grad_, in_data_, out_data_, aux_data_;
  vector<TBlob*> arg_data_ptr_;
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
  vector<TBlob> in_data_, out_data_;
};

void AttachOpExecsRec(const Graph& g,
                      const Column<TShape>* vshape,
                      const Column<int>* vdtype,
                      const Column<Context>* vctx,
                      const Column<vector<uint32_t>>* mutate_index,
                      const Column<shared_ptr<OpExecutor>>* fwd_execs,
                      Column<shared_ptr<OpExecutor>>* execs) {
  static auto& fcreate_layer_op = nnvm::Op::GetAttr<FCreateLayerOp>("FCreateLayerOp");
  static auto& is_layer_backward = nnvm::Op::GetAttr<bool>("TIsLayerOpBackward");
  const auto& idx = g.indexed_graph();

  if (fwd_execs != nullptr) {
    // This is a standalone backward graph.
    CHECK(g.global_attrs.count("gradient_node_mapping"));
    const auto& gradient_node_mapping =
      g.GetGlobalAttr<vector<uint32_t>>("gradient_node_mapping");
    CHECK_EQ(gradient_node_mapping.size(), idx.num_nodes());
    for (uint32_t nid = 0; nid < gradient_node_mapping.size(); ++nid) {
      const Node* node = idx[nid].source;
      const uint32_t fwd_nid = gradient_node_mapping[nid];
      if (fwd_nid < fwd_execs->value.size()) {
        shared_ptr<OpExecutor> fwd_opexec =
          CHECK_NOTNULL(fwd_execs->value[fwd_nid]);
        execs->value[nid] = std::make_shared<BackwardOpExecutor>(
            dynamic_cast<ForwardOpExecutor*>(fwd_opexec.get())->op(),
            mxnet::op::OpPropGetOpProperty(node->attrs),
            mutate_index->value[nid]);
      }
    }
  }
  
  for (size_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    if (execs->value[nid] != nullptr) {
      continue;
    }
    if (node->is_variable()) {
      continue;
    }
    if (node->is_graph()) {
      // XXX(minjie): Always create new executors for subgraphs. This can be saved if
      // (1) all executors in the subgraphs are stateless; (2) all the given arguments
      // are the same. However, be aware of the cost of this check.
      auto subgraph = node->graph();
      auto subref = subgraph->CreateNodeColumn<shared_ptr<OpExecutor>>();
      const Column<shared_ptr<OpExecutor>>* sub_fwd_execs = nullptr;
      if (subgraph->global_attrs.count("gradient_node_mapping")) {
        if (fwd_execs != nullptr) {
          // Forward node is in another subgraph.
          const auto& node_mapping =
            subgraph->GetGlobalAttr<vector<uint32_t>>("gradient_node_mapping");
          const uint32_t fwd_nid = node_mapping[nid];
          CHECK_LT(fwd_nid, fwd_execs->children.size());
          sub_fwd_execs = fwd_execs->children[fwd_nid].get();
        } else {
          // Foward node is connected by the first control dependency.
          CHECK_GE(node->control_deps.size(), 1U)
            << "Backward node need to have control_deps to its forward node";
          const NodePtr& fwd_ptr = node->control_deps[0];
          CHECK(fwd_ptr->is_graph());
          const uint32_t fwd_nid = idx.node_id(fwd_ptr.get());
          sub_fwd_execs = execs->children[fwd_nid].get();
        }
      }
      AttachOpExecsRec(*subgraph,
                       vshape->children[nid].get(),
                       vdtype->children[nid].get(),
                       vctx->children[nid].get(),
                       mutate_index->children[nid].get(),
                       sub_fwd_execs,
                       subref.CopyOnWrite());
      execs->children[nid] = subref;
    } else {
      const vector<uint32_t>& midx = mutate_index->value[nid];
      FCompute fcompute = FComputeExecutor::GetFCompute(node->op(), vctx->value[nid]);
      if (fcreate_layer_op.count(node->op())) {
        vector<TShape> ishape;
        vector<int> itype;
        for (const auto& e : node->inputs) {
          ishape.emplace_back(vshape->value[idx.entry_id(e)]);
          itype.emplace_back(vdtype->value[idx.entry_id(e)]);
        }
        shared_ptr<Operator> opr;
        opr.reset(fcreate_layer_op[node->op()](
              node->attrs, vctx->value[nid], ishape, itype));
        execs->value[nid] = std::make_shared<ForwardOpExecutor>(opr, midx);
      } else if (is_layer_backward.get(node->op(), false)) {
        CHECK_GE(node->control_deps.size(), 1);
        const uint32_t fwd_nid = idx.node_id(node->control_deps[0].get());
        CHECK(vctx->value[fwd_nid] == vctx->value[nid]);
        shared_ptr<OpExecutor> fwd_opexec =
          CHECK_NOTNULL(execs->value[fwd_nid]);
        execs->value[nid] = std::make_shared<BackwardOpExecutor>(
            dynamic_cast<ForwardOpExecutor*>(fwd_opexec.get())->op(),
            mxnet::op::OpPropGetOpProperty(node->attrs),
            midx);
      } else if (fcompute != nullptr) {
        execs->value[nid] = std::make_shared<FComputeExecutor>(fcompute, node->attrs);
      } else {
        LOG(INFO) << "FCompute not registered for operator " << node->op()->name;
      }
    }
  }
}

Graph MXAttachOpExecs(Graph &&graph) {
  if (graph.node_attrs.count(attach_op::key) == 0) {
    auto ref = graph.CreateNodeColumn<shared_ptr<OpExecutor>>();
    const auto* shapes = graph.entry_attrs.GetColumn<TShape>(shape::key).get();
    const auto* dtypes = graph.entry_attrs.GetColumn<int>(dtype::key).get();
    const auto* ctx = graph.node_attrs.GetColumn<Context>(ctx::key).get();
    const auto* mutate = graph.node_attrs.GetColumn<vector<uint32_t>>(mutate::key).get();
    using FwdExecsType = const Column<shared_ptr<OpExecutor>>*;
    FwdExecsType fwd_execs = nullptr;
    if (graph.global_attrs.count("forward_execs")) {
      fwd_execs = graph.GetGlobalAttr<FwdExecsType>("forward_execs");
    }
    AttachOpExecsRec(graph,
                     shapes,
                     dtypes,
                     ctx,
                     mutate,
                     fwd_execs,
                     ref.CopyOnWrite());
    graph.node_attrs.SetColumn(attach_op::key, ref);
  }
  return graph;
}

NNVM_REGISTER_PASS(MXAttachOpExecs)
.describe("Attach operator executor for each node in the graph.")
.set_body(MXAttachOpExecs)
.set_change_graph(false)
.depend_entry_attr(shape::key)
.depend_entry_attr(dtype::key)
.depend_node_attr(ctx::key)
.depend_node_attr(mutate::key)
.provide_node_attr(attach_op::key);

}  // namespace pass
}  // namespace mxnet
