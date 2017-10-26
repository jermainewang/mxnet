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
  void Run(const OpContext& op_ctx) override {
    op_->Forward(op_ctx, in_data_, req, out_data_, aux_data_);
#if MKL_EXPERIMENTAL == 1
    mkl_tblobs_prv_to_cpu(in_data_);
    mkl_tblobs_prv_to_cpu(out_data_);
    mkl_tblobs_prv_to_cpu(aux_data_);
#endif
  }
  ExecType exec_type() const override {
    return op_->exec_type();
  }
  shared_ptr<Operator> op() const {
    return op_;
  }
  explicit ForwardOpExecutor(shared_ptr<Operator> op,
                             const OperatorProperty* prop,
                             vector<uint32_t> aux_index)
      : op_(op) {
    const size_t num_inputs = prop->ListArguments().size() + aux_index.size();
    const size_t num_outputs = prop->NumOutputs();
    this->Reset(num_inputs, num_outputs);
    out_data_.resize(prop->NumOutputs());
    in_data_.resize(prop->ListArguments().size());
    aux_data_.resize(aux_index.size());
    // Setup in tblob pointer.
    std::sort(aux_index.begin(), aux_index.end());
    size_t nml_top = 0, aux_top = 0;
    for (size_t i = 0; i < num_inputs; ++i) {
      if (!std::binary_search(aux_index.begin(), aux_index.end(), i)) {
        CHECK_GT(in_data_.size(), nml_top);
        in_tblob_ptr_[i] = &in_data_[nml_top++];
      } else {
        CHECK_GT(aux_data_.size(), aux_top);
        in_tblob_ptr_[i] = &aux_data_[aux_top++];
      }
    }
    // Setup out tblob pointer.
    for (size_t i = 0; i < num_outputs; ++i) {
      out_tblob_ptr_[i] = &out_data_[i];
    }
  }

 private:
  shared_ptr<Operator> op_;
  vector<TBlob> in_data_, out_data_, aux_data_;
};

// backward executor
class BackwardOpExecutor : public OpExecutor {
 public:
  void Run(const OpContext& op_ctx) override {
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
  ExecType exec_type() const override {
    return op_->exec_type();
  }
  explicit BackwardOpExecutor(shared_ptr<Operator> op,
                              const OperatorProperty* prop,
                              vector<uint32_t> aux_index)
      : op_(op) {
    out_grad_.resize(prop->NumVisibleOutputs());
    in_data_.resize(prop->ListArguments().size());
    in_grad_.resize(in_data_.size());
    out_data_.resize(prop->NumOutputs());
    aux_data_.resize(aux_index.size());
    // Compute backward dependencies.
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
    vector<TBlob*> bwd_in_ptr = prop->BackwardInputs(
        out_grad_ptr, in_data_ptr, out_data_ptr);
    const size_t num_inputs = bwd_in_ptr.size() + aux_index.size();
    const size_t num_outputs = in_data_.size();
    this->Reset(num_inputs, num_outputs);
    // Setup input tblob pointers.
    std::sort(aux_index.begin(), aux_index.end());
    size_t nml_top = 0, aux_top = 0;
    for (size_t i = 0; i < num_inputs; ++i) {
      if (!std::binary_search(aux_index.begin(), aux_index.end(), i)) {
        CHECK_GT(bwd_in_ptr.size(), nml_top);
        in_tblob_ptr_[i] = bwd_in_ptr[nml_top++];
      } else {
        CHECK_GT(aux_data_.size(), aux_top);
        in_tblob_ptr_[i] = &aux_data_[aux_top++];
      }
    }
    // Setup output tblob pointers.
    for (size_t i = 0; i < out_tblob_ptr_.size(); ++i) {
      out_tblob_ptr_[i] = &in_grad_[i];
    }
  }

 private:
  shared_ptr<Operator> op_;
  vector<TBlob> out_grad_, in_data_, out_data_, aux_data_;
  vector<TBlob> in_grad_;
};

// fcompute executor executor
class FComputeExecutor : public OpExecutor {
 public:
  void Run(const OpContext& op_ctx) override {
    fcompute_(attrs_, op_ctx, in_data_, req, out_data_);
#if MKL_EXPERIMENTAL == 1
    mkl_tblobs_prv_to_cpu(in_data_);
    mkl_tblobs_prv_to_cpu(out_data_);
#endif
  }
  ExecType exec_type() const override {
    return ExecType::kSync;
  }
  explicit FComputeExecutor(FCompute fcompute,
                            const NodeAttrs& attrs,
                            size_t num_inputs,
                            size_t num_outputs)
      : fcompute_(fcompute), attrs_(attrs) {
    in_data_.resize(num_inputs);
    out_data_.resize(num_outputs);
    this->Reset(num_inputs, num_outputs);
    // Setup input tblob pointers.
    for (size_t i = 0; i < num_inputs; ++i) {
      in_tblob_ptr_[i] = &in_data_[i];
    }
    // Setup output tblob pointers.
    for (size_t i = 0; i < num_outputs; ++i) {
      out_tblob_ptr_[i] = &out_data_[i];
    }
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
                      const Column<plan_memory::StorageRef>* mem_plan,
                      const Column<int>* vdevice,
                      const Column<vector<uint32_t>>* mutate_index,
                      const Column<shared_ptr<OpExecutor>>* fwd_execs,
                      Column<shared_ptr<OpExecutor>>* execs) {
  using plan_memory::StorageRef;
  using plan_memory::kNull;
  static auto& fcreate_layer_op = nnvm::Op::GetAttr<FCreateOpState>("FCreateOpState");
  static auto& is_layer_backward = nnvm::Op::GetAttr<bool>("TIsLayerOpBackward");
  const auto& idx = g.indexed_graph();
  const auto& context = g.GetGlobalAttr<vector<Context>>(ctx::ctx_key);

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
                       mem_plan->children[nid].get(),
                       vdevice->children[nid].get(),
                       mutate_index->children[nid].get(),
                       sub_fwd_execs,
                       subref.CopyOnWrite());
      execs->children[nid] = subref;
    } else {
      const vector<uint32_t>& midx = mutate_index->value[nid];
      FCompute fcompute = FComputeExecutor::GetFCompute(
          node->op(), context.at(vdevice->value[nid]));
      if (fcreate_layer_op.count(node->op())) {
        vector<TShape> ishape;
        vector<int> itype;
        for (const auto& e : node->inputs) {
          ishape.emplace_back(vshape->value[idx.entry_id(e)]);
          itype.emplace_back(vdtype->value[idx.entry_id(e)]);
        }
        // TODO(minjie): hack for new OperatorState change.
        shared_ptr<Operator> opr;
        auto state = fcreate_layer_op[node->op()](
              node->attrs, context.at(vdevice->value[nid]), ishape, itype);
        state.get_state<op::OperatorState>().GiveTo(opr);
        const auto* opprop = mxnet::op::OpPropGetOpProperty(node->attrs);
        execs->value[nid] = std::make_shared<ForwardOpExecutor>(opr, opprop, midx);
      } else if (is_layer_backward.get(node->op(), false)) {
        CHECK_GE(node->control_deps.size(), 1);
        const uint32_t fwd_nid = idx.node_id(node->control_deps[0].get());
        CHECK(vdevice->value[fwd_nid] == vdevice->value[nid]);
        shared_ptr<OpExecutor> fwd_opexec =
          CHECK_NOTNULL(execs->value[fwd_nid]);
        const auto* opprop = mxnet::op::OpPropGetOpProperty(node->attrs);
        execs->value[nid] = std::make_shared<BackwardOpExecutor>(
            dynamic_cast<ForwardOpExecutor*>(fwd_opexec.get())->op(),
            opprop, midx);
      } else if (fcompute != nullptr) {
        execs->value[nid] = std::make_shared<FComputeExecutor>(
            fcompute, node->attrs, node->inputs.size(), node->num_outputs());
      } else {
        LOG(INFO) << "FCompute not registered for operator " << node->op()->name;
      }
      // Setup output requests.
      OpExecutor* exec = execs->value[nid].get();
      for (size_t i = 0; i < node->num_outputs(); ++i) {
        const uint32_t eid = idx.entry_id(nid, i);
        const StorageRef& store_ref = mem_plan->value[eid];
        const int storageid = mem_plan->value[eid].storage_id;
        // Output request.
        if (false) {
          // TODO(minjie): addto inplace optimization.
        } else if (store_ref.inplace_index >= 0) {
          exec->req.push_back(kWriteInplace);
        } else if (storageid == kNull) {
          // TODO(minjie): need double-check.
          exec->req.push_back(kNullOp);
        } else {
          exec->req.push_back(kWriteTo);
        }
      }
    }
  }
}

Graph MXAttachOpExecs(Graph &&graph) {
  using plan_memory::StorageRef;
  using plan_memory::ref_key;
  if (graph.node_attrs.count(attach_op::key) == 0) {
    auto ref = graph.CreateNodeColumn<shared_ptr<OpExecutor>>();
    const auto* shapes = graph.entry_attrs.GetColumn<TShape>(shape::key).get();
    const auto* dtypes = graph.entry_attrs.GetColumn<int>(dtype::key).get();
    const auto* mem_plan = graph.entry_attrs.GetColumn<StorageRef>(ref_key).get();
    const auto* vdevice = graph.node_attrs.GetColumn<int>(ctx::device_key).get();
    const auto* mutate = graph.node_attrs.GetColumn<vector<uint32_t>>(mutate::key).get();
    using FwdExecsType = const Column<shared_ptr<OpExecutor>>*;
    FwdExecsType fwd_execs = nullptr;
    if (graph.global_attrs.count("forward_execs")) {
      fwd_execs = graph.GetGlobalAttr<FwdExecsType>("forward_execs");
    }
    AttachOpExecsRec(graph,
                     shapes,
                     dtypes,
                     mem_plan,
                     vdevice,
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
.depend_entry_attr(plan_memory::ref_key)
.depend_node_attr(ctx::device_key)
.depend_global_attr(ctx::ctx_key)
.depend_node_attr(mutate::key)
.provide_node_attr(attach_op::key);

}  // namespace pass
}  // namespace mxnet
