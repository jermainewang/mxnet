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
namespace exec {
FCompute GetFCompute(const Op* op, Context ctx) {
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

void AttachFunctorInfoRec(
    const Graph& g,
    const Column<TShape>* vshape,
    const Column<int>* vdtype,
    const Column<int>* vdevice,
    const Column<FunctorInfo>* fwd_infos,
    const vector<Context>& context,
    Column<FunctorInfo>* infos) {
  static auto& fcreate_layer_op = nnvm::Op::GetAttr<FCreateOpState>("FCreateOpState");
  static auto& is_layer_backward = nnvm::Op::GetAttr<bool>("TIsLayerOpBackward");
  const auto& idx = g.indexed_graph();

  if (fwd_infos != nullptr) {
    // This is a standalone backward graph.
    CHECK(g.global_attrs.count("gradient_node_mapping"));
    const auto& gradient_node_mapping =
      g.GetGlobalAttr<vector<uint32_t>>("gradient_node_mapping");
    CHECK_EQ(gradient_node_mapping.size(), idx.num_nodes());
    for (uint32_t nid = 0; nid < gradient_node_mapping.size(); ++nid) {
      const Node* node = idx[nid].source;
      const uint32_t fwd_nid = gradient_node_mapping[nid];
      if (fwd_nid < fwd_infos->value.size()) {
        const auto& fwd_info = fwd_infos->value[fwd_nid];
        if (fwd_info.type == FunctorType::kForward) {
          infos->value[nid].type = FunctorType::kBackward;
          infos->value[nid].state = fwd_info.state;
        }
      }
    }
  }
  
  for (size_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    if (infos->value[nid].type != FunctorType::kUndefined) {
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
      const Column<FunctorInfo>* sub_fwd_infos = nullptr;
      if (subgraph->global_attrs.count("gradient_node_mapping")) {
        if (fwd_infos != nullptr) {
          // Forward node info is provided.
          const auto& node_mapping =
            g.GetGlobalAttr<vector<uint32_t>>("gradient_node_mapping");
          const uint32_t fwd_nid = node_mapping[nid];
          CHECK_LT(fwd_nid, fwd_infos->children.size());
          sub_fwd_infos = fwd_infos->children[fwd_nid].get();
        } else {
          // Foward node is connected by the first control dependency.
          CHECK_GE(node->control_deps.size(), 1U)
            << "Backward node need to have control_deps to its forward node";
          const NodePtr& fwd_ptr = node->control_deps[0];
          CHECK(fwd_ptr->is_graph());
          const uint32_t fwd_nid = idx.node_id(fwd_ptr.get());
          sub_fwd_infos = infos->children[fwd_nid].get();
        }
      }
      AttachFunctorInfoRec(*subgraph,
                       vshape->children[nid].get(),
                       vdtype->children[nid].get(),
                       vdevice->children[nid].get(),
                       sub_fwd_infos,
                       context,
                       infos->children[nid].CopyOnWrite());
    } else {
      auto& info = infos->value[nid];
      FCompute fcompute = GetFCompute(node->op(), context.at(vdevice->value[nid]));
      if (fcompute != nullptr) {
        info.type = FunctorType::kFCompute;
      } else if (fcreate_layer_op.count(node->op())) {
        vector<TShape> ishape;
        vector<int> itype;
        for (const auto& e : node->inputs) {
          ishape.emplace_back(vshape->value[idx.entry_id(e)]);
          itype.emplace_back(vdtype->value[idx.entry_id(e)]);
        }
        info.type = FunctorType::kForward;
        info.state = fcreate_layer_op[node->op()](
              node->attrs, context.at(vdevice->value[nid]), ishape, itype);
      } else if (is_layer_backward.get(node->op(), false)) {
        CHECK_GE(node->control_deps.size(), 1);
        const uint32_t fwd_nid = idx.node_id(node->control_deps[0].get());
        CHECK(vdevice->value[fwd_nid] == vdevice->value[nid]);
        info.type = FunctorType::kBackward;
        info.state = infos->value[fwd_nid].state;
      } else {
        // Do nothing.
      }
    }
  }
}

}  // namespace exec
}  // namespace mxnet
