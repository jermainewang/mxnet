/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file c_api_symbolic.cc
 * \brief C API of mxnet
 */
#include <mxnet/base.h>
#include <mxnet/c_api.h>
#include <mxnet/ndarray.h>
#include <mxnet/imperative.h>

#include <nnvm/c_api.h>
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include <nnvm/symbolic.h>

#include "./c_api_common.h"
#include "../operator/operator_common.h"
#include "../nnvm/mx_passes.h"
#include "../nnvm/graph_executor_v2.h"
#include "../executor/exec_pass.h"
#include "../imperative/autograd.h"

namespace {
void RunGraph(exec::GraphExecutorV2* exec,
              int num_inputs,
              NDArrayHandle *inputs,
              int *num_outputs,
              NDArrayHandle **outputs,
              const exec::GraphExecutorV2::RunOption& opt) {
  using nnvm::GraphPtr;
  using exec::GraphExecutorV2;
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  auto gptr = exec->graph();
  const int num_visible_outputs =
    (gptr->global_attrs.count("num_visible_outputs")) ?
    gptr->GetGlobalAttr<size_t>("num_visible_outputs") :
    gptr->outputs.size();
  const int num_all_outputs = gptr->outputs.size();

  std::vector<NDArray> arguments, results;
  // Feed argument arrays.
  NDArray** args_ptr = reinterpret_cast<NDArray**>(inputs);
  for (int i = 0; i < num_inputs; ++i) {
    arguments.push_back(*args_ptr[i]);
  }
  // Create result arrays.
  NDArray** rsts_ptr = *reinterpret_cast<NDArray***>(outputs);
  if (rsts_ptr != nullptr) {
    CHECK_EQ(*num_outputs, num_visible_outputs);
    results.resize(num_all_outputs);
    for (int i = 0; i < num_visible_outputs; ++i) {
      results[i] = *rsts_ptr[i];
    }
  } else {
    *num_outputs = num_visible_outputs;
  }
  exec->Run(arguments, &results, opt);

  if (Imperative::Get()->is_recording()) {
    // TODO(minjie): output arrays are not supported in recording mode.
    CHECK(rsts_ptr == nullptr);
    NodeAttrs attrs;
    attrs.graph = gptr;
    std::vector<NDArray*> ndinputs(arguments.size()), ndoutputs(results.size());
    for (size_t i = 0; i < arguments.size(); ++i) {
      ndinputs[i] = args_ptr[i];
    }
    for (size_t i = 0; i < results.size(); ++i) {
      ndoutputs[i] = &results[i];
    }
    ag::AutogradTape::Get().Record(attrs, ndinputs, ndoutputs);
  }

  if (rsts_ptr == nullptr) {
    ret->ret_handles.clear();
    for (int i = 0; i < num_visible_outputs; ++i) {
      ret->ret_handles.push_back(
          reinterpret_cast<NDArrayHandle>(new NDArray(std::move(results[i]))));
    }
    *outputs = dmlc::BeginPtr(ret->ret_handles);
  }
}
}  // namespace

void _SpecializeByNDArrays(nnvm::Graph* graph, int num_arrays, NDArrayHandle *arrays) {
  using nnvm::any;
  using pass::shape::MXInferShapeArgs;
  using pass::dtype::MXInferTypeArgs;
  using pass::plan_memory::MXPlanMemoryArgs;
  std::unordered_map<std::string, std::shared_ptr<any>> kwargs_any;
  std::vector<TShape> shape_inputs(num_arrays, TShape());
  std::vector<int> dtype_inputs(num_arrays, -1);
  for (int i = 0; i < num_arrays; ++i) {
    NDArray* arr = static_cast<NDArray*>(arrays[i]);
    if (arr && !arr->is_none()) {
      shape_inputs[i] = arr->shape();
      dtype_inputs[i] = arr->dtype();
    }
  }
  MXInferShapeArgs shape_args;
  MXInferTypeArgs dtype_args;
  shape_args.shape_inputs = std::move(shape_inputs);
  dtype_args.dtype_inputs = std::move(dtype_inputs);
  kwargs_any[pass::shape::arg_name] = std::make_shared<any>(std::move(shape_args));
  kwargs_any[pass::dtype::arg_name] = std::make_shared<any>(std::move(dtype_args));
  nnvm::Specialize(graph, kwargs_any);
}

int MXGraphCreate(SymbolHandle symbol, GraphHandle *out) {
  using nnvm::GraphPtr;
  using nnvm::Graph;
  using nnvm::Symbol;
  using nnvm::any;
  GraphPtr* ppg = new GraphPtr();
  *ppg = Graph::Create();
  API_BEGIN();
  Graph& graph = **ppg;
  graph.outputs = static_cast<Symbol*>(symbol)->outputs;
  *out = ppg;
  API_END_HANDLE_ERROR(delete ppg;);
}

int MXGraphFree(GraphHandle graph) {
  using nnvm::GraphPtr;
  API_BEGIN();
  delete static_cast<GraphPtr*>(graph);
  API_END();
}

int MXGraphSpecialize(GraphHandle graph,
                      mx_uint num_param,
                      const char **keys,
                      const char **vals) {
  using nnvm::GraphPtr;
  using nnvm::any;
  API_BEGIN();
  GraphPtr pg = *static_cast<GraphPtr*>(graph);
  const auto& kwargs = _ExtractSymbolKWArgs(num_param, keys, vals);
  std::unordered_map<std::string, std::shared_ptr<any>> kwargs_any;
  for (const auto& kv : kwargs) {
    kwargs_any[kv.first] = std::make_shared<any>(kv.second);
  }
  nnvm::Specialize(pg.get(), kwargs_any);
  API_END();
}

MXNET_DLL int MXGraphSpecializeByNDArrays(GraphHandle graph,
                                          int num_arrays,
                                          NDArrayHandle *arrays) {
  using nnvm::GraphPtr;
  using nnvm::any;
  API_BEGIN();
  GraphPtr pg = *static_cast<GraphPtr*>(graph);
  _SpecializeByNDArrays(pg.get(), num_arrays, arrays);
  API_END();
}

int MXGraphTransform(GraphHandle graph,
                     mx_uint num_passes,
                     const char **pass_names,
                     mx_uint num_param,
                     const char **keys,
                     const char **vals,
                     GraphHandle *out) {
  using nnvm::GraphPtr;
  using nnvm::Graph;
  using nnvm::any;
  GraphPtr* pp_out_graph = new GraphPtr();
  *pp_out_graph = Graph::Create();
  API_BEGIN();
  GraphPtr p_in_graph = *static_cast<GraphPtr*>(graph);
  std::vector<std::string> passes;
  passes.reserve(num_passes);
  for (nn_uint i = 0; i < num_passes; ++i) {
    passes.push_back(std::string(pass_names[i]));
  }
  const auto& kwargs = _ExtractSymbolKWArgs(num_param, keys, vals);
  std::unordered_map<std::string, std::shared_ptr<any>> kwargs_any;
  for (const auto& kv : kwargs) {
    kwargs_any[kv.first] = std::make_shared<any>(kv.second);
  }
  **pp_out_graph = nnvm::Transform(*p_in_graph, passes, kwargs_any);
  *out = pp_out_graph;
  API_END_HANDLE_ERROR(delete pp_out_graph;);
}

int MXGraphTransformToOpCompatible(GraphHandle ghdl,
                                   mx_uint grad_order,
                                   GraphHandle *out) {
  using nnvm::GraphPtr;
  using nnvm::Graph;
  using nnvm::any;
  GraphPtr* pp_out_graph = new GraphPtr();
  *pp_out_graph = Graph::Create();
  API_BEGIN();
  CHECK_LE(grad_order, 1)
    << "Up to first-order gradient is supported right now.";
  GraphPtr pg = *static_cast<GraphPtr*>(ghdl);
  std::unordered_map<std::string, std::shared_ptr<any>> kwargs_any;
  **pp_out_graph = nnvm::Transform(*pg, {"MXExposeInvisibleOutputs"}, kwargs_any);
  if (grad_order > 0) {
    kwargs_any["mx_gradient_args"] = std::make_shared<any>(pass::grad::MXGradientArgs());
    **pp_out_graph = nnvm::Transform(**pp_out_graph, {"MXGradient"}, kwargs_any);
  }
  *out = pp_out_graph;
  API_END_HANDLE_ERROR(delete pp_out_graph;);
}

int MXGraphGetGlobalAttrJSON(GraphHandle graph,
                             const char *key,
                             const char **out) {
  using nnvm::GraphPtr;
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  GraphPtr pg = *static_cast<GraphPtr*>(graph);
  std::string skey(key);
  auto it = pg->global_attrs.find(skey);
  *out = "";
  if (it != pg->global_attrs.end()) {
    std::ostringstream oss;
    dmlc::JSONWriter writer(&oss);
    writer.Write(*it->second.get());
    ret->ret_str = oss.str();
    *out = (ret->ret_str).c_str();
  }
  API_END();
}

int MXGraphGetNodeAttrJSON(GraphHandle graph,
                           const char *key,
                           const char **out) {
  using nnvm::GraphPtr;
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  GraphPtr pg = *static_cast<GraphPtr*>(graph);
  std::string skey(key);
  *out = "";
  if (pg->node_attrs.count(skey)) {
    std::ostringstream oss;
    dmlc::JSONWriter writer(&oss);
    pg->node_attrs.SaveColumn(skey, &writer);
    ret->ret_str = oss.str();
    *out = (ret->ret_str).c_str();
  }
  API_END();
}

int MXGraphGetNodeEntryAttrJSON(GraphHandle graph,
                                const char *key,
                                const char **out) {
  using nnvm::GraphPtr;
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  GraphPtr pg = *static_cast<GraphPtr*>(graph);
  std::string skey(key);
  *out = "";
  if (pg->entry_attrs.count(skey)) {
    std::ostringstream oss;
    dmlc::JSONWriter writer(&oss);
    pg->entry_attrs.SaveColumn(skey, &writer);
    ret->ret_str = oss.str();
    *out = (ret->ret_str).c_str();
  }
  API_END();
}

int MXSymbolCreateGraphSymbol(GraphHandle ghdl,
                              mx_uint num_param,
                              const char **keys,
                              const char **vals,
                              SymbolHandle *out) {
  using nnvm::Symbol;
  using nnvm::GraphPtr;
  Symbol *s = new Symbol();
  API_BEGIN();
  GraphPtr* pg = static_cast<GraphPtr*>(ghdl);
  *s = Symbol::CreateFunctor(*pg, _ExtractSymbolKWArgs(num_param, keys, vals));
  *out = s;
  API_END_HANDLE_ERROR(delete s;);
}

int MXGraphCreateInputArrays(GraphHandle graph,
                             int *num_inputs,
                             NDArrayHandle **in_arrays) {
  using nnvm::GraphPtr;
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  GraphPtr pg = *static_cast<GraphPtr*>(graph);
  const auto& idx = pg->indexed_graph();
  *num_inputs = idx.input_nodes().size();
  CHECK(pg->entry_attrs.count("shape") && pg->entry_attrs.count("dtype"))
    << "Cannot create input arrays. Graph need to be specialized"
    << " for attributes \"shape\" and \"dtype\" beforehand.";
  const auto& shapes = pg->entry_attrs.GetColumn<TShape>("shape");
  const auto& dtypes = pg->entry_attrs.GetColumn<int>("dtype");
  ret->ret_handles.clear();
  // TODO(minjie): context.
  const Context& ctx = Context::CPU();
  for (int i = 0; i < *num_inputs; ++i) {
    const uint32_t eid = idx.entry_id(idx.input_nodes()[i], 0);
    ret->ret_handles.push_back(
        reinterpret_cast<NDArrayHandle>(new NDArray(
            shapes->value[eid], ctx, true, dtypes->value[eid])));
  }
  *in_arrays = dmlc::BeginPtr(ret->ret_handles);
  API_END();
}

int MXGraphCreateOutputArrays(GraphHandle graph,
                              int *num_outputs,
                              NDArrayHandle **out_arrays) {
  using nnvm::GraphPtr;
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  GraphPtr pg = *static_cast<GraphPtr*>(graph);
  const auto& idx = pg->indexed_graph();
  *num_outputs = pg->outputs.size();
  CHECK(pg->entry_attrs.count("shape") && pg->entry_attrs.count("dtype"))
    << "Cannot create input arrays. Graph need to be specialized"
    << " for attributes \"shape\" and \"dtype\" beforehand.";
  const auto& shapes = pg->entry_attrs.GetColumn<TShape>("shape");
  const auto& dtypes = pg->entry_attrs.GetColumn<int>("dtype");
  ret->ret_handles.clear();
  // TODO(minjie): context.
  const Context& ctx = Context::CPU();
  for (int i = 0; i < *num_outputs; ++i) {
    const uint32_t eid = idx.entry_id(pg->outputs[i]);
    ret->ret_handles.push_back(
        reinterpret_cast<NDArrayHandle>(new NDArray(
            shapes->value[eid], ctx, true, dtypes->value[eid])));
  }
  *out_arrays = dmlc::BeginPtr(ret->ret_handles);
  API_END();
}

int MXGraphEval(GraphHandle ghdl,
                int num_inputs,
                NDArrayHandle *inputs,
                int *num_outputs,
                NDArrayHandle **outputs,
                int is_training) {
  using nnvm::GraphPtr;
  using exec::GraphExecutorV2;

  API_BEGIN();
  GraphPtr pg = *static_cast<GraphPtr*>(ghdl);

  // TODO(minjie): hard-code for testing.
  using nnvm::any;
  std::unordered_map<std::string, std::shared_ptr<any>> kwargs_any;
  std::vector<Context> ctx = {Context::CPU(0)};
  kwargs_any[pass::ctx::ctx_key] = std::make_shared<any>(std::move(ctx));
  nnvm::Specialize(pg.get(), kwargs_any);

  GraphExecutorV2 exec(pg);
  GraphExecutorV2::RunOption opt;
  if (Imperative::Get()->is_training()) {
    opt.is_train = true;
  } else {
    opt.is_train = (is_training == 1);
  }
  RunGraph(&exec, num_inputs, inputs, num_outputs, outputs, opt);
  API_END();
}

int MXExecV2Create(GraphHandle ghdl, 
                   int dynamic_allocation,
                   int zero_copy,
                   GraphExecutorV2Handle *out) {
  using nnvm::GraphPtr;
  using exec::GraphExecutorV2;
  GraphExecutorV2::Config cfg;
  cfg.dynamic_allocation = (dynamic_allocation == 1);
  cfg.zero_copy = (zero_copy == 1);
  GraphPtr pg = *static_cast<GraphPtr*>(ghdl);

  // TODO(minjie): hard-code for testing.
  using nnvm::any;
  std::unordered_map<std::string, std::shared_ptr<any>> kwargs_any;
  std::vector<Context> ctx = {Context::CPU(0)};
  kwargs_any["context"] = std::make_shared<any>(std::move(ctx));
  nnvm::Specialize(pg.get(), kwargs_any);

  GraphExecutorV2::ExecState fwd_state;
  GraphExecutorV2* exec = new GraphExecutorV2(pg, fwd_state, cfg);
  API_BEGIN();
  *out = exec;
  API_END_HANDLE_ERROR(delete exec;);
}

int MXExecV2Run(GraphExecutorV2Handle ehdl,
                int num_inputs,
                NDArrayHandle *inputs,
                int *num_outputs,
                NDArrayHandle **outputs,
                int is_training) {
  using nnvm::any;
  using exec::GraphExecutorV2;

  API_BEGIN();
  GraphExecutorV2* exec = static_cast<GraphExecutorV2*>(ehdl);
  GraphExecutorV2::RunOption opt;
  if (Imperative::Get()->is_training()) {
    opt.is_train = true;
  } else {
    opt.is_train = (is_training == 1);
  }
  RunGraph(exec, num_inputs, inputs, num_outputs, outputs, opt);
  API_END();
}
