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
#include "../nnvm/graph_executor_v2.h"
#include "../executor/exec_pass.h"

int MXGraphCreate(SymbolHandle symbol, GraphHandle *out) {
  using nnvm::GraphPtr;
  using nnvm::Graph;
  using nnvm::Symbol;
  GraphPtr* pg = new GraphPtr();
  *pg = Graph::Create();
  API_BEGIN();
  (*pg)->outputs = static_cast<Symbol*>(symbol)->outputs;
  *out = pg;
  API_END_HANDLE_ERROR(delete pg;);
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
  std::unordered_map<std::string, std::shared_ptr<any>> kwargs_any;
  std::vector<TShape> shape_inputs(num_arrays, TShape());
  std::vector<int> dtype_inputs(num_arrays, -1);
  shape_inputs.reserve(num_arrays);
  dtype_inputs.reserve(num_arrays);
  for (int i = 0; i < num_arrays; ++i) {
    NDArray* arr = static_cast<NDArray*>(arrays[i]);
    if (arr && !arr->is_none()) {
      shape_inputs[i] = arr->shape();
      dtype_inputs[i] = arr->dtype();
    }
  }
  kwargs_any["shape_inputs"] = std::make_shared<any>(std::move(shape_inputs));
  kwargs_any["dtype_inputs"] = std::make_shared<any>(std::move(dtype_inputs));
  kwargs_any["graph_frozen"] = std::make_shared<any>((int)1);
  nnvm::Specialize(pg.get(), kwargs_any);
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
  std::vector<Context> ctx = {Context::GPU(0)};
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
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();

  API_BEGIN();
  GraphExecutorV2* exec = static_cast<GraphExecutorV2*>(ehdl);
  std::vector<NDArray> arguments, results;
  // Feed argument arrays.
  NDArray** args_ptr = reinterpret_cast<NDArray**>(inputs);
  for (int i = 0; i < num_inputs; ++i) {
    arguments.push_back(*args_ptr[i]);
  }
  // Create result arrays.
  NDArray** rsts_ptr = *reinterpret_cast<NDArray***>(outputs);
  if (rsts_ptr != nullptr) {
    for (int i = 0; i < *num_outputs; ++i) {
      results.push_back(*rsts_ptr[i]);
    }
  } else {
    const auto& graph = exec->graph();
    if (graph.global_attrs.count("num_visible_outputs")) {
      *num_outputs = graph.GetGlobalAttr<size_t>("num_visible_outputs");
    } else {
      *num_outputs = graph.outputs.size();
    }
  }
  GraphExecutorV2::RunOption opt;
  opt.is_train = (is_training == 1);
  exec->Run(arguments, &results, opt);
  //TODO(state)
  //exec->GetState();

  if (Imperative::Get()->is_recording()) {
    NodeAttrs attrs;
    //Imperative::Get()->RecordOp(std::move(attrs), arguments, results);
  }

  if (rsts_ptr == nullptr) {
    // TODO(minjie): num visible outputs
    ret->ret_handles.clear();
    for (int i = 0; i < *num_outputs; ++i) {
      ret->ret_handles.push_back(
          reinterpret_cast<NDArrayHandle>(new NDArray(std::move(results[i]))));
    }
    *outputs = dmlc::BeginPtr(ret->ret_handles);
  }
  API_END();
}
