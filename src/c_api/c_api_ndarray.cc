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
 *  Copyright (c) 2016 by Contributors
 * \file c_api_ndarray.cc
 * \brief C API of mxnet
 */

#include <mxnet/base.h>
#include <mxnet/c_api.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/imperative.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <string>
#include "./c_api_common.h"
#include "../common/utils.h"
#include "../common/exec_utils.h"
#include "../imperative/taping.h"
#include "../imperative/autograd.h"

using namespace mxnet;

nnvm::NodeAttrs ParseAttrs(const nnvm::Op *op,
                           const int& num_inputs,
                           const int& num_params,
                           const char **param_keys,
                           const char **param_vals) {
  static auto& num_args = nnvm::Op::GetAttr<std::string>("key_var_num_args");

  nnvm::NodeAttrs attrs;
  attrs.op = op;
  attrs.dict.reserve(num_params+1);
  for (int i = 0; i < num_params; ++i) {
    attrs.dict.emplace(param_keys[i], param_vals[i]);
  }
  if (num_args.count(op)) {
    attrs.dict.emplace(num_args[op], std::to_string(num_inputs));
  }
  if (op->attr_parser != nullptr) {
    op->attr_parser(&attrs);
  }

  return attrs;
}

void SetNumOutputs(const nnvm::Op *op,
                   const nnvm::NodeAttrs& attrs,
                   const int& num_inputs,
                   int* infered_num_outputs,
                   int* num_visible_outputs) {
  static auto& visible_out = nnvm::Op::GetAttr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs");
  int infered_num_inputs;
  if (op->get_num_inputs != nullptr) {
    infered_num_inputs = op->get_num_inputs(attrs);
  } else {
    infered_num_inputs = op->num_inputs;
  }
  CHECK_EQ(num_inputs, infered_num_inputs)
    << "Operator " << op->name << " expects " << infered_num_inputs
    << " inputs, but got " << num_inputs << " instead.";
  if (op->get_num_outputs != nullptr) {
    *infered_num_outputs = op->get_num_outputs(attrs);
  } else {
    *infered_num_outputs = op->num_outputs;
  }
  *num_visible_outputs = *infered_num_outputs;
  if (visible_out.count(op)) {
    *num_visible_outputs = visible_out[op](attrs);
    CHECK_LE(*num_visible_outputs, *infered_num_outputs);
  }
}

void SetNDInputsOutputs(const nnvm::Op* op,
                        std::vector<NDArray*>* ndinputs,
                        std::vector<NDArray*>* ndoutputs,
                        int num_inputs,
                        const NDArrayHandle *inputs,
                        int *num_outputs,
                        int infered_num_outputs,
                        int num_visible_outputs,
                        NDArrayHandle **outputs) {
  NDArray** out_array = *reinterpret_cast<NDArray***>(outputs);

  ndinputs->clear();
  ndinputs->reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    ndinputs->emplace_back(reinterpret_cast<NDArray*>(inputs[i]));
  }

  ndoutputs->clear();
  ndoutputs->reserve(infered_num_outputs);
  if (out_array == nullptr) {
    for (int i = 0; i < infered_num_outputs; ++i) {
      ndoutputs->emplace_back(new NDArray());
    }
    *num_outputs = num_visible_outputs;
  } else {
    CHECK(*num_outputs == infered_num_outputs || *num_outputs == num_visible_outputs)
      << "Operator expects " << infered_num_outputs << " (all) or "
      << num_visible_outputs << " (visible only) outputs, but got "
      << *num_outputs << " instead.";
    for (int i = 0; i < *num_outputs; ++i) {
      ndoutputs->emplace_back(out_array[i]);
    }
    for (int i = *num_outputs; i < infered_num_outputs; ++i) {
      ndoutputs->emplace_back(new NDArray());
    }
  }
}

void MXImperativeInvokeImpl(AtomicSymbolCreator creator,
                            int num_inputs,
                            NDArrayHandle *inputs,
                            int *num_outputs,
                            NDArrayHandle **outputs,
                            int num_params,
                            const char **param_keys,
                            const char **param_vals) {
  using nnvm::Op;
  using nnvm::Symbol;
  using nnvm::Graph;
  const Op* op = static_cast<nnvm::Op*>(creator);
  //LOG(INFO) << "ImperativeInvoke op: " << op->name;
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();

  nnvm::NodeAttrs attrs = ParseAttrs(op, num_inputs, num_params, param_keys, param_vals);

  int infered_num_outputs;
  int num_visible_outputs;
  SetNumOutputs(op, attrs, num_inputs, &infered_num_outputs, &num_visible_outputs);

  std::vector<NDArray*> ndinputs, ndoutputs;
  SetNDInputsOutputs(op, &ndinputs, &ndoutputs, num_inputs, inputs,
      num_outputs, infered_num_outputs, num_visible_outputs, outputs);

  auto state = Imperative::Get()->Invoke(Context::CPU(), attrs, ndinputs, ndoutputs);
#ifdef USE_LEGACY_AUTOGRAD
  if (Imperative::Get()->is_recording()) {
    Imperative::Get()->RecordOp(std::move(attrs), ndinputs, ndoutputs, state);
  }
#else
  if (Imperative::Get()->is_recording()) {
    ag::AutogradTape::Get().Record(attrs, ndinputs, ndoutputs);
  }
#endif

  for (int i = *num_outputs; i < infered_num_outputs; ++i) delete ndoutputs[i];

  if (*outputs == nullptr) {
    ret->ret_handles.clear();
    ret->ret_handles.reserve(*num_outputs);
    for (int i = 0; i < *num_outputs; ++i) ret->ret_handles.push_back(ndoutputs[i]);
    *outputs = reinterpret_cast<NDArrayHandle*>(dmlc::BeginPtr(ret->ret_handles));
  }
}

int MXImperativeInvoke(AtomicSymbolCreator creator,
                       int num_inputs,
                       NDArrayHandle *inputs,
                       int *num_outputs,
                       NDArrayHandle **outputs,
                       int num_params,
                       const char **param_keys,
                       const char **param_vals) {
  API_BEGIN();
  MXImperativeInvokeImpl(creator, num_inputs, inputs, num_outputs, outputs,
                         num_params, param_keys, param_vals);
  API_END();
}

int MXImperativeInvokeEx(AtomicSymbolCreator creator,
                         int num_inputs,
                         NDArrayHandle *inputs,
                         int *num_outputs,
                         NDArrayHandle **outputs,
                         int num_params,
                         const char **param_keys,
                         const char **param_vals,
                         const int **out_stypes) {  // outputs storage types
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  MXImperativeInvokeImpl(creator, num_inputs, inputs, num_outputs, outputs,
                         num_params, param_keys, param_vals);
  NDArray** out_array = *reinterpret_cast<NDArray***>(outputs);
  ret->out_types.clear();
  ret->out_types.reserve(*num_outputs);
  for (int i = 0; i < *num_outputs; ++i) {
    ret->out_types.emplace_back(out_array[i]->storage_type());
  }
  *out_stypes = dmlc::BeginPtr(ret->out_types);
  API_END();
}

int MXCreateCachedOp(SymbolHandle handle,
                     CachedOpHandle *out) {
  LOG(FATAL) << "Cached op has been disabled.";
  nnvm::Symbol* sym = static_cast<nnvm::Symbol*>(handle);

  API_BEGIN();
  *out = new std::shared_ptr<Imperative::CachedOp>(
      new Imperative::CachedOp(*sym));
  API_END();
}

int MXFreeCachedOp(CachedOpHandle handle) {
  CachedOpPtr* g = static_cast<CachedOpPtr*>(handle);
  API_BEGIN();
  delete g;
  API_END();
}

int MXInvokeCachedOp(CachedOpHandle handle,
                     int num_inputs,
                     NDArrayHandle *inputs,
                     int *num_outputs,
                     NDArrayHandle **outputs) {
  LOG(FATAL) << "Cached op has been disabled.";

  static const auto cached_op = nnvm::Op::Get("_CachedOp");
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();

  API_BEGIN();
  CachedOpPtr op = *static_cast<CachedOpPtr*>(handle);
  std::vector<NDArray*> ndinputs;
  ndinputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    ndinputs.push_back(reinterpret_cast<NDArray*>(inputs[i]));
  }

  std::vector<NDArray*> ndoutputs;
  ndoutputs.reserve(op->num_outputs());
  if (*outputs == nullptr) {
    *num_outputs = op->num_outputs();
    for (int i = 0; i < *num_outputs; ++i) ndoutputs.push_back(new NDArray());
  } else {
    CHECK_EQ(*num_outputs, op->num_outputs())
        << "CachedOp expects " << op->num_outputs() << " outputs, but "
        << *num_outputs << " was given.";
    for (int i = 0; i < *num_outputs; ++i) {
      ndoutputs.push_back(reinterpret_cast<NDArray*>((*outputs)[i]));
    }
  }

  OpStatePtr state = op->Forward(ndinputs, ndoutputs);
  if (Imperative::Get()->is_recording()) {
    nnvm::NodeAttrs attrs;
    attrs.op = cached_op;
    attrs.name = "_cachedop";
    attrs.parsed = op;
    Imperative::Get()->RecordOp(
        std::move(attrs), ndinputs, ndoutputs, state,
        &op->save_inputs(), &op->save_outputs());
  }

  if (*outputs == nullptr) {
    ret->ret_handles.clear();
    ret->ret_handles.reserve(*num_outputs);
    for (int i = 0; i < *num_outputs; ++i) {
      ret->ret_handles.push_back(ndoutputs[i]);
    }
    *outputs = dmlc::BeginPtr(ret->ret_handles);
  }

  API_END();
}

int MXInvokeCachedOpEx(CachedOpHandle handle,
                       int num_inputs,
                       NDArrayHandle *inputs,
                       int *num_outputs,
                       NDArrayHandle **outputs,
                       const int **out_stypes) {  // outputs storage types
  LOG(FATAL) << "Cached op has been disabled.";

  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  int err = MXInvokeCachedOp(handle, num_inputs, inputs, num_outputs, outputs);
  if (err != 0) return err;
  API_BEGIN();
  NDArray** out_array = reinterpret_cast<NDArray**>(*outputs);
  ret->out_types.clear();
  ret->out_types.reserve(*num_outputs);
  for (int i = 0; i < *num_outputs; ++i) {
    ret->out_types.emplace_back(out_array[i]->storage_type());
  }
  *out_stypes = dmlc::BeginPtr(ret->out_types);
  API_END();
}

int MXAutogradIsTraining(bool* curr) {
  API_BEGIN();
  *curr = Imperative::Get()->is_training();
  API_END();
}

int MXAutogradSetIsTraining(int is_training, int* prev) {
  API_BEGIN();
  *prev = Imperative::Get()->set_is_training(static_cast<bool>(is_training));
  API_END();
}

int MXAutogradIsRecording(bool* curr) {
  API_BEGIN();
  *curr = Imperative::Get()->is_recording();
  API_END();
}

int MXAutogradSetIsRecording(int is_recording, int* prev) {
  API_BEGIN();
#ifndef USE_LEGACY_AUTOGRAD
  if (Imperative::Get()->is_recording() && is_recording == false) {
    // Turning off previous recording tape.
    ag::AutogradTape::Get().NewSession();
  }
#endif
  *prev = Imperative::Get()->set_is_recording(static_cast<bool>(is_recording));
  API_END();
}

int MXAutogradMarkVariables(mx_uint num_var,
                            NDArrayHandle *var_handles,
                            mx_uint *reqs_array,
                            NDArrayHandle *grad_handles) {
  API_BEGIN();
#ifdef USE_LEGACY_AUTOGRAD
  std::vector<NDArray*> variables, gradients;
  std::vector<mx_uint> grad_reqs;
  variables.reserve(num_var);
  gradients.reserve(num_var);
  grad_reqs.reserve(num_var);
  for (mx_uint i = 0; i < num_var; ++i) {
    variables.emplace_back(static_cast<NDArray*>(var_handles[i]));
    gradients.emplace_back(static_cast<NDArray*>(grad_handles[i]));
    grad_reqs.emplace_back(reqs_array[i]);
  }
  Imperative::Get()->MarkVariables(variables, grad_reqs, gradients);
#else
  for (mx_uint i = 0; i < num_var; ++i) {
    NDArray* var = static_cast<NDArray*>(var_handles[i]);
    NDArray* grad = static_cast<NDArray*>(grad_handles[i]);
    OpReqType req = static_cast<OpReqType>(reqs_array[i]);
    var->AttachGrad(req, *grad);
    ag::AutogradTape::Get().AttachGrad(var->tape_entry_id(), req, *grad);
  }
#endif
  API_END();
}

int MXAutogradComputeGradient(mx_uint num_output,
                              NDArrayHandle *output_handles) {
  return MXAutogradBackward(num_output, output_handles, nullptr, 0);
}

int MXAutogradBackward(mx_uint num_output,
                       NDArrayHandle *output_handles,
                       NDArrayHandle *ograd_handles,
                       int retain_graph) {
  return MXAutogradBackwardEx(num_output, output_handles, ograd_handles,
                              0, nullptr, retain_graph, false, true,
                              nullptr, nullptr);
}

int MXAutogradBackwardEx(mx_uint num_output,
                         NDArrayHandle *output_handles,
                         NDArrayHandle *ograd_handles,
                         mx_uint num_variables,
                         NDArrayHandle *var_handles,
                         int retain_graph,
                         int create_graph,
                         int is_train,
                         NDArrayHandle **grad_handles,
                         int **grad_stypes) {
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();

#ifdef USE_LEGACY_AUTOGRAD
  std::vector<NDArray*> outputs, ograds, variables;
#else
  std::vector<const NDArray*> outputs, ograds, variables;
#endif
  outputs.reserve(num_output);
  for (mx_uint i = 0; i < num_output; ++i) {
    outputs.emplace_back(reinterpret_cast<NDArray*>(output_handles[i]));
  }

  ograds.reserve(num_output);
  for (mx_uint i = 0; i < num_output; ++i) {
    if (ograd_handles != nullptr) {
      ograds.emplace_back(reinterpret_cast<NDArray*>(ograd_handles[i]));
    } else {
      ograds.emplace_back(nullptr);
    }
  }

  variables.reserve(num_variables);
  for (mx_uint i = 0; i < num_variables; ++i) {
    variables.emplace_back(reinterpret_cast<NDArray*>(var_handles[i]));
  }

#ifdef USE_LEGACY_AUTOGRAD
  auto grads = Imperative::Get()->Backward(outputs, ograds, variables, is_train,
                                                  retain_graph, create_graph);
  if (num_variables != 0) {
    ret->ret_handles.clear();
    ret->out_types.clear();
    ret->ret_handles.reserve(grads.size());
    ret->out_types.reserve(grads.size());
    for (const auto& i : grads) {
      ret->ret_handles.push_back(i);
      ret->out_types.push_back(i->storage_type());
    }
    *grad_handles = dmlc::BeginPtr(ret->ret_handles);
    *grad_stypes = dmlc::BeginPtr(ret->out_types);
  }
#else
  ag::AutogradTape::Get().GetSpecializedBackwardGraph(
      outputs, variables, ograds);
  // Evaluate the backward graph.
  // 0. Specialize for context.
  // 1. Make the correct input and output args.
  
  //LOG(FATAL) << "Not Implemented.";

  if (num_variables != 0) {
    //ret->ret_handles.clear();
    //ret->out_types.clear();
    //ret->ret_handles.reserve(grads.size());
    //ret->out_types.reserve(grads.size());
    //for (const auto& i : grads) {
      //ret->ret_handles.push_back(i);
      //ret->out_types.push_back(i->storage_type());
    //}
    //*grad_handles = dmlc::BeginPtr(ret->ret_handles);
    //*grad_stypes = dmlc::BeginPtr(ret->out_types);
  }
#endif
  API_END();
}

int MXAutogradGetSymbol(NDArrayHandle handle, SymbolHandle *out) {
  API_BEGIN();
  NDArray *head = reinterpret_cast<NDArray*>(handle);
  auto sym = new nnvm::Symbol(head->get_autograd_symbol());
  *out = reinterpret_cast<SymbolHandle>(sym);
  API_END();
}
