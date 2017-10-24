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
#include <nnvm/c_api.h>
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include <nnvm/symbolic.h>
#include "./c_api_common.h"
#include "../operator/operator_common.h"
#include "../nnvm/graph_executor_v2.h"
#include "../executor/exec_pass.h"

namespace mxnet {
namespace op {
void RegisterLegacyOpProp();
void RegisterLegacyNDFunc();
}
const std::vector<std::string> kHiddenKeys = {
  "ctx_group", "lr_mult", "wd_mult", "force_mirroring", "mirror_stage"
};
const std::vector<std::string> kReplacedHiddenKeys = {
  "__ctx_group__", "__lr_mult__", "__wd_mult__", "__force_mirroring__", "__mirror_stage__"
};
const char *kNamespaceSeparator = "$";


DMLC_JSON_ENABLE_ANY(int, int);

// convert nnvm symbol to a nnvm graph.
nnvm::Graph Symbol2Graph(const nnvm::Symbol &s) {
  nnvm::Graph g;
  g.outputs = s.outputs;
  g.attrs["mxnet_version"] = std::make_shared<nnvm::any>(static_cast<int>(MXNET_VERSION));
  return g;
}

std::vector<uint32_t> ReadOnlyArgIndices(const nnvm::IndexedGraph& idx) {
  std::vector<uint32_t> ret;
  auto& arg_nodes = idx.input_nodes();
  for (uint32_t i = 0; i < arg_nodes.size(); ++i) {
    if (idx.mutable_input_nodes().count(arg_nodes[i]) == 0) {
      ret.push_back(i);
    }
  }
  return ret;
}

}  // namespace mxnet

namespace {

std::unordered_map<std::string, std::string>
_ExtractSymbolKWArgs(mx_uint num_param,
                     const char** keys,
                     const char** vals) {
  std::unordered_map<std::string, std::string> kwargs;
  for (nn_uint i = 0; i < num_param; ++i) {
    bool flag = false;
    for (const auto &k : kHiddenKeys) {
      std::string tmp(keys[i]);
      size_t pos = tmp.rfind(k);
      if (pos == 0) {
        kwargs.insert({"__" + tmp + "__", std::string(vals[i])});
        flag = true;
        break;
      } else if (pos != std::string::npos && pos == tmp.length() - k.length()) {
        std::ostringstream os;
        os << "setting variable attributes with " << keys[i] << " is deprecated. "
           << "please instead use\nw = Variable(" << k << "=" << vals[i] << ")\n"
           << "sym = YourSymbolName(" << tmp.substr(0, pos-1) << "=w)";
        throw dmlc::Error(os.str());
      }
    }
    if (!flag)
      kwargs.insert({std::string(keys[i]), std::string(vals[i])});
  }
  return kwargs;
}

}  // namespace

// symbolic configuration generation API.
// Redirect to NNVM's C API
int MXListAllOpNames(nn_uint *out_size,
                     const char ***out_array) {
  mxnet::op::RegisterLegacyOpProp();
  mxnet::op::RegisterLegacyNDFunc();
  return NNListAllOpNames(out_size, out_array);
}

int MXSymbolListAtomicSymbolCreators(mx_uint *out_size,
                                     AtomicSymbolCreator **out_array) {
  mxnet::op::RegisterLegacyOpProp();
  mxnet::op::RegisterLegacyNDFunc();
  return NNListUniqueOps(out_size, out_array);
}

int MXSymbolGetAtomicSymbolInfo(AtomicSymbolCreator creator,
                                const char **name,
                                const char **description,
                                mx_uint *num_args,
                                const char ***arg_names,
                                const char ***arg_type_infos,
                                const char ***arg_descriptions,
                                const char **key_var_num_args,
                                const char **return_type) {
  static auto& map_key_var_args = nnvm::Op::GetAttr<std::string>("key_var_num_args");
  const Op* op = static_cast<Op*>(creator);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  ret->ret_str.resize(0);

  if (map_key_var_args.count(op) != 0) {
    *key_var_num_args = map_key_var_args[op].c_str();
  } else {
    *key_var_num_args = ret->ret_str.c_str();
  }
  return NNGetOpInfo(
      creator, name, description,
      num_args, arg_names, arg_type_infos,
      arg_descriptions, return_type);
}

int MXSymbolCreateAtomicSymbol(AtomicSymbolCreator creator,
                               mx_uint num_param,
                               const char **keys,
                               const char **vals,
                               SymbolHandle *out) {
  nnvm::Symbol *s = new nnvm::Symbol();
  API_BEGIN();
  const nnvm::Op* op = static_cast<const nnvm::Op*>(creator);
  *s = nnvm::Symbol::CreateFunctor(op, _ExtractSymbolKWArgs(num_param, keys, vals));
  *out = s;
  API_END_HANDLE_ERROR(delete s;);
}

int MXSymbolCreateVariable(const char *name, SymbolHandle *out) {
  return NNSymbolCreateVariable(name, out);
}

int MXSymbolCreateGroup(mx_uint num_symbols,
                        SymbolHandle *symbols,
                        SymbolHandle *out) {
  return NNSymbolCreateGroup(num_symbols, symbols, out);
}

int MXSymbolGetOutput(SymbolHandle symbol,
                      mx_uint index,
                      SymbolHandle *out) {
  return NNSymbolGetOutput(symbol, index, out);
}

int MXSymbolGetInternals(SymbolHandle symbol,
                         SymbolHandle *out) {
  nnvm::Symbol *s = new nnvm::Symbol();
  API_BEGIN();
  *s = static_cast<nnvm::Symbol*>(symbol)->GetInternals();
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolGetChildren(SymbolHandle symbol,
                        SymbolHandle *out) {
  nnvm::Symbol *s = new nnvm::Symbol();
  API_BEGIN();
  *s = static_cast<nnvm::Symbol*>(symbol)->GetChildren();
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolFree(SymbolHandle symbol) {
  return NNSymbolFree(symbol);
}

int MXSymbolCopy(SymbolHandle symbol, SymbolHandle *out) {
  return NNSymbolCopy(symbol, out);
}

int MXSymbolPrint(SymbolHandle symbol, const char **out_str) {
  return NNSymbolPrint(symbol, out_str);
}

int MXSymbolGetName(SymbolHandle symbol,
                    const char** out,
                    int* success) {
  return NNSymbolGetAttr(symbol, "name", out, success);
}

int MXSymbolGetAttr(SymbolHandle symbol,
                    const char* key,
                    const char** out,
                    int* success) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(symbol);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  if (s->GetAttr(key, &(ret->ret_str))) {
    *out = (ret->ret_str).c_str();
    *success = 1;
  } else {
    *out = nullptr;
    *success = 0;
    if (std::find(kHiddenKeys.begin(), kHiddenKeys.end(), key) != kHiddenKeys.end()) {
      std::string skey = "__" + std::string(key) + "__";
      if (s->GetAttr(skey, &(ret->ret_str))) {
        *out = (ret->ret_str).c_str();
        *success = 1;
      }
    }
  }
  API_END();
}

int MXSymbolSetAttr(SymbolHandle symbol,
                    const char* key,
                    const char* value) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(symbol);
  API_BEGIN();
  std::vector<std::pair<std::string, std::string> > kwargs;
  std::string skey(key), sval(value);
  for (const auto &k : kHiddenKeys) {
    size_t pos = skey.rfind(k);
    if (pos == 0 && k.length() == skey.length()) {
      skey = "__" + skey + "__";
      break;
    } else if (pos != std::string::npos && pos + k.length() == skey.length()) {
      std::ostringstream os;
      os << "setting variable attributes with " << key << " is deprecated. "
         << "please instead use\nw = Variable(" << k << "=" << value << ")\n"
         << "sym = YourSymbolName(" << skey.substr(0, pos-1) << "=w)";
      throw dmlc::Error(os.str());
    }
  }
  kwargs.emplace_back(std::make_pair(std::move(skey), std::move(sval)));
  s->SetAttrs(kwargs);
  API_END();
}

int MXSymbolListAttr(SymbolHandle symbol,
                     mx_uint *out_size,
                     const char*** out) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(symbol);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  std::vector<std::tuple<std::string, std::string, std::string> > attr =
      s->ListAttrsRecursive();

  std::vector<std::string>& attr_list = ret->ret_vec_str;
  attr_list.clear();
  for (const auto& tp : attr) {
    attr_list.emplace_back(std::get<0>(tp) + kNamespaceSeparator + std::get<1>(tp));
    attr_list.emplace_back(std::get<2>(tp));
    if (find(kReplacedHiddenKeys.begin(), kReplacedHiddenKeys.end(), std::get<1>(tp))
          != kReplacedHiddenKeys.end()) {
      attr_list.push_back(std::get<0>(tp) + kNamespaceSeparator +
                          std::get<1>(tp).substr(2, std::get<1>(tp).length() - 4));
      attr_list.push_back(std::get<2>(tp));
    }
  }
  *out_size = attr_list.size()/2;
  ret->ret_vec_charp.clear();
  for (size_t i = 0; i < attr_list.size(); ++i) {
    ret->ret_vec_charp.push_back(attr_list[i].c_str());
  }
  *out = dmlc::BeginPtr(ret->ret_vec_charp);
  API_END();
}

int MXSymbolListAttrShallow(SymbolHandle symbol,
                            mx_uint *out_size,
                            const char*** out) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(symbol);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  std::unordered_map<std::string, std::string> attr =
      s->ListAttrs(static_cast<nnvm::Symbol::ListAttrOption>(1));  // NOLINT(*)

  std::vector<std::string>& attr_list = ret->ret_vec_str;
  attr_list.clear();
  for (const auto& kv : attr) {
    attr_list.push_back(kv.first);
    attr_list.push_back(kv.second);
    if (find(kReplacedHiddenKeys.begin(), kReplacedHiddenKeys.end(), kv.first)
          != kReplacedHiddenKeys.end()) {
      attr_list.push_back(kv.first.substr(2, kv.first.length() - 4));
      attr_list.push_back(kv.second);
    }
  }
  *out_size = attr_list.size()/2;
  ret->ret_vec_charp.clear();
  for (size_t i = 0; i < attr_list.size(); ++i) {
    ret->ret_vec_charp.push_back(attr_list[i].c_str());
  }
  *out = dmlc::BeginPtr(ret->ret_vec_charp);
  API_END();
}

int MXSymbolListOutputs(SymbolHandle symbol,
                        mx_uint *out_size,
                        const char ***out_str_array) {
  return NNSymbolListOutputNames(symbol, out_size, out_str_array);
}

int MXSymbolCompose(SymbolHandle sym,
                    const char *name,
                    mx_uint num_args,
                    const char** keys,
                    SymbolHandle* args) {
  return NNSymbolCompose(sym, name, num_args, keys, args);
}

// adapter functions that re-implements the functions.
int MXSymbolListArguments(SymbolHandle symbol,
                          mx_uint *out_size,
                          const char ***out_str_array) {
  return NNSymbolListInputNames(symbol, 1, out_size, out_str_array);
}

int MXSymbolListAuxiliaryStates(SymbolHandle symbol,
                                mx_uint *out_size,
                                const char ***out_str_array) {
  return NNSymbolListInputNames(symbol, 2, out_size, out_str_array);
}

int MXSymbolGetAtomicSymbolName(AtomicSymbolCreator creator,
                                const char **out) {
  API_BEGIN();
  Op *e = static_cast<Op *>(creator);
  *out = e->name.c_str();
  API_END();
}

int MXSymbolCreateFromFile(const char *fname, SymbolHandle *out) {
  nnvm::Symbol *s = new nnvm::Symbol();
  API_BEGIN();
  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname, "r"));
  dmlc::istream is(fi.get());
  nnvm::Graph g;
  g.attrs["json"] = std::make_shared<nnvm::any>(
    std::string(std::istreambuf_iterator<char>(is), std::istreambuf_iterator<char>()));
  s->outputs = nnvm::ApplyPass(g, "LoadLegacyJSON").outputs;
  *out = s;
  is.set_stream(nullptr);
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolCreateFromJSON(const char *json, SymbolHandle *out) {
  nnvm::Symbol *s = new nnvm::Symbol();
  API_BEGIN();
  nnvm::Graph g;
  g.attrs["json"] = std::make_shared<nnvm::any>(std::string(json));
  s->outputs = nnvm::ApplyPass(g, "LoadLegacyJSON").outputs;
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXSymbolSaveToFile(SymbolHandle symbol, const char *fname) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(symbol);
  API_BEGIN();
  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(fname, "w"));
  dmlc::ostream os(fo.get());
  os << nnvm::pass::SaveJSON(Symbol2Graph(*s));
  // reset file pointer, force flush
  os.set_stream(nullptr);
  API_END();
}

int MXSymbolSaveToJSON(SymbolHandle symbol, const char **out_json) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(symbol);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  ret->ret_str = nnvm::pass::SaveJSON(Symbol2Graph(*s));
  *out_json = ret->ret_str.c_str();
  API_END();
}

namespace mxnet {

template<typename AttrType>
void MatchArguments(
    const nnvm::IndexedGraph& idx,
    const std::unordered_map<std::string, AttrType>& known_arg_attrs,
    std::vector<AttrType>* arg_attrs,
    const char* source) {
  auto& arg_nodes = idx.input_nodes();
  CHECK_EQ(arg_attrs->size(), arg_nodes.size());
  size_t nmatched = 0;
  for (size_t i = 0; i < arg_nodes.size(); ++i) {
    const std::string& name = idx[arg_nodes[i]].source->attrs.name;
    auto it = known_arg_attrs.find(name);
    if (it != known_arg_attrs.end()) {
      arg_attrs->at(i) = it->second;
      ++nmatched;
    }
  }
  if (nmatched != known_arg_attrs.size()) {
    std::unordered_set<std::string> keys;
    std::ostringstream head, msg;
    msg << "\nCandidate arguments:\n";
    for (size_t i = 0; i < arg_nodes.size(); ++i) {
      std::string arg_name = idx[arg_nodes[i]].source->attrs.name;
      keys.insert(arg_name);
      msg << "\t[" << i << ']' << arg_name << '\n';
    }
    for (const auto& kv : known_arg_attrs) {
      const std::string& key = kv.first;
      if (keys.count(key) == 0) {
        LOG(FATAL) << source
                   << "Keyword argument name " << key << " not found."
                   << msg.str();
      }
    }
  }
}

}  // namespace mxnet

int MXSymbolInferShape(SymbolHandle sym,
                       mx_uint num_args,
                       const char** keys,
                       const mx_uint *arg_ind_ptr,
                       const mx_uint *arg_shape_data,
                       mx_uint *in_shape_size,
                       const mx_uint **in_shape_ndim,
                       const mx_uint ***in_shape_data,
                       mx_uint *out_shape_size,
                       const mx_uint **out_shape_ndim,
                       const mx_uint ***out_shape_data,
                       mx_uint *aux_shape_size,
                       const mx_uint **aux_shape_ndim,
                       const mx_uint ***aux_shape_data,
                       int *complete) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(sym);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  nnvm::Graph g = Symbol2Graph(*s);
  nnvm::ShapeVector arg_shapes(g.indexed_graph().input_nodes().size(), TShape());
  if (keys == nullptr && num_args != 0) {
    std::vector<uint32_t> read_only_args = mxnet::ReadOnlyArgIndices(g.indexed_graph());
    CHECK_LE(num_args, read_only_args.size());
    for (mx_uint i = 0; i < num_args; ++i) {
      arg_shapes[read_only_args[i]] = nnvm::ShapeTypeCast(
          arg_shape_data + arg_ind_ptr[i], arg_shape_data + arg_ind_ptr[i+1]);
    }
  } else {
    std::unordered_map<std::string, TShape> kwargs;
    for (mx_uint i = 0; i < num_args; ++i) {
      kwargs[keys[i]] = nnvm::ShapeTypeCast(
          arg_shape_data + arg_ind_ptr[i], arg_shape_data + arg_ind_ptr[i+1]);
    }
    mxnet::MatchArguments(g.indexed_graph(), kwargs, &arg_shapes, "InferShape");
  }

  try {
    g = mxnet::exec::InferShape(std::move(g), std::move(arg_shapes), "__shape__");
  } catch (const mxnet::op::InferShapeError &err) {
    throw dmlc::Error(err.msg);
  }

  // copy back
  CopyAttr(g.indexed_graph(), g.GetAttr<nnvm::ShapeVector>("shape"),
           &(ret->arg_shapes), &(ret->out_shapes), &(ret->aux_shapes));

  // copy data back
  MXAPIThreadLocalEntry::SetupShapeArrayReturnWithBuffer(ret->arg_shapes,
      &(ret->arg_shape_ndim), &(ret->arg_shape_data), &(ret->arg_shape_buffer));
  MXAPIThreadLocalEntry::SetupShapeArrayReturnWithBuffer(ret->out_shapes,
      &(ret->out_shape_ndim), &(ret->out_shape_data), &(ret->out_shape_buffer));
  MXAPIThreadLocalEntry::SetupShapeArrayReturnWithBuffer(ret->aux_shapes,
      &(ret->aux_shape_ndim), &(ret->aux_shape_data), &(ret->aux_shape_buffer));
  *in_shape_size = static_cast<mx_uint>(ret->arg_shapes.size());
  *in_shape_ndim = dmlc::BeginPtr(ret->arg_shape_ndim);
  *in_shape_data = dmlc::BeginPtr(ret->arg_shape_data);
  *out_shape_size = static_cast<mx_uint>(ret->out_shapes.size());
  *out_shape_ndim = dmlc::BeginPtr(ret->out_shape_ndim);
  *out_shape_data = dmlc::BeginPtr(ret->out_shape_data);
  *aux_shape_size = static_cast<mx_uint>(ret->aux_shapes.size());
  *aux_shape_ndim = dmlc::BeginPtr(ret->aux_shape_ndim);
  *aux_shape_data = dmlc::BeginPtr(ret->aux_shape_data);
  // mark complete
  *complete = (g.GetAttr<size_t>("shape_num_unknown_nodes") == 0);
  API_END();
}

int MXSymbolInferShapePartial(SymbolHandle sym,
                              mx_uint num_args,
                              const char** keys,
                              const mx_uint *arg_ind_ptr,
                              const mx_uint *arg_shape_data,
                              mx_uint *in_shape_size,
                              const mx_uint **in_shape_ndim,
                              const mx_uint ***in_shape_data,
                              mx_uint *out_shape_size,
                              const mx_uint **out_shape_ndim,
                              const mx_uint ***out_shape_data,
                              mx_uint *aux_shape_size,
                              const mx_uint **aux_shape_ndim,
                              const mx_uint ***aux_shape_data,
                              int *complete) {
  int succ;
  *complete = 1;
  return MXSymbolInferShape(sym, num_args, keys,
                            arg_ind_ptr, arg_shape_data,
                            in_shape_size, in_shape_ndim, in_shape_data,
                            out_shape_size, out_shape_ndim, out_shape_data,
                            aux_shape_size, aux_shape_ndim, aux_shape_data,
                            &succ);
}

int MXSymbolInferType(SymbolHandle sym,
                      mx_uint num_args,
                      const char** keys,
                      const int *arg_type_data,
                      mx_uint *in_type_size,
                      const int **in_type_data,
                      mx_uint *out_type_size,
                      const int **out_type_data,
                      mx_uint *aux_type_size,
                      const int **aux_type_data,
                      int *complete) {
  nnvm::Symbol *s = static_cast<nnvm::Symbol*>(sym);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  nnvm::Graph g = Symbol2Graph(*s);
  nnvm::DTypeVector arg_types(g.indexed_graph().input_nodes().size(), -1);
  if (keys == nullptr && num_args != 0) {
    std::vector<uint32_t> read_only_args = mxnet::ReadOnlyArgIndices(g.indexed_graph());
    CHECK_LE(num_args, read_only_args.size());
    for (mx_uint i = 0; i < num_args; ++i) {
      arg_types[read_only_args[i]] = arg_type_data[i];
    }
  } else {
    std::unordered_map<std::string, int> kwargs;
    for (mx_uint i = 0; i < num_args; ++i) {
      kwargs[keys[i]] = arg_type_data[i];
    }
    mxnet::MatchArguments(g.indexed_graph(), kwargs, &arg_types, "InferType");
  }

  g = mxnet::exec::InferType(std::move(g), std::move(arg_types), "__dtype__");
  // copy back
  CopyAttr(g.indexed_graph(), g.GetAttr<nnvm::DTypeVector>("dtype"),
           &(ret->arg_types), &(ret->out_types), &(ret->aux_types));

  *in_type_size = static_cast<mx_uint>(ret->arg_types.size());
  *in_type_data = dmlc::BeginPtr(ret->arg_types);
  *out_type_size = static_cast<mx_uint>(ret->out_types.size());
  *out_type_data = dmlc::BeginPtr(ret->out_types);
  *aux_type_size = static_cast<mx_uint>(ret->aux_types.size());
  *aux_type_data = dmlc::BeginPtr(ret->aux_types);
  *complete = (g.GetAttr<size_t>("dtype_num_unknown_nodes") == 0);
  API_END();
}

int MXSymbolGrad(SymbolHandle sym, mx_uint num_wrt, const char** wrt, SymbolHandle* out) {
  API_BEGIN();
  LOG(FATAL) << "not implemented";
  API_END();
}

/////////////// Subgraph APIs
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

int MXGraphEval(GraphHandle graph,
                int num_inputs,
                NDArrayHandle *inputs,
                int *num_outputs,
                NDArrayHandle **outputs,
                int is_training) {
  using nnvm::GraphPtr;
  using nnvm::any;
  using exec::GraphExecutorV2;
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();

  API_BEGIN();
  GraphPtr pg = *static_cast<GraphPtr*>(graph);
  const auto& idx = pg->indexed_graph();
  std::unordered_map<std::string, std::shared_ptr<any>> kwargs_any;
  // TODO(minjie): hard-code for testing.
  std::vector<Context> ctx = {Context::GPU(0)};
  kwargs_any["context"] = std::make_shared<any>(std::move(ctx));
  nnvm::Specialize(pg.get(), kwargs_any);

  // TODO(minjie): how to expose to python?
  GraphExecutorV2::Config cfg;
  cfg.zero_copy = true;
  GraphExecutorV2 executor(*pg, cfg);

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
  }
  // TODO(minjie): how to expose to python?
  GraphExecutorV2::RunOption opt;
  opt.is_train = (is_training == 1);
  executor.Run(arguments, &results, opt);
  if (rsts_ptr == nullptr) {
    *num_outputs = results.size();
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
/////////////// Subgraph APIs
