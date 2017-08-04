/*!
 *  Copyright (c) 2016 by Contributors
 * \file infer_shape.cc
 * \brief Inference the shapes given existin information.
 */
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>

using namespace std;
using namespace nnvm;

namespace mxnet {
namespace {

/* Pseudo-code
 *
 * void InferShapePass(Graph graph, vector<TShape> hints) {  // hints also include input shapes
 *   SaveShapes(hints);
 *   for (uint32_t nid = 0; nid < num_nodes; ++nid) {
 *     if (IsSubgraph(nid)) {
 *       Graph subgraph = GetSubgraph(nid);
 *       vector<TShape> sub_hints = GetSubhints(nid);
 *       InferShapePass(subgraph, sub_hints);  // no matter forward or backward.
 *       SaveInputShapes(nid, GetGraphInputShapes(subgraph));
 *       SaveOutputShapes(nid, GetGraphOutputShapes(subgraph));
 *     } else {
 *       FInferShape finfer = GetFInferShape(nid);  // no matter forward or backward.
 *       vector<TShape> ishapes = GetInputShapes(nid);
 *       vector<TShape> oshapes = GetOutputShapes(nid);
 *       finfer(&ishapes, &oshapes);
 *       SaveInputShapes(nid, ishapes);
 *       SaveOutputShapes(nid, oshapes);
 *     }
 *   }
 * }
 */
template<typename AttrType>
class InferAttrPass {
 public:
  typedef vector<AttrType> AttrVector;
  InferAttrPass(const AttrType& empty_val,
                const string& infer_name,
                const string& attr_name)
    : empty_val_(empty_val), infer_name_(infer_name), attr_name_(attr_name)
  { }

  void Infer(const Graph* graph,
             Column<AttrType>* attr_col,
             const Column<AttrType>* fwd_attr_col = nullptr) {
    // Do infer.
    const IndexedGraph& idx = graph->indexed_graph();

    if (fwd_attr_col != nullptr) {
      CHECK(graph->global_attrs.count("gradient_entry_mapping"));
      const auto& entry_mapping = graph->GetGlobalAttr<vector<uint32_t>>("gradient_entry_mapping");
      CHECK_EQ(entry_mapping.size(), idx.num_node_entries());
      cout << "entry_mapping=[";
      for (const auto& m : entry_mapping) {
        cout << m << " ";
      }
      cout << "]" << endl;
      for (uint32_t eid = 0; eid < entry_mapping.size(); ++eid) {
        const uint32_t fwd_eid = entry_mapping[eid];
        if (fwd_eid < fwd_attr_col->value.size()) {
          LOG(INFO) << "Use fwd_eid=" << fwd_eid << " for bwd_eid=" << eid;
          attr_col->value[eid] = fwd_attr_col->value[fwd_eid];
        }
      }
    }

    // TODO(minjie): Only one pass right now. Need multiple passes.
    for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
      LOG(INFO) << "Infer node#" << nid << idx[nid].source->attrs.name;
      InferOneNode(graph, nid, attr_col, fwd_attr_col);
    }
  }

 private:
  bool AllKnown(const ColumnRef<AttrType>& ref) {
    for (const AttrType& a : ref->value) {
      if (a == empty_val_) {
        return false;
      }
    }
    return true;
  }

  // Check whether the attribute exists for the subgraph and if exists,
  // whether they are the same. If so, simply use the attributes of output entries.
  bool TryReuse(const IndexedGraph& idx,
                uint32_t nid,
                Column<AttrType>* attr) {
    const Node* node = idx[nid].source;
    auto sg = node->graph();
    if (sg->entry_attrs.count(attr_name_)) {
      attr->children[nid] = sg->entry_attrs.GetColumn<AttrType>(attr_name_);
      // Check whether all input attributes are equal.
      for (size_t i = 0; i < node->inputs.size(); ++i) {
        const uint32_t ineid = idx.entry_id(node->inputs[i]);
        if (attr->value[ineid] != empty_val_ &&
            attr->value[ineid] != attr->children[nid]->value[i]) {
          return false;
        }
      }
      // Check whether all output attributes are equal.
      const uint32_t outent_st = idx.entry_id(nid, 0);
      const uint32_t outent_ed = outent_st + node->num_outputs();
      return std::equal(attr->value.begin() + outent_st,
                        attr->value.begin() + outent_ed,
                        attr->children[nid]->value.end() - node->num_outputs(),
                        [this] (const AttrType& a1, const AttrType& a2) {
                          return a1 == empty_val_ || a1 == a2;
                        });
    }
    attr->children[nid] = sg->CreateEntryColumn(empty_val_);
    return false;
  }

  void InferGraphNode(const Graph* graph,
                      uint32_t nid,
                      Column<AttrType>* attr,
                      const Column<AttrType>* fwd_attr_col) {
    const IndexedGraph& idx = graph->indexed_graph();
    const Node* node = idx[nid].source;
    auto sg = node->graph();
    const uint32_t outent_st = idx.entry_id(nid, 0);
    const uint32_t outent_ed = outent_st + node->num_outputs();
    if (!TryReuse(idx, nid, attr)) {
      Column<AttrType>* subattr = attr->children[nid].CopyOnWrite();
      // Reset the column.
      for (size_t i = 0; i < subattr->value.size(); ++i) {
        subattr->value[i] = empty_val_;
      }
      // Copy input/output shapes.
      for (size_t i = 0; i < node->inputs.size(); ++i) {
        subattr->value[i] = attr->value[idx.entry_id(node->inputs[i])];
      }
      std::copy(attr->value.begin() + outent_st,
                attr->value.begin() + outent_ed,
                subattr->value.end() - node->num_outputs());
      // TODO: handle backward graph.
      const Column<AttrType>* sub_fwd_attr_col = nullptr;
      if (sg->global_attrs.count("gradient_entry_mapping")) { // A backward graph node.
        if (fwd_attr_col != nullptr) {
          // Foward node is in another graph.
          LOG(FATAL) << "Not implemented.";
          const auto& node_mapping =
            graph->GetGlobalAttr<vector<uint32_t>>("gradient_node_mapping");
          const uint32_t fwd_nid = node_mapping[nid];
          CHECK_LT(fwd_nid, fwd_attr_col->children.size());
          sub_fwd_attr_col = fwd_attr_col->children[fwd_nid].get();
        } else {
          // Foward node is connected by the first control dependency.
          CHECK_GE(node->control_deps.size(), 1U)
            << "Backward node need to have control_deps to its forward node";
          const NodePtr& fwd_ptr = node->control_deps[0];
          CHECK(fwd_ptr->is_graph());
          const uint32_t fwd_nid = idx.node_id(fwd_ptr.get());
          sub_fwd_attr_col = attr->children[fwd_nid].get();
        }
      }
      LOG(INFO) << ">>>>Infer subgraph node: " << node->attrs.name;
      this->Infer(sg.get(), subattr, sub_fwd_attr_col);
    }
    // Fetch input/output attribute.
    for (size_t i = 0; i < node->inputs.size(); ++i) {
      attr->value[idx.entry_id(node->inputs[i])] = attr->children[nid]->value[i];
    }
    std::copy(attr->children[nid]->value.end() - node->num_outputs(),
              attr->children[nid]->value.end(),
              attr->value.begin() + outent_st);
  }

  void InferOpNode(const Graph* graph,
                   uint32_t nid,
                   Column<AttrType>* attr) {
    const IndexedGraph& idx = graph->indexed_graph();
    static auto& is_backward = Op::GetAttr<TIsBackward>("TIsBackward");
    const auto& inode = idx[nid];
    if (is_backward.get(inode.source->op(), false) && inode.control_deps.size()) {
      InferLegacyBackwardNode(graph, nid, attr);
    } else {
      InferNormalOpNode(graph, nid, attr);
    }
  }

  void InferLegacyBackwardNode(const Graph* graph,
                               uint32_t nid,
                               Column<AttrType>* attr) {
    const IndexedGraph& idx = graph->indexed_graph();
    const auto& inode = idx[nid];

    // Backward node.
    CHECK_GE(inode.control_deps.size(), 1U)
      << "Backward node need to have control_deps to its forward node";
    const NodePtr& fwd_ptr = inode.source->control_deps[0];
    const uint32_t fwd_nid = idx.node_id(fwd_ptr.get());
    CHECK(fwd_ptr->op() != nullptr) << "Forward op cannot be a variable";
    /*LOG(INFO) << "\tforward node: " << fwd_ptr->attrs.name;
    LOG(INFO) << "\tnum backward outputs: " << inode.source->num_outputs();
    for (size_t i = 0; i < fwd_ptr->inputs.size(); ++i) {
      const uint32_t fwd_eid = idx.entry_id(fwd_ptr->inputs[i]);
      const uint32_t bwd_eid = idx.entry_id(nid, i);
      LOG(INFO) << "\tFwd ent#" << fwd_eid << ": " << attr->value[fwd_eid]
        << " Bwd ent#" << bwd_eid << ": " << attr->value[bwd_eid];
    }*/
    
    // Attributes of the grad_in entries are equal to what of the input entries
    // of the forward node.
    CHECK_LE(inode.source->num_outputs(), fwd_ptr->inputs.size());
    for (size_t i = 0; i < inode.source->num_outputs(); ++i) {
      const uint32_t fwd_eid = idx.entry_id(fwd_ptr->inputs[i]);
      const uint32_t bwd_eid = idx.entry_id(nid, i);
      if (attr->value[bwd_eid] == empty_val_) {
        attr->value[bwd_eid] = attr->value[fwd_eid];
      } else {
        CHECK_EQ(attr->value[bwd_eid], attr->value[fwd_eid])
            << "Backward " << attr_name_ << " is inconsistent with the forward "
            << attr_name_;
      }
    }
    // TODO(minjie): handle grad_out entries.
    // Attributes of the grad_out entries are equal to what of the output
    // entries of the forward node.
    /*for (size_t i = 0; i < inode.inputs.size(); ++i) {
      const uint32_t fwd_eid = idx.entry_id(fwd_nid, i);
      const uint32_t bwd_eid = idx.entry_id(inode.inputs[i]);
      if (attr->value[bwd_eid] == empty_val_) {
        attr->value[bwd_eid] = attr->value[fwd_eid];
      } else {
        CHECK_EQ(attr->value[bwd_eid], attr->value[fwd_eid])
            << "Backward " << attr_name_ << " is inconsistent with the forward "
            << attr_name_;
      }
    }*/
  }

  void InferNormalOpNode(const Graph* graph,
                         uint32_t nid,
                         Column<AttrType>* attr) {
    static auto& finfer_shape = Op::GetAttr<FInferNodeEntryAttr<AttrType> >(infer_name_);

    const IndexedGraph& idx = graph->indexed_graph();
    const auto& inode = idx[nid];
    const uint32_t num_inputs = inode.inputs.size();
    const uint32_t num_outputs = inode.source->num_outputs();
    // Temp space for shape inference.
    vector<AttrType> iattr(num_inputs), oattr(num_outputs);
    for (uint32_t i = 0; i < num_inputs; ++i) {
      const uint32_t in_ent_id = idx.entry_id(inode.inputs[i]);
      iattr[i] = attr->value[in_ent_id];
    }
    for (uint32_t i = 0; i < num_outputs; ++i) {
      const uint32_t out_ent_id = idx.entry_id(nid, i);
      oattr[i] = attr->value[out_ent_id];
    }

    auto finfer = finfer_shape.get(inode.source->op(), nullptr);
    CHECK(finfer != nullptr)
      << "Attribute " << infer_name_
      << " is not registed by op " << inode.source->op()->name
      << " we are not able to complete the inference because of this";
    // Call inference function of the operator.
    try {
      finfer(inode.source->attrs, &iattr, &oattr);
    } catch (const exception& e) {
      throw dmlc::Error("Infer \"" + attr_name_ + "\" error for operator "
          + inode.source->attrs.name + ": " + e.what());
    }

    // Save to the result map.
    for (uint32_t i = 0; i < num_inputs; ++i) {
      attr->value[idx.entry_id(inode.inputs[i])] = iattr[i];
      LOG(INFO) << "\tFwd ent#" << idx.entry_id(inode.inputs[i]) << ": " << iattr[i];
    }
    for (uint32_t i = 0; i < num_outputs; ++i) {
      attr->value[idx.entry_id(nid, i)] = oattr[i];
    }
  }

  void InferOneNode(const Graph* graph,
                    uint32_t nid,
                    Column<AttrType>* attr,
                    const Column<AttrType>* fwd_attr_col) {

    const IndexedGraph& idx = graph->indexed_graph();
    const auto& inode = idx[nid];

    // Short cut if all input/output attributes has known.
    bool all_known = true;
    for (uint32_t i = 0; i < inode.inputs.size(); ++i) {
      const uint32_t in_ent_id = idx.entry_id(inode.inputs[i]);
      all_known &= (attr->value[in_ent_id] != empty_val_);
    }
    for (uint32_t i = 0; i < inode.source->num_outputs(); ++i) {
      const uint32_t out_ent_id = idx.entry_id(nid, i);
      all_known &= (attr->value[out_ent_id] != empty_val_);
    }
    if (all_known) {
      // All is known, no need to do infer.
      return;
    }

    if (inode.source->is_graph()) {
      InferGraphNode(graph, nid, attr, fwd_attr_col);
    } else if (!inode.source->is_variable()) {
      InferOpNode(graph, nid, attr); 
    }
  }

 private:
  const AttrType empty_val_;
  const string infer_name_;
  const string attr_name_;
};

template<typename AttrType>
Graph InferAttrHelper(Graph &&graph,
                      const AttrType& empty_val,
                      const string& infer_name,
                      const string& input_name,
                      const string& attr_key_name,
                      const string& attr_name) {
  // Preprocess all kinds of hints into the attribute column.
  using AttrVector = vector<AttrType>;
  const IndexedGraph& idx = graph.indexed_graph();
  // TODO(minjie): short cut for the same shape inputs.
  if (graph.entry_attrs.count(attr_name) == 0) {
    ColumnRef<AttrType> ref = graph.CreateEntryColumn<AttrType>(empty_val);
    graph.entry_attrs.SetColumn(attr_name, ref);
  }
  Column<AttrType>* attr_col = graph.entry_attrs.GetColumn<AttrType>(attr_name).CopyOnWrite();

  // Get input shapes.
  const AttrVector& shape_args = graph.GetGlobalAttr<AttrVector>(input_name);
  CHECK_LE(shape_args.size(), idx.input_nodes().size())
      << "More provided shapes than number of arguments.";
  for (size_t i = 0; i < shape_args.size(); ++i) {
    attr_col->value[idx.entry_id(idx.input_nodes()[i], 0)] = shape_args[i];
  }

  // Get variable shapes.
  if (graph.global_attrs.count(attr_key_name) != 0) {
    const string& attr_key = graph.GetGlobalAttr<string>(attr_key_name);
    CHECK(attr_key.length() != 0)
      << "Invalid attribute key for \"" << attr_key_name << "\"";
    // Save all variable shapes to the column.
    for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
      const auto& inode = idx[nid];
      if (!inode.source->is_variable()) {
        continue;
      }
      CHECK_EQ(inode.source->num_outputs(), 1U);
      const uint32_t out_ent_id = idx.entry_id(nid, 0);
      if (attr_col->value[out_ent_id] != empty_val
          && inode.source->attrs.dict.count(attr_key) != 0) {
        const string& attr_val = inode.source->attrs.dict.at(attr_key);
        istringstream is(attr_val);
        CHECK(is >> attr_col->value[out_ent_id])
          << "Invalid attribute value \"" << attr_val << "\" for key \""
          << attr_key << "\".";
      }
    }
    // Erase the provided arguments.
    graph.global_attrs.erase(attr_key_name);
  }

  InferAttrPass<AttrType> pass(empty_val, infer_name, attr_name);
  pass.Infer(&graph, attr_col);
  return graph;
}

Graph MXInferShape(Graph &&graph) {
  static const TShape empty_val = TShape();
  static const string infer_name = "FInferShape";
  static const string input_name = "shape_inputs";
  static const string attr_key_name = "shape_attr_key";
  static const string attr_name = "shape";
  return InferAttrHelper<TShape>(std::move(graph),
                                 empty_val,
                                 infer_name,
                                 input_name,
                                 attr_key_name,
                                 attr_name);
}
NNVM_REGISTER_PASS(MXInferShape)
.describe("Infer the shape of each node entries.")
.set_body(MXInferShape)
.set_change_graph(false)
.depend_graph_attr("shape_inputs")
.provide_entry_attr("shape");

struct MXInferShapeArgs {
  vector<TShape> shape_inputs;
  void Load(dmlc::JSONReader *reader) {
    dmlc::JSONObjectReadHelper helper;
    vector<vector<int>> raw_shapes;
    helper.DeclareOptionalField("shape_inputs", &raw_shapes);
    helper.ReadAllFields(reader);
    for (const auto& rs : raw_shapes) {
      shape_inputs.emplace_back(rs.begin(), rs.end());
    }
  }
};

Graph MXInferShapeAPI(Graph &&graph) {
  static const TShape empty_val = TShape();
  static const string infer_name = "FInferShape";
  static const string input_name = "shape_inputs";
  static const string attr_key_name = "shape_attr_key";
  static const string attr_name = "shape";
  const string& json_args = graph.GetGlobalAttr<string>("mx_infer_shape_args");
  MXInferShapeArgs args;
  istringstream is(json_args);
  dmlc::JSONReader reader(&is);
  reader.Read(&args);
  const auto& idx = graph.indexed_graph();
  CHECK_LE(args.shape_inputs.size(), idx.input_nodes().size())
    << "Pass argument error: more input shapes are provided than required.";
  args.shape_inputs.resize(idx.input_nodes().size());
  graph.global_attrs[input_name] = std::make_shared<any>(std::move(args.shape_inputs));
  auto&& ret = InferAttrHelper<TShape>(std::move(graph),
                                       empty_val,
                                       infer_name,
                                       input_name,
                                       attr_key_name,
                                       attr_name);
  ret.global_attrs.erase(input_name);
  return ret;
}
NNVM_REGISTER_PASS(MXInferShapeAPI)
.describe("Infer the shape of each node entries.")
.set_body(MXInferShapeAPI)
.set_change_graph(false)
.depend_graph_attr("mx_infer_shape_args")
.provide_entry_attr("shape");

DMLC_JSON_ENABLE_ANY(ColumnRef<TShape>, column_shape);

DMLC_JSON_ENABLE_ANY(ShapeVector, list_shape);
DMLC_JSON_ENABLE_ANY(DTypeVector, list_int);
DMLC_JSON_ENABLE_ANY(size_t, size_t);


}  // namespace
}  // namespace mxnet
