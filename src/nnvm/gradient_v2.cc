/*!
 *  Copyright (c) 2016 by Contributors
 * \file gradients.cc
 * \brief Passes that takes gradient of the graph
 * This code was modified based on mxnet codebase by Min Lin.
 */
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <algorithm>
#include <functional>
#include <unordered_set>
#include <queue>

using namespace nnvm;
using namespace std;

namespace mxnet {
namespace {

nnvm::NodeEntry AggregateGradient(std::vector<nnvm::NodeEntry>&& v) {
  using nnvm::Op;
  static size_t inplace_sum_cap = dmlc::GetEnv("MXNET_EXEC_INPLACE_GRAD_SUM_CAP", 8);
  static const Op* ewise_plus_op = Op::Get("_grad_add");
  static const Op* ewise_sum_op = Op::Get("ElementWiseSum");
  static const Op* identity_op = Op::Get("identity");
  static const Op* zeros_op = Op::Get("_zeros");
  static const Op* zeros_like_op = Op::Get("zeros_like");

  if (v.size() == 0) {
    nnvm::NodePtr ng = nnvm::Node::Create();
    ng->attrs.op = zeros_op;
    ng->attrs.name = "zeros";
    ng->attrs.op->attr_parser(&(ng->attrs));
    return nnvm::NodeEntry{ng, 0, 0};
  }

  // remove zero in the sum. at least keep 1.
  size_t begin = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    if (v[i].node->op() != zeros_op && v[i].node->op() != zeros_like_op) {
      if (begin != i) {
        v[begin] = std::move(v[i]);
      }
      ++begin;
    }
  }
  if (begin == 0) begin = 1;
  v.resize(begin);

  if (v.size() == 1) {
    return std::move(v[0]);
  } else {
    if (v.size() < inplace_sum_cap) {
      nnvm::NodePtr sum_node = nnvm::Node::Create();
      sum_node->attrs.op = ewise_sum_op;
      sum_node->attrs.name = "sum_grad";
      sum_node->attrs.dict["num_args"] = std::to_string(v.size());
      sum_node->attrs.op->attr_parser(&(sum_node->attrs));
      sum_node->inputs = std::move(v);
      return nnvm::NodeEntry{sum_node, 0, 0};
    } else {
      // use a stream line of plus instead
      nnvm::NodeEntry ret = v[0];
      for (size_t i = 1; i < v.size(); ++i) {
        // Add control flow dependency from to previous node
        // This enforces the gradient sum order will be in the inverse
        // order of forward traversal
        // NOTE: adding control dependency can be dangerous and cause cycle in the dep.
        // The curent usage is correct, because of the following invariant:
        // assert: v[i-1] do not depend on v[i]
        // To put in plain text: v is gradient vector that get pushed in the order
        // that can generate them, which means if v[i] is not yet pushed,
        // all previous gradient cannot depend on it.
        v[i].node->control_deps.push_back(ret.node);

        std::ostringstream os;
        os << "sum_grad_" << i;
        nnvm::NodePtr x = nnvm::Node::Create();
        x->attrs.op = ewise_plus_op;
        x->attrs.name = os.str();
        x->inputs = {ret, v[i]};
        ret = nnvm::NodeEntry{x, 0, 0};
      }
      // identity node is used to avoid exposure of dummy plus node
      // when its output get assigned to another space.
      nnvm::NodePtr id_node = nnvm::Node::Create();
      id_node->attrs.op = identity_op;
      id_node->attrs.name = "sum_grad_final";
      id_node->inputs = {ret};
      return nnvm::NodeEntry{id_node, 0, 0};
    }
  }
}

static const vector<const Op*> kZeroOps = {
  nnvm::Op::Get("zeros_like"),
  nnvm::Op::Get("_zeros")
};

bool CheckGradAllZero(const vector<NodeEntry>& grads) {
  if (!grads.size() || !kZeroOps.size()) return false;
  for (const auto& g : grads) {
    bool found = false;
    for (const auto& op : kZeroOps) {
      if (g.node->op() == op) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

inline vector<NodeEntry> CreateVariableEntries(const std::string& name_prefix, size_t size) {
  vector<NodeEntry> ret;
  for (size_t i = 0; i < size; ++i) {
    NodePtr var_node = Node::Create();
    var_node->attrs.name = name_prefix + std::to_string(i);
    ret.push_back(NodeEntry{var_node, 0, 0});
  }
  return ret;
}

vector<NodeEntry> GenerateAllZeroGradients(NodePtr node) {
  vector<NodeEntry> ret;
  for (size_t i = 0; i < node->num_inputs(); ++i) {
    ostringstream os;
    if (1 == node->num_inputs()) {
      os << node->attrs.name << "_backward";
    } else {
      os << node->attrs.name << "_in" << i << "_backward";
    }
    auto p = Node::Create();
    p->attrs.op = kZeroOps[0];
    p->attrs.name = os.str();
    p->inputs.push_back(node->inputs[i]);
    p->control_deps.emplace_back(node);
    if (p->op()->attr_parser != nullptr) {
      p->op()->attr_parser(&(p->attrs));
    }
    ret.emplace_back(nnvm::NodeEntry{p, 0, 0});
  }
  return ret;
}

using AggFun = function<NodeEntry (vector<NodeEntry>&& inputs)>;

// helper entry
class GradEntry {
 public:
  GradEntry(): sum_({nullptr, 0, 0}) {}
  void AddGrad(const NodeEntry& ent) {
    grads_.push_back(ent);
  }
  NodeEntry GetSum() {
    if (sum_.node == nullptr) {
      if (grads_.size() == 1) {
        sum_ = std::move(grads_[0]);
      } else {
        sum_ = AggregateGradient(std::move(grads_));
      }
    }
    return sum_;
  }
  bool IsNone() const {
    return sum_.node == nullptr && grads_.empty();
  }

 private:
  NodeEntry sum_;
  vector<NodeEntry> grads_;
};

inline vector<NodeEntry> FetchSumGradients(vector<GradEntry>& ent) {
  vector<NodeEntry> ret;
  ret.reserve(ent.size());
  for (auto& e : ent) {
    ret.push_back(e.GetSum());
  }
  return ret;
}

void SplitForwardBackward(
    const vector<NodeEntry>& in_grads,
    const IndexedGraph& fwd_graph_idx,
    NodeEntryMap<NodeEntry>* fwdent2bwdent) {
  unordered_set<const Node*> visited;
  queue<Node*> search;
  for (const NodeEntry& e : in_grads) {
    search.push(e.node.get());
  }
  while (!search.empty()) {
    Node* node = search.front();
    search.pop();
    if (fwd_graph_idx.exist(node)) {
      // Nothing to be done for forward node.
      continue;
    }
    if (visited.count(node)) {
      continue;
    }
    visited.insert(node);
    // Split data dependencies.
    for (size_t i = 0; i < node->inputs.size(); ++i) {
      const NodeEntry& in_ent = node->inputs[i];
      const NodePtr& in_node = in_ent.node;
      if (fwd_graph_idx.exist(in_node.get())) {
        // The entry is between two forward and backward nodes.
        // Replace the entry with a variable input.
        if (!fwdent2bwdent->count(in_ent)) {
          ostringstream oss;
          if (in_node->is_variable()) {
            oss << in_node->attrs.name;
          } else {
            oss << in_node->attrs.name << "$" << "output" << in_ent.index;
          }
          NodePtr new_in_node = Node::Create();
          new_in_node->attrs.op = nullptr;
          new_in_node->attrs.name = oss.str();
          fwdent2bwdent->insert({in_ent,  NodeEntry{new_in_node, 0, 0}});
        }
        node->inputs[i] = fwdent2bwdent->at(in_ent);
      } else {
        // Traverse the input.
        search.push(in_node.get());
      }
    }
    // XXX(minjie): Remove control dependencies to forward graph? This should be
    // fine since the backward subgraph will also have a control dependency to
    // the forward subgraph.
    auto& cdeps = node->control_deps;
    cdeps.erase(std::remove_if(cdeps.begin(),
                               cdeps.end(), 
                               [&] (const NodePtr& prev) {
                                 return fwd_graph_idx.exist(prev.get());
                               }),
                cdeps.end());
  }
}

struct GradNodeInEntry {
  enum Type {
    kFromGradOut = 0,
    kFromFwdOut,
    kFromFwdIn,
  };
  // Whether the input should be fetched from out_grads. If false, the input entry
  // should be fetched from forward outputs.
  int type = -1;
  // The index of the entry to fetch from. If `from_out_grads` is true, the input
  // should be fetched from out_grads[index]; otherwise, it should be fetched from
  // forward_outputs[index].
  size_t index = 0;
  
  inline static GradNodeInEntry CreateFromOutGrads(size_t idx) {
    GradNodeInEntry ret;
    ret.type = kFromGradOut;
    ret.index = idx;
    return ret;
  }

  inline static GradNodeInEntry CreateFromForwardOut(size_t idx) {
    GradNodeInEntry ret;
    ret.type = kFromFwdOut;
    ret.index = idx;
    return ret;
  }

  inline static GradNodeInEntry CreateFromForwardIn(size_t idx) {
    GradNodeInEntry ret;
    ret.type = kFromFwdIn;
    ret.index = idx;
    return ret;
  }
};

struct GradNodeOutEntry {
  enum Type {
    kFromZero = 0,
    kFromOut,
  };
  int type = -1;
  size_t index = 0;

  inline static GradNodeOutEntry CreateFromBackwardOut(size_t idx) {
    GradNodeOutEntry ret;
    ret.type = kFromOut;
    ret.index = idx;
    return ret;
  }

  inline static GradNodeOutEntry CreateFromZero() {
    GradNodeOutEntry ret;
    ret.type = kFromZero;
    return ret;
  }
};

}  // namespace

struct MXEntryArg {
  uint32_t node = 0;
  uint32_t index = 0;
  uint32_t version = 0;
  void Load(dmlc::JSONReader *reader) {
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("node", &node);
    helper.DeclareOptionalField("index", &index);
    helper.DeclareOptionalField("version", &version);
    helper.ReadAllFields(reader);
  }
};

struct MXGradientArgs {
  // The targets to be differentiated. If none is given,
  // all the input entries will be included.
  // Note: Only one of "xs" and "xs_blacklist" should be provided.
  // If none is provided, blacklist rule is applied.
  vector<MXEntryArg> xs;
  // Alternative way to specify the gradable targets. The list
  // specifies which input entries do NOT need to be differentiated.
  // Note: Only one of "xs" and "xs_blacklist" should be provided.
  // If none is provided, blacklist rule is applied.
  vector<MXEntryArg> xs_blacklist;

  // The objective entries to compute gradients from. If none is
  // given, all the output entries will be included.
  // Note: Only one of "ys" and "ys_blacklist" should be provided.
  // If none is provided, blacklist rule is applied.
  vector<MXEntryArg> ys;
  // Alternative way to specify the objective entries. The list
  // specifies which objective entries are NOT included in gradient
  // computation.
  // Note: Only one of "ys" and "ys_blacklist" should be provided.
  // If none is provided, blacklist rule is applied.
  vector<MXEntryArg> ys_blacklist;

  // Whether generate full graph instead of seperating the gradient
  // as a subgraph.
  bool full_graph = false;

  void Load(dmlc::JSONReader *reader) {
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareOptionalField("xs", &xs);
    helper.DeclareOptionalField("xs_blacklist", &xs_blacklist);
    helper.DeclareOptionalField("ys", &ys);
    helper.DeclareOptionalField("ys_blacklist", &ys_blacklist);
    helper.DeclareOptionalField("full_graph", &full_graph);
    helper.ReadAllFields(reader);
  }
};

inline MXGradientArgs LoadArgs(const string& json_str) {
  MXGradientArgs ret;
  istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  reader.Read(&ret);
  return ret;
}

const vector<IndexedGraph::NodeEntry> ExtractXS(
    const IndexedGraph& idx, const MXGradientArgs& args) {
  CHECK(args.xs.empty() || args.xs_blacklist.empty())
    << "Only one of \"xs\" and \"xs_blacklist\" arguments should be provided.";
  vector<IndexedGraph::NodeEntry> xs;
  if (args.xs.empty()) {
    // Blacklist rule.
    for (const uint32_t nid : idx.input_nodes()) {
      const Node* node = idx[nid].source;
      for (uint32_t i = 0; i < node->num_outputs(); ++i) {
        bool blacked = false;
        for (const auto& ent : args.xs_blacklist) {
          if (ent.node == nid && ent.index == i) {  // XXX: version?
            blacked = true;
            break;
          }
        }
        if (!blacked) {
          xs.emplace_back(IndexedGraph::NodeEntry{nid, i, 0});
        }
      }
    }
  } else {
    // Whitelist rule.
    for (const auto& ent : args.xs) {
      xs.emplace_back(IndexedGraph::NodeEntry{ent.node, ent.index, ent.version});
    }
  }
  return xs;
}

const vector<IndexedGraph::NodeEntry> ExtractYS(
    const IndexedGraph& idx, const MXGradientArgs& args) {
  CHECK(args.ys.empty() || args.ys_blacklist.empty())
    << "Only one of \"ys\" and \"ys_blacklist\" arguments should be provided.";
  vector<IndexedGraph::NodeEntry> ys;
  if (args.ys.empty()) {
    // Blacklist rule.
    for (const auto& out_ent : idx.outputs()) {
      bool blacked = false;
      for (const auto& ent : args.ys_blacklist) {
        if (ent.node == out_ent.node_id && ent.index == out_ent.index) {  // XXX: version?
          blacked = true;
          break;
        }
      }
      if (!blacked) {
        ys.emplace_back(out_ent);
      }
    }
  } else {
    // Whitelist rule.
    for (const auto& ent : args.ys) {
      ys.emplace_back(IndexedGraph::NodeEntry{ent.node, ent.index, ent.version});
    }
  }
  return ys;
}

Graph GradientRec(const Graph& fwd_graph,
                  const MXGradientArgs& args) {
  using nnvm::FGradient;
  static auto& op_fgrad_map = Op::GetAttr<FGradient>("FGradient");

  if (fwd_graph.global_attrs.count("FGradient")) {
    // XXX: What if the gradient function is called twice but with different
    // arguments?
    return fwd_graph;
  }

  const auto& fwd_graph_idx = fwd_graph.indexed_graph();
  const auto& xs = ExtractXS(fwd_graph_idx, args);
  const auto& ys = ExtractYS(fwd_graph_idx, args);

  // The map contains how the forward output is fed into backward graph.
  // The key is the output entry in the forward graph. The value is the
  // in input entry in the backward graph it corresponds to.
  NodeEntryMap<NodeEntry> fwdent2bwdent;
  unordered_map<const Node*, GradNodeInEntry> inent_map;
  // Maintain a mapping from a node to its output gradient entries.
  unordered_map<const Node*, vector<GradEntry> > node2outgrads;

  // Topological order of all nodes.
  // XXX(minjie): An ideal way is to dfs only the dependent nodes.
  // However, I cannot find a good way to convert from IndexedGraph::NodeEntry
  // to nnvm::NodeEntry.
  vector<NodePtr> topo_order;
  DFSVisit(fwd_graph.outputs, [&](const NodePtr& node) {
      if (node2outgrads.count(node.get()) == 0) {
        node2outgrads[node.get()].resize(node->num_outputs());
      }
      topo_order.push_back(node);
    });

  // Feed ys_grad into the map.
  for (size_t i = 0; i < ys.size(); ++i) {
    const Node* ys_node = fwd_graph_idx[ys[i].node_id].source;
    ostringstream oss;
    oss << ys_node->attrs.name << "$" << "output" << ys[i].index
      << "$" << "grad";
    NodePtr ys_grad_var = Node::Create();
    ys_grad_var->attrs.op = nullptr;
    ys_grad_var->attrs.name = oss.str();
    node2outgrads.at(ys_node)[ys[i].index].AddGrad(
        NodeEntry{ys_grad_var, 0, 0});
    inent_map.insert({ys_grad_var.get(), GradNodeInEntry::CreateFromOutGrads(i)});
  }

  // Traverse in reverse topological order to compute input gradients of each node.
  for (auto rit = topo_order.rbegin(); rit != topo_order.rend(); ++rit) {
    const NodePtr& node = *rit;
    if (node->inputs.size() == 0) {
      // This includes all variable nodes and input nodes like zeros, random, etc.
      continue;
    }
    // Fetch output gradients.
    const vector<NodeEntry>& out_grads = FetchSumGradients(
        node2outgrads.at(node.get()));
    FGradient fgrad = nullptr;
    if (node->is_graph()) {
      // Subgraph node.
      CHECK(node->graph()->global_attrs.count("FGradient"))
        << "Graph node " << node->attrs.name << " is non-differentiable "
        << "because its subgraph did not specialize gradient.";
      fgrad = node->graph()->GetGlobalAttr<FGradient>("FGradient");
      CHECK_NOTNULL(fgrad);
    } else if (op_fgrad_map.count(node->op())) {
      // Op node with gradient function registered.
      fgrad = op_fgrad_map[node->op()];
      CHECK_NOTNULL(fgrad);
    } else if (CheckGradAllZero(out_grads)) {
      // No gradient function registered, but can still compute gradients
      // if all the output gradients are zeros.
      fgrad = [] (const NodePtr& nodeptr, const vector<NodeEntry>& ) {
          return GenerateAllZeroGradients(nodeptr);
        };
    } else {
      LOG(FATAL) << "Operator " << node->op()->name << " is non-differentiable "
                 << "because it didn't register FGradient attribute.";
    }
    // Compute input gradients.
    const vector<NodeEntry>& in_grads = fgrad(node, out_grads);
    CHECK_EQ(node->inputs.size(), in_grads.size())
        << "Gradient function for node \"" << node->attrs.name
        << "\" does not return enough gradient entries.";
    if (!args.full_graph) {
      SplitForwardBackward(in_grads, fwd_graph_idx, &fwdent2bwdent);
    }
    // Save input gradients.
    for (size_t i = 0; i < node->inputs.size(); ++i) {
      const NodeEntry& in_ent = node->inputs[i];
      const NodeEntry& grad_in_ent = in_grads[i];
      node2outgrads.at(in_ent.node.get())[in_ent.index].AddGrad(grad_in_ent);
    }
  }

  // Create gradient subgraph.
  shared_ptr<Graph> grad_g = std::make_shared<Graph>();
  grad_g->outputs.reserve(xs.size());
  for (const auto& e : xs) {
    const Node* n = fwd_graph_idx[e.node_id].source;
    GradEntry& ge = node2outgrads.at(n)[e.index];
    grad_g->outputs.push_back(ge.GetSum());
  }
  if (args.full_graph) {
    // The returned full graph will not be splitted into forward and backward parts.
    return *grad_g;
  }

  // Create new forward graph.
  // 1. First inserts visible outputs.
  Graph new_fwd_graph;
  NodeEntrySet all_fwd_outputs;
  for (size_t idx = 0; idx < fwd_graph.outputs.size(); ++idx) {
    const NodeEntry& outent = fwd_graph.outputs[idx];
    new_fwd_graph.outputs.push_back(outent);
    all_fwd_outputs.insert(outent);
    if (fwdent2bwdent.count(outent)) {
      const Node* bwdvar = fwdent2bwdent[outent].node.get();
      inent_map[bwdvar] = GradNodeInEntry::CreateFromForwardOut(idx);
    }
  }
  // 2. Add an attribute to the graph about visible outputs.
  new_fwd_graph.global_attrs["num_visible_outputs"] = std::make_shared<any>(
      fwd_graph.outputs.size());
  // 3. Insert invisible outputs (required by gradient graph).
  for (const auto& kv : fwdent2bwdent) {
    const NodeEntry& fwd_ent = kv.first;
    if (all_fwd_outputs.count(fwd_ent)) {
      continue;
    }
    if (fwd_ent.node->is_variable()) {
      // This is the input of the forward graph.
      continue;
    }
    // New forward output entry.
    all_fwd_outputs.insert(fwd_ent);
    const size_t idx = new_fwd_graph.outputs.size();
    new_fwd_graph.outputs.push_back(fwd_ent);
    const Node* bwdvar = kv.second.node.get();
    inent_map[bwdvar] = GradNodeInEntry::CreateFromForwardOut(idx);
  }
  
  const auto& new_fwd_graph_idx = new_fwd_graph.indexed_graph();

  // Understand how the input entries of the gradient graph come from.
  vector<GradNodeInEntry> inent_info;
  // 1. Come from the inputs of the forward graph.
  const vector<uint32_t>& new_fwd_in_nodes = new_fwd_graph_idx.input_nodes();
  for (const auto& kv : fwdent2bwdent) {
    const NodeEntry& fwd_ent = kv.first;
    if (fwd_ent.node->is_variable()) {
      const uint32_t nid = new_fwd_graph_idx.node_id(fwd_ent.node.get());
      const auto it = std::find(new_fwd_in_nodes.begin(),
                                new_fwd_in_nodes.end(),
                                nid);
      CHECK(it != new_fwd_in_nodes.end());
      const size_t idx = it - new_fwd_in_nodes.begin();
      const Node* bwdvar = kv.second.node.get();
      inent_map[bwdvar] = GradNodeInEntry::CreateFromForwardIn(idx);
    }
  }
  // 2. Come from the outputs of the forward graph (both visible and invisible).
  const auto& grad_g_idx = grad_g->indexed_graph();
  for (const uint32_t grad_in_nid : grad_g_idx.input_nodes()) {
    const Node* grad_in_node = grad_g_idx[grad_in_nid].source;
    CHECK(grad_in_node->is_variable());
    CHECK(inent_map.count(grad_in_node))
      << "Cannot find way to feed input node \""
      << grad_in_node->attrs.name << "\" of the gradient graph.";
    inent_info.push_back(std::move(inent_map[grad_in_node]));
  }

  // Create how the output entry of the gradient values come from.
  vector<GradNodeOutEntry> outent_info;
  unordered_map<uint32_t, size_t> bwdoutnid2index;
  for (size_t i = 0; i < xs.size(); ++i) {
    bwdoutnid2index[xs[i].node_id] = i;
  }
  for (const uint32_t fwd_in_nid : new_fwd_graph_idx.input_nodes()) {
    if (bwdoutnid2index.count(fwd_in_nid)) {
      const size_t bwdidx = bwdoutnid2index[fwd_in_nid];
      outent_info.emplace_back(GradNodeOutEntry::CreateFromBackwardOut(bwdidx));
    } else {
      outent_info.emplace_back(GradNodeOutEntry::CreateFromZero());
    }
  }

  // Register FGradient for the graph. Note that all local variables must be passed by value.
  FGradient graph_fgrad = [grad_g, inent_info, outent_info]
    (const NodePtr& fwd_node, const vector<NodeEntry>& out_grads) {
      CHECK(fwd_node->is_graph());
      NodePtr grad_node = Node::Create();
      grad_node->attrs.name = fwd_node->attrs.name + "/backward";
      grad_node->attrs.graph = grad_g;
      // Insert inputs.
      for (const auto& info : inent_info) {
        switch (info.type) {
        case GradNodeInEntry::kFromGradOut:
          CHECK_LT(info.index, out_grads.size());
          grad_node->inputs.emplace_back(out_grads[info.index]);
          break;
        case GradNodeInEntry::kFromFwdOut:
          grad_node->inputs.emplace_back(
              NodeEntry{fwd_node, static_cast<uint32_t>(info.index), 0});
          break;
        case GradNodeInEntry::kFromFwdIn:
          grad_node->inputs.emplace_back(fwd_node->inputs[info.index]);
          break;
        default:
          LOG(FATAL) << "Cannot differentiate the subgraph.";
        }
      }
      // XXX(minjie): Add control dependency from the grad node to its forward node.
      // This can be used later to fetch attributes of the forward node.
      grad_node->control_deps.push_back(fwd_node);
      vector<NodeEntry> ret;
      auto zero_node = Node::Create();
      zero_node->attrs.op = kZeroOps[0];
      zero_node->attrs.name = "_NoGradient";
      for (const auto& info : outent_info) {
        switch (info.type) {
        case GradNodeOutEntry::kFromZero:
          ret.emplace_back(NodeEntry{zero_node, 0, 0});
          break;
        case GradNodeOutEntry::kFromOut:
          ret.emplace_back(NodeEntry{grad_node, static_cast<uint32_t>(info.index), 0});
          break;
        default:
          LOG(FATAL) << "Cannot differentiate the subgraph.";
        }
      }
      return ret;
    };

  new_fwd_graph.global_attrs["FGradient"] = std::make_shared<any>(graph_fgrad);
  return new_fwd_graph;
}



vector<IndexedGraph::NodeEntry> ExtractEntries(
    const Graph& graph, const vector<MXEntryArg>& entry_args) {
  // Convert to IndexGraph::NodeEntry.
  const auto& idx = graph.indexed_graph();
  vector<IndexedGraph::NodeEntry> ret;
  ret.reserve(entry_args.size());
  for (size_t i = 0; i < entry_args.size(); ++i) {
    const MXEntryArg& arg = entry_args[i];
    CHECK(arg.node < idx.num_nodes())
      << "Argument Error: invalid node id \"" << arg.node << "\". Only "
      << idx.num_nodes() << " nodes in the graph.";
    ret.emplace_back(IndexedGraph::NodeEntry{arg.node, arg.index, 0});
  }
  return ret;
}

Graph MXGradient(const Graph& graph) {
  const MXGradientArgs& args = LoadArgs(graph.GetGlobalAttr<string>("mx_gradient_args"));
  return GradientRec(graph, args);
}

// register pass
NNVM_REGISTER_PASS(MXGradient)
.describe("Return a gradient graph of src.attrs[\"ys\"] wrt src.attrs[\"xs\"]")
.set_body(MXGradient)
.set_change_graph(true)
.depend_global_attr("mx_gradient_args")
;

}  // namespace mxnet
