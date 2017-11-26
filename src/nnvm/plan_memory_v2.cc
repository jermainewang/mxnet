/*!
 *  Copyright (c) 2016 by Contributors
 * \file plan_memory.cc
 * \brief Assign memory tag to each of the data entries.
 */
#include <memory>

#include "./mx_passes.h"

using namespace std;
using namespace nnvm;

namespace mxnet {
namespace pass {
namespace {
/*!
 * \brief Find best path in the DAG, with reward defined
 *  by sum of reward of each node along the path.
 * \param graph the original static graph.
 * \param topo_order topo order of the nodes in the graph.
 * \param node_reward the reward of each node.
 * \param path the output path of nodes.
 * \return the total reward of best path.
 */
inline uint32_t FindBestPath(
    const IndexedGraph& graph,
    const std::vector<uint32_t>& node_reward,
    std::vector<uint32_t>* path) {
  const uint32_t num_nodes = static_cast<uint32_t>(graph.num_nodes());
  CHECK_EQ(num_nodes, node_reward.size());

  std::vector<uint32_t> best_reward(node_reward.size(), 0);
  std::vector<uint32_t> next_node(node_reward.size(), num_nodes);
  uint32_t best_solution = 0, best_start_node = 0;

  // traverse in reverse topo order
  for (uint32_t i = static_cast<uint32_t>(graph.num_nodes()); i != 0; --i) {
    const uint32_t nid = i - 1;
    best_reward[nid] += node_reward[nid];
    if (best_reward[nid] > best_solution) {
      best_solution = best_reward[nid];
      best_start_node = nid;
    }
    for (const auto& e : graph[nid].inputs) {
      const uint32_t prev = e.node_id;
      if (best_reward[nid] > best_reward[prev]) {
        best_reward[prev] = best_reward[nid];
        next_node[prev] = nid;
      }
    }
  }
  path->clear();
  uint32_t reward = 0;
  for (uint32_t nid = best_start_node; nid < num_nodes; nid = next_node[nid]) {
    path->push_back(nid); reward += node_reward[nid];
  }
  CHECK_EQ(reward, best_solution);
  return best_solution;
}

/*!
 * \brief Color the nodes in the graph into index.
 *
 *  The coloring algorithm tries to assign node group
 *  such that nodes in different groups can run in parallel.
 *  Note that the opposite may not be true though. Nodes
 *  that can run in parallel may have the same color. This
 *  is fine since treating concurrent nodes sequentially is
 *  safe.
 *
 * \param graph the original indexed graph.
 * \param node_importance The importance of the node
 * \param max_ncolor maximum number of colors allowed.
 * \param color the color index of each of the node.
 * \return the total number of colors.
 */
inline uint32_t ColorNodeGroup(
    const IndexedGraph &graph,
    std::vector<uint32_t> node_importance,
    uint32_t max_ncolor,
    std::vector<uint32_t> *color) {
  CHECK_NE(max_ncolor, 0U);
  CHECK_EQ(graph.num_nodes(), node_importance.size());

  color->clear();
  color->resize(graph.num_nodes(), max_ncolor);
  uint32_t cindex;
  // Greedy algorithm. Every time
  // find a path with best reward and assign a new color
  // All the nodes in the path cannot run in parallel.
  for (cindex = 0; cindex < max_ncolor - 1; ++cindex) {
    std::vector<uint32_t> path;
    uint32_t reward = FindBestPath(graph, node_importance, &path);
    if (reward == 0) break;
    for (uint32_t nid : path) {
      if (node_importance[nid] != 0) {
        CHECK_EQ(color->at(nid), max_ncolor);
        color->at(nid) = cindex;
        // make the importance 0 after color is decided.
        node_importance[nid] = 0;
      }
    }
  }
  // assign i for rest of the node
  for (uint32_t i = 0; i < graph.num_nodes(); ++i) {
    if (color->at(i) == max_ncolor) {
      color->at(i) = cindex;
    }
  }
  return cindex + 1;
}
}  // namespace

using plan_memory::StorageRef;
using plan_memory::Storage;
using inplace::InplaceOption;

// simple graph based allocator.
class GraphAllocator {
 public:
  // storage id equals integer.
  using StorageID = int;

  StorageID Request(int dev_id, int dtype, TShape shape,
                    uint32_t node_id, uint32_t refcount) {
    const uint32_t color = (node_color_.size() != 0)?
      node_color_[node_id] : num_match_color_;
    return RequestColor(dev_id, dtype, shape, color, refcount);
  }

  // request a free storage
  StorageID RequestColor(int dev_id, int dtype, TShape shape,
                         uint32_t color, uint32_t refcount) {
    if (shape.ndim() == 0) {
      return plan_memory::kBadStorageID;
    }
    if (refcount == 0) {
      return plan_memory::kNull;
    }
    // search memory block in [size / match_range_, size * match_range_)
    size_t size = shape.Size() * mshadow::mshadow_sizeof(dtype);
    if (match_range_ == 0) {
      // Always allocate new space if no match range is set.
      return this->Alloc(dev_id, size, refcount);
    }
    auto begin = free_.lower_bound(size / match_range_);
    auto mid = free_.lower_bound(size);
    auto end = free_.upper_bound(size * match_range_);
    // First search for memory blocks larger than requested.
    for (auto it = mid; it != end; ++it) {
      StorageEntry *e = it->second;
      if (e->device_id != dev_id) {
        // Ignore storage on different device.
        continue;
      }
      if (node_color_.size() != 0 &&
          node_color_[e->released_by_node] != color) {
        // The storage entry is used by node that has different color with the
        // request node. This means this storage entry *may* be used by nodes
        // that can run concurrently with the requesting node. Thus, this storage
        // entry cannot be assigned to the requesting node to avoid data race.
        continue;
      }
      // Use exect matching strategy
      e->max_bytes = std::max(size, e->max_bytes);
      e->refcount = refcount;
      // find a exact match, erase from map and return
      free_.erase(it);
      return e->id;
    }
    // Then search for memory blocks smaller than requested space.
    for (auto it = mid; it != begin;) {
      --it;
      StorageEntry *e = it->second;
      if (e->device_id != dev_id) {
        // Ignore storage on different device.
        continue;
      }
      if (node_color_.size() != 0 &&
          node_color_[e->released_by_node] != color) {
        // The storage entry is used by node that has different color with the
        // request node. This means this storage entry *may* be used by nodes
        // that can run concurrently with the requesting node. Thus, this storage
        // entry cannot be assigned to the requesting node to avoid data race.
        continue;
      }
      // Use exect matching strategy
      e->max_bytes = std::max(size, e->max_bytes);
      e->refcount = refcount;
      // erase from map and return
      free_.erase(it);
      return e->id;
    }
    // Cannot find anything, so return a new one.
    return this->Alloc(dev_id, size, refcount);
  }
  // release a memory space.
  void Release(StorageID id, uint32_t node_id) {
    CHECK_NE(id, plan_memory::kBadStorageID);
    CHECK_NE(id, plan_memory::kNull);
    if (id == plan_memory::kExternalStorageID) {
      // Nothing to be done for external storage.
      return;
    }
    CHECK_LT(id, data_.size());
    StorageEntry *e = data_[id].get();
    CHECK_GT(e->refcount, 0);
    if (--e->refcount == 0) {
      e->released_by_node = node_id;
      free_.insert({e->max_bytes, e});
    }
  }

  void ReUse(StorageID id, uint32_t node_id, uint32_t refcount) {
    // TODO(minjie): check node color.
    CHECK_GE(id, 0);
    CHECK_LT(id, data_.size());
    data_[id]->refcount += refcount;
  }

  uint32_t GetRefCount(StorageID id) const {
    CHECK_GE(id, 0);
    CHECK_LT(id, data_.size());
    return data_[id]->refcount;
  }

  // totoal number of bytes allocated
  size_t TotalAllocBytes() const {
    size_t total = 0;
    for (auto &p : data_) {
      total += p->max_bytes;
    }
    return total;
  }

  vector<plan_memory::Storage> GetAllStorage() const {
    using plan_memory::Storage;
    vector<Storage> ret;
    for (const auto& d : data_) {
      ret.emplace_back(Storage{d->id, d->device_id, d->max_bytes});
    }
    return ret;
  }

  // Constructor. No graph coloring is used by default.
  explicit GraphAllocator(const IndexedGraph* idx,
                          size_t match_range,
                          uint32_t num_match_color = 0)
    : idx_(idx) {
    // TODO(minjie): coloring is not supported right now.
    CHECK_EQ(num_match_color, 0);
    //this->Init(match_range, dmlc::GetEnv("NNVM_EXEC_NUM_TEMP", 1));
    this->Init(match_range, num_match_color);
  }

 private:
  // initialize the graph allocator
  void Init(const size_t match_range, const uint32_t num_match_color) {
    match_range_ = match_range;
    num_match_color_ = num_match_color;
    if (num_match_color_ > 1) {
      std::vector<uint32_t> importance(idx_->num_nodes(), 0);
      for (uint32_t nid = 0; nid < idx_->num_nodes(); ++nid) {
        if ((*idx_)[nid].source->is_variable()) continue;
        importance[nid] = 1;
      }
      num_match_color_ = ColorNodeGroup(
          *idx_, importance, num_match_color_, &node_color_);
    }
  }

  StorageID Alloc(int dev_id, size_t size, uint32_t refcount) {
    StorageID id = static_cast<StorageID>(data_.size());
    std::unique_ptr<StorageEntry> ptr(new StorageEntry());
    ptr->id = id;
    ptr->device_id = dev_id;
    ptr->max_bytes = size;
    ptr->refcount = refcount;
    data_.emplace_back(std::move(ptr));
    return id;
  }
  // internal storage entry
  struct StorageEntry {
    // the id of the entry.
    StorageID id;
    // the device id of the storage.
    int device_id;
    // maximum size of storage requested.
    size_t max_bytes{0};
    // node index that released it last time
    uint32_t released_by_node{0};
    // reference counts
    uint32_t refcount{0};
  };
  // scale used for rough match
  size_t match_range_;
  // whether use color based match algorithm
  uint32_t num_match_color_{1};
  // the size of each dtype
  std::vector<size_t> dtype_size_dict_;
  // free list of storage entry; map from size to storage.
  std::multimap<size_t, StorageEntry*> free_;
  // all the storage resources available
  std::vector<std::unique_ptr<StorageEntry> > data_;
  // color of nodes in the graph, used for auxiliary policy making.
  std::vector<uint32_t> node_color_;
  // internal indexed graph
  const IndexedGraph* idx_;
};


void InplaceOptimize(const Graph& graph,
                     uint32_t nid,
                     const vector<uint32_t>& entry_ref_counts,
                     const Column<TShape>* shape,
                     const Column<int>* dtype,
                     const Column<vector<InplaceOption>>* inplace_option,
                     Column<StorageRef>* storage,
                     GraphAllocator* allocator) {
  const auto& idx = graph.indexed_graph();
  const Node* node = idx[nid].source;
  // check inplace option
  std::vector<bool> taken(node->inputs.size(), false);
  const vector<InplaceOption>& opts = inplace_option->value[nid];
  for (size_t ipair = 0; ipair < opts.size(); ++ipair) {
    const InplaceOption& opt = opts[ipair];
    const auto& kv = opt.inplace_pair;
    const uint32_t eid_out = idx.entry_id(nid, kv.second);
    const uint32_t eid_in = idx.entry_id(node->inputs[kv.first]);
    const auto sid_out = storage->value[eid_out].storage_id;
    const auto sid_in = storage->value[eid_in].storage_id;
    if (taken[kv.first] == false &&
        sid_out == plan_memory::kNull &&
        sid_in >= 0 &&
        (allocator->GetRefCount(sid_in) == 1 || opt.is_identity) &&
        entry_ref_counts[eid_out] > 0 &&
        shape->value[eid_out].Size() == shape->value[eid_in].Size() &&
        dtype->value[eid_out] == dtype->value[eid_in]) {
      // inplace optimization
      taken[kv.first] = true;
      storage->value[eid_out].storage_id = sid_in;
      storage->value[eid_out].inplace_index = kv.first;
      // Reuse storage for output and add ref count of output
      // to storage. This will get substracted later in free
      // input section.
      allocator->ReUse(sid_in, nid, entry_ref_counts[eid_out]);
    }
  }
}


void NormalAlloc(const Graph& graph,
                 uint32_t nid,
                 const vector<uint32_t>& entry_ref_counts,
                 const Column<TShape>* shape,
                 const Column<int>* dtype,
                 const Column<int>* device,
                 Column<StorageRef>* storage,
                 GraphAllocator* allocator) {
  const auto& idx = graph.indexed_graph();
  const Node* node = idx[nid].source;
  // normal allocation
  const int dev_id = (device != nullptr) ? device->value[nid] : 0;
  // sort output nodes based on size before allocating output
  std::multimap<size_t, uint32_t> eids;
  for (uint32_t index = 0; index < node->num_outputs(); ++index) {
    uint32_t eid = idx.entry_id(nid, index);
    if (storage->value[eid].storage_id == plan_memory::kNull) {
      const TShape &eshape = shape->value[eid];
      const size_t esize = eshape.Size();
      eids.insert(std::make_pair(esize, eid));
    }
  }
  for (auto rit = eids.rbegin(); rit != eids.rend(); ++rit) {
    uint32_t eid = rit->second;
    auto sid = allocator->Request(dev_id,
                                  dtype->value[eid],
                                  shape->value[eid],
                                  nid,
                                  entry_ref_counts[eid]);
    storage->value[eid].storage_id = sid;
  }
}

void FreeInputs(const Graph& graph,
                uint32_t nid,
                const Column<vector<uint32_t>>* ignored_inputs,
                Column<StorageRef>* storage,
                GraphAllocator* allocator) {
  const auto& idx = graph.indexed_graph();
  const Node* node = idx[nid].source;
  // check if certain inputs is ignored.
  std::vector<uint32_t> ignore_inputs = ignored_inputs->value[nid];
  std::sort(ignore_inputs.begin(), ignore_inputs.end());
  // then free inputs
  for (size_t i = 0; i < node->inputs.size(); ++i) {
    // ref counter of ignored input is already decreased.
    if (std::binary_search(ignore_inputs.begin(), ignore_inputs.end(), i)) continue;
    const auto& e = node->inputs[i];
    const uint32_t eid = idx.entry_id(e);
    const auto sid = storage->value[eid].storage_id;
    if (sid >= 0) {
      allocator->Release(sid, nid);
    }
  }
}

// TODO(minjie): change name
// Check if there are outputs that can be freed immediately
// these output are not referenced by any operator.
size_t FreeOutputs(const Graph& graph,
                   uint32_t nid,
                   Column<StorageRef>* storage,
                   GraphAllocator* allocator) {
  const auto& idx = graph.indexed_graph();
  const Node* node = idx[nid].source;
  size_t num_not_allocated = 0;
  for (uint32_t index = 0; index < node->num_outputs(); ++index) {
    const uint32_t eid = idx.entry_id(nid, index);
    const auto sid = storage->value[eid].storage_id;
    if (sid == plan_memory::kBadStorageID) {
      ++num_not_allocated;
    }
  }
  return num_not_allocated;
}

/*
 * Internal method to perform the memory allocation for a graph
 */
  // Graph allocator is created for each graph locally.
  // If the subgraph has been specialized for memory planning,
  //  - We need to check whether the specialization is done for
  //    inplaced inputs or not.
  //  - If the input inplace option is the same, we need to modify
  //    the storage vector so it reflects a global plan.
  //    - This is doable by modifying GraphAllocator
  //  - Otherwise, re-specialize the subgraph plan.
  // This means we can save the memory planning time (no need to
  // color the graph again), but the storage vector is always
  // copied.
  // NOTE: We need to compute ignored inputs for subgraphs.
  // NOTE: The match range search act as normal. If the subgraph has
  //   been specialized, no need to search for the best.
  
  // How to adapt a local plan to a global plan (recursively)?
  // - If the graph has no subgraph, local plan is equal to global plan.
  // - Otherwise, figure out the global plan of each subgraph node. Then, for each
  //   subgraph node:
  //   - Input and output entry should use the same storage id as the parent graph.
  //   - For remaining entries, remap them from the largest to the smallest.
size_t PlanMemoryRec(const Graph& graph,
                     const vector<uint32_t>& input_ref_counts,
                     const vector<uint32_t>& output_ref_counts,
                     const Column<TShape>* shape,
                     const Column<int>* dtype,
                     const Column<vector<InplaceOption>>* inplace_option,
                     const Column<vector<uint32_t>>* ignored_inputs,
                     const Column<int>* device,
                     Column<StorageRef>* storage,
                     GraphAllocator* allocator) {
  const IndexedGraph& idx = graph.indexed_graph();
  // Reference counter of each node
  std::vector<uint32_t> entry_ref_counts(idx.num_node_entries(), 0);
  // 1. Initialize reference count.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    if (node->is_variable()) {
      // Ignore input node.
      continue;
    }
    for (const auto& e : node->inputs) {
      ++entry_ref_counts[idx.entry_id(e)];
    }
    // no dataflow dependency is needed for those are ignored.
    // revoke the dependency counter.
    for (const uint32_t ignored : ignored_inputs->value[nid]) {
      --entry_ref_counts[idx.entry_id(node->inputs[ignored])];
    }
  }
  // 2. Reference count for input nodes.
  CHECK_LE(input_ref_counts.size(), idx.input_nodes().size());
  for (size_t i = 0; i < idx.input_nodes().size(); ++i) {
    const uint32_t eid = idx.entry_id(idx.input_nodes()[i], 0);
    const GraphAllocator::StorageID sid = storage->value[eid].storage_id;
    if (sid >= 0) {
      // This is a subgraph and the input entries are allocated by parent graph.
      allocator->ReUse(sid, idx.input_nodes()[i], entry_ref_counts[eid]);
      // We need to release once for the input entry to simulate it being copied
      // into the subgraph.
      allocator->Release(sid, idx.input_nodes()[i]);
    }
    if (i < input_ref_counts.size()) {
      entry_ref_counts[eid] += input_ref_counts[i];
    }
  }
  // 3. Reference counts for output entries.
  CHECK_LE(output_ref_counts.size(), idx.outputs().size());
  for (size_t i = 0; i < idx.outputs().size(); ++i) {
    const uint32_t outeid = idx.entry_id(idx.outputs()[i]);
    if (i < output_ref_counts.size()) {
      entry_ref_counts[outeid] += output_ref_counts[i];
    } else {
      // Increase the rc of output entries so that the space will not be shared.
      ++entry_ref_counts[outeid];
    }
  }

  /*{
  const auto& idx = graph.indexed_graph();
  ostringstream oss;
  oss << "[" << endl;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    for (size_t i = 0; i < node->num_outputs(); ++i) {
      auto eid = idx.entry_id(nid, i);
      const auto& ref = storage->value[eid];
      oss << "\t@@@" << node->attrs.name << "$" << i
        << " || storage_id=" << ref.storage_id << ", "
        << " graph_rc=" << entry_ref_counts[eid]
        << " alloc_rc=" << ((ref.storage_id >= 0)? allocator->GetRefCount(ref.storage_id) : 0)
        << endl;
    }
  }
  oss << "]";
  LOG(INFO) << oss.str();
  }*/

  size_t num_not_allocated = 0;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    //LOG(INFO) << "Planning node#" << nid << node->attrs.name;
    if (node->is_variable()) {
      // Do nothing for variable inputs. Variable inputs are usually from external
      // storage. If this is a subgraph, the storage has already been specified
      // by the parent graph.
      const uint32_t eid = idx.entry_id(nid, 0);
      if (storage->value[eid].storage_id == plan_memory::kNull) {
        storage->value[eid].storage_id = plan_memory::kExternalStorageID;
      }
    } else if (node->is_graph()) {
      // XXX(minjie): Reuse of subgraph's memory plan can only be done if *all*
      // the given arguments are the same (e.g. shape/type/device/etc.). Since
      // the reuse check itself may be quite costly already, currently we never
      // reuse subgraph's memory plan.
      // Set reference counts and storage plans for the input and output entries
      // of the subgraph.
      auto sg = node->graph();
      const auto& subidx = sg->indexed_graph();
      vector<uint32_t> sub_in_refcounts(subidx.input_nodes().size(), 0);
      vector<uint32_t> sub_out_refcounts(subidx.outputs().size(), 0);
      ColumnRef<StorageRef> subplan_ref
        = sg->CreateEntryColumn<StorageRef>({plan_memory::kNull, -1});
      Column<StorageRef>* subplan = subplan_ref.CopyOnWrite();
      for (size_t i = 0; i < node->inputs.size(); ++i) {
        const uint32_t ineid = idx.entry_id(node->inputs[i]);
        const uint32_t subineid = subidx.entry_id(subidx.input_nodes()[i], 0);
        subplan->value[subineid] = storage->value[ineid];
        sub_in_refcounts[i] = entry_ref_counts[ineid];
      }
      for (size_t i = 0; i < node->num_outputs(); ++i) {
        const uint32_t outeid = idx.entry_id(nid, i);
        const uint32_t subouteid = subidx.entry_id(sg->outputs[i]);
        sub_out_refcounts[i] = entry_ref_counts[outeid];
      }
      // Memory planning for the subgraph.
      num_not_allocated += PlanMemoryRec(
          *sg,
          sub_in_refcounts,
          sub_out_refcounts,
          shape->children[nid].get(),
          dtype->children[nid].get(),
          inplace_option->children[nid].get(),
          ignored_inputs->children[nid].get(),
          (device == nullptr)? nullptr : device->children[nid].get(),
          subplan,
          allocator);
      storage->children[nid] = subplan_ref;
      // Copy the plan for the output entries.
      for (size_t i = 0; i < node->num_outputs(); ++i) {
        const uint32_t outeid = idx.entry_id(nid, i);
        const uint32_t subouteid = subidx.entry_id(sg->outputs[i]);
        // TODO(minjie): is it correct to also copy the inplace index?
        storage->value[outeid] = subplan->value[subouteid];
      }
    } else {
      InplaceOptimize(graph, nid, entry_ref_counts, shape, dtype,
                      inplace_option, storage, allocator);

      NormalAlloc(graph, nid, entry_ref_counts, shape, dtype,
                  device, storage, allocator);
      
      FreeInputs(graph, nid, ignored_inputs, storage, allocator);

      num_not_allocated += FreeOutputs(graph, nid, storage, allocator);
      
    }
  }

  {
  const auto& idx = graph.indexed_graph();
  ostringstream oss;
  oss << "[" << endl;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    for (size_t i = 0; i < node->num_outputs(); ++i) {
      auto eid = idx.entry_id(nid, i);
      const auto& ref = storage->value[eid];
      oss << "\t" << node->attrs.name << "$" << i
        << " || storage_id=" << ref.storage_id << ", "
        << "inplace_index=" << ref.inplace_index << endl;
    }
  }
  oss << "]";
  DLOG(INFO) << oss.str();
  }
  return num_not_allocated;
}

// function to plan memory
Graph MXPlanMemory(Graph&& graph) {
  using plan_memory::MXPlanMemoryArgs;
  const MXPlanMemoryArgs& args =
    GetPassArgument<MXPlanMemoryArgs>(graph, plan_memory::arg_name);
  const auto* shape =
    graph.entry_attrs.GetColumn<TShape>(shape::key).get();
  const auto* dtype =
    graph.entry_attrs.GetColumn<int>(dtype::key).get();
  const auto* inplace_option =
    graph.node_attrs.GetColumn<vector<InplaceOption>>(inplace::key).get();
  const auto* ignored_inputs =
    graph.node_attrs.GetColumn<vector<uint32_t>>("ignored_inputs").get();
  const auto* device = graph.node_attrs.GetColumn<int>(ctx::device_key).get();
  const auto& idx = graph.indexed_graph();

  // Create initial memory plan.
  ColumnRef<StorageRef> init_plan =
    graph.CreateEntryColumn<StorageRef>({plan_memory::kNull, -1});
  for (uint32_t eid : args.external_entry_ids) {
    init_plan.CopyOnWrite()->value[eid].storage_id = plan_memory::kExternalStorageID;
  }

  // Plan memory using different match ranges.
  size_t min_allocated_bytes = -1;
  ColumnRef<StorageRef> min_storage_ref;
  vector<Storage> min_storages;
  const size_t max_match_range = dmlc::GetEnv("NNVM_EXEC_MATCH_RANGE", 16);
  const size_t min_match_range =
         dmlc::GetEnv("NNVM_AUTO_SEARCH_MATCH_RANGE", false) ? 1 : max_match_range;
  for (size_t match_range = min_match_range; match_range <= max_match_range; match_range *= 2) {
    // Make a copy of related fields
    ColumnRef<StorageRef> storage_ref = init_plan;

    // the allocator
    GraphAllocator allocator(&idx, match_range);

    // number of entries that are not statically allocated.
    PlanMemoryRec(graph, vector<uint32_t>(), vector<uint32_t>(), shape, dtype, inplace_option,
                  ignored_inputs, device, storage_ref.CopyOnWrite(), &allocator);

    size_t storage_allocated_bytes = allocator.TotalAllocBytes();
    // Choose the plan which leads to minimal memory usage
    if (min_allocated_bytes > storage_allocated_bytes) {
      min_storage_ref = storage_ref;
      min_allocated_bytes = storage_allocated_bytes;
      min_storages = allocator.GetAllStorage();
    }
  }

  /*{
  ostringstream oss;
  oss << "[" << endl;
  for (size_t i = 0; i < min_storages.size(); ++i) {
    oss << "\tStorage#" << i << " id=" << min_storages[i].id
      << " size=" << min_storages[i].max_bytes << endl;
  }
  oss << "]";
  LOG(INFO) << oss.str();
  }*/

  graph.entry_attrs.SetColumn(plan_memory::ref_key, min_storage_ref);
  graph.global_attrs[plan_memory::storage_key] =
    std::make_shared<any>(std::move(min_storages));

  return graph;
}

NNVM_REGISTER_PASS(MXPlanMemory)
.describe("Plan the memory allocation of each node entries.")
.set_body(MXPlanMemory)
.set_change_graph(false)
.set_argument(plan_memory::arg_name)
.depend_entry_attr(shape::key)
.depend_entry_attr(dtype::key)
.depend_node_attr(inplace::key)
.depend_node_attr(ignore::key)
.depend_node_attr(ctx::device_key)
.provide_entry_attr(plan_memory::ref_key)
.provide_global_attr(plan_memory::storage_key)
;

}  // namespace pass
}  // namespace mxnet
