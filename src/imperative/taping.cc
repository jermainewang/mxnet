#include "./taping.h"

using namespace std;
using namespace nnvm;

namespace mxnet {
namespace tape {

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> ParseTapeEntryId(TapeEntryId id) {
  static const uint64_t mask1 = 0xf;
  static const uint64_t mask2 = 0xfffff;
  static const uint64_t mask3 = 0xff;
  return std::make_tuple(id & mask1,
                         (id >> 4) & mask2,
                         (id >> 24) & mask3,
                         (id >> 32));
}

TapeEntryId MakeTapeEntryId(uint32_t tapeid, uint32_t pos, uint32_t index, uint32_t sid) {
  static const uint64_t mask1 = 0xf;
  static const uint64_t mask2 = 0xfffff;
  static const uint64_t mask3 = 0xff;
  const uint32_t lower = (tapeid & mask1) + ((pos & mask2) << 4) + ((index & mask3) << 24);
  return (static_cast<uint64_t>(sid) << 32) + lower;
}

Tape::Tape(uint32_t id): tape_id_(id) {
}

NodePtr Tape::NewNode(const nnvm::NodeAttrs& attrs) {
  // TODO(minjie): keep original name?
  NodePtr node = Node::Create();
  node->attrs = attrs;
  std::ostringstream oss;
  oss << "tape" << tape_id_ << "/node" << tape_.size();
  node->attrs.name = oss.str();
  return node;
}

NodePtr Tape::NewVariable() {
  NodePtr node = Node::Create();
  node->attrs.op = nullptr;
  std::ostringstream oss;
  oss << "tape" << tape_id_ << "/var" << tape_.size();
  node->attrs.name = oss.str();
  return node;
}

void Tape::RecordValue(uint32_t pos, uint32_t index, const NDArray& value) {
  const NodePtr& node = tape_[pos].node;
  CHECK_LT(index, node->num_outputs());
  auto& values = node->CreateOrGetEntryAttrs<NDArray>("value");
  values.value[index] = value;
}

uint32_t Tape::Record(const NodeAttrs& attrs,
                      const vector<NDArray*>& inputs,
                      const vector<NDArray*>& outputs) {
  NodePtr node = NewNode(attrs);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const TapeEntryId teid = inputs[i]->tape_entry_id();
    uint32_t tapeid = 0, pos = 0, index = 0, sid = 0;
    std::tie(tapeid, pos, index, sid) = ParseTapeEntryId(inputs[i]->tape_entry_id());
    if (inputs[i]->tape_entry_id() == kNotTaped || sid != session_id_) {
      NodePtr var = NewVariable();
      inputs[i]->set_tape_entry_id(MakeTapeEntryId(tape_id_, tape_.size(), 0, session_id_));
      tape_.emplace_back(TapeEntry{var});
      node->inputs.emplace_back(NodeEntry{var, 0, 0});
    } else {
      CHECK(tapeid == tape_id_) << "The same array has been recorded on multiple tapes.";
      CHECK(pos < tape_.size());
      node->inputs.emplace_back(NodeEntry{tape_[pos].node, index, 0});
    }
  }
  CHECK_EQ(node->num_outputs(), outputs.size());
  const uint32_t pos = tape_.size();
  tape_.emplace_back(TapeEntry{node});
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs[i]->set_tape_entry_id(MakeTapeEntryId(tape_id_, pos, i, session_id_));
  }
  return pos;
}
  
Graph Tape::GetGraph(const std::vector<const NDArray*>& arrays) const {
  Graph graph;
  uint32_t tapeid = 0, pos = 0, index = 0, sid = 0;
  for (const NDArray* arr : arrays) {
    CHECK(arr->tape_entry_id() != kNotTaped);
    std::tie(tapeid, pos, index, sid) = ParseTapeEntryId(arr->tape_entry_id());
    CHECK(tapeid == tape_id_ && sid == session_id_)
      << "Cannot extract computation graph of the provided NDArray. The array "
      << "has not been recorded on this tape (session " << session_id_ << ").";
    graph.outputs.emplace_back(NodeEntry{tape_[pos].node, index, 0});
  }
  return graph;
}

void Tape::NewSession() {
  tape_.clear();
  ++session_id_;
}

Tape* Tape::Get(uint32_t tapeid) {
  // TODO(only single tape right now).
  static Tape tape(1);
  return &tape;
}

}  // namespace tape
}  // namespace mxnet
