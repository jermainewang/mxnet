#ifndef MXNET_IMPERATIVE_TAPING_H_
#define MXNET_IMPERATIVE_TAPING_H_

#include <mxnet/c_api.h>
#include <mxnet/graph_attr_types.h>
#include <mxnet/ndarray.h>
#include <nnvm/symbolic.h>
#include <nnvm/op.h>
#include <nnvm/graph.h>

namespace mxnet {
namespace tape {

static const uint32_t kGradTape = 1;

struct TapeEntry {
  nnvm::NodePtr node;
};

// TapeEntryId is currently designed as follows:
// TapeEntryId is a uint64_t type.
// 0. Id 0 is reserved as default value for have not been taped.
// 1. The lower 4 digits are for tape id (a total of 15 tapes at most).
// 2. The next 20 digits are for the position on the tape (a total of 2^20 operations
//    at most on each tape).
// 3. The next 8 digits are for entry index of its operator (each operator can have
//    at most 256 number of output entries).
// 4. The higher 32 digits are for session ids of each tape. Each tape can be cleared and
//    reused. So the session id is used to differentiate those records.
// Note: 2 and 3 can be merged as entry id by maintaining a CSR-like data structure.
std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> ParseTapeEntryId(TapeEntryId id);
TapeEntryId MakeTapeEntryId(uint32_t tapeid, uint32_t pos, uint32_t index, uint32_t sid);

// A general class for tape recording. It should support:
// 1. Multiple taping at the same time.
// 2. Mutable operations.
class Tape {
 public:
  // Record one node and return the position on the tape.
  uint32_t Record(const nnvm::NodeAttrs& attrs,
                  const std::vector<NDArray*>& inputs,
                  const std::vector<NDArray*>& outputs);

  TapeEntry& operator[] (uint32_t pos) { return tape_[pos]; }

  const TapeEntry& operator[] (uint32_t pos) const { return tape_[pos]; }

  size_t size() const { return tape_.size(); }

  uint32_t session_id() const { return session_id_; }

  bool HasTaped(TapeEntryId teid) const {
    uint32_t tapeid = 0, pos = 0, index = 0, sid = 0;
    std::tie(tapeid, pos, index, sid) = tape::ParseTapeEntryId(teid);
    return tapeid == tape_id_ && sid == session_id_;
  }

  nnvm::Graph GetGraph(const std::vector<const NDArray*>& arrays) const;

  void NewSession();

  static Tape& Get(uint32_t tapeid);

 private:
  Tape(uint32_t id);

  nnvm::NodePtr NewNode(const nnvm::NodeAttrs& attrs);
  nnvm::NodePtr NewVariable();

  // Unique id of this tape.
  const uint32_t tape_id_;
  uint32_t session_id_ = 0;
  // Recorded operations.
  std::vector<TapeEntry> tape_;
  // Positions of tape head entries.
  // TODO(minjie): use ring-buffer for better cache-locality.
  //std::unordered_set<uint32_t> tape_heads_;
};

}  // namespace tape
}  // namespace mxnet

#endif  // MXNET_IMPERATIVE_TAPING_H_
