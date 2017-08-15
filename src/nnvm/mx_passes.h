/*!
 *  Copyright (c) 2016 by Contributors
 * \file mx_passes.h
 * \brief Header file for shared pass attributes.
 */

#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>

namespace mxnet {
namespace pass {

namespace inplace {
static const std::string key = "inplace_option";
struct InplaceOption {
  // A map from input to output.
  std::pair<int, int> inplace_pair;
  bool is_identity;
};
}  // namespace inplace

namespace plan_memory {
static const std::string key = "storage";
struct StorageRef {
  int storage_id;
  int inplace_index;
};
struct Storage {
  int id;
  int device_id;
  size_t max_bytes;
};
}  // namespace plan_memory

}  // namespace pass
}  // namespace mxnet
