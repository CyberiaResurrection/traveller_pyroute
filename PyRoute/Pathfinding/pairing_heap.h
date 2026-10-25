#pragma once
#include <cstddef>
#include <limits>

// Heap item. Put extra fields here as needed.
struct Item {
  double key = std::numeric_limits<double>::quiet_NaN();
  double augment = std::numeric_limits<double>::quiet_NaN();
  double dist = std::numeric_limits<double>::quiet_NaN();
  long curnode = std::numeric_limits<long>::quiet_NaN();
  long parent_node = std::numeric_limits<long>::quiet_NaN();
  Item *child = nullptr;
  Item *sibling = nullptr;
  // For nonroots, points back at the `child` or `sibling` field that points at
  // this.
  Item **parent = nullptr;
  bool operator>(Item const &other) const {
    if (this->augment > other.augment) {
      return true;
    }
    if (this->augment < other.augment) {
      return false;
    }
    if (this->dist > other.dist) {
      return true;
    }
    if (this->dist < other.dist) {
      return false;
    }
    return this->curnode > other.curnode;
  }
  bool operator<(Item const &other) const {
    if (this->augment < other.augment) {
      return true;
    }
    if (this->augment > other.augment) {
      return false;
    }
    if (this->dist < other.dist) {
      return true;
    }
    if (this->dist > other.dist) {
      return false;
    }
    return this->curnode < other.curnode;
  }
};

// Pairing heap. The heap owns its items: Insert copies the supplied item into
// a heap-allocated node, and DeleteMin hands back a copy and frees the node.
class PairingHeap {
public:
  explicit PairingHeap() = default;
  ~PairingHeap();

  PairingHeap(const PairingHeap &) = delete;
  PairingHeap &operator=(const PairingHeap &) = delete;

  // Inserts a copy of `x` and returns a handle to it, valid until that item
  // is removed by DeleteMin. Caller must not mutate the item through the
  // handle except via DecreaseKey.
  Item *Insert(Item x);

  // Removes and returns one of the contained items with the minimum key. It
  // will have null tree pointers. Requires a non-empty heap.
  Item DeleteMin(void);

  // Requires (augment, dist) to be no greater than x's current values.
  void DecreaseKey(Item *x, double augment, double dist);

  // O(n); for tests.
  std::size_t CountWhileCheckingInvariant() const;

  // Is heap empty?
  bool IsEmpty();

private:
  void Clear();

  Item *root_ = nullptr;
};
