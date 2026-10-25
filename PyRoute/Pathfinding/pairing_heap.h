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
  Item *parent = nullptr;
  bool operator>(Item const& other) const {
    if (this->augment > other.augment) { return true; }
    if (this->augment < other.augment) { return false; }
    if (this->dist > other.dist) { return true; }
    if (this->dist < other.dist) { return false; }
    return this->curnode > other.curnode;
  }
  bool operator<(Item const& other) const {
    if (this->augment < other.augment) { return true; }
    if (this->augment > other.augment) { return false; }
    if (this->dist < other.dist) { return true; }
    if (this->dist > other.dist) { return false; }
    return this->curnode < other.curnode;
  }
};

// Pairing heap. Caller owns the items.
class PairingHeap {
public:
  explicit PairingHeap() = default;

  PairingHeap(const PairingHeap &) = delete;
  PairingHeap &operator=(const PairingHeap &) = delete;
  PairingHeap(PairingHeap &&) = default;
  PairingHeap &operator=(PairingHeap &&) = default;

  // Inserts the item `x`. Caller must not mutate the key or pointers directly
  // as long as this heap contains `x`.
  void Insert(Item x);

  // Removes and returns one of the contained items with the minimum key. It
  // will have null tree pointers.
  Item DeleteMin(void);

  // Requires `key <= x->key`.
  void DecreaseKey(Item x, double augment, double dist);

  // O(n); for tests.
  std::size_t CountWhileCheckingInvariant() const;

  // Is heap empty?
  bool IsEmpty();

private:
  Item *root_ = nullptr;
};
