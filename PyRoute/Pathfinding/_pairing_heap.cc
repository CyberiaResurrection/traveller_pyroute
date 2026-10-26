#include "pairing_heap.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {

bool IsRootOfTree(Item *x) {
  return x->sibling == nullptr && x->parent == nullptr;
}

void Attach(Item **parent, Item *x) {
  assert(*parent == nullptr);
  assert(x == nullptr || x->parent == nullptr);
  *parent = x;
  if (x != nullptr) {
    x->parent = parent;
  }
}

Item *Detach(Item **parent) {
  Item *x = *parent;
  *parent = nullptr;
  if (x != nullptr) {
    x->parent = nullptr;
  }
  return x;
}

Item *Link(Item *x, Item *y) {
  assert(x != y);
  assert(IsRootOfTree(x));
  assert(IsRootOfTree(y));
  if (*y < *x) {
    std::swap(x, y);
  }
  Attach(&y->sibling, Detach(&x->child));
  Attach(&x->child, y);
  return x;
}

Item *LinkPairsAndReverse(Item *head) {
  assert(head->parent == nullptr);
  Item *x = head;
  Item *previous = nullptr;
  do {
    Item *y = Detach(&x->sibling);
    if (y == nullptr) {
      Attach(&x->sibling, previous);
      return x;
    }
    Item *z = Detach(&y->sibling);
    x = Link(x, y);
    Attach(&x->sibling, previous);
    previous = x;
    x = z;
  } while (x != nullptr);
  assert(previous != nullptr);
  return previous;
}

Item *LinkAll(Item *x) {
  Item *y = Detach(&x->sibling);
  while (y != nullptr) {
    Item *z = Detach(&y->sibling);
    x = Link(x, y);
    y = z;
  }
  return x;
}

} // namespace

PairingHeap::~PairingHeap() { Clear(); }

void PairingHeap::Clear() {
  // Iterative, so deep trees can't overflow the stack.
  std::vector<Item *> pending;
  if (root_ != nullptr) {
    pending.push_back(root_);
  }
  while (!pending.empty()) {
    Item *x = pending.back();
    pending.pop_back();
    if (x->child != nullptr) {
      pending.push_back(x->child);
    }
    if (x->sibling != nullptr) {
      pending.push_back(x->sibling);
    }
    delete x;
  }
  root_ = nullptr;
}

Item *PairingHeap::Insert(Item item) {
  Item *x = new Item(item);
  x->child = nullptr;
  x->sibling = nullptr;
  x->parent = nullptr;
  if (root_ != nullptr) {
    root_ = Link(root_, x);
  } else {
    root_ = x;
  }
  return x;
}

Item PairingHeap::DeleteMin() {
  assert(root_ != nullptr);
  assert(IsRootOfTree(root_));
  Item *child = Detach(&root_->child);
  if (child != nullptr) {
    child = LinkAll(LinkPairsAndReverse(child));
    assert(IsRootOfTree(child));
  }
  Item *min = root_;
  root_ = child;
  assert(min->child == nullptr);
  assert(min->sibling == nullptr);
  assert(min->parent == nullptr);
  Item result = *min;
  delete min;
  return result;
}

void PairingHeap::DecreaseKey(Item *x, double augment, double dist, long nu_parent) {
  assert(root_ != nullptr);
  assert(IsRootOfTree(root_));
  assert(!std::isnan(augment));
  assert(!std::isnan(dist));
  assert(augment <= x->augment);
  assert(augment < x->augment || dist <= x->dist);
  x->augment = augment;
  x->dist = dist;
  x->parent_node = nu_parent;
  Item **parent = x->parent;
  if (parent == nullptr) {
    assert(x == root_);
    return;
  }
  assert(*parent == x);
  Detach(parent);
  Attach(parent, Detach(&x->sibling));
  root_ = Link(root_, x);
}

bool PairingHeap::IsEmpty() { return root_ == nullptr; }

namespace {

std::size_t Walk(Item const *min, Item *const *parent, Item const *x) {
  if (x == nullptr) {
    return 0;
  }
  if (x->parent != parent) {
    std::abort();
  }
  if (min != nullptr && *x < *min) {
    std::abort();
  }
  return 1 + Walk(x, &x->child, x->child) + Walk(min, &x->sibling, x->sibling);
}

} // namespace

std::size_t PairingHeap::CountWhileCheckingInvariant() const {
  if (root_ != nullptr && !IsRootOfTree(root_)) {
    std::abort();
  }
  return Walk(nullptr, nullptr, root_);
}
