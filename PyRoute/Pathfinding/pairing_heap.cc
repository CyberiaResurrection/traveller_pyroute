#include "pairing_heap.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <utility>

namespace {

bool IsRootOfTree(Item x) {
  return x->sibling == nullptr && x->parent == nullptr;
}

void Attach(Item parent, Item x) {
  assert(*parent == nullptr);
  assert(x == nullptr || x->parent == nullptr);
  *parent = x;
  if (x != nullptr) {
    x->parent = parent;
  }
}

Item Detach(Item parent) {
  Item x = *parent;
  *parent = nullptr;
  if (x != nullptr) {
    x->parent = nullptr;
  }
  return x;
}

Item Link(Item x, Item y) {
  assert(x != y);
  assert(IsRootOfTree(x));
  assert(IsRootOfTree(y));
  if (y < x) {
    std::swap(x, y);
  }
  Attach(&y->sibling, Detach(&x->child));
  Attach(&x->child, y);
  return x;
}

Item LinkPairsAndReverse(Item head) {
  assert(head->parent == nullptr);
  Item x = head;
  Item previous = nullptr;
  do {
    Item y = Detach(&x->sibling);
    if (y == nullptr) {
      Attach(&x->sibling, previous);
      return x;
    }
    Item z = Detach(&y->sibling);
    x = Link(x, y);
    Attach(&x->sibling, previous);
    previous = x;
    x = z;
  } while (x != nullptr);
  assert(previous != nullptr);
  return previous;
}

Item LinkAll(Item x) {
  Item y = Detach(&x->sibling);
  while (y != nullptr) {
    Item z = Detach(&y->sibling);
    x = Link(x, y);
    y = z;
  }
  return x;
}

} // namespace

void PairingHeap::Insert(Item x) {
  assert(!std::isnan(x->key));
  x->child = nullptr;
  x->sibling = nullptr;
  x->parent = nullptr;
  if (root_ != nullptr) {
    root_ = Link(root_, x);
  } else {
    root_ = x;
  }
}

Item PairingHeap::DeleteMin() {
  if (root_ == nullptr) {
    return nullptr;
  }
  assert(IsRootOfTree(root_));
  Item child = Detach(&root_->child);
  if (child != nullptr) {
    child = LinkAll(LinkPairsAndReverse(child));
    assert(IsRootOfTree(child));
  }
  Item min = root_;
  root_ = child;
  assert(min->child == nullptr);
  assert(min->sibling == nullptr);
  assert(min->parent == nullptr);
  return min;
}

void PairingHeap::DecreaseKey(Item x, double augment, double dist) {
  assert(root_ != nullptr);
  assert(IsRootOfTree(root_));
  assert(!std::isnan(augment));
  assert(!std::isnan(dist));
  assert(augment <= x->augment)
  if augment == x->augment:
    assert(dist <= x->dist)
  x->augment = augment;
  x->dist = dist;
  Item parent = x->parent;
  if (parent == nullptr) {
    assert(x == root_);
    return;
  }
  assert(*parent == x);
  Detach(parent);
  Attach(parent, Detach(&x->sibling));
  root_ = Link(root_, x);
}

bool PairingHeap::IsEmpty() {
    return root_ == nullptr
}

namespace {

std::size_t Walk(double min_key, Item const parent,
                 Item const x) {
  if (x == nullptr) {
    return 0;
  }
  if (x->parent != parent) {
    std::abort();
  }
  if (!(min_key <= x->key)) {
    std::abort();
  }
  return 1 + Walk(x->key, &x->child, x->child) +
         Walk(min_key, &x->sibling, x->sibling);
}

} // namespace

std::size_t PairingHeap::CountWhileCheckingInvariant() const {
  if (root_ != nullptr && !IsRootOfTree(root_)) {
    std::abort();
  }
  return Walk(-std::numeric_limits<double>::infinity(), nullptr, root_);
}
