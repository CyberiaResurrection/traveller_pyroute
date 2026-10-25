# distutils: language = c++
# distutils: sources = _pairing_heap.cc

from pairing_heap cimport PairingHeap, Item


cdef class PairingHeapHarness:
    """
    Python-facing wrapper around the C++ PairingHeap, so it can be exercised from tests.
    Items are identified by curnode, which must be unique among the items currently in the heap.
    """
    cdef PairingHeap heap
    cdef dict handles  # curnode -> address of the heap's node for that item

    def __cinit__(self):
        self.handles = {}

    def insert(self, double augment, double dist, long curnode, long parent_node):
        cdef Item item
        if curnode in self.handles:
            raise ValueError(f"curnode {curnode} already in heap")
        item.augment = augment
        item.dist = dist
        item.curnode = curnode
        item.parent_node = parent_node
        self.handles[curnode] = <size_t> self.heap.Insert(item)

    def delete_min(self):
        cdef Item item
        if self.heap.IsEmpty():
            raise IndexError("delete_min from empty heap")
        item = self.heap.DeleteMin()
        del self.handles[item.curnode]
        return item.augment, item.dist, item.curnode, item.parent_node

    def decrease_key(self, long curnode, double augment, double dist):
        cdef Item *node = <Item *> <size_t> self.handles[curnode]
        # Check DecreaseKey's precondition here, as the C++ asserts are compiled out.
        if augment > node.augment or (augment == node.augment and dist > node.dist):
            raise ValueError(f"({augment}, {dist}) is not a decrease from ({node.augment}, {node.dist})")
        self.heap.DecreaseKey(node, augment, dist)

    def is_empty(self):
        return self.heap.IsEmpty()

    def count_checking_invariant(self):
        return self.heap.CountWhileCheckingInvariant()
