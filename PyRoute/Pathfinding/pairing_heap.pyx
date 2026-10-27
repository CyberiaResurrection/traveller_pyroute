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

    def decrease_key(self, long curnode, double augment, double dist, long nu_parent):
        cdef Item *node = <Item *> <size_t> self.handles[curnode]
        # Check DecreaseKey's precondition here, as the C++ asserts are compiled out.
        if augment > node.augment or (augment == node.augment and dist > node.dist):
            raise ValueError(f"({augment}, {dist}) is not a decrease from ({node.augment}, {node.dist})")
        self.heap.DecreaseKey(node, augment, dist, nu_parent)

    def is_empty(self):
        return self.heap.IsEmpty()

    def count_checking_invariant(self):
        return self.heap.CountWhileCheckingInvariant()


cdef class ItemWrapper:
    cdef Item* _ptr

    def __cinit__(self):
        # The pointer is assigned by the Cython factory below.
        self._ptr = NULL

    # Deliberately no __dealloc__ that frees _ptr.

    property augment:
        def __get__(self):
            self._check_valid()
            return self._ptr.augment

        def __set__(self, double value):
            self._check_valid()
            self._ptr.augment = value

    property dist:
        def __get__(self):
            self._check_valid()
            return self._ptr.dist

        def __set__(self, double value):
            self._check_valid()
            self._ptr.dist = value

    property curnode:
        def __get__(self):
            self._check_valid()
            return self._ptr.curnode

        def __set__(self, long value):
            self._check_valid()
            self._ptr.curnode = value

    property parent_node:
        def __get__(self):
            self._check_valid()
            return self._ptr.parent_node

        def __set__(self, long value):
            self._check_valid()
            self._ptr.parent_node = value

    cdef void _check_valid(self):
        if self._ptr == NULL:
            raise ReferenceError("Item pointer is NULL")


cdef ItemWrapper wrap_item(Item*ptr):
    cdef ItemWrapper wrapper
    if ptr == NULL:
        raise ValueError("Cannot wrap a NULL Item pointer")
    wrapper = ItemWrapper()
    wrapper._ptr = ptr
    return wrapper
