# distutils: language = c++
# Thanks to @GamesByDavidE for providing this code

from libcpp cimport bool

cdef extern from "pairing_heap.h" namespace "":
    cdef struct Item:
        double key;
        double augment;
        double dist;
        long curnode;
        long parent_node;
        Item *child;
        Item *sibling;
        Item **parent;

    cdef cppclass PairingHeap:
        PairingHeap()
        Item *Insert(Item)
        Item DeleteMin()
        void DecreaseKey(Item *, double, double)
        size_t CountWhileCheckingInvariant()
        bool IsEmpty()
