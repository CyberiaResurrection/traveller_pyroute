# distutils: language = c++
# Thanks to @GamesByDavidE for providing this code

from libcpp cimport bool

cdef extern from "pairing_heap.h" namespace "":
    cdef struct Item:
        double augment;
        double dist;
        int curnode;
        int parent_node;
        Item *child;
        Item *sibling;
        Item *parent;

    cdef cppclass PairingHeap:
        PairingHeap()
        void Insert(Item)
        Item DeleteMin()
        DecreaseKey(Item, double, double)
        bool IsEmpty()
