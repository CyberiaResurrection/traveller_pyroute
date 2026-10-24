# distutils: language = c++
# Thanks to @GamesByDavidE for providing this code

from libcpp.vector cimport vector

cdef extern from "pairing_heap.h" namespace "pairing_heap":
    cdef struct Item:
        double key;
        Item *child;
        Item *sibling;
        Item *parent;

    cdef cppclass PairingHeap:
        PairingHeap()
        void Insert(Item)
        Item DeleteMin()
        DecreaseKey(Item, double)
