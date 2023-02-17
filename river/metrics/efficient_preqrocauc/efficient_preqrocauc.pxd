cdef extern from "cpp/PreqROCAUC.cpp":
    pass

cdef extern from "cpp/PreqROCAUC.hpp" namespace "preqrocauc":
    cdef cppclass PreqROCAUC:
        PreqROCAUC(int positiveLabel, int windowSize) except +
        void update(int label, double score)
        void revert(int label, double score)
        double get()
