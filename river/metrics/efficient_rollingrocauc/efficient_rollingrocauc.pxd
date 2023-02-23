from libcpp.vector cimport vector

cdef extern from "cpp/RollingROCAUC.cpp":
    pass

cdef extern from "cpp/RollingROCAUC.hpp" namespace "rollingrocauc":
    cdef cppclass RollingROCAUC:
        RollingROCAUC(int positiveLabel, int windowSize) except +
        void update(int label, double score)
        void revert(int label, double score)
        double get()
        vector[int] getTrueLabels()
        vector[double] getScores()
