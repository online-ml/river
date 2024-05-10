from libcpp.vector cimport vector

cdef extern from "cpp/RollingPRAUC.cpp":
    pass

cdef extern from "cpp/RollingPRAUC.hpp" namespace "rollingprauc":
    cdef cppclass RollingPRAUC:
        RollingPRAUC(int positiveLabel, int windowSize) except +
        void update(int label, double score)
        void revert(int label, double score)
        double get()
        vector[int] getTrueLabels()
        vector[double] getScores()
