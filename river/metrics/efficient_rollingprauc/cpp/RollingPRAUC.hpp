#ifndef ROLLINGPRAUC_HPP
#define ROLLINGPRAUC_HPP

#include <deque>
#include <set>
#include <tuple>
#include <vector>

namespace rollingprauc {

class RollingPRAUC {
    public:
        RollingPRAUC();
        RollingPRAUC(const int positiveLabel, const long unsigned windowSize);

        virtual ~RollingPRAUC() = default;

        // Calls insert() and removeLast() if needed
        virtual void update(const int label, const double score);

        // Erase the most recent instance with content equal to params
        virtual void revert(const int label, const double score);

        // Calculates the PRAUC and returns it
        virtual double get() const;

        // Returns y_true as a vector
        virtual std::vector<int> getTrueLabels() const;

        // Returns y_score as a vector
        virtual std::vector<double> getScores() const;

    private:
        // Insert instance based on params
        virtual void insert(const int label, const double score);

        // Remove oldest instance
        virtual void removeLast();

        // Calculates the trapezoid area
        double trapzArea(double x1, double x2, double y1, double y2) const;

        int positiveLabel;

        std::size_t windowSize;
        std::size_t positives;

        // window maintains a queue of the instances to store the temporal
        // aspect of the stream. Using deque to allow revert()
        std::deque<std::tuple<double, int>> window;

        // orderedWindow maintains a multiset (implemented as a tree)
        // to store the ordered instances
        std::multiset<std::tuple<double, int>> orderedWindow;
};

} // namespace rollingprauc

#endif
