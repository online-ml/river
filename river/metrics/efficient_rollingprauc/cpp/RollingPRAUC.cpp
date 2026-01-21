#include "RollingPRAUC.hpp"

#include <limits>
#include <stdlib.h>

namespace rollingprauc {

RollingPRAUC::RollingPRAUC(): positiveLabel{1}, windowSize{1000}, positives{0} {
}

RollingPRAUC::RollingPRAUC(int positiveLabel, long unsigned windowSize):
    positiveLabel{positiveLabel}, windowSize{windowSize}, positives{0} {
}

void RollingPRAUC::update(int label, double score) {
    if (this->window.size() == this->windowSize)
        this->removeLast();

    this->insert(label, score);

    return;
}

void RollingPRAUC::revert(int label, double score) {
    int normalizedLabel = 0;
    if (label == this->positiveLabel)
        normalizedLabel = 1;

    std::deque<std::tuple<double, int>>::const_iterator it{this->window.cbegin()};
    for (; it != this->window.cend(); ++it)
        if (std::get<0>(*it) == score && std::get<1>(*it) == normalizedLabel)
            break;

    if (it == this->window.cend())
        return;

    if (normalizedLabel)
        this->positives--;

    this->window.erase(it);

    std::multiset<std::tuple<double, int>>::const_iterator itr{
        this->orderedWindow.find(std::make_tuple(score, label))
    };
    this->orderedWindow.erase(itr);

    return;
}

double RollingPRAUC::get() const {
    unsigned long windowSize{this->window.size()};

    // If there is only one class in the window, it will lead to a
    // division by zero. So, zero is returned.
    if (!this->positives || !(windowSize - this->positives))
        return 0;

    unsigned long fp{windowSize - this->positives};
    unsigned long tp{this->positives}, tpPrev{tp};

    double auc{0}, scorePrev{std::numeric_limits<double>::max()};

    double prec{tp / (double) (tp + fp)}, precPrev{prec};

    std::multiset<std::tuple<double, int>>::const_iterator it{this->orderedWindow.begin()};
    double score;
    int label;

    for (; it != this->orderedWindow.end(); ++it) {
        score = std::get<0>(*it);
        label = std::get<1>(*it);

        if (score != scorePrev) {
            prec = tp / (double) (tp + fp);

            if (precPrev > prec)
                prec = precPrev; // Monotonic. decreasing

            auc += this->trapzArea(tp, tpPrev, prec, precPrev);

            scorePrev = score;
            tpPrev = tp;
            precPrev = prec;
        }

        if (label) tp--;
        else fp--;
    }

    auc += this->trapzArea(tp, tpPrev, 1.0, precPrev);

    return auc / this->positives; // Scale the x axis
}

void RollingPRAUC::insert(int label, double score) {
    // Normalize label to 0 (negative) or 1 (positive)
    int l = 0;
    if (label == this->positiveLabel) {
        l = 1;
        this->positives++;
    }

    this->window.emplace_back(score, l);
    this->orderedWindow.emplace(score, l);

    return;
}

void RollingPRAUC::removeLast() {
    std::tuple<double, int> last{this->window.front()};

    if (std::get<1>(last))
        this->positives--;

    this->window.pop_front();

    // Erase using a iterator to avoid multiple erases with equivalent instances
    std::multiset<std::tuple<double, int>>::iterator it{
        this->orderedWindow.find(last)
    };
    this->orderedWindow.erase(it);

    return;
}

std::vector<int> RollingPRAUC::getTrueLabels() const {
    std::vector<int> trueLabels;

    std::deque<std::tuple<double, int>>::const_iterator it{this->window.begin()};
    for (; it != this->window.end(); ++it)
        trueLabels.push_back(std::get<1>(*it));

    return trueLabels;
}

std::vector<double> RollingPRAUC::getScores() const {
    std::vector<double> scores;

    std::deque<std::tuple<double, int>>::const_iterator it{this->window.begin()};
    for (; it != this->window.end(); ++it)
        scores.push_back(std::get<0>(*it));

    return scores;
}

double RollingPRAUC::trapzArea(double x1, double x2, double y1, double y2) const {
    return abs(x1 - x2) * (y1 + y2) / 2;
}

} // namespace rollingprauc
