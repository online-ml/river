#include "RollingROCAUC.hpp"

namespace rollingrocauc {

RollingROCAUC::RollingROCAUC(): positiveLabel{1}, windowSize{1000}, positives{0} {
}

RollingROCAUC::RollingROCAUC(int positiveLabel, long unsigned windowSize):
    positiveLabel{positiveLabel}, windowSize{windowSize}, positives{0} {
}

void RollingROCAUC::update(int label, double score) {
    if (this->window.size() == this->windowSize)
        this->removeLast();

    this->insert(label, score);

    return;
}

void RollingROCAUC::revert(int label, double score) {
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

double RollingROCAUC::get() const {
    // If there is only one class in the window, it will lead to a
    // division by zero. So, zero is returned.
    if (!this->positives || !(this->orderedWindow.size() - this->positives))
        return 0;

    double auc{0}, score, lastPosScore{-1};
    int c{0}, prevC{0}, label;

    std::multiset<std::tuple<double, int>>::const_reverse_iterator it{this->orderedWindow.crbegin()};
    for (; it != this->orderedWindow.crend(); ++it) {
        score = std::get<0>(*it);
        label = std::get<1>(*it);

        if (label) {
            if (score != lastPosScore) {
                prevC = c;
                lastPosScore = score;
            }

            c++;
        } else {
            if (score == lastPosScore)
                auc += (c + prevC) / 2;
            else
                auc += c;
        }
    }

    auc /= this->positives * (this->orderedWindow.size() - this->positives);

    return auc;
}

void RollingROCAUC::insert(int label, double score) {
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

void RollingROCAUC::removeLast() {
    std::tuple<double, int> last{this->window.front()};

    if (std::get<1>(last))
        this->positives--;

    this->window.pop_front();

    // Erase using a iterator to avoid multiple erases with equivalent instances
    std::multiset<std::tuple<double, int>>::const_iterator it{
        this->orderedWindow.find(std::make_tuple(std::get<0>(last), std::get<1>(last)))
    };
    this->orderedWindow.erase(it);

    return;
}

std::vector<int> RollingROCAUC::getTrueLabels() const {
    std::vector<int> trueLabels;

    std::deque<std::tuple<double, int>>::const_iterator it{this->window.begin()};
    for (; it != this->window.end(); ++it)
        trueLabels.push_back(std::get<1>(*it));

    return trueLabels;
}

std::vector<double> RollingROCAUC::getScores() const {
    std::vector<double> scores;

    std::deque<std::tuple<double, int>>::const_iterator it{this->window.begin()};
    for (; it != this->window.end(); ++it)
        scores.push_back(std::get<0>(*it));

    return scores;
}

} // namespace rollingrocauc
