#pragma once

#include <eigen3/Eigen/Dense>

#include <cstdint>
#include <string>
#include <vector>
#include <chrono>

namespace teegnn {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using IntVector = std::vector<int>;

struct TimerRecord {
    std::string name;
    double milliseconds = 0.0;
};

class ScopedTimer {
public:
    ScopedTimer(std::vector<TimerRecord>& records, std::string name)
        : records_(records), name_(std::move(name)), start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        const auto end = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        records_.push_back({name_, ms});
    }

private:
    std::vector<TimerRecord>& records_;
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace teegnn

