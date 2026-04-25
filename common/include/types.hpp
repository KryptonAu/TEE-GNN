#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <string>
#include <vector>

namespace teegnn {

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::VectorXd;
using IntVector = std::vector<int>;

struct TimerRecord {
    std::string name;
    double milliseconds = 0.0;
};

}  // namespace teegnn

