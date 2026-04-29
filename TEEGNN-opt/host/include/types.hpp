#pragma once

#include <Eigen/Dense>

#include <Eigen/src/Core/Matrix.h>
#include <cstdint>
#include <string>
#include <vector>

namespace teegnn {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using IntVector = std::vector<int>;

struct TimerRecord {
    std::string name;
    double milliseconds = 0.0;
};

}  // namespace teegnn

