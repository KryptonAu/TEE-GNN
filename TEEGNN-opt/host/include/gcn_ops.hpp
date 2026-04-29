#pragma once

#include "dataset_loader.hpp"
#include "types.hpp"

#include <vector>

namespace teegnn {

struct InferenceResult {
    Matrix logits;
    IntVector predictions;
    double accuracy = 0.0;
    std::vector<TimerRecord> timings;
};

Matrix relu(const Matrix& input);
IntVector argmax_rows(const Matrix& logits);
double accuracy(const IntVector& predictions, const IntVector& labels);
InferenceResult run_plaintext_inference(const Dataset& dataset);

}  // namespace teegnn

