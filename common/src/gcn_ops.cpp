#include "gcn_ops.hpp"

#include <chrono>
#include <stdexcept>

namespace teegnn {
namespace {

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

}  // namespace

Matrix relu(const Matrix& input) {
    return input.cwiseMax(0.0);
}

IntVector argmax_rows(const Matrix& logits) {
    IntVector pred(static_cast<std::size_t>(logits.rows()));
    for (int i = 0; i < logits.rows(); ++i) {
        Eigen::Index index = 0;
        logits.row(i).maxCoeff(&index);
        pred[static_cast<std::size_t>(i)] = static_cast<int>(index);
    }
    return pred;
}

double accuracy(const IntVector& predictions, const IntVector& labels) {
    if (predictions.size() != labels.size()) {
        throw std::runtime_error("prediction count does not match label count");
    }
    std::size_t correct = 0;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correct += 1;
        }
    }
    return labels.empty() ? 0.0 : static_cast<double>(correct) / static_cast<double>(labels.size());
}

InferenceResult run_plaintext_inference(const Dataset& dataset) {
    InferenceResult result;
    Matrix ax;
    Matrix hidden_linear;
    Matrix hidden;
    Matrix ah;

    {
        ScopedTimer timer(result.timings, "A_hat * X");
        ax = sparse_dense_mul(dataset.graph, dataset.features);
    }
    {
        ScopedTimer timer(result.timings, "(A_hat * X) * W1");
        hidden_linear = ax * dataset.w1;
    }
    {
        ScopedTimer timer(result.timings, "ReLU");
        hidden = relu(hidden_linear);
    }
    {
        ScopedTimer timer(result.timings, "A_hat * H");
        ah = sparse_dense_mul(dataset.graph, hidden);
    }
    {
        ScopedTimer timer(result.timings, "(A_hat * H) * W2");
        result.logits = ah * dataset.w2;
    }
    {
        ScopedTimer timer(result.timings, "argmax + accuracy");
        result.predictions = argmax_rows(result.logits);
        result.accuracy = accuracy(result.predictions, dataset.labels);
    }
    return result;
}

}  // namespace teegnn

