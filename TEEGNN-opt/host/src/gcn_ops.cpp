#include "gcn_ops.hpp"

#include <stdexcept>

namespace teegnn {

Matrix relu(const Matrix& input) {
    return input.cwiseMax(0.0);
}
Matrix softmax(const Matrix &input) {
    Matrix output = input;
    for (int i = 0; i < input.rows(); ++i) {
        double max_val = input.row(i).maxCoeff();
        auto tmp_array = (input.row(i).array() - max_val).exp();
        double sum_exp = tmp_array.sum();
        output.row(i) = tmp_array / sum_exp;
    }
    return output;
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
    Matrix H = dataset.features;
    for (int layer = 0; layer < 2; ++layer) {
        if (layer == 0) {
            H = sparse_dense_mul(dataset.graph, H * dataset.w1);
            H = relu(H);
        } else {
            H = sparse_dense_mul(dataset.graph, H * dataset.w2);
            H = softmax(H);
        }
    }
    result.logits = H;
    return result;
}

}  // namespace teegnn

