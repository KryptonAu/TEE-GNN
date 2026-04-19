#pragma once

#include <Eigen/Dense>
#include <iostream>
#include "graph.h"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

Matrix message_passing_layer(const Graph &g, const Matrix &H) {
    auto n = g.num_vertices();
    Matrix result = Eigen::MatrixXd::Zero(H.rows(), H.cols());
    for (int u = 0; u < n; ++u) {
        for (auto v : g.neighbors(u)) {
            result.row(u) += H.row(v);
        }
    }
    return result;
}

Matrix ReLU(const Matrix &input) {
    return input.unaryExpr([](double x) { return std::max(0.0, x); });
}

Matrix Softmax(const Matrix &input) {
    Matrix output = input;
    for (int i = 0; i < input.rows(); ++i) {
        double max_val = input.row(i).maxCoeff();
        auto tmp_array = (input.row(i).array() - max_val).exp();
        double sum_exp = tmp_array.sum();
        output.row(i) = tmp_array / sum_exp;
    }
    return output;
}

void degree_normalization(const Vector &sqrt_degree, Matrix &input) {
    for (int i = 0; i < input.rows(); ++i) {
        if (sqrt_degree(i) > 0) {
            input.row(i) /= sqrt_degree(i);
        }
    }
}

Matrix GCN_inference(const Graph &g, const Matrix &features, const std::vector<Matrix> &W) {
    auto k = W.size();
    Matrix output = features;
    Vector sqrt_degree(g.num_vertices());
    for (int i = 0; i < g.num_vertices(); ++i) {
        sqrt_degree(i) = std::sqrt(static_cast<double>(g.degree(i)));
        // std::cout<<g.degree(i)<<' ';
    }
    // std::cout<<'\n';
    for (size_t i = 0; i < k; ++i) {
        Matrix tmp = output * W[i];
        degree_normalization(sqrt_degree, tmp);
        output = message_passing_layer(g, tmp);
        degree_normalization(sqrt_degree, output);
        // std::cout << output << std::endl;
        if (i != k - 1) {
            output = ReLU(output);
        } else {
            output = Softmax(output);
        }
    }
    return output;
}
