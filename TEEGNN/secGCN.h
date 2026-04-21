#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>

#include "GCN.h"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// 以下代码之后会移植到TEE
// 封装为一个类
class secure_computation {
public:
    secure_computation() = default;
    secure_computation(std::vector<std::vector<int>> vals, std::vector<std::vector<int>> perms, std::vector<int> degree)
        : values(std::move(vals)), perms(std::move(perms)) {
        int n = degree.size();
        sqrt_degree = Eigen::VectorXd::Zero(n);
        for (int i = 0; i < n; ++i) {
            sqrt_degree(i) = std::sqrt(static_cast<double>(degree[i]));
        }
    }
    void non_linear_layer(int k, Matrix &input1, Matrix &input2, const std::string &activation) {
        Matrix input = decrypt(input1, 0, 5 + 4 * k) + decrypt(input2, 2, 7 + 4 * k);
        degree_normalization(input);
        std::cout << "第" << k << "层运算结果还原已完成\n";
        // std::cout << input << '\n';
        if (activation == "ReLU") {
            Matrix activated = ReLU(input);
            degree_normalization(activated);
            input1 = encrypt(activated, 1, 8 + 4 * k);
            input2 = encrypt(activated, 3, 10 + 4 * k);
        } else if (activation == "Softmax") {
            input1 = input2 = Softmax(input);
        } else {
            throw std::invalid_argument("Unsupported activation function");
        }
    }
private:
    std::vector<std::vector<int>> values;
    std::vector<std::vector<int>> perms;
    Vector sqrt_degree;
    Matrix encrypt(const Matrix &input, int x, int y) {
        int r = input.rows();
        int c = input.cols();
        Matrix encrypted = Eigen::MatrixXd::Zero(r, c);
        for (int i = 0; i < r; ++i) {
            int new_i = perms[x][i];
            for (int j = 0; j < c; ++j) {
                int new_j = perms[y][j];
                encrypted(i, j) = input(new_i, new_j) * values[x][new_i] / values[y][new_j];
            }
        }
        return encrypted;
    }

    Matrix decrypt(const Matrix &input, int x, int y) {
        int r = input.rows();
        int c = input.cols();
        Matrix decrypted = Eigen::MatrixXd::Zero(r, c);
        for (int i = 0; i < r; ++i) {
            int new_i = perms[x][i];
            for (int j = 0; j < c; ++j) {
                int new_j = perms[y][j];
                decrypted(new_i, new_j) = input(i, j) * values[y][new_j] / values[x][new_i];
            }
        }
        return decrypted;
    }
    void degree_normalization(Matrix &input) {
        for (int i = 0; i < input.rows(); ++i) {
            if (sqrt_degree(i) > 0) {
                input.row(i) /= sqrt_degree(i);
            }
        }
    }
};


// -------------------------------------------------------------------------------

Matrix secure_message_passing_layer(const Weighted_Graph &g, const Matrix &H) {
    auto n = g.num_vertices();
    Matrix result = Eigen::MatrixXd::Zero(H.rows(), H.cols());
    for (int u = 0; u < n; ++u) {
        for (auto [v, w] : g.neighbors(u)) {
            result.row(u) += H.row(v) * w;
        }
    }
    return result;
}

Matrix secure_GCN_inference(const Weighted_Graph &g1, const Weighted_Graph &g2,
             const std::vector<Matrix> &features, const std::vector<Matrix> &W, secure_computation &sec_comp) {
    auto k = W.size() / 2;
    Matrix output1 = features[0];
    Matrix output2 = features[1];

    for (size_t i = 0; i < k; ++i) {
        output1 = secure_message_passing_layer(g1, output1 * W[2 * i]);
        output2 = secure_message_passing_layer(g2, output2 * W[2 * i + 1]);
        std::cout << "第" << i << "层线性部分计算已完成\n";
        if (i != k - 1) {
            sec_comp.non_linear_layer(i, output1, output2, "ReLU");
            // std::cout << output1 <<'\n';
            // std::cout << output2 <<'\n';
        } else {
            sec_comp.non_linear_layer(i, output1, output2, "Softmax");
        }
        std::cout << "第" << i << "层计算已完成\n";
    }
    return output1;
}

