#include <iostream>
#include <eigen3/Eigen/Dense>

#include "masks.hpp"

using Matrix = Eigen::MatrixXd;

int main() {
    teegnn::RandomEngine rng(114514, 0, "teegnn-sim");
    teegnn::SDIMMask L = teegnn::SDIMMask::random(3, rng);
    teegnn::SDIMMask R = teegnn::SDIMMask::random(4, rng);
    teegnn::SDIMMask M3 = teegnn::SDIMMask::random(5, rng);
    Matrix A(3, 4);
    Matrix B(4, 5);
    for (int i = 0; i < A.rows(); ++i){
        for (int j = 0; j < A.cols(); ++j) {
            A(i, j) = rng.random_matrix_value();
            // A(i, j) = 1;
        }
    }
    for (int i = 0; i < B.rows(); ++i){
        for (int j = 0; j < B.cols(); ++j) {
            B(i, j) = rng.random_matrix_value();
        }
    }

    for (int i = 0; i < 10; ++i) {
        std::cout << rng.random_matrix_value()<<' ';
    }
    std::cout<<'\n';
    
    Matrix masked_a = teegnn::apply_SDIM(L, R, A);
    Matrix masked_b = teegnn::apply_SDIM(R, M3, B);
    std::cout<< A << '\n';
    std::cout<< A - teegnn::apply_SDIM_inv(L, R, masked_a) << '\n';
    std::cout<< A * B - teegnn::apply_SDIM_inv(L, M3, masked_a * masked_b) << '\n';
    

    return 0;
}