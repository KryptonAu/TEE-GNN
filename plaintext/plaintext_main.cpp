#include "dataset_loader.hpp"
#include "gcn_ops.hpp"

#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>

namespace {

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " /dataset/cora\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        print_usage(argv[0]);
        return 2;
    }

    try {
        const auto total_start = std::chrono::steady_clock::now();
        const auto load_start = std::chrono::steady_clock::now();
        teegnn::Dataset dataset = teegnn::load_dataset(argv[1]);
        const auto load_end = std::chrono::steady_clock::now();

        teegnn::InferenceResult result = teegnn::run_plaintext_inference(dataset);
        const auto total_end = std::chrono::steady_clock::now();

        const double load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
        const double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        std::cout << std::fixed << std::setprecision(8);
        std::cout << "nodes: " << dataset.graph.num_nodes() << "\n";
        std::cout << "directed_edges_without_self: " << dataset.graph.raw_directed_edges() << "\n";
        std::cout << "normalized_support_edges: " << dataset.graph.edges().size() << "\n";
        std::cout << "feature_dim: " << dataset.features.cols() << "\n";
        std::cout << "hidden_dim: " << dataset.w1.cols() << "\n";
        std::cout << "class_count: " << dataset.w2.cols() << "\n";
        std::cout << "accuracy: " << result.accuracy << "\n";
        std::cout << "timing_ms.load_dataset: " << load_ms << "\n";
        for (const auto& record : result.timings) {
            std::cout << "timing_ms." << record.name << ": " << record.milliseconds << "\n";
        }
        std::cout << "timing_ms.total: " << total_ms << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "gcn_infer: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}

