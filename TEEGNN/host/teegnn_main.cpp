#include "dataset_loader.hpp"
#include "gcn_ops.hpp"
#include "util.hpp"
#include "types.hpp"
#include "tee_gnn_client.hpp"

#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct InferencePhaseResult {
    teegnn::Matrix logits;
};

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " /dataset/cora [--confusion-rate 0.2] [--mask-rank 2] [--seed 1234]\n";
}

teegnn::Options parse_args(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        throw std::runtime_error("missing dataset directory");
    }
    teegnn::Options options;
    options.dataset_dir = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--seed" && i + 1 < argc) {
            options.seed_data = static_cast<std::uint64_t>(std::stoull(argv[++i]));
        } else {
            print_usage(argv[0]);
            throw std::runtime_error("unknown or incomplete option: " + arg);
        }
    }
    return options;
}

InferencePhaseResult run_secure_inference(teegnn::MaskedData& masked_data,
                                          const std::unique_ptr<teegnn::TEEGNNClient> &secure) {
    InferencePhaseResult result;

    size_t num_nodes = masked_data.num_nodes;
    
    teegnn::Matrix output;
    teegnn::Matrix &y = result.logits;
    y = masked_data.features;
    for (int layer = 0; layer < 2; ++layer) {
        y *= masked_data.weights[layer];
        secure->secure_compute(masked_data.graphs[layer].get(), y);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const teegnn::Options options = parse_args(argc, argv);
        std::vector<teegnn::TimerRecord> timings;
        const auto total_start = std::chrono::steady_clock::now();

        teegnn::Dataset dataset;
        {
            teegnn::ScopedTimer timer(timings, "load_dataset");
            dataset = teegnn::load_dataset(options.dataset_dir);
        }

        teegnn::InferenceResult plaintext;
        {
            teegnn::ScopedTimer timer(timings, "plaintext_inference");
            plaintext = teegnn::run_plaintext_inference(dataset);
        }

        teegnn::MaskPhaseResult masks;
        {
            teegnn::ScopedTimer timer(timings, "mask_phase");
            masks = teegnn::run_mask_phase(dataset, options);
        }

        InferencePhaseResult inference;
        {
            std::unique_ptr<teegnn::TEEGNNClient> secure = std::make_unique<teegnn::TEEGNNClient>();
            if (!secure->initialize()) {
                throw std::runtime_error("Failed to initialize TEE client");
            }
            teegnn::ScopedTimer timer(timings, "teegnn_inference");
            if (!secure->init_GNNContext(masks.data.weights[0], masks.secrets,
                    masks.data.feature_dim, masks.data.hidden_dim,
                    masks.data.num_nodes, masks.data.class_dim)) {
                throw std::runtime_error("Failed to initialize GNN context in TEE");
            }
            inference = run_secure_inference(masks.data, secure);
        }
        
        plaintext.predictions = teegnn::argmax_rows(plaintext.logits);
        plaintext.accuracy = teegnn::accuracy(plaintext.predictions, dataset.labels);
        teegnn::IntVector predictions = teegnn::argmax_rows(inference.logits);
        double teegnn_accuracy = teegnn::accuracy(predictions, dataset.labels);

        std::vector<int> mismatches;
        for (std::size_t i = 0; i < predictions.size(); ++i) {
            if (predictions[i] != plaintext.predictions[i]) {
                mismatches.push_back(static_cast<int>(i));
                if (mismatches.size() >= 10) {
                    break;
                }
            }
        }

        const auto total_end = std::chrono::steady_clock::now();
        const double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        double error = (inference.logits - plaintext.logits).norm();

        std::cout << std::fixed << std::setprecision(10);
        std::cout << "nodes: " << dataset.graph.num_nodes() << "\n";
        std::cout << "normalized_support_edges: " << dataset.graph.num_edges() << "\n";
        std::cout << "feature_dim: " << dataset.features.cols() << "\n";
        std::cout << "hidden_dim: " << dataset.w1.cols() << "\n";
        std::cout << "class_count: " << dataset.w2.cols() << "\n";
        std::cout << "plaintext_accuracy: " << plaintext.accuracy << "\n";
        std::cout << "teegnn_accuracy: " << teegnn_accuracy << "\n";
        std::cout << "logits_error: " << error << "\n";
        
        if (mismatches.empty()) {
            std::cout << "comparison: PASS\n";
        } else {
            std::cout << "comparison: FAIL\n";
            std::cout << "first_mismatched_nodes:";
            for (int node : mismatches) {
                std::cout << " " << node << "(plain=" << plaintext.predictions[static_cast<std::size_t>(node)]
                          << ",sim=" << predictions[static_cast<std::size_t>(node)] << ")";
            }
            std::cout << "\n";
        }
        
        for (const auto& record : timings) {
            std::cout << "timing_ms." << record.name << ": " << record.milliseconds << "\n";
        }
        std::cout << "timing_ms.total: " << total_ms << "\n";

    } catch (const std::exception& ex) {
        std::cerr << "teegnn: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
