#include "dataset_loader.hpp"
#include "gcn_ops.hpp"
#include "masks.hpp"
#include "types.hpp"

#include "tee_gnn_client.hpp"

#include <Eigen/src/Core/Matrix.h>
#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct InferencePhaseResult {
    teegnn::Matrix logits;
};

teegnn::Matrix ree_masked_sparse_dense(
    int rows,
    const std::vector<std::vector<std::pair<int, double>>>& adjacency,
    const teegnn::Matrix& x) {
    if (static_cast<int>(adjacency.size()) != rows) {
        throw std::runtime_error("masked adjacency row count does not match expected rows");
    }
    teegnn::Matrix output = teegnn::Matrix::Zero(rows, x.cols());
    for (int row = 0; row < rows; ++row) {
        for (const auto& [col, value] : adjacency[static_cast<std::size_t>(row)]) {
            if (col < 0 || col >= x.rows()) {
                throw std::runtime_error("masked adjacency column out of range");
            }
            output.row(row).noalias() += value * x.row(col);
        }
    }
    return output;
}

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
        if (arg == "--confusion-rate" && i + 1 < argc) {
            options.confusion_rate = std::stod(argv[++i]);
        } else if (arg == "--mask-rank" && i + 1 < argc) {
            options.mask_rank = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            options.seed = static_cast<std::uint64_t>(std::stoull(argv[++i]));
        } else {
            print_usage(argv[0]);
            throw std::runtime_error("unknown or incomplete option: " + arg);
        }
    }
    if (options.confusion_rate < 0.0) {
        throw std::runtime_error("--confusion-rate must be non-negative");
    }
    if (options.mask_rank < 0) {
        throw std::runtime_error("--mask-rank must be non-negative");
    }
    return options;
}

InferencePhaseResult run_secure_inference(teegnn::MaskedData& masked_data,
                                          const std::unique_ptr<teegnn::TEEGNNClient> &secure) {
    InferencePhaseResult result;

    size_t num_nodes = masked_data.graph_shares.a1.size();
    
    teegnn::Matrix output;
    teegnn::Matrix &y1 = masked_data.features;
    teegnn::Matrix y2;
    for (int layer = 0; layer < 2; ++layer) {
        y1 *= masked_data.weights[layer];
        y2 = Eigen::MatrixXd::Zero(y1.rows(), y1.cols()); // dummy for secure computation

        secure->remask(layer, y1, y2);

        y1 = ree_masked_sparse_dense(num_nodes, 
            masked_data.graph_shares.a1, y1);
        y2 = ree_masked_sparse_dense(num_nodes, 
            masked_data.graph_shares.a2, y2);
        
        if (layer == 0) {
            secure->nonlinear_layer(layer, y1, y2);
        } else {
            secure->nonlinear_layer(layer, y1, y2);
        }
    }
    result.logits = y1; 
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
            teegnn::ScopedTimer timer(timings, "teegnn_inference");
            std::unique_ptr<teegnn::TEEGNNClient> secure = std::make_unique<teegnn::TEEGNNClient>();
            if (!secure->initialize()) {
                throw std::runtime_error("Failed to initialize TEE client");
            }
            if (!secure->init_GNNContext(dataset.graph.num_nodes(), options.mask_rank, 
                                         masks.matrices.feature_dim, masks.matrices.hidden_dim,
                                         masks.data.weights[0], masks.matrices.precompute_Ahat_u)) {
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
        std::cout << "directed_edges_without_self: " << dataset.graph.raw_directed_edges() << "\n";
        std::cout << "normalized_support_edges: " << dataset.graph.edges().size() << "\n";
        std::cout << "confusion_edges: " << masks.data.graph_shares.confusion_edges << "\n";
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
