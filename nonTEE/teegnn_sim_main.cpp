#include "csprng_adapter.hpp"
#include "dataset_loader.hpp"
#include "gcn_ops.hpp"
#include "masks.hpp"
#include "types.hpp"

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

struct Options {
    std::string dataset_dir;
    double confusion_rate = 0.0;
    int mask_rank = 2;
    std::uint64_t seed = 1;
};

struct MaskedTensorShares {
    teegnn::Matrix share1;
    teegnn::Matrix share2;
};

struct MaskMatrices {
    int node_count = 0;
    int feature_dim = 0;
    int hidden_dim = 0;

    teegnn::ScaledPermutation p1;
    teegnn::ScaledPermutation p2;
    teegnn::ScaledPermutation p3;
    teegnn::ScaledPermutation p4;
    teegnn::ScaledPermutation p5;
    teegnn::ScaledPermutation p6;

    std::vector<teegnn::LowRankMask> lr_masks;

    std::vector<teegnn::SDIMMask> sdim_masks;
};

struct MaskedData {
    teegnn::ProtectedGraphShares graph_shares;
    MaskedTensorShares input_features;
    std::vector<teegnn::Matrix> masked_weights;
};

struct MaskPhaseResult {
    MaskMatrices matrices;
    MaskedData data;
};

struct InferencePhaseResult {
    teegnn::Matrix logits;
    double layer1_restore_error = 0.0;
    double layer1_dense_error = 0.0;
    double layer2_restore_error = 0.0;
};

class ScopedTimer {
public:
    ScopedTimer(std::vector<teegnn::TimerRecord>& records, std::string name)
        : records_(records), name_(std::move(name)), start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        const auto end = std::chrono::steady_clock::now();
        records_.push_back({name_, std::chrono::duration<double, std::milli>(end - start_).count()});
    }

private:
    std::vector<teegnn::TimerRecord>& records_;
    std::string name_;
    std::chrono::steady_clock::time_point start_;
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

Options parse_args(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        throw std::runtime_error("missing dataset directory");
    }
    Options options;
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

teegnn::Matrix remask_for_ree(const teegnn::Matrix& x,
                              const teegnn::LowRankMask& low_rank_mask,
                              const teegnn::ScaledPermutation& left,
                              const teegnn::ScaledPermutation& right) {
    teegnn::Matrix masked = x + low_rank_mask.materialize();
    masked = left.apply_left(masked);
    masked = right.apply_right_inv(masked);
    return masked;
}

MaskedTensorShares make_masked_tensor_shares(const teegnn::Matrix& x,
                                             const teegnn::LowRankMask& low_rank_mask,
                                             const teegnn::ScaledPermutation& left1,
                                             const teegnn::ScaledPermutation& right1,
                                             const teegnn::ScaledPermutation& left2,
                                             const teegnn::ScaledPermutation& right2) {
    return {remask_for_ree(x, low_rank_mask, left1, right1),
            remask_for_ree(x, low_rank_mask, left2, right2)};
}

class secure_computation {
public:
    secure_computation(const teegnn::Graph& graph,
                       MaskMatrices&& masks,
                       teegnn::RandomEngine& rng)
        : graph_(graph), masks_(std::move(masks)), rng_(rng) {}

    teegnn::Matrix restore_aggregation(const teegnn::Matrix& y1,
                                       const teegnn::Matrix& y2,
                                       int layer) {
        teegnn::Matrix restored = teegnn::apply_SPM_inv(masks_.p1, masks_.p3, y1);
        teegnn::Matrix second = teegnn::apply_SPM_inv(masks_.p4, masks_.p6, y2);
        restored.noalias() += second;
        restored.noalias() -= masks_.lr_masks[layer].ahat_times_mask(graph_);

        temp = std::move(teegnn::SDIMMask::random(restored.rows(), rng_));
        return teegnn::apply_SDIM(temp, masks_.sdim_masks[layer * 2], restored);
    }

    MaskedTensorShares nonlinear_layer(const teegnn::Matrix& linear_output, int layer) {
        teegnn::Matrix z = teegnn::apply_SDIM_inv(temp, masks_.sdim_masks[layer * 2 + 1], linear_output);
        if (layer == 0) {
            z = teegnn::relu(z);
            masks_.p3 = teegnn::ScaledPermutation::random(z.cols(), rng_);
            masks_.p6 = teegnn::ScaledPermutation::random(z.cols(), rng_);
            return make_masked_tensor_shares(
                z, masks_.lr_masks[1], masks_.p2, masks_.p3,
                masks_.p5, masks_.p6);      
        } else {
            return {z, z};  // identity for the second layer
        }
    }

private:
    const teegnn::Graph& graph_;
    MaskMatrices masks_;
    teegnn::SDIMMask temp;
    teegnn::RandomEngine& rng_;
};

MaskPhaseResult run_mask_phase(const teegnn::Dataset& dataset,
                               const Options& options,
                               teegnn::RandomEngine& rng,
                               std::vector<teegnn::TimerRecord>& timings) {
    ScopedTimer timer(timings, "teegnn_mask_phase");

    MaskPhaseResult result;
    MaskMatrices& matrices = result.matrices;
    MaskedData& masked_data = result.data;

    matrices.node_count = dataset.graph.num_nodes();
    matrices.feature_dim = static_cast<int>(dataset.features.cols());
    matrices.hidden_dim = static_cast<int>(dataset.w1.cols());
    const int class_dim = static_cast<int>(dataset.w2.cols());

    matrices.p1 = teegnn::ScaledPermutation::random(matrices.node_count, rng);
    matrices.p2 = teegnn::ScaledPermutation::random(matrices.node_count, rng);
    matrices.p4 = teegnn::ScaledPermutation::random(matrices.node_count, rng);
    matrices.p5 = teegnn::ScaledPermutation::random(matrices.node_count, rng);
    masked_data.graph_shares =
        teegnn::protect_graph_edges(dataset.graph, options.confusion_rate, matrices.p1, matrices.p2,
                                    matrices.p4, matrices.p5, rng);

    matrices.p3 = teegnn::ScaledPermutation::random(matrices.feature_dim, rng);
    matrices.p6 = teegnn::ScaledPermutation::random(matrices.feature_dim, rng);
    matrices.lr_masks.emplace_back(
        teegnn::make_low_rank_mask(matrices.node_count, 
                                   matrices.feature_dim, 
                                   options.mask_rank, rng));
    masked_data.input_features = make_masked_tensor_shares(
        dataset.features, matrices.lr_masks[0], matrices.p2, matrices.p3,
        matrices.p5, matrices.p6);

    matrices.lr_masks.emplace_back(
        teegnn::make_low_rank_mask(matrices.node_count, 
                                   matrices.hidden_dim, 
                                   options.mask_rank, rng));
    {
        teegnn::SDIMMask s_left = teegnn::SDIMMask::random(matrices.feature_dim, rng);
        teegnn::SDIMMask s_right = teegnn::SDIMMask::random(matrices.hidden_dim, rng);
        masked_data.masked_weights.push_back(teegnn::apply_SDIM(s_left, s_right, dataset.w1));
        matrices.sdim_masks.push_back(std::move(s_left));
        matrices.sdim_masks.push_back(std::move(s_right));
    }
    {
        teegnn::SDIMMask s_left = teegnn::SDIMMask::random(matrices.hidden_dim, rng);
        teegnn::SDIMMask s_right = teegnn::SDIMMask::random(class_dim, rng);
        masked_data.masked_weights.push_back(teegnn::apply_SDIM(s_left, s_right, dataset.w2));
        matrices.sdim_masks.push_back(std::move(s_left));
        matrices.sdim_masks.push_back(std::move(s_right));
    }

    return result;
}

InferencePhaseResult run_inference_phase(const teegnn::Dataset& dataset,
                                         MaskedData& masked_data,
                                         secure_computation& secure,
                                         std::vector<teegnn::TimerRecord>& timings) {
    ScopedTimer timer(timings, "teegnn_inference_phase");

    InferencePhaseResult result;
    
    teegnn::Matrix output;
    for (int layer = 0; layer < 2; ++layer) {
        teegnn::Matrix y1 = ree_masked_sparse_dense(dataset.graph.num_nodes(), 
            masked_data.graph_shares.a1, masked_data.input_features.share1);
        teegnn::Matrix y2 = ree_masked_sparse_dense(dataset.graph.num_nodes(), 
            masked_data.graph_shares.a2, masked_data.input_features.share2);
        
        teegnn::Matrix masked_xbar = secure.restore_aggregation(y1, y2, layer);

        masked_data.input_features = secure.nonlinear_layer(masked_xbar * masked_data.masked_weights[layer], layer);
    }
    result.logits = masked_data.input_features.share1; 

    /*
    teegnn::Matrix y1;
    teegnn::Matrix y2;
    {
        ScopedTimer stage(timings, "teegnn_layer1_masked_aggregation_ree");
        y1 = ree_masked_sparse_dense(
            matrices.node_count, masked_data.graph_shares.a1, masked_data.input_features.share1);
        y2 = ree_masked_sparse_dense(
            matrices.node_count, masked_data.graph_shares.a2, masked_data.input_features.share2);
    }

    teegnn::Matrix xbar1;
    {
        ScopedTimer stage(timings, "teegnn_layer1_restore_tee");
        xbar1 = secure.restore_input_aggregation(y1, y2);
    }
    result.layer1_restore_error =
        (xbar1 - teegnn::sparse_dense_mul(dataset.graph, dataset.features)).cwiseAbs().maxCoeff();

    teegnn::Matrix hidden_linear;
    {
        ScopedTimer stage(timings, "teegnn_layer1_masked_dense");
        hidden_linear = secure.masked_dense_protocol(xbar1, dataset.w1);
    }
    result.layer1_dense_error = (hidden_linear - xbar1 * dataset.w1).cwiseAbs().maxCoeff();

    teegnn::Matrix hidden;
    {
        ScopedTimer stage(timings, "teegnn_relu_tee");
        hidden = secure.relu(hidden_linear);
    }

    const MaskedTensorShares hidden_shares = make_masked_tensor_shares(
        hidden, matrices.hidden_mask, matrices.p2, matrices.p3_hidden,
        matrices.p5, matrices.p6_hidden);

    {
        ScopedTimer stage(timings, "teegnn_layer2_masked_aggregation_ree");
        y1 = ree_masked_sparse_dense(
            matrices.node_count, masked_data.graph_shares.a1, hidden_shares.share1);
        y2 = ree_masked_sparse_dense(
            matrices.node_count, masked_data.graph_shares.a2, hidden_shares.share2);
    }

    teegnn::Matrix xbar2;
    {
        ScopedTimer stage(timings, "teegnn_layer2_restore_tee");
        xbar2 = secure.restore_hidden_aggregation(y1, y2);
    }
    result.layer2_restore_error =
        (xbar2 - teegnn::sparse_dense_mul(dataset.graph, hidden)).cwiseAbs().maxCoeff();

    {
        ScopedTimer stage(timings, "teegnn_layer2_masked_dense");
        result.logits = secure.masked_dense_protocol(xbar2, dataset.w2);
    }
    */

    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        std::vector<teegnn::TimerRecord> timings;
        const auto total_start = std::chrono::steady_clock::now();

        teegnn::Dataset dataset;
        {
            ScopedTimer timer(timings, "load_dataset");
            dataset = teegnn::load_dataset(options.dataset_dir);
        }

        teegnn::InferenceResult plaintext;
        {
            ScopedTimer timer(timings, "plaintext_reference");
            plaintext = teegnn::run_plaintext_inference(dataset);
        }

        teegnn::RandomEngine rng(options.seed, 0, "teegnn-sim");

        MaskPhaseResult masks = run_mask_phase(dataset, options, rng, timings);
        secure_computation secure(dataset.graph, std::move(masks.matrices), rng);
        const InferencePhaseResult inference =
            run_inference_phase(dataset, masks.data, secure, timings);

        teegnn::IntVector predictions;
        double teegnn_accuracy = 0.0;
        {
            ScopedTimer timer(timings, "teegnn_argmax_accuracy");
            predictions = teegnn::argmax_rows(inference.logits);
            teegnn_accuracy = teegnn::accuracy(predictions, dataset.labels);
        }

        // const teegnn::Matrix abs_error = (inference.logits - plaintext.logits).cwiseAbs();
        // const double max_abs_error = abs_error.maxCoeff();
        // const double mean_abs_error = abs_error.sum() / static_cast<double>(abs_error.size());

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

        std::cout << std::fixed << std::setprecision(10);
        std::cout << "nodes: " << dataset.graph.num_nodes() << "\n";
        std::cout << "directed_edges_without_self: " << dataset.graph.raw_directed_edges() << "\n";
        std::cout << "normalized_support_edges: " << dataset.graph.edges().size() << "\n";
        std::cout << "confusion_edges: " << masks.data.graph_shares.confusion_edges << "\n";
        std::cout << "feature_dim: " << dataset.features.cols() << "\n";
        std::cout << "hidden_dim: " << dataset.w1.cols() << "\n";
        std::cout << "class_count: " << dataset.w2.cols() << "\n";
        std::cout << "plaintext_accuracy: " << plaintext.accuracy << "\n";
        std::cout << "teegnn_sim_accuracy: " << teegnn_accuracy << "\n";
        // std::cout << "logits_max_abs_error: " << max_abs_error << "\n";
        // std::cout << "logits_mean_abs_error: " << mean_abs_error << "\n";
        // std::cout << "debug_layer1_restore_max_abs_error: " << inference.layer1_restore_error << "\n";
        // std::cout << "debug_layer1_dense_max_abs_error: " << inference.layer1_dense_error << "\n";
        // std::cout << "debug_layer2_restore_max_abs_error: " << inference.layer2_restore_error << "\n";
        
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
        std::cerr << "teegnn_sim: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
