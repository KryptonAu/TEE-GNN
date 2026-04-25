#include "csprng_adapter.hpp"
#include "dataset_loader.hpp"
#include "gcn_ops.hpp"
#include "masks.hpp"

#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string dataset_dir;
    double confusion_rate = 0.0;
    int mask_rank = 2;
    std::uint64_t seed = 1;
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

class REESimulator {
public:
    teegnn::Matrix masked_sparse_dense(int rows,
                                       const std::vector<teegnn::WeightedEdge>& edges,
                                       const teegnn::Matrix& x) const {
        return teegnn::sparse_dense_mul_edges(rows, edges, x);
    }

    teegnn::Matrix masked_dense(const teegnn::Matrix& x, const teegnn::Matrix& w) const {
        return x * w;
    }
};

class TEESimulator {
public:
    teegnn::Matrix restore_aggregation(const teegnn::Matrix& y1,
                                       const teegnn::Matrix& y2,
                                       const teegnn::LowRankMask& mask,
                                       const teegnn::Graph& graph,
                                       const teegnn::ScaledPermutation& p1,
                                       const teegnn::ScaledPermutation& p3,
                                       const teegnn::ScaledPermutation& p4,
                                       const teegnn::ScaledPermutation& p6) const {
        teegnn::Matrix restored = p1.apply_left_inv(y1);
        restored = p3.apply_right(restored);
        teegnn::Matrix second = p4.apply_left_inv(y2);
        second = p6.apply_right(second);
        restored.noalias() += second;
        restored.noalias() -= mask.ahat_times_mask(graph);
        return restored;
    }

    teegnn::Matrix masked_dense_protocol(const teegnn::Matrix& x,
                                         const teegnn::Matrix& w,
                                         teegnn::RandomEngine& rng,
                                         const REESimulator& ree) const {
        teegnn::SDIMMask s_left = teegnn::SDIMMask::random(static_cast<int>(w.rows()), rng);
        teegnn::SDIMMask s_right = teegnn::SDIMMask::random(static_cast<int>(w.cols()), rng);
        teegnn::SDIMMask s_tmp = teegnn::SDIMMask::random(static_cast<int>(x.rows()), rng);

        teegnn::Matrix w_masked = s_left.apply_left(w);
        w_masked = s_right.apply_right_inv(w_masked);

        teegnn::Matrix x_masked = s_tmp.apply_left(x);
        x_masked = s_left.apply_right_inv(x_masked);

        teegnn::Matrix z_masked = ree.masked_dense(x_masked, w_masked);
        teegnn::Matrix z = s_tmp.apply_left_inv(z_masked);
        z = s_right.apply_right(z);
        return z;
    }
};

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
        REESimulator ree;
        TEESimulator tee;

        teegnn::Matrix logits;
        std::size_t confusion_edges = 0;
        double layer1_restore_error = 0.0;
        double layer1_dense_error = 0.0;
        double layer2_restore_error = 0.0;
        {
            ScopedTimer timer(timings, "teegnn_mask_generation");
            const int n = dataset.graph.num_nodes();
            const int feature_dim = static_cast<int>(dataset.features.cols());
            const int hidden_dim = static_cast<int>(dataset.w1.cols());

            teegnn::ScaledPermutation p1 = teegnn::ScaledPermutation::random(n, rng);
            teegnn::ScaledPermutation p2 = teegnn::ScaledPermutation::random(n, rng);
            teegnn::ScaledPermutation p4 = teegnn::ScaledPermutation::random(n, rng);
            teegnn::ScaledPermutation p5 = teegnn::ScaledPermutation::random(n, rng);
            teegnn::ProtectedGraphShares graph_shares =
                teegnn::protect_graph_edges(dataset.graph, options.confusion_rate, p1, p2, p4, p5, rng);
            confusion_edges = graph_shares.confusion_edges;

            teegnn::ScaledPermutation p3_0 = teegnn::ScaledPermutation::random(feature_dim, rng);
            teegnn::ScaledPermutation p6_0 = teegnn::ScaledPermutation::random(feature_dim, rng);
            teegnn::LowRankMask m0 =
                teegnn::make_low_rank_mask(n, feature_dim, options.mask_rank, rng);

            teegnn::Matrix h1 = remask_for_ree(dataset.features, m0, p2, p3_0);
            teegnn::Matrix h2 = remask_for_ree(dataset.features, m0, p5, p6_0);

            teegnn::Matrix y1;
            teegnn::Matrix y2;
            {
                ScopedTimer stage(timings, "teegnn_layer1_masked_aggregation_ree");
                y1 = ree.masked_sparse_dense(n, graph_shares.a1, h1);
                y2 = ree.masked_sparse_dense(n, graph_shares.a2, h2);
            }

            teegnn::Matrix xbar1;
            {
                ScopedTimer stage(timings, "teegnn_layer1_restore_tee");
                xbar1 = tee.restore_aggregation(y1, y2, m0, dataset.graph, p1, p3_0, p4, p6_0);
            }
            layer1_restore_error = (xbar1 - teegnn::sparse_dense_mul(dataset.graph, dataset.features))
                                       .cwiseAbs()
                                       .maxCoeff();

            teegnn::Matrix hidden_linear;
            {
                ScopedTimer stage(timings, "teegnn_layer1_masked_dense");
                hidden_linear = tee.masked_dense_protocol(xbar1, dataset.w1, rng, ree);
            }
            layer1_dense_error = (hidden_linear - xbar1 * dataset.w1).cwiseAbs().maxCoeff();

            teegnn::Matrix hidden;
            {
                ScopedTimer stage(timings, "teegnn_relu_tee");
                hidden = teegnn::relu(hidden_linear);
            }

            teegnn::ScaledPermutation p3_1 = teegnn::ScaledPermutation::random(hidden_dim, rng);
            teegnn::ScaledPermutation p6_1 = teegnn::ScaledPermutation::random(hidden_dim, rng);
            teegnn::LowRankMask m1 =
                teegnn::make_low_rank_mask(n, hidden_dim, options.mask_rank, rng);
            h1 = remask_for_ree(hidden, m1, p2, p3_1);
            h2 = remask_for_ree(hidden, m1, p5, p6_1);

            {
                ScopedTimer stage(timings, "teegnn_layer2_masked_aggregation_ree");
                y1 = ree.masked_sparse_dense(n, graph_shares.a1, h1);
                y2 = ree.masked_sparse_dense(n, graph_shares.a2, h2);
            }

            teegnn::Matrix xbar2;
            {
                ScopedTimer stage(timings, "teegnn_layer2_restore_tee");
                xbar2 = tee.restore_aggregation(y1, y2, m1, dataset.graph, p1, p3_1, p4, p6_1);
            }
            layer2_restore_error = (xbar2 - teegnn::sparse_dense_mul(dataset.graph, hidden))
                                       .cwiseAbs()
                                       .maxCoeff();

            {
                ScopedTimer stage(timings, "teegnn_layer2_masked_dense");
                logits = tee.masked_dense_protocol(xbar2, dataset.w2, rng, ree);
            }
        }

        teegnn::IntVector predictions;
        double teegnn_accuracy = 0.0;
        {
            ScopedTimer timer(timings, "teegnn_argmax_accuracy");
            predictions = teegnn::argmax_rows(logits);
            teegnn_accuracy = teegnn::accuracy(predictions, dataset.labels);
        }

        const teegnn::Matrix abs_error = (logits - plaintext.logits).cwiseAbs();
        const double max_abs_error = abs_error.maxCoeff();
        const double mean_abs_error = abs_error.sum() / static_cast<double>(abs_error.size());

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
        std::cout << "confusion_edges: " << confusion_edges << "\n";
        std::cout << "feature_dim: " << dataset.features.cols() << "\n";
        std::cout << "hidden_dim: " << dataset.w1.cols() << "\n";
        std::cout << "class_count: " << dataset.w2.cols() << "\n";
        std::cout << "plaintext_accuracy: " << plaintext.accuracy << "\n";
        std::cout << "teegnn_sim_accuracy: " << teegnn_accuracy << "\n";
        std::cout << "logits_max_abs_error: " << max_abs_error << "\n";
        std::cout << "logits_mean_abs_error: " << mean_abs_error << "\n";
        std::cout << "debug_layer1_restore_max_abs_error: " << layer1_restore_error << "\n";
        std::cout << "debug_layer1_dense_max_abs_error: " << layer1_dense_error << "\n";
        std::cout << "debug_layer2_restore_max_abs_error: " << layer2_restore_error << "\n";
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
