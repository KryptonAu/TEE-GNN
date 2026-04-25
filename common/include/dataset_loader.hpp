#pragma once

#include "graph.hpp"
#include "types.hpp"

#include <string>

namespace teegnn {

struct Dataset {
    Graph graph;
    Matrix features;
    Matrix w1;
    Matrix w2;
    IntVector labels;
};

Dataset load_dataset(const std::string& dataset_dir);

}  // namespace teegnn

