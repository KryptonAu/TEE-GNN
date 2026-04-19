#pragma once

#include <vector>
#include <queue>
#include <stack>
#include <algorithm>
#include <optional>
#include <fstream>
#include <iostream>
#include <iomanip>

// 简单的图类（无权），支持有向/无向图和基本操作
class Graph {
public:
    using Vertex = int;

    // 构造函数：directed=false 表示无向图
    explicit Graph(bool directed = false) : directed_(directed), edge_count_(0) {}

    // 添加一个顶点，返回顶点 id（从 0 开始）
    Vertex add_vertex() {
        adj_.emplace_back();
        return static_cast<Vertex>(adj_.size() - 1);
    }

    // 确保顶点存在（如果 id 超出当前范围，则扩展）
    void ensure_vertex(Vertex v) {
        if (v >= static_cast<Vertex>(adj_.size())) {
            adj_.resize(v + 1);
        }
    }

    // 添加边 u->v（若无向图同时添加 v->u），自动扩展顶点集合
    // 返回 true 如果添加成功（不存在重复边）
    bool add_edge(Vertex u, Vertex v) {
        ensure_vertex(std::max(u, v));
        //if (has_edge(u, v)) return false;
        adj_[u].push_back(v);
        if (!directed_ && u != v) adj_[v].push_back(u);
        ++edge_count_;
        return true;
    }

    // 添加自环
    void add_self_loop() {
        for (Vertex v = 0; v < num_vertices(); ++v) {
            add_edge(v, v);
        }
    }

    // 删除边 u->v（若无向图同时删除 v->u），返回是否删除成功
    bool remove_edge(Vertex u, Vertex v) {
        if (!valid_vertex(u) || !valid_vertex(v)) return false;
        bool removed = remove_one(adj_[u], v);
        if (!directed_ && remove_one(adj_[v], u)) removed = true;
        if (removed) --edge_count_;
        return removed;
    }

    // 判断是否存在边 u->v
    bool has_edge(Vertex u, Vertex v) const {
        if (!valid_vertex(u)) return false;
        const auto &list = adj_[u];
        return std::find(list.begin(), list.end(), v) != list.end();
    }

    // 返回顶点 v 的邻居列表（按引用返回以避免拷贝）
    const std::vector<Vertex>& neighbors(Vertex v) const {
        static const std::vector<Vertex> empty;
        if (!valid_vertex(v)) return empty;
        return adj_[v];
    }

    // 顶点数量
    Vertex num_vertices() const { return static_cast<Vertex>(adj_.size()); }

    // 边数量（对无向图，每条边计一次）
    size_t num_edges() const { return edge_count_; }

    // 清空图
    void clear() {
        adj_.clear();
        edge_count_ = 0;
    }

    // 度（无向图为度，有向图为出度）
    size_t degree(Vertex v) const {
        if (!valid_vertex(v)) return 0;
        return adj_[v].size();
    }

    // 从简单的文本文件加载图（每行 "u v"），若是无向图请确保文件只列出一次每条边
    bool load_from_edge_list(const std::string &path) {
        std::ifstream in(path);
        if (!in) return false;
        clear();
        Vertex u, v;
        while (in >> u >> v) {
            ensure_vertex(std::max(u, v));
            add_edge(u, v);
        }
        return true;
    }

    // 将图以边列表形式写入文件（每行 "u v"）
    bool save_to_edge_list(const std::string &path) const {
        std::ofstream out(path);
        if (!out) return false;
        if (directed_) {
            for (Vertex u = 0; u < num_vertices(); ++u)
                for (Vertex v : adj_[u]) out << u << ' ' << v << '\n';
        } else {
            // 对无向图只输出 u<v 的边以避免重复
            for (Vertex u = 0; u < num_vertices(); ++u)
                for (Vertex v : adj_[u])
                    if (u <= v) out << u << ' ' << v << '\n';
        }
        return true;
    }

private:
    std::vector<std::vector<Vertex>> adj_;
    bool directed_;
    size_t edge_count_;

    bool valid_vertex(Vertex v) const { return v >= 0 && v < num_vertices(); }

    static bool remove_one(std::vector<Vertex> &vec, Vertex value) {
        auto it = std::find(vec.begin(), vec.end(), value);
        if (it == vec.end()) return false;
        vec.erase(it);
        return true;
    }
};

// 带权重图类（权重类型为 double）
class Weighted_Graph {
public:
    using Vertex = int;

    // 构造函数：directed=false 表示无向图
    explicit Weighted_Graph(bool directed = false) : directed_(directed), edge_count_(0) {}

    // 添加一个顶点，返回顶点 id（从 0 开始）
    Vertex add_vertex() {
        adj_.emplace_back();
        return static_cast<Vertex>(adj_.size() - 1);
    }

    // 确保顶点存在（如果 id 超出当前范围，则扩展）
    void ensure_vertex(Vertex v) {
        if (v >= static_cast<Vertex>(adj_.size())) {
            adj_.resize(v + 1);
        }
    }

    // 添加边 u->v（若无向图同时添加 v->u），自动扩展顶点集合
    // 返回 true 如果添加成功（不存在重复边）
    bool add_edge(Vertex u, Vertex v, double weight) {
        ensure_vertex(std::max(u, v));
        adj_[u].push_back({v, weight});
        if (!directed_ && u != v) adj_[v].push_back({u, weight});
        ++edge_count_;
        return true;
    }

    // 添加自环
    void add_self_loop() {
        for (Vertex v = 0; v < num_vertices(); ++v) {
            add_edge(v, v, 1.0);
        }
    }

    // 返回顶点 v 的邻居列表（按引用返回以避免拷贝）
    const std::vector<std::pair<Vertex, double>>& neighbors(Vertex v) const {
        static const std::vector<std::pair<Vertex, double>> empty;
        if (!valid_vertex(v)) return empty;
        return adj_[v];
    }

    // 顶点数量
    Vertex num_vertices() const { return static_cast<Vertex>(adj_.size()); }

    // 边数量（对无向图，每条边计一次）
    size_t num_edges() const { return edge_count_; }

    // 清空图
    void clear() {
        adj_.clear();
        edge_count_ = 0;
    }

    // 度（无向图为度，有向图为出度）
    size_t degree(Vertex v) const {
        if (!valid_vertex(v)) return 0;
        return adj_[v].size();
    }

    // 输出图的边表
    void print() const {
        int u = 0;
        for (auto nbr : adj_) {
            for (auto [v, w] : nbr) {
                std::cout << u << ' ' << v << ' ' << w << '\n';
            }
            u++;
        }
    }

private:
    std::vector<std::vector<std::pair<Vertex, double>>> adj_;
    bool directed_;
    size_t edge_count_;

    bool valid_vertex(Vertex v) const { return v >= 0 && v < num_vertices(); }

    static bool remove_one(std::vector<Vertex> &vec, Vertex value) {
        auto it = std::find(vec.begin(), vec.end(), value);
        if (it == vec.end()) return false;
        vec.erase(it);
        return true;
    }
};
