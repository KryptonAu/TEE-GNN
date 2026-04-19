#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <set>
#include <cstdlib>
#include <random>
#include <ctime>

#include "secGCN.h"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

std::string DATASET_PATH = "../dataset/";

void read_matrix_from_file(const std::string &filename, Matrix &mat) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    std::vector<std::vector<double>> data;
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::vector<double> row;
        double value;
        while (iss >> value) {
            row.push_back(value);
        }
        data.push_back(row);
    }
    infile.close();
    if (data.empty()) {
        mat = Matrix(0, 0);
        return;
    }
    int rows = data.size();
    int cols = data[0].size();
    mat = Matrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat(i, j) = data[i][j];
        }
    }
}

std::vector<int> random_permutation(int n) {
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) {
        perm[i] = i;
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(perm.begin(), perm.end(), g);
    return perm;
}

std::vector<int> random_vector(int n, int min_val = -1000, int max_val = 1000) {
    std::vector<int> vec(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min_val, max_val);
    // 确保不生成0
    for (int i = 0; i < n; ++i) {
        while (true) {
            vec[i] = dis(gen);
            if (vec[i] != 0) {
                break;
            }
        }
    }
    return vec;
}

// 读取图并进行划分和转换,将结果存入g1和g2,具体来说,假设图有m条边,随机生成rm(1<r<2)条权重为0的边,
// 然后生成 m+rm 个随机值,对于每条边,x+value存入g1,-value存入g2
// 最后返回原始图的度序列
void read_split_transform(const Graph &g, Weighted_Graph &g1, Weighted_Graph &g2, 
                        const std::vector<std::vector<int>> &values, 
                        const std::vector<std::vector<int>> &perms, 
                        double r = 1.01) {
    static std::set<std::pair<int, int>> edge_set;
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(-1000, 1000);

    std::vector<std::vector<int>> inv_perms;
    for (int i = 0; i < 4; ++i) {
        std::vector<int> inv_perm(perms[i].size());
        for (size_t j = 0; j < perms[i].size(); ++j) {
            inv_perm[perms[i][j]] = j;
        }
        inv_perms.emplace_back(inv_perm);
    }
    int n = g.num_vertices();
    int random_value;
    for (int u = 0; u < n; ++u) {
        for (auto v : g.neighbors(u)) {
            random_value = dis(gen);
            g1.add_edge(inv_perms[0][u], inv_perms[1][v], 1.0 * (1 + random_value) * values[0][u] / values[1][v]);
            g2.add_edge(inv_perms[2][u], inv_perms[3][v], 1.0 * ( - random_value) * values[2][u] / values[3][v]);
            edge_set.insert({u, v});
        }
    }

    int m = g1.num_edges();
    int rm = static_cast<int>(m * r);
    for (int i = 0; i < rm; ++i) {
        int u, v;
        do {
            u = rand() % n;
            v = rand() % n;
            if (edge_set.count({u, v}) == 0) {
                random_value = dis(gen);
                g1.add_edge(inv_perms[0][u], inv_perms[1][v], 1.0 * random_value * values[0][u] / values[1][v]);
                g2.add_edge(inv_perms[2][u], inv_perms[3][v], -1.0 * random_value * values[2][u] / values[3][v]);
                edge_set.insert({u, v});
                break;
            }
        } while (1);
    }
}

Matrix encrypt(const Matrix &input, const std::vector<int> &value1, const std::vector<int> &perm1,
                                    const std::vector<int> &value2, const std::vector<int> &perm2) {
    int r = input.rows();
    int c = input.cols();
    Matrix encrypted = Eigen::MatrixXd::Zero(r, c);
    for (int i = 0; i < r; ++i) {
        int new_i = perm1[i];
        // std::cout<<r<<' '<<c<<' '<<i<<'\n';
        for (int j = 0; j < c; ++j) {
            int new_j = perm2[j];
            encrypted(i, j) = input(new_i, new_j) * value1[new_i] / value2[new_j];
        }
    }
    return encrypted;
}

Matrix decrypt(const Matrix &input, const std::vector<int> &value1, const std::vector<int> &perm1,
                                    const std::vector<int> &value2, const std::vector<int> &perm2) {
    int r = input.rows();
    int c = input.cols();
    Matrix decrypted = Eigen::MatrixXd::Zero(r, c);
    for (int i = 0; i < r; ++i) {
        int new_i = perm1[i];
        // std::cout<<r<<' '<<c<<' '<<i<<'\n';
        for (int j = 0; j < c; ++j) {
            int new_j = perm2[j];
            decrypted(new_i, new_j) = input(i, j) * value2[new_j] / value1[new_i];
        }
    }
    return decrypted;
}

// 图数据读入
void load_graph(Graph &g, Matrix &features, std::vector<Matrix> &weights) {
    g.load_from_edge_list(DATASET_PATH + "edges.txt");
    g.add_self_loop();
    read_matrix_from_file(DATASET_PATH + "features.txt", features);
    Matrix W1, W2;
    read_matrix_from_file(DATASET_PATH + "w1.txt", W1);
    read_matrix_from_file(DATASET_PATH + "w2.txt", W2);
    W1.transposeInPlace();
    W2.transposeInPlace();
    weights.push_back(W1);
    weights.push_back(W2);
    std::cout << "--------图数据已加载--------\n";
}

// 生成变换矩阵
void generate_transformation_matrices(int n, int num_features, int hidden, int classes,
    std::vector<std::vector<int>> &values, std::vector<std::vector<int>> &perms) {
    // 生成随机的变换参数
    // p_0 A P_1^-1   p_2 B p_3^-1
    // 用于变换g1,g2的邻接矩阵
    for (int i = 0 ; i < 4; ++i) {
        values.emplace_back(random_vector(n));
        perms.emplace_back(random_permutation(n));
    }
    // 用于变换第一层权重矩阵
    for (int i = 0 ; i < 2; ++i) {
        values.emplace_back(random_vector(num_features));  
        perms.emplace_back(random_permutation(num_features));
        values.emplace_back(random_vector(hidden));  
        perms.emplace_back(random_permutation(hidden));
    }
    // 用于变换第二层权重矩阵
    for (int i = 0 ; i < 2; ++i) {
        values.emplace_back(random_vector(hidden));  
        perms.emplace_back(random_permutation(hidden));
        values.emplace_back(random_vector(classes));  
        perms.emplace_back(random_permutation(classes));
    }

    std::cout << "--------变换矩阵已生成--------\n";
}

// 特征矩阵和权重矩阵变换
void transform_matrices(const Matrix &features, const std::vector<Matrix> &weights,
                        const std::vector<std::vector<int>> &values, 
                        const std::vector<std::vector<int>> &perms,
                        std::vector<Matrix> &trans_features, 
                        std::vector<Matrix> &trans_weights) {
    trans_features.emplace_back(encrypt(features, values[1], perms[1], values[4], perms[4]));
    trans_features.emplace_back(encrypt(features, values[3], perms[3], values[6], perms[6]));

    std::cout << "--------特征矩阵已变换--------\n";

    trans_weights.emplace_back(encrypt(weights[0], values[4], perms[4], values[5], perms[5]));
    trans_weights.emplace_back(encrypt(weights[0], values[6], perms[6], values[7], perms[7]));
    trans_weights.emplace_back(encrypt(weights[1], values[8], perms[8], values[9], perms[9]));
    trans_weights.emplace_back(encrypt(weights[1], values[10], perms[10], values[11], perms[11]));

    std::cout << "--------权重矩阵已变换--------\n";
}

// 计算推理准确率
double compute_accuracy(const Matrix &output) {
    // 读入标签
    static std::ifstream fin(DATASET_PATH + "labels.txt");
    static std::vector<int> labels;
    if (labels.empty()) {
        if (!fin.is_open()) {
            throw std::runtime_error("Could not open file: " + DATASET_PATH + "labels.txt");
        }
        
        int label;
        while (fin >> label) {
            labels.push_back(label);
        }
        fin.close();
    }
    int num_vertices = output.rows();
    // 计算准确率
    int correct = 0;
    for (int i = 0; i < num_vertices; ++i) {
        int predicted_label;
        output.row(i).maxCoeff(&predicted_label);
        if (predicted_label == labels[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / num_vertices;
}

Graph g(true);
Weighted_Graph g1(true);
Weighted_Graph g2(true);

// 变换用的随机值和排列
std::vector<std::vector<int>> values;
std::vector<std::vector<int>> perms;
// 变换后的特征矩阵和权重
std::vector<Matrix> trans_features;
std::vector<Matrix> trans_weights;
int main(int argc, char* argv[]) {
    // 参数解析
    if (argc >= 2) {
        DATASET_PATH += std::string(argv[1]);
        if (DATASET_PATH.back() != '/') {
            DATASET_PATH += '/';
        }
    } else {
        DATASET_PATH += "naive_test/";
        std::cout << "使用默认数据集路径: " << DATASET_PATH << std::endl;
    }
    // 普通GCN测试
    Matrix features;
    std::vector<Matrix> weights;
    load_graph(g, features, weights);
    int num_vertices = g.num_vertices();
    int num_features = features.cols();
    int hidden = weights[0].cols();
    int classes = weights[1].cols();
    std::cout << num_vertices << '\n'
              << num_features << '\n'
              << hidden << '\n'
              << classes << '\n';

    // 记录运行时间，单位毫秒
    std::clock_t start_time = std::clock();

    Matrix output = GCN_inference(g, features, weights);

    std::clock_t end_time = std::clock();
    double elapsed_secs = double(end_time - start_time) / CLOCKS_PER_SEC * 1000;
    std::cout << "普通GCN推理运行时间: " << elapsed_secs << " 毫秒" << std::endl;

    // 输出到文件
    std::ofstream fout(DATASET_PATH + "gcn_output.txt");
    if (fout.is_open()) {
        fout << output.array().log() << std::endl;
        fout.close();
    } else {
        std::cerr << "无法打开文件进行写入: " << DATASET_PATH + "gcn_output.txt" << std::endl;
    }
    // std::cout << "GCN Inference Output:\n" << output.array().log() << std::endl;

    // SecGCN测试

    generate_transformation_matrices(num_vertices, num_features, hidden, classes, values, perms);

    read_split_transform(g, g1, g2, values, perms);
    
    std::cout << "--------原始图已变换--------\n";

    std::vector<int> degree(num_vertices, 0);
    for (int u = 0; u < num_vertices; ++u) {
        degree[u] = g.degree(u);
    }
    for (int i = 0; i < features.rows(); ++i) {
        features.row(i) /= std::sqrt(static_cast<double>(degree[i]));
    }
    transform_matrices(features, weights, values, perms, trans_features, trans_weights);
    
    // 实际解密需要的编号为0,2,5,7,9,11....
    // 从TEE返回的结果需加密编号为1,3,8,10,12,14....
    secure_computation sec_comp(values, perms, degree);

    start_time = std::clock();

    Matrix sec_output = secure_GCN_inference(g1, g2, trans_features, trans_weights, sec_comp);

    end_time = std::clock();
    elapsed_secs = double(end_time - start_time) / CLOCKS_PER_SEC * 1000;
    std::cout << "SecGCN推理运行时间: " << elapsed_secs << " 毫秒" << std::endl;

    fout.open(DATASET_PATH + "secgcn_output.txt");
    if (fout.is_open()) {
        fout << sec_output.array().log() << std::endl;
        fout.close();
    } else {
        std::cerr << "无法打开文件进行写入: " << DATASET_PATH + "secgcn_output.txt" << std::endl;
    }
    //  std::cout << "secure GCN Inference Output:\n" << sec_output.array().log() << std::endl;

    // 比较结果矩阵的相似度
    double error = (output - sec_output).norm();
    std::cout << "两种方法结果矩阵的误差范数为: " << error << std::endl;
    std::cout << "GCN分类准确率: " << compute_accuracy(output) << std::endl;
    std::cout << "SecGCN分类准确率: " << compute_accuracy(sec_output) << std::endl;
    return 0;
}