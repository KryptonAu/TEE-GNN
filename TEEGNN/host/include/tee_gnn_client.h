#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <tee_client_api.h>

#include "teegnn_ta.h"

using Matrix = Eigen::MatrixXd;

class TEEGNNClient {
public:
    TEEGNNClient();
    ~TEEGNNClient();
    
    // 初始化TEE连接
    bool initialize();
    
    // 初始化GNN上下文
    bool init_GNNContext(int num_vertices, int rank, 
                         const std::vector<uint32_t>& lmm_u);
    
    bool restore_aggregation(uint32_t layer_idx, Matrix& y1, Matrix& y2);

    // 执行非线性层
    bool nonlinear_layer(uint32_t layer_idx,
                         Matrix& linear_output,
                         Matrix& h_share,
                         const std::string& activation);
    
    // 清理资源
    void cleanup();
    
private:
    TEEC_Context context_;
    TEEC_Session session_;
    bool initialized_;
     
    // 检查TEE操作结果
    bool checkResult(TEEC_Result result, const std::string& operation);
};
