// tee_gcn_client.cpp
#include "tee_gnn_client.hpp"
#include <iostream>
#include <cstring>
#include <stdexcept>

namespace teegnn {

// TA UUID
static const TEEC_UUID ta_uuid = TEEGNN_TA_UUID;

TEEGNNClient::TEEGNNClient() : initialized_(false) {
    memset(&context_, 0, sizeof(context_));
    memset(&session_, 0, sizeof(session_));
}

TEEGNNClient::~TEEGNNClient() {
    cleanup();
}

bool TEEGNNClient::initialize() {
    TEEC_Result result = TEEC_InitializeContext(NULL, &context_);
    if (!checkResult(result, "InitializeContext")) {
        return false;
    }
    
    result = TEEC_OpenSession(&context_, &session_, &ta_uuid,
                              TEEC_LOGIN_PUBLIC, NULL, NULL, NULL);
    if (!checkResult(result, "OpenSession")) {
        TEEC_FinalizeContext(&context_);
        return false;
    }
    
    initialized_ = true;
    std::cout << "TEE GNN Client initialized successfully" << std::endl;
    return true;
}

bool TEEGNNClient::init_GNNContext(
    int num_vertices, int rank, 
    const std::vector<Matrix>& lmm_u) {
    
    if (!initialized_) {
        std::cerr << "TEE client not initialized" << std::endl;
        return false;
    }
    if (num_vertices <= 0 || rank < 0 || lmm_u.size() != 2) {
        std::cerr << "Invalid GNN context dimensions" << std::endl;
        return false;
    }

    // 设置操作
    TEEC_Operation op;
    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(
        TEEC_MEMREF_TEMP_INPUT,   // low_rank_mask precompute
        TEEC_VALUE_INPUT,         // num_vertices, rank
        TEEC_NONE,
        TEEC_NONE
    );

    std::vector<double> lmm_u_data;
    size_t lmm_u_size = 0;
    for (const auto& mat : lmm_u) {
        if (mat.rows() != num_vertices || mat.cols() != rank) {
            std::cerr << "Low-rank precompute matrix dimension mismatch" << std::endl;
            return false;
        }
        lmm_u_size += static_cast<size_t>(mat.size());
    }
    lmm_u_data.reserve(lmm_u_size);
    for (const auto& mat : lmm_u) {
        const size_t mat_size = static_cast<size_t>(mat.size());
        if (mat_size > 0) {
            lmm_u_data.insert(lmm_u_data.end(), mat.data(), mat.data() + mat_size);
        }
    }
    
    op.params[0].tmpref.buffer = lmm_u_data.empty() ? nullptr : (void*)lmm_u_data.data();
    op.params[0].tmpref.size = lmm_u_data.size() * sizeof(double);
    
    op.params[1].value.a = num_vertices;
    op.params[1].value.b = rank;
    
    // 调用TA
    TEEC_Result result = TEEC_InvokeCommand(
        &session_, TEEGNN_CMD_INIT_CONTEXT, &op, NULL);
    
    return checkResult(result, "InitGNNContext");
}

bool TEEGNNClient::restore_aggregation(uint32_t layer_idx, Matrix& y1, Matrix& y2) {
    if (!initialized_) {
        std::cerr << "TEE client not initialized" << std::endl;
        return false;
    }
    
    // 验证矩阵尺寸
    if (y1.rows() != y2.rows() ||
        y1.cols() != y2.cols()) {
        std::cerr << "Input matrices have different dimensions" << std::endl;
        return false;
    }
    
    uint32_t rows = y1.rows();
    uint32_t cols = y1.cols();

    // Matrix debug_info(rows, cols);
    
    // 设置操作
    TEEC_Operation op;
    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(
        TEEC_MEMREF_TEMP_INOUT,
        TEEC_MEMREF_TEMP_INPUT, 
        TEEC_VALUE_INPUT, 
        TEEC_NONE
    );
    
    op.params[0].tmpref.buffer = (void*)y1.data();
    op.params[0].tmpref.size = rows * cols * sizeof(double);
    
    op.params[1].tmpref.buffer = (void*)y2.data();
    op.params[1].tmpref.size = rows * cols * sizeof(double);

    op.params[2].value.a = layer_idx;  // layer_idx
    op.params[2].value.b = cols;       // feature_dim
    
    // 调用TA
    TEEC_Result result = TEEC_InvokeCommand(
        &session_, TEEGNN_CMD_RESTORE_AGGREGATION, &op, NULL);
    
    return checkResult(result, "ComputeNonlinearLayer");
}

bool TEEGNNClient::nonlinear_layer(
    uint32_t layer_idx,
    Matrix& linear_output,
    Matrix& h_share,
    const std::string& activation) {
    
    if (!initialized_) {
        std::cerr << "TEE client not initialized" << std::endl;
        return false;
    }
    
    // 验证矩阵尺寸
    if (linear_output.rows() != h_share.rows() ||
        linear_output.cols() != h_share.cols()) {
        std::cerr << "Input matrices have different dimensions" << std::endl;
        return false;
    }
    
    uint32_t rows = linear_output.rows();
    uint32_t cols = linear_output.cols();
    
    // 设置操作
    TEEC_Operation op;
    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(
        TEEC_MEMREF_TEMP_INOUT,   // masked H1_0
        TEEC_MEMREF_TEMP_OUTPUT,   // masked H1_1
        TEEC_VALUE_INPUT,  // layer_idx and activation
        TEEC_NONE
    );
    
    // 编码额外信息到memref.size
    uint32_t activation_code = (activation == "ReLU") ? 0 : 1;
    uint32_t encoded_size = (layer_idx << 16) | activation_code;
    
    op.params[0].tmpref.buffer = (void*)linear_output.data();
    op.params[0].tmpref.size = rows * cols * sizeof(double);
    
    op.params[1].tmpref.buffer = (void*)h_share.data();
    op.params[1].tmpref.size = rows * cols * sizeof(double);
    
    op.params[2].value.a = encoded_size;  // 高16位: layer_idx, 低16位: activation
    op.params[2].value.b = cols;  // 特征维度
    
    // 调用TA
    TEEC_Result result = TEEC_InvokeCommand(
        &session_, TEEGNN_CMD_APPLY_ACTIVATION, &op, NULL);
    
    return checkResult(result, "ComputeNonlinearLayer");
}

bool TEEGNNClient::get_debug_info(IntVector& debug_info) {
    if (!initialized_) {
        std::cerr << "TEE client not initialized" << std::endl;
        return false;
    }
    TEEC_Operation op;
    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(
        TEEC_MEMREF_TEMP_OUTPUT,
        TEEC_NONE, 
        TEEC_NONE, 
        TEEC_NONE
    );
    op.params[0].tmpref.buffer = (void*)debug_info.data();
    op.params[0].tmpref.size = debug_info.size() * sizeof(int);
    
    TEEC_Result result = TEEC_InvokeCommand(
        &session_, TEEGNN_CMD_GET_DEBUG_INFO, &op, NULL);

    for (int i = 0; i < debug_info.size(); ++i) {
        std::cout << debug_info[i] << " ";
    }
    std::cout << std::endl;
    return checkResult(result, "GetDebugInfo");
}

void TEEGNNClient::cleanup() {
    if (initialized_) {
        // 清理TA上下文
        TEEC_Operation op;
        memset(&op, 0, sizeof(op));
        TEEC_InvokeCommand(&session_, TEEGNN_CMD_FINALIZE_RESULT, &op, NULL);
        
        // 关闭会话
        TEEC_CloseSession(&session_);
        TEEC_FinalizeContext(&context_);
        
        initialized_ = false;
        std::cout << "TEE GNN Client cleaned up" << std::endl;
    }
}

bool TEEGNNClient::checkResult(TEEC_Result result, const std::string& operation) {
    if (result != TEEC_SUCCESS) {
        std::cerr << operation << " failed: 0x" << std::hex << result << std::dec;
        switch (result) {
            case TEEC_ERROR_GENERIC:
                std::cerr << " (Generic error)" << std::endl; break;
            case TEEC_ERROR_ACCESS_DENIED:
                std::cerr << " (Access denied)" << std::endl; break;
            case TEEC_ERROR_CANCEL:
                std::cerr << " (Operation cancelled)" << std::endl; break;
            case TEEC_ERROR_ACCESS_CONFLICT:
                std::cerr << " (Access conflict)" << std::endl; break;
            case TEEC_ERROR_EXCESS_DATA:
                std::cerr << " (Excess data)" << std::endl; break;
            case TEEC_ERROR_BAD_FORMAT:
                std::cerr << " (Bad format)" << std::endl; break;
            case TEEC_ERROR_BAD_PARAMETERS:
                std::cerr << " (Bad parameters)" << std::endl; break;
            case TEEC_ERROR_BAD_STATE:
                std::cerr << " (Bad state)" << std::endl; break;
            case TEEC_ERROR_ITEM_NOT_FOUND:
                std::cerr << " (Item not found)" << std::endl; break;
            case TEEC_ERROR_NOT_IMPLEMENTED:
                std::cerr << " (Not implemented)" << std::endl; break;
            case TEEC_ERROR_NOT_SUPPORTED:
                std::cerr << " (Not supported)" << std::endl; break;
            case TEEC_ERROR_NO_DATA:
                std::cerr << " (No data)" << std::endl; break;
            case TEEC_ERROR_OUT_OF_MEMORY:
                std::cerr << " (Out of memory)" << std::endl; break;
            case TEEC_ERROR_BUSY:
                std::cerr << " (Busy)" << std::endl; break;
            case TEEC_ERROR_COMMUNICATION:
                std::cerr << " (Communication error)" << std::endl; break;
            case TEEC_ERROR_SECURITY:
                std::cerr << " (Security error)" << std::endl; break;
            case TEEC_ERROR_SHORT_BUFFER:
                std::cerr << " (Short buffer)" << std::endl; break;
            case TEEC_ERROR_EXTERNAL_CANCEL:
                std::cerr << " (External cancel)" << std::endl; break;
            default:
                std::cerr << " (Unknown error)" << std::endl; break;
        }
        return false;
    }
    return true;
}

}  // namespace teegnn
