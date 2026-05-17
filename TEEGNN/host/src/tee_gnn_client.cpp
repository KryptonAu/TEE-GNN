// tee_gcn_client.cpp
#include "tee_gnn_client.hpp"
#include "crypto.h"
#include "tee_client_api.h"
#include "teegnn_ta.h"

#include <cstdint>
#include <iostream>
#include <cstring>

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

    std::cout<<"context_.imp.reg_mem:"<<context_.imp.reg_mem<< '\n';
    
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

bool TEEGNNClient::init_GNNContext(Matrix& w1, const Secrets& secrets, uint32_t feature_dim, uint32_t hidden_dim) {
    if (!initialized_) {
        std::cerr << "TEE client not initialized" << std::endl;
        return false;
    }

    TEEC_Operation op;
    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(
        TEEC_MEMREF_TEMP_INOUT,   // w1
        TEEC_MEMREF_TEMP_INPUT,   // seeds and keys
        TEEC_VALUE_INPUT,         // feature_dim, hidden_dim
        TEEC_NONE       
    );

    row_block_size = secrets.row_block_size;

    std::vector<uint8_t> pack = secret_pack(secrets);

    op.params[0].tmpref.buffer = (void*)w1.data();
    op.params[0].tmpref.size = feature_dim * hidden_dim * sizeof(double);
    
    op.params[1].tmpref.buffer = pack.data();
    op.params[1].tmpref.size = pack.size();

    op.params[2].value.a = feature_dim;
    op.params[2].value.b = hidden_dim; 

    
    TEEC_Result result = TEEC_InvokeCommand(
        &session_, TEEGNN_CMD_INIT_CONTEXT, &op, NULL);
    
    return checkResult(result, "InitGNNContext");
}

bool TEEGNNClient::secure_compute(EncryptedBlockedEdgeList *lst, Matrix& y) {
    if (!initialized_) {
        std::cerr << "TEE client not initialized" << std::endl;
        return false;
    }
    TEEC_Result res;
    TEEC_SharedMemory shm;
    std::memset(&shm, 0, sizeof(shm));

    shm.buffer = (void *)lst;
    shm.size = lst->total_size;
    shm.flags = TEEC_MEM_INPUT;

    res = TEEC_RegisterSharedMemory(&context_, &shm);
    if (res != TEEC_SUCCESS) {
        std::cerr << "TEEC_RegisterSharedMemory failed: 0x"
                  << std::hex << res << std::dec << "\n";
        return false;
    }

    TEEC_Operation op;
    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(
        TEEC_MEMREF_TEMP_INOUT,
        TEEC_MEMREF_WHOLE, 
        TEEC_MEMREF_TEMP_INOUT,
        TEEC_VALUE_INPUT
    );

    uint32_t rows = y.rows();
    uint32_t cols = y.cols();
    size_t row_blocks = (rows + row_block_size - 1) / row_block_size;

    temp_ciphertext.resize(row_blocks * (row_block_size + 28) * cols * sizeof(double));
    
    op.params[0].tmpref.buffer = (void*)y.data();
    op.params[0].tmpref.size = rows * cols * sizeof(double);
    
    op.params[1].memref.parent = &shm;
    op.params[1].memref.size = shm.size;
    op.params[1].memref.offset = 0;

    // op.params[1].tmpref.buffer = (void*)lst;
    // op.params[1].tmpref.size = lst->total_size;

    op.params[2].tmpref.buffer = temp_ciphertext.data();
    op.params[2].tmpref.size = rows * cols * sizeof(double) + 4096;

    op.params[3].value.a = rows;        // num_nodes
    op.params[3].value.b = cols;        // feature_dim
    
    TEEC_Result result = TEEC_InvokeCommand(
        &session_, TEEGNN_CMD_SECURE_COMPUTE, &op, NULL);
    
    TEEC_ReleaseSharedMemory(&shm);

    return checkResult(result, "secure compute");
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

std::vector<uint8_t> TEEGNNClient::secret_pack(const Secrets& secrets) {
    size_t total_size = 0;
    total_size += sizeof(uint32_t);
    total_size += 2 * sizeof(uint64_t);
    total_size += 2 * TEEGNN_AES128_KEY_LEN * sizeof(uint8_t);
    
    std::vector<uint8_t> pack(total_size);
    size_t offset = 0;

    memcpy(pack.data() + offset, &secrets.row_block_size, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    memcpy(pack.data() + offset, &secrets.seed_data, sizeof(uint64_t));
    offset += sizeof(uint64_t);
    memcpy(pack.data() + offset, &secrets.seed_model, sizeof(uint64_t));
    offset += sizeof(uint64_t);
    
    memcpy(pack.data() + offset, secrets.key1.data(), secrets.key1.size());
    offset += secrets.key1.size();
    memcpy(pack.data() + offset, secrets.key2.data(), secrets.key2.size());
    offset += secrets.key2.size();

    return pack;
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
    std::cout << operation << " succeeded" << std::endl;
    return true;
}

}  // namespace teegnn
