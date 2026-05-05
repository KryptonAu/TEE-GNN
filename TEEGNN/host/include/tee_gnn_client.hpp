#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <tee_client_api.h>

#include "types.hpp"
#include "blocked_csc.h"
#include "util.hpp"

namespace teegnn {

class TEEGNNClient {
public:
    TEEGNNClient();
    ~TEEGNNClient();
    
    bool initialize();
    
    bool init_GNNContext(Matrix& w1, const Secrets& secrets, uint32_t feature_dim, uint32_t hidden_dim);
    
    // message passing and activation function
    bool secure_compute(const EncryptedBlockedCSC* csc, Matrix& y);

    bool get_debug_info(IntVector& debug_info);
    
    void cleanup();
    
private:
    TEEC_Context context_;
    TEEC_Session session_;
    bool initialized_;
    
    std::vector<uint8_t> secret_pack(const Secrets& secrets);
    bool checkResult(TEEC_Result result, const std::string& operation);
};

}  // namespace teegnn