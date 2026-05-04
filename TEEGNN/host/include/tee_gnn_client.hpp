#pragma once

#include <vector>
#include <string>
#include <tee_client_api.h>

#include "types.hpp"
#include "teegnn_ta.h"

namespace teegnn {

class TEEGNNClient {
public:
    TEEGNNClient();
    ~TEEGNNClient();
    
    bool initialize();
    
    bool init_GNNContext(int num_vertices, int rank, 
                         int feature_dim, int hidden_dim, 
                         Matrix& w1, const std::vector<Matrix>& lmm_u);
    
    // message passing and activation function
    bool secure_compute(uint32_t layer_idx, Matrix& y1, Matrix& y2);

    bool get_debug_info(IntVector& debug_info);
    
    void cleanup();
    
private:
    TEEC_Context context_;
    TEEC_Session session_;
    bool initialized_;
     
    bool checkResult(TEEC_Result result, const std::string& operation);
};

}  // namespace teegnn