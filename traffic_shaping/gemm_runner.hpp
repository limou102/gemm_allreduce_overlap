#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <stdexcept>
#include <iostream>

#define CHECK_ZERO(x) \
    do { \
        auto __error_code = (x); \
        if (static_cast<int32_t>(__error_code) != 0) { \
            std::cerr << "[" << __LINE__ << "] " << "non zero error code : " \
                << __error_code << std::endl; \
            std::terminate(); \
        } \
    } while(false)


class GemmRunner final
{
public:
    // A (mxk)
    // B (nxk)
    // C (mxn)
    // C = 1.0f * A * B(T) + 0.0f * C
    GemmRunner() = default;

    ~GemmRunner()
    {
        // TODO : destroy resources
        // ...
    }

    void Init(uint64_t workspace_size, int32_t batch_size, uint64_t m, uint64_t n, uint64_t k) {
        CHECK_ZERO(hipblasLtCreate(&handle_));
        CHECK_ZERO(hipblasLtMatrixLayoutCreate(&mat_layout_a_, HIP_R_32F, m, k, k));
        CHECK_ZERO(hipblasLtMatrixLayoutCreate(&mat_layout_b_, HIP_R_32F, n, k, n));
        CHECK_ZERO(hipblasLtMatrixLayoutCreate(&mat_layout_c_, HIP_R_32F, m, n, n));
        if (batch_size > 1) {
            int64_t stride_a = m * k;
            int64_t stride_b = n * k;
            int64_t stride_c = m * n;
            CHECK_ZERO(hipblasLtMatrixLayoutSetAttribute(
                mat_layout_a_, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size)));
            CHECK_ZERO(hipblasLtMatrixLayoutSetAttribute(
                mat_layout_a_, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a)));
            CHECK_ZERO(hipblasLtMatrixLayoutSetAttribute(
                mat_layout_b_, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size)));
            CHECK_ZERO(hipblasLtMatrixLayoutSetAttribute(
                mat_layout_b_, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b)));
            CHECK_ZERO(hipblasLtMatrixLayoutSetAttribute(
                mat_layout_c_, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size)));
            CHECK_ZERO(hipblasLtMatrixLayoutSetAttribute(
                mat_layout_c_, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c)));
        }

        CHECK_ZERO(hipblasLtMatmulDescCreate(&matmul_desc_, HIPBLAS_COMPUTE_32F, HIP_R_32F));

        hipblasOperation_t trans_b = HIPBLAS_OP_T;
        CHECK_ZERO(hipblasLtMatmulDescSetAttribute(matmul_desc_, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

        int returnedAlgoCount = 0;
        hipblasLtMatmulPreference_t pref;
        CHECK_ZERO(hipblasLtMatmulPreferenceCreate(&pref));
        CHECK_ZERO(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
        CHECK_ZERO(hipblasLtMatmulAlgoGetHeuristic(handle_,
                    matmul_desc_,
                    mat_layout_a_,
                    mat_layout_b_,
                    mat_layout_c_,
                    mat_layout_c_,
                    pref,
                    1,
                    &result_,
                    &returnedAlgoCount));

        CHECK_ZERO(!(returnedAlgoCount == 1));

        CHECK_ZERO(!(result_.workspaceSize <= workspace_size));
        ws_size_ = result_.workspaceSize;

        CHECK_ZERO(hipblasLtMatmulPreferenceDestroy(pref));

        CHECK_ZERO(hipMalloc(&d_a_, m*k*batch_size*sizeof(float)));
        CHECK_ZERO(hipMemset(d_a_, 0, m*k*batch_size*sizeof(float)));
        CHECK_ZERO(hipMalloc(&d_b_, n*k*batch_size*sizeof(float)));
        CHECK_ZERO(hipMemset(d_b_, 0, n*k*batch_size*sizeof(float)));
        CHECK_ZERO(hipMalloc(&d_c_, m*n*batch_size*sizeof(float)));
        CHECK_ZERO(hipMemset(d_c_, 0, m*n*batch_size*sizeof(float)));

        CHECK_ZERO(hipMalloc(&d_ws_, result_.workspaceSize));
        CHECK_ZERO(hipMemset(d_c_, 0, m*n*batch_size*sizeof(float)));
        CHECK_ZERO(hipDeviceSynchronize());
    }

    void RunSelf(hipStream_t stream) {
        CHECK_ZERO(hipblasLtMatmul(
                    handle_,
                    matmul_desc_,
                    &alpha_,
                    d_a_,
                    mat_layout_a_,
                    d_b_,
                    mat_layout_b_,
                    &beta_,
                    d_c_,
                    mat_layout_c_,
                    d_c_,
                    mat_layout_c_,
                    &result_.algo,
                    d_ws_,
                    ws_size_,
                    stream));
    }

private:
    hipblasLtHandle_t handle_;
    hipblasLtMatrixLayout_t mat_layout_a_;
    hipblasLtMatrixLayout_t mat_layout_b_;
    hipblasLtMatrixLayout_t mat_layout_c_;
    hipblasLtMatmulDesc_t matmul_desc_;
    hipblasLtMatmulHeuristicResult_t result_;

    float *d_a_ {nullptr};
    float *d_b_ {nullptr};
    float *d_c_ {nullptr};
    uint8_t *d_ws_ {nullptr};
    uint64_t ws_size_ {0};

    const float alpha_ {1.0f};
    const float beta_ {0.0f};
};

