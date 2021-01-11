#ifndef GEMM_H
#define GEMM_H

#include <stdint.h>
#include "calloc.h"
void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);
        
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

void gemm_nn_int8_int16(int M, int N, int K, int8_t ALPHA,
    int8_t *A, int lda,
    int8_t *B, int ldb,
    int16_t *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

#endif
