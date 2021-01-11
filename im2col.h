#ifndef IM2COL_H
#define IM2COL_H

#include <stdint.h>
void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);

void im2col_cpu_int8(int8_t* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, int8_t* data_col);
#endif
