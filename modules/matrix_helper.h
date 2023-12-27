#pragma once
#include <iostream>
#include <complex>
#include <vector>
#include <array>
#include <cmath>
// #include "webrtc_AEC3/audio_processing/aec3/aec3_fft.h"
// #include "webrtc_AEC3/audio_processing/aec3/fft_data.h"

using namespace std;
// using namespace webrtc;
using V = vector<float>&;

void dot_product(V output, V weights, int rows, int cols, V input, V bias, int start);

void add_vector(V output, V v1, V v2);

void subtract_vector(V output, V v1, V v2);

void mult_vector(V output, V v1, V v2, int start);

// input shape: fftPts * rows
// weights shape: cols * rows
// output shape: fftPts * cols
void dense(V output, V weights, int rows, int cols, int fftPts, V input, V bias, float *temp);

// input1 (dim0, dim1, dim2)
// input2 (dim0, dim2, dim3)
// output (dim0, dim1, dim3)
void matmulFft(V output_re, V output_im, int dim0, int dim1, int dim2, int dim3, array<array<complex<float>, 513>*, 4>& inputFft, V input2_re, V input2_im);

// input1 (dim0, dim1, dim2)
// input2 (dim0, dim2, dim3)
// output (dim0, dim1, dim3)
void matmul(V output_re, V output_im, int dim0, int dim1, int dim2, int dim3, V input1_re, V input1_im, V input2_re, V input2_im);
