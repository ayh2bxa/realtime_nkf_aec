#include "matrix_helper.h"
#include "simd_size.h"

using namespace std;

void dot_product(V output, V weights, int rows, int cols, V input, V bias, int start)
{
   for (int i = 0; i < rows; i++)
   {
      float sum = 0.0f;
      for (int j = 0; j < cols; j++) {
         int pos = i*cols+j;
         sum += weights[pos]*input[j+start];
      }
      output[i] = sum+bias[i];
   }
}

void add_vector(V output, V v1, V v2) {
   int i = 0;
   for (i = 0; i <= v1.size() - (v1.size()%SIMD_SIZE); i += SIMD_SIZE) {
      __mx a = _mm_loadu_ps(&v1[i]);
      __mx b = _mm_loadu_ps(&v2[i]);
      __mx c = _mm_add_ps(a, b);
      _mm_storeu_ps(&output[i], c);
   }
   for (; i < v1.size(); i++) {
      output[i] = v1[i] + v2[i];
   }
   // for (int i = 0; i < v1.size(); i++) {
   //    output[i] = v1[i]+v2[i];
   // }
}

void subtract_vector(V output, V v1, V v2) {
   int i = 0;
   for (i = 0; i <= v2.size() - (v2.size()%SIMD_SIZE); i += SIMD_SIZE) {
      __mx a = _mm_loadu_ps(&v1[i]);
      __mx b = _mm_loadu_ps(&v2[i]);
      __mx c = _mm_sub_ps(a, b);
      _mm_storeu_ps(&output[i], c);
   }
   for (; i < v2.size(); i++) {
      output[i] = v1[i] - v2[i];
   }
   // for (int i = 0; i < v2.size(); i++) {
   //    output[i] = v1[i]-v2[i];
   // }
}

void mult_vector(V output, V v1, V v2, int start) {
   // for (int i = 0; i < v1.size(); i++) {
   //    output[i] = v1[i]*v2[start+i];
   // }
   int i = 0;
   for (i = 0; i <= v1.size() - (v1.size()%SIMD_SIZE); i += SIMD_SIZE) {
      __mx a = _mm_loadu_ps(&v1[i]);
      __mx b = _mm_loadu_ps(&v2[start+i]);
      __mx c = _mm_mul_ps(a, b);
      _mm_storeu_ps(&output[i], c);
   }
   for (; i < v1.size(); i++) {
      output[i] = v1[i] * v2[start+i];
   }
}

void transpose(vector<float>& res, vector<float>& input, int rows, int cols) {
   for (int i = 0; i < rows; i++) {
         for (int j = 0; j < cols; j++) {
            int pos1 = i*cols+j;
            int pos2 = j*rows+i;
            res[pos2] = input[pos1];
         }
   }
}

// input shape: fftPts * rows
// weights shape: cols * rows
// output shape: fftPts * cols
void dense(V output, V weights, int rows, int cols, int fftPts, V input, V bias, float *temp) {
   if (rows < SIMD_SIZE) {
      for (int b = 0; b < fftPts; b++) {
         for (int j = 0; j < cols; j++) {
            float sum = 0.0f;
            for (int i = 0; i < rows; i++) {
                  sum += input[b*rows+i]*weights[i*cols+j];
            }
            output[b*cols+j] = sum + bias[j];
         }
      }
   }
   else {
      __m128 sum;
      __m128 a;
      __m128 b;
      for (int i = 0; i < fftPts; ++i) { // m = fftPts
         for (int j = 0; j < cols; ++j) { // n = cols
            sum = _mm_setzero_ps();
            output[i*cols+j] = 0.f;
            for (int k = 0; k < rows-(rows%SIMD_SIZE); k += SIMD_SIZE) { // k = rows
                  a = _mm_loadu_ps(&input[i * rows + k]);
                  b = _mm_loadu_ps(&weights[j * rows + k]);
                  sum = _mm_add_ps(sum, _mm_mul_ps(a, b));
            }
            for (int l = 0; l < SIMD_SIZE; l++) {
               output[i*cols+j] += sum[l];
            }
            // Handle remaining elements
            for (int k = rows - (rows % 4); k < rows; ++k) {
                  output[i * cols + j] += input[i * rows + k] * weights[j * rows + k];
            }
            // Add bias
            output[i * cols + j] += bias[j];
         }
      }
   }
}

// input1 (dim0, dim1, dim2)
// input2 (dim0, dim2, dim3)
// output (dim0, dim1, dim3)
void matmulFft(V output_re, V output_im, int dim0, int dim1, int dim2, int dim3, array<array<complex<float>, 513>*, 4>& inputFft, V input2_re, V input2_im) {
   for (int i = 0; i < dim0; i++) {
      for (int j = 0; j < dim1; j++) {
         for (int k = 0; k < dim3; k++) {
            float sum_re = 0.0f;
            float sum_im = 0.0f;
            for (int l = 0; l < dim2; l++) {
               array<complex<float>, 513>* fft = inputFft[l];
               float fft_re = (*fft)[i].real();
               float fft_im = (*fft)[i].imag();
               float input2_re_val = input2_re[i*dim2*dim3+l*dim3+k];
               // if (input2_re_val >= 1.0f) {
               //    cout<<"matmulfft input2: "<<input2_re_val<<endl;;
               // }
               float input2_im_val = input2_im[i*dim2*dim3+l*dim3+k];
               sum_re += fft_re * input2_re_val
                        - fft_im * input2_im_val;
               sum_im += fft_re * input2_im_val 
                            + fft_im * input2_re_val;
            }
            output_re[i*dim1*dim3+j*dim3+k] = sum_re;
            output_im[i*dim1*dim3+j*dim3+k] = sum_im;
         }
      }
   }
}

// input1 (dim0, dim1, dim2)
// input2 (dim0, dim2, dim3)
// output (dim0, dim1, dim3)
void matmul(V output_re, V output_im, int dim0, int dim1, int dim2, int dim3, V input1_re, V input1_im, V input2_re, V input2_im) {
   for (int i = 0; i < dim0; i++) {
      for (int j = 0; j < dim1; j++) {
         for (int k = 0; k < dim3; k++) {
            float sum_re = 0.0f;
            float sum_im = 0.0f;
            for (int l = 0; l < dim2; l++) {
               sum_re += input1_re[i*dim1*dim2+j*dim2+l] * input2_re[i*dim2*dim3+l*dim3+k]
                        - input1_im[i*dim1*dim2+j*dim2+l] * input2_im[i*dim2*dim3+l*dim3+k];
               sum_im += input1_re[i*dim1*dim2+j*dim2+l] * input2_im[i*dim2*dim3+l*dim3+k]
                            + input1_im[i*dim1*dim2+j*dim2+l] * input2_re[i*dim2*dim3+l*dim3+k];
            }
            output_re[i*dim1*dim3 + j*dim3 + k] = sum_re;
            output_im[i*dim1*dim3 + j*dim3 + k] = sum_im;
         }
      }
   }
}