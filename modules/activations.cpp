#include "activations.h"
#include "simd_size.h"

const float tansig_table[201] = {
0.000000f, 0.039979f, 0.079830f, 0.119427f, 0.158649f,
0.197375f, 0.235496f, 0.272905f, 0.309507f, 0.345214f,
0.379949f, 0.413644f, 0.446244f, 0.477700f, 0.507977f,
0.537050f, 0.564900f, 0.591519f, 0.616909f, 0.641077f,
0.664037f, 0.685809f, 0.706419f, 0.725897f, 0.744277f,
0.761594f, 0.777888f, 0.793199f, 0.807569f, 0.821040f,
0.833655f, 0.845456f, 0.856485f, 0.866784f, 0.876393f,
0.885352f, 0.893698f, 0.901468f, 0.908698f, 0.915420f,
0.921669f, 0.927473f, 0.932862f, 0.937863f, 0.942503f,
0.946806f, 0.950795f, 0.954492f, 0.957917f, 0.961090f,
0.964028f, 0.966747f, 0.969265f, 0.971594f, 0.973749f,
0.975743f, 0.977587f, 0.979293f, 0.980869f, 0.982327f,
0.983675f, 0.984921f, 0.986072f, 0.987136f, 0.988119f,
0.989027f, 0.989867f, 0.990642f, 0.991359f, 0.992020f,
0.992631f, 0.993196f, 0.993718f, 0.994199f, 0.994644f,
0.995055f, 0.995434f, 0.995784f, 0.996108f, 0.996407f,
0.996682f, 0.996937f, 0.997172f, 0.997389f, 0.997590f,
0.997775f, 0.997946f, 0.998104f, 0.998249f, 0.998384f,
0.998508f, 0.998623f, 0.998728f, 0.998826f, 0.998916f,
0.999000f, 0.999076f, 0.999147f, 0.999213f, 0.999273f,
0.999329f, 0.999381f, 0.999428f, 0.999472f, 0.999513f,
0.999550f, 0.999585f, 0.999617f, 0.999646f, 0.999673f,
0.999699f, 0.999722f, 0.999743f, 0.999763f, 0.999781f,
0.999798f, 0.999813f, 0.999828f, 0.999841f, 0.999853f,
0.999865f, 0.999875f, 0.999885f, 0.999893f, 0.999902f,
0.999909f, 0.999916f, 0.999923f, 0.999929f, 0.999934f,
0.999939f, 0.999944f, 0.999948f, 0.999952f, 0.999956f,
0.999959f, 0.999962f, 0.999965f, 0.999968f, 0.999970f,
0.999973f, 0.999975f, 0.999977f, 0.999978f, 0.999980f,
0.999982f, 0.999983f, 0.999984f, 0.999986f, 0.999987f,
0.999988f, 0.999989f, 0.999990f, 0.999990f, 0.999991f,
0.999992f, 0.999992f, 0.999993f, 0.999994f, 0.999994f,
0.999994f, 0.999995f, 0.999995f, 0.999996f, 0.999996f,
0.999996f, 0.999997f, 0.999997f, 0.999997f, 0.999997f,
0.999997f, 0.999998f, 0.999998f, 0.999998f, 0.999998f,
0.999998f, 0.999998f, 0.999999f, 0.999999f, 0.999999f,
0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f,
0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f,
1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
1.000000f,
};

float fastTanh(float x)
{
    int i;
    float y, dy;
    float sign=1;
    if (x<0)
    {
       x=-x;
       sign=-1;
    }
    i = (int)floor(.5f+25*x);
    i = IMAX(0, IMIN(200, i));
    x -= .04f*i;
    y = tansig_table[i];
    dy = 1-y*y;
    y = y + x*dy*(1 - y*x);
    return sign*y;
}

// inline __mx fastTanh(__mx x)
// {
//     __mx sign = _mm_set1_ps(1.0f);
//     __mx mask = _mm_cmp_ps(x, _mm_setzero_ps(), _CMP_LT_OS);
//     x = _mm_andnot_ps(mask, x); // abs(x)
//     sign = _mm_blendv_ps(sign, _mm_set1_ps(-1.0f), mask);

//     __mx i = _mm_floor_ps(_mm_add_ps(_mm_mul_ps(x, _mm_set1_ps(25.0f)), _mm_set1_ps(0.5f)));
//     i = _mm_max_ps(_mm_setzero_ps(), _mm_min_ps(_mm_set1_ps(200.0f), i));
    
//     x = _mm_sub_ps(x, _mm_mul_ps(i, _mm_set1_ps(0.04f)));
    
//     // You will need to vectorize the tansig_table lookup
//     __mx y = ...; // vectorized lookup in tansig_table using indices in 'i'
//     __mx dy = _mm_sub_ps(_mm_set1_ps(1.0f), _mm_mul_ps(y, y));
    
//     y = _mm_add_ps(y, _mm_mul_ps(x, _mm_mul_ps(dy, _mm_sub_ps(_mm_set1_ps(1.0f), _mm_mul_ps(y, x)))));
    
//     return _mm_mul_ps(sign, y);
// }

void sigmoid(V arr) {
   for (int i = 0; i < arr.size(); i++) {
      arr[i] = 0.5f+0.5f*fastTanh(0.5f*arr[i]);
   }
   // __mx half = _mm_set1_ps(0.5f);
   // for (int i = 0; i < arr.size()-SIMD_SIZE-+1; i += SIMD_SIZE) {
   //    __mx vec = _mm_loadu_ps(&arr[i]);
   //    vec = _mm_mul_ps(half, _mm_add_ps(half, _mm_mul_ps(half, fastTanh(_mm_mul_ps(half, vec)))));
   //    _mm_storeu_ps(&arr[i], vec);
   // }
}

void tanh(V arr) {
   for (int i = 0; i < arr.size(); i++) {
      arr[i] = fastTanh(arr[i]);
   }
}

void pRelu(V x, float a) {
   __m128 a_vec = _mm_set1_ps(a);
   __m128 zero = _mm_setzero_ps();

   for (int i = 0; i < x.size()-(x.size()%SIMD_SIZE); i += SIMD_SIZE) {
      __m128 x_vec = _mm_loadu_ps(&x[i]);
      __m128 mask = _mm_cmpge_ps(x_vec, zero);
      __m128 result = _mm_or_ps(
         _mm_and_ps(mask, x_vec),
         _mm_andnot_ps(mask, _mm_mul_ps(a_vec, x_vec))
      );
      _mm_storeu_ps(&x[i], result);
   }

   // Handle remaining elements
   for (int i = x.size() - (x.size() % SIMD_SIZE); i < x.size(); i++) {
      x[i] = x[i] >= 0 ? x[i] : a * x[i];
   }
}