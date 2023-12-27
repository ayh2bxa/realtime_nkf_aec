#include "complex_gru.h"
#include "complex.h"
#include <array>

using namespace std;
using V = vector<float>&;

class KGNet {
public:
    KGNet(int L, int fc_dim, int rnn_layers, int rnn_dim, int fftPoints);
    ~KGNet();
    // template <typename V>
    void forward(V input_feature_re, V input_feature_im);
    vector<float>& get_kg_re();
    vector<float>& get_kg_im();
    void transpose(vector<float>& res, vector<float>& input, int rows, int cols);
private:
    int num_frames;
    int input_dim;
    int num_layers;
    int hidden_dim;
    vector<float> feat_re;
    vector<float> feat_im;

    float a1, a2;
    ComplexGRU complex_gru;

    vector<float> h_rr;
    vector<float> h_ir;
    vector<float> h_ri;
    vector<float> h_ii;

    vector<float> weights1_real;
    vector<float> weights1_imag;
    vector<float> weights2_real;
    vector<float> weights2_imag;
    vector<float> weights3_real;
    vector<float> weights3_imag;

    vector<float> bias1_real;
    vector<float> bias1_imag;
    vector<float> bias2_real;
    vector<float> bias2_imag;
    vector<float> bias3_real;
    vector<float> bias3_imag;

    vector<float> kg_re;
    vector<float> kg_im;

    int fftPts;

    vector<float> rnn_out_re1;
    vector<float> rnn_out_im1;

    float temp[4];
};