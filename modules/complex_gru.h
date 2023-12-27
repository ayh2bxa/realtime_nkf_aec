#include "gru.h"
#include <vector>
#include <complex>

using namespace std;
using V = vector<float>&;

class ComplexGRU {
private:
    int h;
    std::vector<float> out_real;
    std::vector<float> out_imag;
    GRUCell gru_r;
    GRUCell gru_i;

public:
    ComplexGRU(int in_dim, int hidden_dim);
    ~ComplexGRU();
    void forward(V x_real, V x_imag, V h_rr, V h_ir, V h_ri, V h_ii);
    vector<float>& get_out_re();
    vector<float>& get_out_im();
    // vector<float>& Frr;
    // vector<float>& Fir;
    // vector<float>& Fri;
    // vector<float>& Fii;
};
