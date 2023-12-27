// ComplexGRU(input_size = 18, hidden_size = 18, num_layers = 1, batch_first = True, bias = True, dropout = 0, bidirectional = False)
#pragma once
#include <vector>
#include <iostream>

using namespace std;
using V = vector<float>&;

class GRUCell {
private:
    int d;  // in_dim
    int h;  // hidden_dim

public:
    GRUCell(int in_dim, int hidden_dim);
    ~GRUCell();
    // template <typename V>
    void forward(V x, V h_prev_t);
    vector<float> x;

    vector<float> Wrx, Wzx, Wnx, Wrh, Wzh, Wnh;
    vector<float> brx, bzx, bnx, brh, bzh, bnh;
    
    vector<float> r, r1, r2;
    vector<float> z, z1, z2, z1m;
    vector<float> n, n1, n2, n21;
    vector<float> h_t, h1, h2;
    vector<float> output;
    int numBins;
    std::vector<float>& get_h();
    std::vector<float>& get_output();
};
