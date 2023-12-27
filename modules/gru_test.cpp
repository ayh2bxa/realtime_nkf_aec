#include "gru.h"

using namespace std;

int main() {
    GRUCell gru(3, 2);
    gru.Wrx.resize(6, 2.0);
    gru.Wzx.resize(6, 2.0);
    gru.Wnx.resize(6, 2.0);
    gru.Wrh.resize(6, 1.2);
    gru.Wzh.resize(6, 1.2);
    gru.Wnh.resize(6, 1.2);
    gru.brx.resize(2, 0.5);
    gru.bzx.resize(2, 0.5);
    gru.bnx.resize(2, 0.5);
    gru.brh.resize(2, 0.1);
    gru.bzh.resize(2, 0.1);
    gru.bnh.resize(2, 0.1);
    vector<float> input;
    input.resize(12, 2);
    vector<float> state;
    state.resize(2);
    state[0] = 1.5;
    state[1] = 2.5;
    gru.forward(input, state);
    vector<float> &h_t = gru.get_h();
    // for (int i = 0; i < 8; i++) {
    //     cout<<h_t[i]<<endl;
    // }
    return 0;
}
