#include "gru.h"
#include "matrix_helper.h"
#include "activations.h"
#include <stdlib.h>

using namespace std;

GRUCell::GRUCell(int in_dim, int hidden_dim): d(in_dim), h(hidden_dim) {
   numBins = 513;
   Wrx.resize(h*d);
   Wzx.resize(h*d);
   Wnx.resize(h*d);
   Wrh.resize(h*h);
   Wzh.resize(h*h);
   Wnh.resize(h*h);

   brx.resize(h);
   bzx.resize(h);
   bnx.resize(h);
   brh.resize(h);
   bzh.resize(h);
   bnh.resize(h);

   r.resize(h);
   r1.resize(h);
   r2.resize(h);

   z.resize(h);
   z1m.resize(h);
   z1.resize(h);
   z2.resize(h);

   n.resize(h);
   n1.resize(h);
   n2.resize(h);
   n21.resize(h);

   h_t.resize(h);
   h1.resize(h);
   h2.resize(h);
   
   x.resize(d*numBins);
   output.resize(numBins*h);
}

GRUCell::~GRUCell() {}

std::vector<float>& GRUCell::get_h() {
   return output;
}

std::vector<float>& GRUCell::get_output() {
   return output;
}

void GRUCell::forward(V x, V h_prev_t) {
   // h, d are ints
   // N = 513, L = 1, Hin = 18
   // since we're dealing only with L = 1 the output IS h_n
   for (int i = 0; i < numBins; i++) {
      dot_product(r1, Wrx, h, d, x, brx, i*d);
      dot_product(r2, Wrh, h, d, x, brh, i*d);
      add_vector(r, r1, r2);
      sigmoid(r);
      // for (int i = 0; i < h; i++) {
      //    if (r[i] > 1.0f || r[i] < -0.0f) {
      //       cout<<"gru r value exceeded range: "<<r[i]<<endl;
      //    }
      // }
      dot_product(z1, Wzx, h, d, x, bzx, i*d);
      dot_product(z2, Wzh, h, d, h_prev_t, bzh, i*d);
      add_vector(z, z1, z2);
      sigmoid(z);
      // for (int i = 0; i < h; i++) {
      //    if (z[i] > 1.0f || z[i] < -0.0f) {
      //       cout<<"gru z value exceeded range: "<<z[i]<<endl;
      //    }
      // }
      for (int j = 0; j < h; j++) {
         z1m[j] = 1.0f-z[j];
      }
      // for (int i = 0; i < h; i++) {
      //    if (z1m[i] > 1.0f || z1m[i] < -0.0f) {
      //       cout<<"gru z value exceeded range: "<<z1m[i]<<endl;
      //    }
      // }
      dot_product(n1, Wnx, h, d, x, bnx, i*d);
      dot_product(n21, Wnh, h, d, h_prev_t, bnh, i*d);
      mult_vector(n2, r, n21, 0);
      add_vector(n, n1, n2);
      for (int i = 0; i < h; i++) {
         n[i] = fastTanh(n[i]);
      }
      mult_vector(h1, z1m, n, 0);
      mult_vector(h2, z, h_prev_t, i*h);
      add_vector(h_t, h1, h2);
      for (int j = 0; j < h; j++) {
         int pos = i*h+j;
         output[pos] = h_t[j];
      }
   }
}
