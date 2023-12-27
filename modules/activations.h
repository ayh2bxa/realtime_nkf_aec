#pragma once
#include <complex>
#include <cmath>
#include <vector>
#define IMAX(a, b) ((a) > (b) ? (a) : (b))
#define IMIN(a, b) ((a) < (b) ? (a) : (b))

using namespace std;
using V = vector<float>&;

float fastTanh(float x);

void sigmoid(V arr);

void tanh(V arr);

void pRelu(V x, float a);