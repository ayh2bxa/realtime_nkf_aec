#include <iostream>
#include "nkf.h"

int main() {
    NKF RT_NKF(FFTSIZE, MAX_BUFFER_SIZE, AMP, HOPSIZE);
    RT_NKF.process();
    return 0;
}