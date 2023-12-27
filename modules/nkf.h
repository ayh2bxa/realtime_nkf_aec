#include <array>
#include <complex>
#include "kgnet.h"
#include "signalsmith-fft.h"

#define SAMPLE_RATE 16000
#define FRAMES_PER_BUFFER 512
#define FFTSIZE 1024
#define HOPSIZE 256
#define AMP 0.5f
#define MAX_BUFFER_SIZE 16384

constexpr int L = 4;
constexpr int fftPoints = FFTSIZE;
constexpr int hopSize = HOPSIZE;
constexpr float ampScale = AMP;
constexpr int numBins = 513;
constexpr int bufSize = MAX_BUFFER_SIZE;
constexpr int in_feat_len = 2*L+1;

using namespace std;
// using namespace webrtc;
class NKF {
private:
    int outWtPtr;
    int outRdPtr;
    array<float, fftPoints> window;
    array<float, L*fftPoints> delay_window;
    int inPtr;
    array<float, bufSize> inBuf {};
    array<float, fftPoints> orderedInBuf;
    // array<float, bufSize> outBuf {};
    float *outBuf;
    int smpCnt;
    vector<float> h_prior_re;
    vector<float> h_prior_im;
    vector<float> h_posterior_re;
    vector<float> h_posterior_im;
    array<float, L*numBins> dh_re;
    array<float, L*numBins> dh_im;
    int l;
    signalsmith::RealFFT<float> fft_mic;
    signalsmith::RealFFT<float> fft_ref;
    array<array<complex<float>, FFTSIZE/2+1>*, L> M;
    array<array<complex<float>, FFTSIZE/2+1>*, L> R;
    vector<float> echo_hat_re;
    vector<float> echo_hat_im;
    signalsmith::RealFFT<float> s_hat;
    // array<std::complex<float>, numBins> diff {};
    vector<float> diff_re;
    vector<float> diff_im;
    // array<std::complex<float>, in_feat_len> input_feature {};
    vector<float> input_feature_re;
    vector<float> input_feature_im;
    vector<float> y_hat_re;
    vector<float> y_hat_im;
    array<float, bufSize> refBuf {};
    array<float, fftPoints> orderedRefBuf {};
    int refPtr;
    int halfFftPlus1;
    int in_feat_len;
    KGNet kgnet;
    vector<float> kg_e_re;
    vector<float> kg_e_im;
    array<float, fftPoints> ifftBuf {};
    // vector<float> kg_re;
    // vector<float> kg_im;
public:
    NKF(int fftSize, int bufferSize, float ampFactor, int hopLength);
    ~NKF();
    // void process(const float *in_data, float *out_data, int numSamples);
    int process();
};
