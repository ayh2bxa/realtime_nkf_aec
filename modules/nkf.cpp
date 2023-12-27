#include "nkf.h"
#include <iostream>
#include <cmath>
#include <cerrno>
#include <thread>
#include <memory>
#include <sndfile.h>
#include <portaudio.h>
#include <stdio.h>
#include <stdlib.h>
#include <array>
#include <fstream>
#include "matrix_helper.h"

using namespace std;

NKF::NKF(int fftSize, int bufferSize, float ampFactor, int hopLength): kgnet(L, 18, 1, 18, numBins), fft_mic(fftPoints), fft_ref(fftPoints), s_hat(fftPoints) {
    inPtr = 0;
    outWtPtr = fftPoints+2*hopSize;
    outRdPtr = 0;
    smpCnt = 0;
    for (int i = 0; i < fftPoints; i++) {
        window[i] = (float)(0.5*(1.0-cos(2.0*M_PI*i/(float)(fftPoints))));
    }
    for (int i = 0; i < L*fftPoints; i++) {
        delay_window[i] = (float)(0.5*(1.0-cos(2.0*M_PI*i/(float)(L*fftPoints-1))));
    }
    l = 0;
    for (int i = 0; i < L; i++) {
        M[i] = (array<complex<float>, FFTSIZE/2+1>*)malloc(sizeof(array<complex<float>, FFTSIZE/2+1>));
        R[i] = (array<complex<float>, FFTSIZE/2+1>*)malloc(sizeof(array<complex<float>, FFTSIZE/2+1>));
        for (int j = 0; j < FFTSIZE/2+1; j++) {
            (*M[i])[j].real(0);
            (*M[i])[j].imag(0);
            (*R[i])[j].real(0);
            (*R[i])[j].imag(0);
        }
    }
    refPtr = 0;
    y_hat_re.resize(numBins);
    y_hat_im.resize(numBins);
    float init = 0.0f; // small value used for debugging stability of matrix operations
    h_prior_re.resize(L*numBins, init);
    h_prior_im.resize(L*numBins, init);
    h_posterior_re.resize(L*numBins, init);
    h_posterior_im.resize(L*numBins, init);
    echo_hat_re.resize(numBins);
    echo_hat_im.resize(numBins);
    diff_re.resize(numBins, init);
    diff_im.resize(numBins, init);
    in_feat_len = 2*L+1;
    input_feature_re.resize(in_feat_len*numBins);
    input_feature_im.resize(in_feat_len*numBins);
    kg_e_re.resize(numBins*L, init);
    kg_e_im.resize(numBins*L, init);
}

NKF::~NKF() {
    for (int i = 0; i < L; i++) {
        free(M[i]);
        free(R[i]);
    }
    free(outBuf);
}

int NKF::process()
{
    SNDFILE *micfile;
    SNDFILE *reffile;
    SF_INFO sfinfo;
    float *mic_buffer;
    float *ref_buffer;
    int numChannels;
    sf_count_t numFrames;
    PaStream *stream;
    PaError err;
    int numWindows = 500;
    uint64_t startPtr = 0;

    array<float, fftPoints> window;
    for (int i = 0; i < fftPoints; i++) {
        window[i] = (float)(0.5*(1.0-cos(2.0*M_PI*i/(float)(fftPoints))));
    }

    // Open the WAV file for reading
    micfile = sf_open("/Users/anthony/Desktop/gerzz/rt_nkf/audio_files/mic.wav", SFM_READ, &sfinfo);
    if (!micfile) {
        cout<<sf_strerror(micfile)<<endl;
        printf("Error: could not open file\n");
        return 1;
    }

    numChannels = sfinfo.channels;
    numFrames = sfinfo.frames;

    // Allocate memory for the buffer
    mic_buffer = (float *)malloc(numFrames * numChannels * sizeof(float));
    if (!mic_buffer) {
        printf("Error: could not allocate memory for mic buffer\n");
        return 1;
    }

    // Read the audio data into the buffer
    sf_readf_float(micfile, mic_buffer, numFrames);
    sf_close(micfile);
    float *padded_mic_buffer = (float *)calloc((fftPoints+numFrames * numChannels), sizeof(float));
    for (int i = 0; i < fftPoints/2; i++) {
        padded_mic_buffer[i] = mic_buffer[fftPoints/2-i-1];
    }
    for (int i = 0; i < numFrames*numChannels; i++) {
        padded_mic_buffer[fftPoints/2+i] = mic_buffer[i];
    }
    // Open the WAV file for reading
    reffile = sf_open("/Users/anthony/Desktop/gerzz/rt_nkf/audio_files/ref.wav", SFM_READ, &sfinfo);
    if (!reffile) {
        printf("Error: could not open file\n");
        return 1;
    }

    // Allocate memory for the buffer
    ref_buffer = (float *)malloc(numFrames * numChannels*sizeof(float));
    if (!ref_buffer) {
        printf("Error: could not allocate memory for ref buffer\n");
        return 1;
    }

    // Read the audio data into the buffer
    sf_readf_float(reffile, ref_buffer, numFrames);
    sf_close(reffile);

    int delay = 188;
    float *aligned_ref = (float *)calloc(delay+numFrames * numChannels, sizeof(float));
    for (int i = 0; i < numFrames*numChannels; i++) {
        aligned_ref[i+delay] = ref_buffer[i];
    }

    float *padded_ref_buffer = (float *)calloc((delay+fftPoints+numFrames * numChannels), sizeof(float));
    for (int i = 0; i < fftPoints/2; i++) {
        padded_ref_buffer[i] = aligned_ref[fftPoints/2-i-1];
    }
    for (int i = 0; i < numFrames*numChannels; i++) {
        padded_ref_buffer[fftPoints/2+i] = aligned_ref[i];
    }

    SNDFILE* outfile = sf_open("/Users/anthony/Desktop/gerzz/rt_nkf/audio_files/output.wav", SFM_WRITE, &sfinfo);
    if (!outfile) {
        std::cerr << "Error opening file" << std::endl;
        return -1;
    }
    outBuf = (float *)calloc(fftPoints*numWindows, sizeof(float));
    auto start = std::chrono::high_resolution_clock::now();
    for (int n = 0; n < numWindows-3; n++) {
        for (int i = 0; i < fftPoints; i++) {
            orderedInBuf[i] = window[i]*padded_mic_buffer[i+startPtr];
            orderedRefBuf[i] = window[i]*padded_ref_buffer[i+startPtr];
        }
        array<complex<float>, FFTSIZE/2+1> *tmpR = R[0];
        for (int i = 1; i < L; i++) {
            R[i-1] = R[i];
        }
        R[L-1] = tmpR;
        fft_ref.fft(orderedRefBuf, *R[L-1]);
        float sum = 0.f;
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < numBins; j++) {
                sum += sqrt((*R[i])[j].real()*(*R[i])[j].real()+(*R[i])[j].imag()*(*R[i])[j].imag());
            }
        }
        sum = sum/(float)(L*numBins);
        if (sum < 1e-5) {
            continue;
        }
        array<complex<float>, FFTSIZE/2+1> *tmpM = M[0];
        for (int i = 1; i < L; i++) {
            M[i-1] = M[i];
        }
        M[L-1] = tmpM;
        fft_ref.fft(orderedInBuf, *M[L-1]);
        for (int i = 0; i < numBins*L; i++) {
            dh_re[i] = h_posterior_re[i]-h_prior_re[i];
            dh_im[i] = h_posterior_im[i]-h_prior_im[i];
            h_prior_re[i] = h_posterior_re[i];
            h_prior_im[i] = h_posterior_im[i];
        }
        matmulFft(y_hat_re, y_hat_im, numBins, 1, L, 1, R, h_prior_re, h_prior_im);
        // compute diff = y-y_hat
        for (int i = 0; i < numBins; i++) {
            diff_re[i] = (*M[L-1])[i].real()-y_hat_re[i];
            diff_im[i] = (*M[L-1])[i].imag()-y_hat_im[i];
        }
        // input_feature = torch.cat([xt, e.unsqueeze(1), dh.squeeze()], dim=1)
        for (int i = 0; i < numBins; i++) {
            for (int j = 0; j < L; j++) {
                input_feature_re[i*in_feat_len+j] = (*R[j])[i].real();
                input_feature_im[i*in_feat_len+j] = (*R[j])[i].imag();
            }
            input_feature_re[i*in_feat_len+L] = diff_re[i];
            input_feature_im[i*in_feat_len+L] = diff_im[i];
            for (int j = L+1; j < in_feat_len; j++) {
                input_feature_re[i*in_feat_len+j] = dh_re[i*L+(j-L-1)];
                input_feature_im[i*in_feat_len+j] = dh_im[i*L+(j-L-1)];
            }
        }

        kgnet.forward(input_feature_re, input_feature_im);
        vector<float>& kg_re = kgnet.get_kg_re();
        vector<float>& kg_im = kgnet.get_kg_im();

        matmul(kg_e_re, kg_e_im, numBins, L, 1, 1, kg_re, kg_im, diff_re, diff_im);

        for (int i = 0; i < numBins; i++) {
            for (int j = 0; j < L; j++) {
                int pos = i*L+j;
                h_posterior_re[pos] = h_prior_re[pos]+kg_e_re[pos];
                h_posterior_im[pos] = h_prior_im[pos]+kg_e_im[pos];
            }
        }
        matmulFft(echo_hat_re, echo_hat_im, numBins, 1, L, 1, R, h_posterior_re, h_posterior_im);
        for (int i = 0; i < numBins; i++) {
            float m_real = (*M[L-1])[i].real();
            (*M[L-1])[i].real(m_real-echo_hat_re[i]);
            float m_imag = (*M[L-1])[i].imag();
            (*M[L-1])[i].imag(m_imag-echo_hat_im[i]);
        }
        (*M[L-1])[0].imag((*M[L-1])[FFTSIZE/2].real());
        (*M[L-1])[FFTSIZE/2].real(0);
        s_hat.ifft(*(M[L-1]), ifftBuf);
        for (int i = 0; i < fftPoints; i++) {
            outBuf[i+startPtr] += (window[i]*ifftBuf[i]/(float)FFTSIZE);
        }
        startPtr += hopSize;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double rtf = duration / (1000.0*(double)numFrames/16000.0);
    cout<<"Processing finished. Real-time factor: "<<rtf<<endl;
    sf_writef_float(outfile, outBuf, numFrames);
    sf_close(outfile);
    // Free the buffer
    free(mic_buffer);
    free(ref_buffer);
    free(aligned_ref);
    free(padded_ref_buffer);
    free(padded_mic_buffer);
    return 0;
}
