#include <cmath>
#include <cstdint>
#include <vector>

extern "C" {

void range_doppler_peak_scores_f32(
    const float* y,
    const int n_codes,
    const int n_ch,
    const int code_len,
    const float* shifted_codes,
    const int n_tx,
    const int n_range,
    const float* window,
    const float window_norm,
    float* out_power,
    int32_t* out_peak_bin
) {
    const float two_pi = 6.2831853071795864769f;
    const float inv_code_len = 1.0f / static_cast<float>(code_len);
    const float norm = 1.0f / (window_norm * static_cast<float>(n_ch * n_tx));

    std::vector<float> tw_re(n_codes * n_codes);
    std::vector<float> tw_im(n_codes * n_codes);
    for (int ks = 0; ks < n_codes; ++ks) {
        const int k = (ks + n_codes / 2) % n_codes;
        for (int n = 0; n < n_codes; ++n) {
            const float phase = two_pi * static_cast<float>(k * n) / static_cast<float>(n_codes);
            tw_re[ks * n_codes + n] = std::cos(phase);
            tw_im[ks * n_codes + n] = -std::sin(phase);
        }
    }

    std::vector<float> power(n_codes);
    std::vector<float> amp_re(n_codes * n_tx);
    std::vector<float> amp_im(n_codes * n_tx);

    for (int ri = 0; ri < n_range; ++ri) {
        std::fill(power.begin(), power.end(), 0.0f);

        for (int chi = 0; chi < n_ch; ++chi) {
            for (int ni = 0; ni < n_codes; ++ni) {
                for (int txi = 0; txi < n_tx; ++txi) {
                    float ar = 0.0f;
                    float ai = 0.0f;
                    const float* yp = y + (((ni * n_ch + chi) * code_len) * 2);
                    const float* cp = shifted_codes + (((txi * n_range + ri) * code_len) * 2);
                    #pragma clang loop vectorize(enable)
                    for (int si = 0; si < code_len; ++si) {
                        const float yr = yp[2 * si];
                        const float yi = yp[2 * si + 1];
                        const float cr = cp[2 * si];
                        const float ci = cp[2 * si + 1];
                        ar += yr * cr + yi * ci;
                        ai += yi * cr - yr * ci;
                    }
                    amp_re[ni * n_tx + txi] = ar * inv_code_len * window[ni];
                    amp_im[ni * n_tx + txi] = ai * inv_code_len * window[ni];
                }
            }

            for (int ks = 0; ks < n_codes; ++ks) {
                const float* tr = tw_re.data() + ks * n_codes;
                const float* ti = tw_im.data() + ks * n_codes;
                for (int txi = 0; txi < n_tx; ++txi) {
                    float sr = 0.0f;
                    float si = 0.0f;
                    for (int ni = 0; ni < n_codes; ++ni) {
                        const float ar = amp_re[ni * n_tx + txi];
                        const float ai = amp_im[ni * n_tx + txi];
                        sr += ar * tr[ni] - ai * ti[ni];
                        si += ar * ti[ni] + ai * tr[ni];
                    }
                    power[ks] += (sr * sr + si * si) * norm;
                }
            }
        }

        int best = 0;
        float best_power = power[0];
        for (int ks = 1; ks < n_codes; ++ks) {
            if (power[ks] > best_power) {
                best_power = power[ks];
                best = ks;
            }
        }
        out_power[ri] = best_power;
        out_peak_bin[ri] = best;
    }
}

}
