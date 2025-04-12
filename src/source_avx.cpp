/*
    Copyright (C) 2022 Devin Davila

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "source.hpp"
#include "simd_helpers.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cstring>
#include <cassert>

// adaptation of WAVSourceAVX2 to support CPUs without AVX2
// see comments of WAVSourceAVX2
// FIXME: this specialization should be removed.
// The only CPUs with FMA3 but not AVX2 are ancient AMD chips that prefer SSE code anyway.
void WAVSourceAVX::tick_spectrum([[maybe_unused]] float seconds)
{
    //std::lock_guard lock(m_mtx); // now locked in tick()

    const auto bufsz = m_fft_size * sizeof(float);
    const auto outsz = m_fft_size / 2;
    constexpr auto step = sizeof(__m256) / sizeof(float);

    const auto dtcapture = m_tick_ts - m_capture_ts;

    if(!m_show || (dtcapture > CAPTURE_TIMEOUT))
    {
        if(m_last_silent)
            return;
        for(auto channel = 0u; channel < m_capture_channels; ++channel)
            if(m_tsmooth_buf[channel] != nullptr)
                memset(m_tsmooth_buf[channel].get(), 0, outsz * sizeof(float));
        for(auto channel = 0; channel < (m_stereo ? 2 : 1); ++channel)
            for(size_t i = 0; i < outsz; ++i)
                m_decibels[channel][i] = DB_MIN;
        m_last_silent = true;
        return;
    }

    const int64_t dtaudio = get_audio_sync(m_tick_ts);
    const size_t dtsize = ((dtaudio > 0) ? size_t(ns_to_audio_frames(m_audio_info.samples_per_sec, (uint64_t)dtaudio)) * sizeof(float) : 0) + bufsz;
    auto silent_channels = 0u;
    for(auto channel = 0u; channel < m_capture_channels; ++channel)
    {
        if(m_capturebufs[channel].size >= dtsize)
        {
            circlebuf_pop_front(&m_capturebufs[channel], nullptr, m_capturebufs[channel].size - dtsize);
            circlebuf_peek_front(&m_capturebufs[channel], m_fft_input.get(), bufsz);
        }
        else
            continue;

        bool silent = true;
        const auto zero = _mm256_setzero_ps();
        for(auto i = 0u; i < m_fft_size; i += step)
        {
            auto mask = _mm256_cmp_ps(zero, _mm256_load_ps(&m_fft_input[i]), _CMP_EQ_OQ);
            if(_mm256_movemask_ps(mask) != 0xff)
            {
                silent = false;
                m_last_silent = false;
                break;
            }
        }

        if(silent)
        {
            if(m_last_silent)
                continue;
            bool outsilent = true;
            auto floor = _mm256_set1_ps((float)(m_floor - 10));
            for(size_t i = 0; i < outsz; i += step)
            {
                const auto ch = (m_stereo) ? channel : 0u;
                auto mask = _mm256_cmp_ps(floor, _mm256_load_ps(&m_decibels[ch][i]), _CMP_GT_OQ);
                if(_mm256_movemask_ps(mask) != 0xff)
                {
                    outsilent = false;
                    break;
                }
            }
            if(outsilent)
            {
                if(++silent_channels >= m_capture_channels)
                    m_last_silent = true;
                continue;
            }
        }

        if(m_window_func != FFTWindow::NONE)
        {
            auto inbuf = m_fft_input.get();
            auto mulbuf = m_window_coefficients.get();
            for(auto i = 0u; i < m_fft_size; i += step)
                _mm256_store_ps(&inbuf[i], _mm256_mul_ps(_mm256_load_ps(&inbuf[i]), _mm256_load_ps(&mulbuf[i])));
        }

        if(m_fft_plan != nullptr)
            fftwf_execute(m_fft_plan);
        else
            continue;

        constexpr auto shuffle_mask_r = 0 | (2 << 2) | (0 << 4) | (2 << 6);
        constexpr auto shuffle_mask_i = 1 | (3 << 2) | (1 << 4) | (3 << 6);
        const auto mag_coefficient = _mm256_set1_ps(2.0f / m_window_sum);
        const auto g = _mm256_set1_ps(get_gravity(seconds));
        const auto g2 = _mm256_sub_ps(_mm256_set1_ps(1.0), g);
        const bool slope = m_slope > 0.0f;
        for(size_t i = 0; i < outsz; i += step)
        {
            // load 8 real/imaginary pairs and group the r/i components in the low/high halves
            // de-interleaving 256-bit float vectors is nigh impossible without AVX2, so we'll
            // use 128-bit vectors and merge them, but i question if this is better than a 128-bit loop
            const float *buf = &m_fft_output[i][0];
            auto chunk1 = _mm_load_ps(buf);
            auto chunk2 = _mm_load_ps(&buf[4]);
            auto rvec = _mm256_castps128_ps256(_mm_shuffle_ps(chunk1, chunk2, shuffle_mask_r)); // group octwords
            auto ivec = _mm256_castps128_ps256(_mm_shuffle_ps(chunk1, chunk2, shuffle_mask_i));
            chunk1 = _mm_load_ps(&buf[8]);
            chunk2 = _mm_load_ps(&buf[12]);
            rvec = _mm256_insertf128_ps(rvec, _mm_shuffle_ps(chunk1, chunk2, shuffle_mask_r), 1); // pack r/i octwords into separate 256-bit vecs
            ivec = _mm256_insertf128_ps(ivec, _mm_shuffle_ps(chunk1, chunk2, shuffle_mask_i), 1);

            auto mag = _mm256_sqrt_ps(_mm256_fmadd_ps(ivec, ivec, _mm256_mul_ps(rvec, rvec)));
            mag = _mm256_mul_ps(mag, mag_coefficient);

            if(slope)
                mag = _mm256_mul_ps(mag, _mm256_load_ps(&m_slope_modifiers[i]));

            auto oldval = _mm256_load_ps(&m_tsmooth_buf[channel][i]);

            mag = _mm256_fmadd_ps(g, oldval, _mm256_mul_ps(g2, mag));
            _mm256_store_ps(&m_tsmooth_buf[channel][i], mag);

            _mm256_store_ps(&m_decibels[channel][i], mag);
        }
    }

    if(m_last_silent)
        return;

    if(m_output_channels > m_capture_channels)
        memcpy(m_decibels[1].get(), m_decibels[0].get(), outsz * sizeof(float));

    if(m_stereo)
    {
        for(auto channel = 0; channel < 2; ++channel)
            for(size_t i = 0; i < outsz; ++i)
                m_decibels[channel][i] = dbfs(m_decibels[channel][i]);
    }
    else if(m_capture_channels > 1)
    {
        for(size_t i = 0; i < outsz; ++i)
            m_decibels[0][i] = dbfs((m_decibels[0][i] + m_decibels[1][i]) * 0.5f);
    }
    else
    {
        for(size_t i = 0; i < outsz; ++i)
            m_decibels[0][i] = dbfs(m_decibels[0][i]);
    }

    if(m_normalize_volume)
    {
        const auto volume_compensation = _mm256_set1_ps(std::min(m_volume_target - dbfs(m_input_rms), m_max_gain));
        for(auto channel = 0; channel < (m_stereo ? 2 : 1); ++channel)
            for(size_t i = 0; i < outsz; i += step)
                _mm256_store_ps(&m_decibels[channel][i], _mm256_add_ps(volume_compensation, _mm256_load_ps(&m_decibels[channel][i])));
    }

    if((m_rolloff_q > 0.0f) && (m_rolloff_rate > 0.0f))
    {
        const auto dbmin = _mm256_set1_ps(DB_MIN);
        for(auto channel = 0; channel < (m_stereo ? 2 : 1); ++channel)
        {
            for(size_t i = 0; i < outsz; i += step)
            {
                auto val = _mm256_sub_ps(_mm256_load_ps(&m_decibels[channel][i]), _mm256_load_ps(&m_rolloff_modifiers[i]));
                _mm256_store_ps(&m_decibels[channel][i], _mm256_max_ps(val, dbmin));
            }
        }
    }
}

void WAVSourceAVX::tick_meter([[maybe_unused]] float seconds)
{
    // handle audio dropouts
    const auto dtcapture = m_tick_ts - m_capture_ts;
    if(dtcapture > CAPTURE_TIMEOUT)
    {
        if(m_last_silent)
            return;
        constexpr auto step = sizeof(__m256) / sizeof(float);
        const auto zero = _mm256_setzero_ps();
        for(auto channel = 0u; channel < m_capture_channels; ++channel)
            for(size_t i = 0u; i < m_fft_size; i += step)
                _mm256_store_ps(&m_decibels[channel][i], zero);

        for(auto& i : m_meter_buf)
            i = 0.0f;
        for(auto& i : m_meter_val)
            i = DB_MIN;
        m_last_silent = true;
        return;
    }

    const int64_t dtaudio = get_audio_sync(m_tick_ts);
    const size_t dtsize = (dtaudio > 0) ? size_t(ns_to_audio_frames(m_audio_info.samples_per_sec, (uint64_t)dtaudio)) * sizeof(float) : 0;

    // repurpose m_decibels as circular buffer for sample data
    for(auto channel = 0u; channel < m_capture_channels; ++channel)
    {
        while(m_capturebufs[channel].size > dtsize)
        {
            auto consume = m_capturebufs[channel].size - dtsize;
            auto max = (m_fft_size - m_meter_pos[channel]) * sizeof(float);
            if(consume >= max)
            {
                circlebuf_pop_front(&m_capturebufs[channel], &m_decibels[channel][m_meter_pos[channel]], max);
                m_meter_pos[channel] = 0;
            }
            else
            {
                circlebuf_pop_front(&m_capturebufs[channel], &m_decibels[channel][m_meter_pos[channel]], consume);
                m_meter_pos[channel] += consume / sizeof(float);
            }
        }
    }

    if(!m_show)
        return;

    for(auto channel = 0u; channel < m_capture_channels; ++channel)
    {
        float out = 0.0f;
        constexpr auto step = (sizeof(__m256) / sizeof(float)) * 2; // buffer size is 64-byte multiple
        constexpr auto halfstep = step / 2;
        if(m_meter_rms)
        {
            auto sum1 = _mm256_setzero_ps(); // split sum into 2 'lanes' for better pipelining
            auto sum2 = _mm256_setzero_ps();
            for(size_t i = 0; i < m_fft_size; i += step)
            {
                auto chunk1 = _mm256_load_ps(&m_decibels[channel][i]);
                sum1 = _mm256_fmadd_ps(chunk1, chunk1, sum1);
                auto chunk2 = _mm256_load_ps(&m_decibels[channel][i + halfstep]); // unroll loop to cache line size
                sum2 = _mm256_fmadd_ps(chunk2, chunk2, sum2);
            }

            out = std::sqrt(horizontal_sum(_mm256_add_ps(sum1, sum2)) / m_fft_size);
        }
        else
        {
            const auto signbit = _mm256_set1_ps(-0.0f);
            auto max1 = _mm256_setzero_ps(); // split max into 2 'lanes' for better pipelining
            auto max2 = _mm256_setzero_ps();
            for(size_t i = 0; i < m_fft_size; i += step)
            {
                auto chunk1 = _mm256_andnot_ps(signbit, _mm256_load_ps(&m_decibels[channel][i])); // absolute value
                max1 = _mm256_max_ps(max1, chunk1);
                auto chunk2 = _mm256_andnot_ps(signbit, _mm256_load_ps(&m_decibels[channel][i + halfstep])); // unroll loop to cache line size
                max2 = _mm256_max_ps(max2, chunk2);
            }

            out = horizontal_max(_mm256_max_ps(max1, max2));
        }

        const auto g = get_gravity(seconds);
        const auto g2 = 1.0f - g;
        if (out <= m_meter_buf[channel])
            out = (g * m_meter_buf[channel]) + (g2 * out);

        m_meter_buf[channel] = out;
        m_meter_val[channel] = dbfs(out);
    }

    // hide on silent
    auto silent_channels = 0u;
    for(auto channel = 0u; channel < m_capture_channels; ++channel)
        if(m_meter_val[channel] < (m_floor - 10))
            ++silent_channels;

    m_last_silent = (silent_channels >= m_capture_channels);
}

void WAVSourceAVX::update_input_rms()
{
    assert(m_normalize_volume);

    if(!sync_rms_buffer())
        return;

    constexpr auto step = (sizeof(__m256) / sizeof(float)) * 2; // buffer size is 64-byte multiple
    constexpr auto halfstep = step / 2;
    auto sum1 = _mm256_setzero_ps();
    auto sum2 = _mm256_setzero_ps();
    for(size_t i = 0; i < m_input_rms_size; i += step)
    {
        sum1 = _mm256_add_ps(sum1, _mm256_load_ps(&m_input_rms_buf[i])); // split sum into 2 'lanes' for better pipelining
        sum2 = _mm256_add_ps(sum2, _mm256_load_ps(&m_input_rms_buf[i + halfstep]));
    }
    m_input_rms = std::sqrt(horizontal_sum(_mm256_add_ps(sum1, sum2)) / m_input_rms_size);
}
