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
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cassert>

// portable non-SIMD implementation
// see comments of WAVSourceAVX2 and WAVSourceAVX
void WAVSourceGeneric::tick_spectrum([[maybe_unused]] float seconds)
{
    const auto bufsz = m_fft_size * sizeof(float);
    const auto outsz = m_fft_size / 2;
    constexpr auto step = 1;

    const auto dtcapture = m_tick_ts - m_capture_ts;

    auto& decibels = m_decibels[0];

    if(!m_show || (dtcapture > CAPTURE_TIMEOUT))
    {
        if(m_last_silent)
            return;
        for(auto channel = 0u; channel < m_capture_channels; ++channel)
            if(m_tsmooth_buf[channel] != nullptr)
                memset(m_tsmooth_buf[channel].get(), 0, outsz * sizeof(float));
        for(size_t i = 0; i < outsz; ++i)
            decibels[i] = DB_MIN;
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
        for(auto i = 0u; i < m_fft_size; i += step)
        {
            if(m_fft_input[i] != 0.0f)
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
            auto floor = (float)(m_floor - 10);
            for(size_t i = 0; i < outsz; i += step)
            {
                if(decibels[i] > floor)
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

        auto inbuf = m_fft_input.get();
        auto mulbuf = m_window_coefficients.get();
        for(auto i = 0u; i < m_fft_size; i += step)
            inbuf[i] *= mulbuf[i];

        if(m_fft_plan != nullptr)
            fftwf_execute(m_fft_plan);
        else
            continue;

        const auto mag_coefficient = 2.0f / m_window_sum;
        const auto g = get_gravity(seconds);
        const auto g2 = 1.0f - g;
        const bool slope = m_slope > 0.0f;
        for(size_t i = 0; i < outsz; i += step)
        {
            auto real = m_fft_output[i][0];
            auto imag = m_fft_output[i][1];

            auto mag = std::hypot(real, imag) * mag_coefficient;

            if(slope)
                mag *= m_slope_modifiers[i];

            auto oldval = m_tsmooth_buf[channel][i];

            mag = (g * oldval) + (g2 * mag);
            m_tsmooth_buf[channel][i] = mag;

            m_decibels[channel][i] = mag;
        }
    }

    if(m_last_silent)
        return;

    auto& decibels1 = m_decibels[1];
    if(m_output_channels > m_capture_channels)
        memcpy(decibels1.get(), decibels.get(), outsz * sizeof(float));

    if(m_capture_channels > 1)
    {
        for(size_t i = 0; i < outsz; ++i)
            decibels[i] = dbfs((decibels[i] + decibels1[i]) * 0.5f);
    }
    else
    {
        for(size_t i = 0; i < outsz; ++i)
            decibels[i] = dbfs(decibels[i]);
    }

    if(m_normalize_volume)
    {
        const auto volume_compensation = std::min(m_volume_target - dbfs(m_input_rms), m_max_gain);
        for(auto channel = 0; channel < 1; ++channel)
            for(size_t i = 1; i < outsz; ++i)
                m_decibels[channel][i] += volume_compensation;
    }

    if((m_rolloff_q > 0.0f) && (m_rolloff_rate > 0.0f))
    {
        for(size_t i = 1; i < outsz; ++i)
        {
            auto val = decibels[i] - m_rolloff_modifiers[i];
            decibels[i] = std::max(val, DB_MIN);
        }
    }
}

void WAVSourceGeneric::update_input_rms()
{
    assert(m_normalize_volume);

    if(!sync_rms_buffer())
        return;

    float sum = 0.0f;
    for(size_t i = 0; i < m_input_rms_size; ++i)
        sum += m_input_rms_buf[i];
    m_input_rms = std::sqrt(sum / m_input_rms_size);
}
