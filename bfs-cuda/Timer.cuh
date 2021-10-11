#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date November, 2017
 * @version v1.4
 *
 * @copyright Copyright © 2017 XLib. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 *
 * @file
 */
#pragma once

#include "Timer.hpp"

namespace timer {

template<typename ChronoPrecision>
class Timer<DEVICE, ChronoPrecision> final :
                      public timer::detail::TimerBase<DEVICE, ChronoPrecision> {
public:
    explicit Timer(int decimals = 1, int space = 15,
                   xlib::Color color = xlib::Color::FG_DEFAULT) noexcept;

    ~Timer() noexcept;

    void start() noexcept override;

    void stop()  noexcept override;

private:
    using detail::TimerBase<DEVICE, ChronoPrecision>::_time_elapsed;
    using detail::TimerBase<DEVICE, ChronoPrecision>::_start_flag;

    cudaEvent_t _start_event, _stop_event;

    using detail::TimerBase<DEVICE, ChronoPrecision>::register_time;
};

} // namespace timer

/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date November, 2017
 * @version v1.4
 *
 * @copyright Copyright © 2017 XLib. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 */
#include <cassert>

namespace timer {

template<typename ChronoPrecision>
Timer<DEVICE, ChronoPrecision>
::Timer(int decimals, int space, xlib::Color color) noexcept :
     timer::detail::TimerBase<DEVICE, ChronoPrecision>(decimals, space, color) {
    cudaEventCreate(&_start_event);
    cudaEventCreate(&_stop_event);
}

template<typename ChronoPrecision>
Timer<DEVICE, ChronoPrecision>::~Timer() noexcept {
    cudaEventDestroy(_start_event);
    cudaEventDestroy(_stop_event);
}

template<typename ChronoPrecision>
void Timer<DEVICE, ChronoPrecision>::start() noexcept {
    assert(!_start_flag);
    _start_flag = true;
    cudaEventRecord(_start_event, 0);
}

template<typename ChronoPrecision>
void Timer<DEVICE, ChronoPrecision>::stop() noexcept {
    float cuda_time_elapsed;
    cudaEventRecord(_stop_event, 0);
    cudaEventSynchronize(_stop_event);
    cudaEventElapsedTime(&cuda_time_elapsed, _start_event, _stop_event);
    auto  time_ms = timer::milli(cuda_time_elapsed);
    _time_elapsed = std::chrono::duration_cast<ChronoPrecision>(time_ms);
    register_time();
}

} // namespace timer

