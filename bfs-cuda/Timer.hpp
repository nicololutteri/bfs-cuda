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

#include <iostream>

#include <chrono>               //std::chrono::duration
#include <string>               //std::string
#if defined(__linux__)
    #include <sys/times.h>      //::tms
#endif

//#define COLOR

#if defined(COLOR)
    #include "Host/PrintExt.hpp"
#else

namespace xlib {
    enum class Color { FG_DEFAULT };
    struct IosFlagSaver {};
} // namespace xlib

inline std::ostream& operator<<(std::ostream& os,
                                const xlib::Color& mod) noexcept {
    return os;
}

#endif

namespace timer {

/// @brief chrono precision : microseconds
using micro   = typename std::chrono::duration<float, std::micro>;
/// @brief default chrono precision (milliseconds)
using milli   = typename std::chrono::duration<float, std::milli>;
/// @brief chrono precision : seconds
using seconds = typename std::chrono::duration<float, std::ratio<1>>;
/// @brief chrono precision : minutes
using minutes = typename std::chrono::duration<float, std::ratio<60>>;
/// @brief chrono precision : minutes
using hours   = typename std::chrono::duration<float, std::ratio<3600>>;

/**
 * @brief timer types
 */
enum timer_type {  HOST = 0       /// Wall (real) clock host time
                 , CPU  = 1       /// CPU User time
            #if defined(__linux__)
                 , SYS  = 2       /// User/Kernel/System time
            #endif
                 , DEVICE = 3     /// GPU device time
};

namespace detail {

/**
 * @brief Timer class
 * @tparam type Timer type (default = HOST)
 * @tparam ChronoPrecision time precision
 */
template<timer_type type, typename ChronoPrecision>
class TimerBase {
    template<typename>
    struct is_duration : std::false_type {};

    template<typename T, typename R>
    struct is_duration<std::chrono::duration<T, R>> : std::true_type {};

    static_assert(is_duration<ChronoPrecision>::value,
                  "Wrong type : typename is not std::chrono::duration");
public:
    /**
     * @brief Default costructor
     * @param[in] decimals precision to print the time elapsed
     * @param[in] space space for the left alignment
     * @param[in] color color of print
     */
    explicit TimerBase(int decimals, int space, xlib::Color color) noexcept;

    virtual ~TimerBase() noexcept = default;

    /**
     * @brief Start the timer
     */
    virtual void start() noexcept = 0;

    /**
     * @brief Stop the timer
     */
    virtual void stop() noexcept = 0;

    /**
     * @brief Get the time elapsed between start() and stop() calls
     * @return time elapsed specified with the \p ChronoPrecision
     */
    virtual float duration() const noexcept final;

    /**
     * @brief Get the time elapsed between the first start() and the last stop()
     *        calls
     * @return time elapsed specified with the \p ChronoPrecision
     */
    virtual float total_duration() const noexcept final;

    /**
     * @brief Get the average time elapsed between the first start() and the
     *        last stop() calls
     * @return average duration
     */
    virtual float average() const noexcept final;

    /**
     * @brief Standard deviation
     * @return Standard deviation
     */
    virtual float std_deviation() const noexcept final;

    /**
     * @brief Standard deviation
     * @return Standard deviation
     */
    virtual float min() const noexcept final;

    /**
     * @brief Standard deviation
     * @return Standard deviation
     */
    virtual float max() const noexcept final;

    /**
     *
     */
    virtual void reset() noexcept final;

    /**
     * @brief Print the time elapsed between start() and stop() calls
     * @param[in] str print string \p str before the time elapsed
     * @warning if start() and stop() not invoked undefined behavior
     */
    virtual void print(const std::string& str) const noexcept;

    virtual void printAll(const std::string& str) const noexcept;

protected:
    ChronoPrecision   _time_elapsed  {};
    const int         _space         { 0 };
    const int         _decimals      { 0 };
    const xlib::Color _default_color { xlib::Color::FG_DEFAULT };
    bool              _start_flag    { false };

    /**
     *
     */
    virtual void register_time() noexcept final;

private:
    ChronoPrecision   _time_squared       {};
    ChronoPrecision   _total_time_elapsed {};
    ChronoPrecision   _time_min           {};
    ChronoPrecision   _time_max           {};
    int               _num_executions     { 0 };
};

} // namespace detail
//------------------------------------------------------------------------------

template<timer_type type, typename ChronoPrecision = milli>
class Timer;

template<typename ChronoPrecision>
class Timer<HOST, ChronoPrecision> final :
            public timer::detail::TimerBase<HOST, ChronoPrecision> {
public:
    explicit Timer(int decimals = 1, int space = 15,
                   xlib::Color color = xlib::Color::FG_DEFAULT) noexcept;

    void start() noexcept override;

    void stop()  noexcept override;

private:
    using timer::detail::TimerBase<HOST, ChronoPrecision>::_time_elapsed;
    using timer::detail::TimerBase<HOST, ChronoPrecision>::_start_flag;

    std::chrono::system_clock::time_point _start_time {};
    std::chrono::system_clock::time_point _stop_time  {};

    using timer::detail::TimerBase<HOST, ChronoPrecision>::register_time;
};

//------------------------------------------------------------------------------

template<typename ChronoPrecision>
class Timer<CPU, ChronoPrecision> final :
            public timer::detail::TimerBase<CPU, ChronoPrecision> {
public:
    explicit Timer(int decimals = 1, int space = 15,
                   xlib::Color color = xlib::Color::FG_DEFAULT) noexcept;

    void start() noexcept override;

    void stop()  noexcept override;

private:
    using timer::detail::TimerBase<CPU, ChronoPrecision>::_time_elapsed;
    using timer::detail::TimerBase<CPU, ChronoPrecision>::_start_flag;

    std::clock_t _start_clock { 0 };
    std::clock_t _stop_clock  { 0 };

    using timer::detail::TimerBase<CPU, ChronoPrecision>::register_time;
};

//------------------------------------------------------------------------------

#if defined(__linux__)

template<typename ChronoPrecision>
class Timer<SYS, ChronoPrecision> final :
            public timer::detail::TimerBase<SYS, ChronoPrecision> {
public:
    explicit Timer(int decimals = 1, int space = 15,
                   xlib::Color color = xlib::Color::FG_DEFAULT) noexcept;

    void start() noexcept override;

    void stop()  noexcept override;

    void print(const std::string& str = "") const noexcept override;

private:
    using timer::detail::TimerBase<SYS, ChronoPrecision>::_start_flag;
    using timer::detail::TimerBase<SYS, ChronoPrecision>::_start_time;
    using timer::detail::TimerBase<SYS, ChronoPrecision>::_stop_time;
    using timer::detail::TimerBase<SYS, ChronoPrecision>::_default_color;
    using timer::detail::TimerBase<SYS, ChronoPrecision>::_space;
    using timer::detail::TimerBase<SYS, ChronoPrecision>::_decimals;

    struct ::tms _start_TMS {};
    struct ::tms _end_TMS   {};
};

#endif

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
#include <cassert>          //assert
#include <cmath>            //std::sqrt
#include <ctime>            //std::clock
#include <iomanip>          //std::setprecision
#include <ratio>            //std::ratio
#if defined(__linux__)
    #include <sys/times.h>  //::times
    #include <unistd.h>     //::sysconf
#endif

namespace timer {

inline std::ostream& operator<<(std::ostream& os,
                                const xlib::Color& mod) noexcept {
    return os;
}

template<class Rep, std::intmax_t Num, std::intmax_t Denom>
std::ostream& operator<<(std::ostream& os,
                         const std::chrono::duration
                            <Rep, std::ratio<Num, Denom>>&) {
    if (Num == 3600 && Denom == 1)    return os << " h";
    if (Num == 60 && Denom == 1)      return os << " min";
    if (Num == 1 && Denom == 1)       return os << " s";
    if (Num == 1 && Denom == 1000)    return os << " ms";
    if (Num == 1 && Denom == 1000000) return os << " us";
    return os << " Unsupported";
}

//==============================================================================
//-------------------------- GENERIC -------------------------------------------
namespace detail {

template<timer_type type, typename ChronoPrecision>
TimerBase<type, ChronoPrecision>
::TimerBase(int decimals, int space, xlib::Color color) noexcept :
                   _decimals(decimals),
                   _space(space),
                   _default_color(color) {}

template<timer_type type, typename ChronoPrecision>
float TimerBase<type, ChronoPrecision>::duration() const noexcept {
    return _time_elapsed.count();
}

template<timer_type type, typename ChronoPrecision>
float TimerBase<type, ChronoPrecision>::total_duration() const noexcept {
    return _total_time_elapsed.count();
}

template<timer_type type, typename ChronoPrecision>
float TimerBase<type, ChronoPrecision>::average() const noexcept {
    auto num_executions = static_cast<float>(_num_executions);
    return _total_time_elapsed.count() / num_executions;
}

template<timer_type type, typename ChronoPrecision>
float TimerBase<type, ChronoPrecision>::std_deviation() const noexcept {
    auto term1 = _num_executions * _time_squared.count();
    auto term2 = _total_time_elapsed.count() * _total_time_elapsed.count();
    return std::sqrt(term1 - term2) / _num_executions;
}

template<timer_type type, typename ChronoPrecision>
float TimerBase<type, ChronoPrecision>::min() const noexcept {
    return _time_min.count();
}

template<timer_type type, typename ChronoPrecision>
float TimerBase<type, ChronoPrecision>::max() const noexcept {
        return _time_max.count();
}

template<timer_type type, typename ChronoPrecision>
void TimerBase<type, ChronoPrecision>::reset() noexcept {
    _time_min           = ChronoPrecision(0);
    _time_max           = ChronoPrecision(0);
    _total_time_elapsed = ChronoPrecision(0);
    _num_executions     = 0;
}

template<timer_type type, typename ChronoPrecision>
void TimerBase<type, ChronoPrecision>::register_time() noexcept {
    assert(_start_flag);
    _time_squared       += _time_elapsed * _time_elapsed.count();
    _total_time_elapsed += _time_elapsed;
    _num_executions++;
    if (_time_elapsed > _time_max)
        _time_max = _time_elapsed;
    else if (_time_elapsed < _time_min)
        _time_min = _time_elapsed;
    _start_flag = false;
}

template<timer_type type, typename ChronoPrecision>
void TimerBase<type, ChronoPrecision>::print(const std::string& str)    //NOLINT
                                             const noexcept {
    xlib::IosFlagSaver tmp;
    std::cout << _default_color
              << std::fixed << std::setprecision(_decimals)
              << std::right << std::setw(_space - 2) << str << "  "
              << duration() << ChronoPrecision()
              << xlib::Color::FG_DEFAULT << std::endl;
}

template<timer_type type, typename ChronoPrecision>
void TimerBase<type, ChronoPrecision>::printAll(const std::string& str) //NOLINT
                                                const noexcept {
    xlib::IosFlagSaver tmp;
    std::cout << _default_color
              << std::right << std::setw(_space - 2) << str << ":"
              << std::fixed << std::setprecision(_decimals)
              << "\n  min: " << min()           << ChronoPrecision()
              << "\n  max: " << max()           << ChronoPrecision()
              << "\n  avg: " << average()       << ChronoPrecision()
              << "\n  dev: " << std_deviation() << ChronoPrecision()
              << xlib::Color::FG_DEFAULT << std::endl;
}

} // namespace detail

//==============================================================================
//-----------------------  HOST ------------------------------------------------

template<typename ChronoPrecision>
Timer<HOST, ChronoPrecision>::Timer(int decimals, int space,
                                           xlib::Color color) noexcept :
      timer::detail::TimerBase<HOST, ChronoPrecision>(decimals, space, color) {}

template<typename ChronoPrecision>
void Timer<HOST, ChronoPrecision>::start() noexcept {
    assert(!_start_flag);
    _start_flag = true;
    _start_time = std::chrono::system_clock::now();
}

template<typename ChronoPrecision>
void Timer<HOST, ChronoPrecision>::stop() noexcept {
    _stop_time     = std::chrono::system_clock::now();
    _time_elapsed  = ChronoPrecision(_stop_time - _start_time);
    register_time();
}

//==============================================================================
//-------------------------- CPU -----------------------------------------------

template<typename ChronoPrecision>
Timer<CPU, ChronoPrecision>::Timer(int decimals, int space, xlib::Color color)
                                   noexcept :
       timer::detail::TimerBase<CPU, ChronoPrecision>(decimals, space, color) {}

template<typename ChronoPrecision>
void Timer<CPU, ChronoPrecision>::start() noexcept {
    assert(!_start_flag);
    _start_flag  = true;
    _start_clock = std::clock();
}

template<typename ChronoPrecision>
void Timer<CPU, ChronoPrecision>::stop() noexcept {
    _stop_clock = std::clock();
    auto clock_time_elapsed = static_cast<float>(_stop_clock - _start_clock) /
                              static_cast<float>(CLOCKS_PER_SEC);
    auto time_seconds = seconds(clock_time_elapsed);
    _time_elapsed  = std::chrono::duration_cast<ChronoPrecision>(time_seconds);
    register_time();
}

//==============================================================================
//-------------------------- SYS -----------------------------------------------

#if defined(__linux__)

template<typename ChronoPrecision>
Timer<SYS, ChronoPrecision>::Timer(int decimals, int space, xlib::Color color)
                                   noexcept :
       timer::detail::TimerBase<SYS, ChronoPrecision>(decimals, space, color) {}

template<typename ChronoPrecision>
void Timer<SYS, ChronoPrecision>::start() noexcept {
    assert(!_start_flag);
    _start_flag = true;
    _start_time = std::chrono::system_clock::now();
    ::times(&_start_TMS);
}

template<typename ChronoPrecision>
void Timer<SYS, ChronoPrecision>::stop() noexcept {
    assert(_start_flag);
    _stop_time = std::chrono::system_clock::now();
    ::times(&_end_TMS);
    _start_flag = false;
}

template<typename ChronoPrecision>
void Timer<SYS, ChronoPrecision>::print(const std::string& str)  //NOLINT
                                        const noexcept {
    xlib::IosFlagSaver tmp;
    auto  wall_time_ms = std::chrono::duration_cast<ChronoPrecision>(
                                             _stop_time - _start_time ).count();

    auto user_diff    = _end_TMS.tms_utime - _start_TMS.tms_utime;
    auto user_float   = static_cast<float>(user_diff) /
                         static_cast<float>(::sysconf(_SC_CLK_TCK));
    auto user_time    = seconds(user_float);
    auto user_time_ms = std::chrono::duration_cast<ChronoPrecision>(user_time);

    auto sys_diff    = _end_TMS.tms_stime - _start_TMS.tms_stime;
    auto sys_float   = static_cast<float>(sys_diff) /
                         static_cast<float>(::sysconf(_SC_CLK_TCK));
    auto sys_time    = seconds(sys_float);
    auto sys_time_ms = std::chrono::duration_cast<ChronoPrecision>(sys_time);

    std::cout << _default_color << std::setw(_space) << str
              << std::fixed << std::setprecision(_decimals)
              << "  Elapsed time: [user " << user_time_ms << ", system "
              << sys_time_ms << ", real "
              << wall_time_ms << ChronoPrecision() << "]"
              << xlib::Color::FG_DEFAULT << std::endl;
}
#endif

} // namespace timer

