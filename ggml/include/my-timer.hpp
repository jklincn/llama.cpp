#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

class Timer {
  public:
    /// @param name  名称，用于打印
    /// @param unit  "s" | "ms" | "us" | "ns"  控制打印单位，默认 "ms"
    /// @param os    输出流，默认 std::cerr
    explicit Timer(std::string name, std::string unit = "ms", std::ostream & os = std::cerr) :
        name_(std::move(name)),
        unit_(std::move(unit)),
        os_(os),
        start_(clock::now()) {}

    /// 离开作用域时自动打印耗时
    ~Timer() {
        auto end = clock::now();

        double       value  = 0.0;
        const char * suffix = nullptr;

        if (unit_ == "s") {
            using dur = std::chrono::duration<double>;
            value     = dur(end - start_).count();
            suffix    = " s";
        } else if (unit_ == "us") {
            using dur = std::chrono::duration<double, std::micro>;
            value     = dur(end - start_).count();
            suffix    = " us";
        } else if (unit_ == "ns") {
            using dur = std::chrono::duration<double, std::nano>;
            value     = dur(end - start_).count();
            suffix    = " ns";
        } else {  // 默认毫秒
            using dur = std::chrono::duration<double, std::milli>;
            value     = dur(end - start_).count();
            suffix    = " ms";
        }

        os_ << std::fixed << std::setprecision(3) << "[TIMER] " << name_ << " took " << value << suffix << '\n';
    }

  private:
    using clock = std::chrono::steady_clock;

    std::string                    name_;
    std::string                    unit_;
    std::ostream &                 os_;
    std::chrono::time_point<clock> start_;
};
