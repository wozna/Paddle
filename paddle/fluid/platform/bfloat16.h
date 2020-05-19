// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#pragma once

#include <stdint.h>
#include <limits>
#if !defined(_WIN32)
#define PADDLE_ALIGN(x) __attribute__((aligned(x)))
#else
#define PADDLE_ALIGN(x) __declspec(align(x))
#endif

namespace paddle {
namespace platform {

// Forward declare bfloat16 for eigen.h
struct bfloat16;

}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace platform {

#include <stdio.h>
#include <cstdint>
#include <cstring>
#include <iostream>

struct PADDLE_ALIGN(2) bfloat16 {
 public:
  uint16_t x;

  // The following defaulted special class member functions
  // are added to make bfloat16 pass the std::is_trivial test
  bfloat16() = default;
  bfloat16(const bfloat16& o) = default;
  bfloat16& operator=(const bfloat16& o) = default;
  bfloat16(bfloat16&& o) = default;
  bfloat16& operator=(bfloat16&& o) = default;
  ~bfloat16() = default;
  // Constructors

  inline explicit bfloat16(float val) {
    std::memcpy(&x, reinterpret_cast<char*>(&val) + 2, 2);
  }

  template <class T>
  inline explicit bfloat16(const T& val)
      : x(bfloat16(static_cast<float>(val)).x) {}

  inline bfloat16& operator=(bool b) {
    x = b ? 0x4049 : 0;
    return *this;
  }

  inline bfloat16& operator=(int8_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  inline bfloat16& operator=(uint8_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  inline bfloat16& operator=(int16_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  inline bfloat16& operator=(uint16_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  inline bfloat16& operator=(int32_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  inline bfloat16& operator=(uint32_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  inline bfloat16& operator=(int64_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  inline bfloat16& operator=(uint64_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  inline bfloat16& operator=(float val) {
    x = bfloat16(val).x;
    return *this;
  }

  inline bfloat16& operator=(double val) {
    x = bfloat16(val).x;
    return *this;
  }

  inline explicit operator float() const {
    float val;
    uint16_t temp = x;
    memcpy(reinterpret_cast<char*>(&val) + 2, reinterpret_cast<char*>(&temp),
           2);
    return val;
  }

  inline explicit operator bool() const { return (x & 0x7fff) != 0; }

  inline explicit operator int8_t() const {
    return static_cast<int8_t>(static_cast<float>(*this));
  }

  inline explicit operator uint8_t() const {
    return static_cast<uint8_t>(static_cast<float>(*this));
  }

  inline explicit operator int16_t() const {
    return static_cast<int16_t>(static_cast<float>(*this));
  }

  inline explicit operator uint16_t() const {
    return static_cast<uint16_t>(static_cast<float>(*this));
  }

  inline explicit operator int32_t() const {
    return static_cast<int32_t>(static_cast<float>(*this));
  }

  inline explicit operator uint32_t() const {
    return static_cast<uint32_t>(static_cast<float>(*this));
  }

  inline explicit operator int64_t() const {
    return static_cast<int64_t>(static_cast<float>(*this));
  }

  inline explicit operator uint64_t() const {
    return static_cast<uint64_t>(static_cast<float>(*this));
  }

  inline explicit operator double() const {
    return static_cast<double>(static_cast<float>(*this));
  }
};

inline bfloat16 operator+(const bfloat16& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) + static_cast<float>(b));
}

inline bfloat16 operator-(const bfloat16& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) - static_cast<float>(b));
}

inline bfloat16 operator*(const bfloat16& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) * static_cast<float>(b));
}

inline bfloat16 operator/(const bfloat16& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) / static_cast<float>(b));
}

inline bfloat16 operator-(const bfloat16& a) {
  bfloat16 res;
  res.x = a.x ^ 0x8000;
  return res;
}

inline bfloat16& operator+=(bfloat16& a, const bfloat16& b) {  // NOLINT
  a = bfloat16(static_cast<float>(a) + static_cast<float>(b));
  return a;
}

inline bfloat16& operator-=(bfloat16& a, const bfloat16& b) {  // NOLINT
  a = bfloat16(static_cast<float>(a) - static_cast<float>(b));
  return a;
}

inline bfloat16& operator*=(bfloat16& a, const bfloat16& b) {  // NOLINT
  a = bfloat16(static_cast<float>(a) * static_cast<float>(b));
  return a;
}

inline bfloat16& operator/=(bfloat16& a, const bfloat16& b) {  // NOLINT
  a = bfloat16(static_cast<float>(a) / static_cast<float>(b));
  return a;
}

inline bfloat16 raw_uint16_to_bfloat16(uint16_t a) {
  bfloat16 res;
  res.x = a;
  return res;
}

inline bool operator==(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) == static_cast<float>(b);
}

inline bool operator!=(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) != static_cast<float>(b);
}

inline bool operator<(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) < static_cast<float>(b);
}

inline bool operator<=(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}

inline bool operator>(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) > static_cast<float>(b);
}

inline bool operator>=(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}

inline bool(isnan)(const bfloat16& a) { return (a.x & 0x7FFF) > 0x7F80; }

inline bool(isinf)(const bfloat16& a) { return (a.x & 0x7F80) == 0x7F80; }

inline bool(isfinite)(const bfloat16& a) {
  return !((isnan)(a)) && !((isinf)(a));
}

inline std::ostream& operator<<(std::ostream& os, const bfloat16& a) {
  os << a.x;
  return os;
}

}  // namespace platform
}  // namespace paddle

namespace std {

// Override the std::is_pod::value for float16
// The reason is that different compilers implemented std::is_pod based on
// different C++ standards. float16 class is a plain old data in C++11 given
// that it is both trivial and standard_layout.
// However, std::is_pod in nvcc 8.0 host c++ compiler follows C++0x and is
// more restricted in that you cannot provide any customized
// constructor in float16. Hence, we override is_pod here following C++11
// so that .cu files can be successfully compiled by nvcc.
template <>
struct is_pod<paddle::platform::bfloat16> {
  static const bool value =
      is_trivial<paddle::platform::bfloat16>::value &&
      is_standard_layout<paddle::platform::bfloat16>::value;
};

template <>
struct is_floating_point<paddle::platform::bfloat16>
    : std::integral_constant<
          bool, std::is_same<paddle::platform::bfloat16,
                             typename std::remove_cv<
                                 paddle::platform::bfloat16>::type>::value> {};
template <>
struct is_signed<paddle::platform::bfloat16> {
  static const bool value = true;
};

template <>
struct is_unsigned<paddle::platform::bfloat16> {
  static const bool value = false;
};

inline bool isnan(const paddle::platform::bfloat16& a) {
  return paddle::platform::isnan(a);
}

inline bool isinf(const paddle::platform::bfloat16& a) {
  return paddle::platform::isinf(a);
}

template <>
struct numeric_limits<paddle::platform::bfloat16> {
  static const bool is_specialized = true;
  static const bool is_signed = true;
  static const bool is_integer = false;
  static const bool is_exact = false;
  static const bool has_infinity = true;
  static const bool has_quiet_NaN = true;
  static const bool has_signaling_NaN = true;
  static const float_denorm_style has_denorm = denorm_present;
  static const bool has_denorm_loss = false;
  static const std::float_round_style round_style = std::round_to_nearest;
  static const bool is_iec559 = false;
  static const bool is_bounded = false;
  static const bool is_modulo = false;
  static const int digits = 11;
  static const int digits10 = 3;
  static const int max_digits10 = 5;
  static const int radix = 2;
  static const int min_exponent = -13;
  static const int min_exponent10 = -4;
  static const int max_exponent = 16;
  static const int max_exponent10 = 4;
  static const bool traps = true;
  static const bool tinyness_before = false;

  static paddle::platform::bfloat16(min)() {
    return paddle::platform::raw_uint16_to_bfloat16(0x0000);
  }
  static paddle::platform::bfloat16 lowest() {
    return paddle::platform::raw_uint16_to_bfloat16(0xfbff);
  }
  static paddle::platform::bfloat16(max)() {
    return paddle::platform::raw_uint16_to_bfloat16(0x7bff);
  }
  static paddle::platform::bfloat16 epsilon() {
    return paddle::platform::raw_uint16_to_bfloat16(0x0800);
  }
  static paddle::platform::bfloat16 round_error() {
    return paddle::platform::bfloat16(0.5);
  }
  static paddle::platform::bfloat16 infinity() {
    return paddle::platform::raw_uint16_to_bfloat16(0x7c00);
  }
  static paddle::platform::bfloat16 quiet_NaN() {
    return paddle::platform::raw_uint16_to_bfloat16(0x7e00);
  }
  static paddle::platform::bfloat16 signaling_NaN() {
    return paddle::platform::raw_uint16_to_bfloat16(0x7e00);
  }
  static paddle::platform::bfloat16 denorm_min() {
    return paddle::platform::raw_uint16_to_bfloat16(0x1);
  }
};

}  // namespace std

namespace Eigen {

using bfloat16 = paddle::platform::bfloat16;

template <>
struct NumTraits<bfloat16> : GenericNumTraits<bfloat16> {
  enum {
    IsSigned = true,
    IsInteger = false,
    IsComplex = false,
    RequireInitialization = false
  };

  static inline bfloat16 epsilon() {
    return paddle::platform::raw_uint16_to_bfloat16(0x0800);
  }
  static inline bfloat16 dummy_precision() { return bfloat16(1e-2f); }
  static inline bfloat16 highest() {
    return paddle::platform::raw_uint16_to_bfloat16(0x7bff);
  }
  static inline bfloat16 lowest() {
    return paddle::platform::raw_uint16_to_bfloat16(0xfbff);
  }
  static inline bfloat16 infinity() {
    return paddle::platform::raw_uint16_to_bfloat16(0x7c00);
  }
  static inline bfloat16 quiet_NaN() {
    return paddle::platform::raw_uint16_to_bfloat16(0x7c01);
  }
};

namespace numext {

template <>
inline bool(isnan)(const bfloat16& a) {
  return (paddle::platform::isnan)(a);
}

template <>
inline bool(isinf)(const bfloat16& a) {
  return (paddle::platform::isinf)(a);
}

template <>
inline bool(isfinite)(const bfloat16& a) {
  return (paddle::platform::isfinite)(a);
}

template <>
inline bfloat16 exp(const bfloat16& a) {
  return bfloat16(::expf(static_cast<float>(a)));
}

template <>
inline bfloat16 erf(const bfloat16& a) {
  return bfloat16(::erff(static_cast<float>(a)));
}

template <>
inline bfloat16 log(const bfloat16& a) {
  return bfloat16(::logf(static_cast<float>(a)));
}

template <>
inline bfloat16 tanh(const bfloat16& a) {
  return bfloat16(::tanhf(static_cast<float>(a)));
}

template <>
inline bfloat16 sqrt(const bfloat16& a) {
  return bfloat16(::sqrtf(static_cast<float>(a)));
}

template <>
inline bfloat16 ceil(const bfloat16& a) {
  return bfloat16(::ceilf(static_cast<float>(a)));
}

template <>
inline bfloat16 floor(const bfloat16& a) {
  return bfloat16(::floorf(static_cast<float>(a)));
}

template <>
inline bfloat16 round(const bfloat16& a) {
  return bfloat16(::roundf(static_cast<float>(a)));
}

template <>
inline bfloat16 pow(const bfloat16& a, const bfloat16& b) {
  return bfloat16(::powf(static_cast<float>(a), static_cast<float>(b)));
}

template <>
inline bfloat16 abs(const bfloat16& a) {
  return bfloat16(::fabs(static_cast<float>(a)));
}

}  // namespace numext

}  // namespace Eigen
