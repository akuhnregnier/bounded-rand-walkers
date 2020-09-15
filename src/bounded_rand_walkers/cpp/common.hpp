#ifndef COMMON_H
#define COMMON_H

#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/time.h>
#include <vector>

typedef std::vector<double> dvect;
typedef xt::xarray<double> dxarray;
typedef std::vector<dvect> vect_dvect;

bool VERBOSE = false;
double max_rand = RAND_MAX;

template <int I> struct Int2Type {
  enum { value = I };
};

struct pdf_data {
  dvect centre;
  double width;
  double decay_rate;
  double exponent;
  double binsize;
};

template <class x_type> inline dvect transform_to_vect(const x_type &arr) {
  dvect arr_vect(arr.size());
  for (long unsigned int it = 0; it < arr.size(); ++it) {
    arr_vect[it] = arr[it];
  }
  return arr_vect;
}

template <class print_type> void print(print_type to_print) {
  std::cout << to_print << std::endl;
}

inline double random_real(double lower, double upper) {
  return ((rand() / max_rand) * (upper - lower)) + lower;
}

template <class T> void print_1d(T v) {
  std::cout << "Vector contents:" << std::endl;
  for (int i = 0; i < v.size(); i++) {
    std::cout << v[i] << ' ';
  }
  std::cout << std::endl;
}

template <class T> std::string format_1d(T v) {
  std::stringstream ss;
  for (int i = 0; i < v.size(); i++) {
    ss.precision(3);
    ss << std::scientific;
    ss << v[i] << ",";
  }
  return ss.str();
}

void set_seed(int seed = -1) {
  if (seed == -1) {
    struct timeval time;
    gettimeofday(&time, NULL);
    // from
    // https://stackoverflow.com/questions/322938/recommended-way-to-initialize-srand
    // microsecond has 1 000 000
    // Assuming you did not need quite that accuracy
    // Also do not assume the system clock has that accuracy.
    srand((time.tv_sec * 1000) + (time.tv_usec / 1000));
    // The trouble here is that the seed will repeat every
    // 24 days or so.

    // If you use 100 (rather than 1000) the seed repeats every 248 days.

    // Do not make the MISTAKE of using just the tv_usec
    // This will mean your seed repeats every second.
  } else {
    srand((unsigned int)seed);
  }
}

inline dvect vect_linspace(double start, double end, unsigned long int N) {
  dxarray xt_xs = xt::linspace<double>(start, end, N);
  dvect xs(xt_xs.begin(), xt_xs.end());
  return xs;
}

#endif
