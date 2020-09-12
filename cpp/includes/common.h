#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <random>
#include <iostream>
#include "cnpy.h"
#include "xtensor/xarray.hpp"
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>

typedef std::vector<double> dvect;
typedef xt::xarray<double> dxarray;
typedef std::vector<dvect> vect_dvect;

bool VERBOSE = false;
double max_rand = RAND_MAX;


template <int I>
struct Int2Type
{
  enum { value = I };
};


struct pdf_data {
    dvect centre;
    double width;
    double decay_rate;
    double exponent;
    double binsize;
};


template <class x_type>
inline dvect transform_to_vect(const x_type& arr);

template <class print_type>
void print(print_type to_print){
    std::cout << to_print << std::endl;
}


inline double random_real(double lower, double upper) {
    return ((rand() / max_rand) * (upper - lower)) + lower;
}


template <class T>
void print_1d(T v){
    std::cout << "Vector contents:" << std::endl;
    for (int i=0; i < v.size(); i++){
        std::cout << v[i] << ' ';
    }
    std::cout << std::endl;
}


template <class T>
std::string format_1d(T v){
    std::stringstream ss;
    for (int i=0; i < v.size(); i++){
        ss.precision(3);
        ss << std::scientific;
        ss << v[i] << ",";
    }
    return ss.str();
}


template <class T, class T2>
void plot_square(const T& v, const T2& shape){
    cnpy::npy_save("/tmp/v_test.npy", &v[0], shape, "w");
    std::system("../visualisation.py /tmp/v_test.npy");
}


template <class T, class T2>
void plot_hist(const T& v, const T2& shape){
    cnpy::npy_save("/tmp/v_test.npy", &v[0], shape, "w");
    std::system("./visualisation.py /tmp/v_test.npy hist");
}


template <class T, class T2>
void save_np(const T& v, const T2& shape, const std::string& filename){
    cnpy::npy_save(filename, &v[0], shape, "w");
}

// return an evenly spaced 1-d grid of doubles.
// from http://rncarpio.github.io/linterp/
dvect linspace(double first, double last, int len) {
  dvect result(len);
  double step = (last-first) / (len - 1);
  for (int i=0; i<len; i++) { result[i] = first + i*step; }
  return result;
}


void set_seed() {
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
}


inline bool file_exists (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

#endif
