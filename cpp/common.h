#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <random>
#include <iostream>
#include "cnpy.h"
#include "xtensor/xarray.hpp"

typedef std::vector<double> dvect;
typedef xt::xarray<double> dxarray;

bool VERBOSE = false;
double max_rand = RAND_MAX;


struct pdf_data {
    dvect centre;
    double width;
};


template <class vect_type>
void print_1d_vect(vect_type vect);

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


template <class T, class T2>
void plot_square(T v, const T2 shape){
    cnpy::npy_save("/tmp/v_test.npy", &v[0], shape, "w");
    std::system("./visualisation.py /tmp/v_test.npy");
}


template <class T, class T2>
void plot_hist(T v, const T2 shape){
    cnpy::npy_save("/tmp/v_test.npy", &v[0], shape, "w");
    std::system("./visualisation.py /tmp/v_test.npy hist");
}


// return an evenly spaced 1-d grid of doubles.
// from http://rncarpio.github.io/linterp/
dvect linspace(double first, double last, int len) {
  dvect result(len);
  double step = (last-first) / (len - 1);
  for (int i=0; i<len; i++) { result[i] = first + i*step; }
  return result;
}


#endif
