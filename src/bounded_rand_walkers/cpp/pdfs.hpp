#ifndef PDFS_H
#define PDFS_H

#include "common.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include <cmath>
#include <iostream>
#include <math.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

const double pi = 3.1415926535897;

typedef double pdf_func(const dvect &, dvect &, void *);

double power(const dvect &x, dvect &grad, void *my_func_data);

double _sinc(const double x) { return std::sin(pi * x) / (pi * x); }

inline double s_funky(const dvect &x, dvect centre, double width) {
  double frequency = 3.7;
  double grad = 1.;

  // Calculate distance to centre.
  dvect squared;
  for (size_t i = 0; i < x.size(); ++i) {
    squared.push_back(std::pow((x[i] - centre[i]), 2.));
  }
  double sum_of_squared(0.); // result of the summation
  // iterate backwards from end to start
  for (int i(squared.size()); i > 0; --i)
    sum_of_squared += squared[i - 1];
  double distance = std::pow(sum_of_squared, 0.5);

  double prob = 1.;
  double arg;
  double constant;
  double scale;
  pdf_data data;
  struct pdf_data *data_ptr = &data;
  dvect dummy{};
  dvect pos;

  if ((distance > 1e-10) && (distance <= ((1 / 3.) * width))) {
    arg = distance * frequency;
    prob = std::abs(_sinc(arg));
    prob *= (1 + (5 * distance));
  } else if ((distance > (width / 3.)) && (distance <= (2 * width / 3.))) {
    constant = std::abs(_sinc(width * frequency / 3.)) * (1 + 5 * width / 3.);
    prob = constant * (1 + grad * (distance - (width / 3.)));
  } else if (distance > (2 * width / 3.)) {
    constant = std::abs(_sinc((width * frequency / 3.))) *
               (1 + 5 * (width / 3.)) * (1 + grad * (width / 3.));

    // pdf params are specified here
    data.centre = dvect{
        2 * width / 3.,
    };
    data.exponent = 0.25;
    data.binsize = 0.001;
    pos = {
        2 * width / 3.,
    };
    scale = power(pos, dummy, data_ptr);
    pos = {
        distance,
    };
    prob = (power(pos, dummy, data_ptr) / scale) * constant;
  }
  return prob;
}

double funky(const dvect &x, dvect &grad_dummy, void *my_func_data) {
  dvect centre_vect = ((pdf_data *)my_func_data)->centre;
  double width = ((pdf_data *)my_func_data)->width;
  return s_funky(x, centre_vect, width);
}

double exponential(const dvect &x, dvect &grad, void *my_func_data) {
  dvect centre_vect = ((pdf_data *)my_func_data)->centre;
  double decay_rate = ((pdf_data *)my_func_data)->decay_rate;

  dvect squared;
  for (size_t i = 0; i < x.size(); ++i) {
    squared.push_back(std::pow((x[i] - centre_vect[i]), 2.));
  }
  double sum_of_squared(0.); // result of the summation
  // iterate backwards from end to start
  for (int i(squared.size()); i > 0; --i)
    sum_of_squared += squared[i - 1];
  double distance = std::pow(sum_of_squared, 0.5);

  return std::exp(-distance * decay_rate);
}

double power(const dvect &x, dvect &grad, void *my_func_data) {
  dvect centre_vect = ((pdf_data *)my_func_data)->centre;
  double exponent = ((pdf_data *)my_func_data)->exponent;
  double binsize = ((pdf_data *)my_func_data)->binsize;

  dvect squared;
  for (size_t i = 0; i < x.size(); ++i) {
    squared.push_back(std::pow((x[i] - centre_vect[i]), 2.));
  }
  double sum_of_squared(0.); // result of the summation
  // iterate backwards from end to start
  for (int i(squared.size()); i > 0; --i)
    sum_of_squared += squared[i - 1];
  double distance = std::pow(sum_of_squared, 0.5);
  distance += binsize;
  return 0.5 * std::pow(distance, -exponent) /
         (std::pow(binsize, 1 - exponent));
}

double tophat(const dvect &x, dvect &grad, void *my_func_data) {
  dvect centre_vect = ((pdf_data *)my_func_data)->centre;
  std::vector<std::size_t> centre_shape = {
      centre_vect.size(),
  };

  dxarray centre = xt::adapt(centre_vect, centre_shape);
  double width = ((pdf_data *)my_func_data)->width;

  // convert the vector into an xarray
  std::vector<std::size_t> shape = {
      x.size(),
  };
  auto pos = xt::adapt(x, shape);
  // print("attempting");
  // print(pos);
  if (xt::all(xt::sqrt(xt::sum(xt::pow(
                  (pos - xt::view(centre, xt::range(0, x.size()))), 2.))) <
              (width / 2.))) {
    return 1.;
  } else {
    return 0.;
  }
}

inline double s_gauss(const dvect &x, double width, dvect centre) {
  dvect power_2(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    power_2[i] = std::pow((x[i] - centre[i]), 2.);
  }
  double sum(0);
  for (size_t i = 0; i < x.size(); ++i) {
    sum += power_2[i];
  }
  return std::exp(-sum / (2 * (std::pow(width, 2.))));
}

double gauss(const dvect &x, dvect &grad, void *my_func_data) {
  /*
   * my_func_data -> width, where width describes the std
   */
  dvect centre_vect = ((pdf_data *)my_func_data)->centre;
  double width = ((pdf_data *)my_func_data)->width;
  return s_gauss(x, width, centre_vect);
}

double arbitrary(const dvect &x, dvect &grad, void *my_func_data) {
  dvect centre_vect = ((pdf_data *)my_func_data)->centre;
  std::vector<std::size_t> centre_shape = {
      centre_vect.size(),
  };

  dxarray centre = xt::adapt(centre_vect, centre_shape);
  double width = ((pdf_data *)my_func_data)->width;
  // convert the vector into an xarray
  std::vector<std::size_t> shape = {
      x.size(),
  };
  auto pos = xt::adapt(x, shape);
  // print("attempting");
  // print(pos);
  auto r =
      -xt::sum(xt::pow((pos - xt::view(centre, xt::range(0, x.size()))), 2.));
  return r[0];
}

std::unordered_map<std::string, pdf_func *> pdf_map = {
    {"funky", funky}, {"arbitrary", arbitrary}, {"exponential", exponential},
    {"gauss", gauss}, {"power", power},         {"tophat", tophat},
};

pdf_func *get_pdf(std::string pdf_name) {
  // Retrieve pdf from `pdf_map`.
  if (auto it{pdf_map.find(pdf_name)}; it != std::end(pdf_map)) {
    auto pdf{it->second};
    return pdf;
  }

  // If no matching pdf could be found, raise an exception containing the names
  // of the available pdfs.
  std::string s = "PDF '" + pdf_name + "' not found. Available pdfs: ";

  std::unordered_map<std::string, pdf_func *>::iterator it = pdf_map.begin();
  while (it != pdf_map.end()) {
    s += it->first;
    if (++it != pdf_map.end()) {
      s += ", ";
    }
  }
  s += '.';
  throw std::invalid_argument(s);
}

#endif
