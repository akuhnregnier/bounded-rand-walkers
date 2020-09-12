#ifndef PDFS_H
#define PDFS_H

#include "common.h"
#include <cmath>
#include <iostream>
#include <math.h>
#include <vector>

const double pi = 3.1415926535897;

double power(const dvect &x, dvect &grad, void *my_func_data);

double sinc(const double x) { return std::sin(pi * x) / (pi * x); }

double funky(const dvect &x, dvect &grad_dummy, void *my_func_data) {
  dvect centre_vect = ((pdf_data *)my_func_data)->centre;
  double width = ((pdf_data *)my_func_data)->width;
  double frequency = 3.7;
  double grad = 1.;

  dvect squared;
  for (size_t i = 0; i < x.size(); ++i) {
    squared.push_back(std::pow((x[i] - centre_vect[i]), 2.));
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
    prob = std::abs(sinc(arg));
    prob *= (1 + (5 * distance));
  } else if ((distance > ((1 / 3.) * width)) &&
             (distance <= (2 / 3.) * width)) {
    constant = std::abs(sinc((1 / 3.) * width * frequency)) *
               (1 + 5 * (1 / 3.) * width);
    prob = constant * (1 + grad * (distance - ((1 / 3.) * width)));
  } else if (distance > ((2 / 3.) * width)) {
    constant = std::abs(sinc((1 / 3.) * width * frequency)) *
               (1 + 5 * (1 / 3.) * width) * (1 + grad * (1 / 3.) * width);

    // pdf params are specified here
    data.centre = dvect{
        (2 / 3.) * width,
    };
    data.exponent = 0.25;
    data.binsize = 0.001;
    pos = {
        (2 / 3.) * width,
    };
    scale = power(pos, dummy, data_ptr);
    pos = {
        distance,
    };
    prob = power(pos, dummy, data_ptr) / (scale * constant);
  }
  return prob;
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
  // This is a radially symmetric tophat function
  dvect centre_vect = ((pdf_data *)my_func_data)->centre;
  double width = ((pdf_data *)my_func_data)->width;

  dvect squared;
  for (size_t i = 0; i < x.size(); ++i) {
    squared.push_back(std::pow((x[i] - centre_vect[i]), 2.));
  }

  double sum_of_squared(0.); // result of the summation
  // iterate backwards from end to start
  for (int i(squared.size()); i > 0; --i)
    sum_of_squared += squared[i - 1];
  double distance = std::pow(sum_of_squared, 0.5);

  // print("distance");
  // print(distance);
  if (distance < (width / 2.)) {
    return 1.;
  } else {
    return 0.;
  }
}

double gauss(const dvect &x, dvect &grad, void *my_func_data) {
  /*
   * my_func_data -> width, where width describes the std
   */
  dvect centre_vect = ((pdf_data *)my_func_data)->centre;
  double width = ((pdf_data *)my_func_data)->width;
  dvect power_2(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    power_2[i] = std::pow((x[i] - centre_vect[i]), 2.);
  }
  double sum(0);
  for (size_t i = 0; i < x.size(); ++i) {
    sum += power_2[i];
  }
  return std::exp(-sum / (2 * (std::pow(width, 2.))));
}

double arbitrary(const dvect &x, dvect &grad, void *my_func_data) {
  dvect centre_vect = ((pdf_data *)my_func_data)->centre;
  // double width = ((pdf_data*) my_func_data) -> width;
  // print("attempting");
  // print(pos);
  dvect power_2(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    power_2[i] = std::pow((x[i] - centre_vect[i]), 2.);
  }
  double sum(0);
  for (size_t i = 0; i < x.size(); ++i) {
    sum += power_2[i];
  }

  return 10000 - sum;
}

#endif
