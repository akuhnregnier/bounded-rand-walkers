#ifndef PDFS_H
#define PDFS_H

#include <math.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "common.h"


double tophat(const dvect &x, dvect &grad, void *my_func_data)
{
    dvect centre_vect = ((pdf_data*) my_func_data) -> centre;
    double width = ((pdf_data*) my_func_data) -> width;

    dvect squared;
    for (size_t i=0; i<x.size(); ++i) {
        squared.push_back(std::pow((x[i] - centre_vect[i]), 2.));
    }

    double sum_of_squared(0.); // result of the summation
    for (int i(squared.size()); i > 0; --i)
        sum_of_squared += squared[i-1];
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
    dvect centre_vect = ((pdf_data*) my_func_data) -> centre;
    double width = ((pdf_data*) my_func_data) -> width;
    dvect power_2(x.size());
    for (size_t i=0; i<x.size(); ++i) {
        power_2[i] = std::pow((x[i] - centre_vect[i]), 2.);
    }
    double sum(0);
    for (size_t i=0; i<x.size(); ++i) {
        sum += power_2[i];
    }
    return std::exp(-sum / (2*(std::pow(width, 2.))));
}


double arbitrary(const dvect &x, dvect &grad, void *my_func_data)
{
    dvect centre_vect = ((pdf_data*) my_func_data) -> centre;
    double width = ((pdf_data*) my_func_data) -> width;
    // print("attempting");
    // print(pos);
    dvect power_2(x.size());
    for (size_t i=0; i<x.size(); ++i) {
        power_2[i] = std::pow((x[i] - centre_vect[i]), 2.);
    }
    double sum(0);
    for (size_t i=0; i<x.size(); ++i) {
        sum += power_2[i];
    }

    return 10000-sum;
}


#endif
