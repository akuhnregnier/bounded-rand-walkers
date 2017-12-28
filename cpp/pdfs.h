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


// overload the above to remove requirement for gradient vector (needed for the
// maximum finding library) and to make the calculations faster.
double tophat(const dvect &x, void *my_func_data)
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
    double rad_dist = std::pow(sum, 0.5);
    return std::exp(-rad_dist / width);
}


double gauss(const dvect &x, void *my_func_data) {
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
    double rad_dist = std::pow(sum, 0.5);
    return std::exp(-rad_dist / width);
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


double arbitrary(const dvect &x, void *my_func_data)
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
