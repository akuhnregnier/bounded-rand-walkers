#include "boost/multi_array.hpp"
#include "cnpy.h"
#include "linterp.h"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <map>
#include <math.h>
#include <nlopt.hpp>
#include <string>
#include <typeinfo>
#include <vector>

typedef std::vector<double> dvect;
typedef xt::xarray<double> dxarray;

template <class print_type> void print(print_type to_print) {
  std::cout << to_print << std::endl;
}

struct pdf_data {
  dvect centre;
  double width;
};

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

xt::xarray<double> find_maximum(nlopt::vfunc func, std::vector<double> &x,
                                std::vector<double> &lb,
                                std::vector<double> &ub, void *my_func_data) {
  /* func: pdf
   * x: initial guess
   * lb: lower bound
   * ub: upper bound
   * All entries (except for func) are n-dimensional vectors, where n is
   * the number of dimensions in the problem.
   */
  nlopt::opt opt;
  opt =
      nlopt::opt(nlopt::LN_COBYLA, x.size()); /* algorithm and dimensionality */
  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);
  opt.set_max_objective(*func, my_func_data);
  opt.set_xtol_rel(1e-4);
  double minf; /* `*`the` `minimum` `objective` `value,` `upon` `return`*` */
  int result = opt.optimize(x, minf);
  if (result < 0) {
    printf("nlopt failed!\n");
    print(result);
  } else {
    if (x.size() == 1) {
      printf("found maximum at f(%g) = %0.10g\n", x[0], minf);
    } else if (x.size() == 2) {
      printf("found maximum at f(%g,%g) = %0.10g\n", x[0], x[1], minf);
    }
  }
  // nlopt_destroy(opt);
  xt::xarray<double> max_pos;
  max_pos = xt::zeros<double>({
      x.size(),
  });
  for (int i = 0; i < x.size(); ++i) {
    max_pos[i] = x[i];
  }
  return max_pos;
}

int main() {

  // the pdf params - giving 2 centre coords for the 1D case results
  // in the first part of the centre coords being used.
  pdf_data data;
  struct pdf_data *data_ptr = &data;
  data.centre = dvect{0.3, 0.0};
  data.width = 0.005;

  /*
  // 2D testing
  dvect lb = { -1, -1 }; // lower bounds
  dvect ub = { 1, 1 }; // upper bounds
  dvect x = {-0.1, 0.002};
  auto output = find_maximum(arbitrary, x, lb, ub, data_ptr);
  print(output);
  */

  // 1D testing
  dvect lb = {
      -1,
  };
  dvect ub = {
      1,
  };
  dvect x = {
      0.2,
  };
  auto output = find_maximum(arbitrary, x, lb, ub, data_ptr);
  print(output);

  /*
  // Additional trials to see how discontinuous functions are handled
  x = {0.15, 0.0};
  auto output2 = find_maximum(tophat, x, lb, ub);
  print(output2);

  x = {0.35, 0.0};
  auto output3 = find_maximum(tophat, x, lb, ub);
  print(output3);
  */
  return 0;
}
