#include "boost/multi_array.hpp"
#include "cnpy.h"
#include "common.h"
#include "linterp.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <map>
#include <string>
#include <vector>

int main() {
  srand(time(NULL));
  float max_rand = RAND_MAX;
  const int D = 1;
  const int l = 10;
  std::vector<double> grid = linspace(0.0, 1.0, l);
  std::vector<std::vector<double>::iterator> grid_iter_list;
  grid_iter_list.push_back(grid.begin());

  array<int, 1> grid_sizes;
  grid_sizes[0] = l;

  int num_elements = grid_sizes[0];

  std::vector<double> f_values(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    f_values[i] = i + 2 * (rand() / max_rand);
  }
  const std::vector<long unsigned int> shape = {l};
  plot_square(f_values, shape);

  // construct the interpolator. the last two arguments are pointers to the
  // underlying data
  InterpMultilinear<D, double> interp_ML(grid_iter_list.begin(),
                                         grid_sizes.begin(), f_values.data(),
                                         f_values.data() + num_elements);

  // interpolate one value
  array<double, 1> args = {0.5};
  printf("%f -> %f\n", args[0], interp_ML.interp(args.begin()));

  std::vector<double> interp_grid = linspace(0.0, 1.0, l * 10);
  int num_interp_elements = interp_grid.size();
  std::vector<double> interp_x1(num_interp_elements);
  for (unsigned int i = 0; i < interp_grid.size(); i++) {
    interp_x1[i] = interp_grid[i];
  }
  // pass in a sequence of iterators, one for each coordinate
  std::vector<std::vector<double>::iterator> interp_x_list;
  interp_x_list.push_back(interp_x1.begin());

  std::vector<double> result(num_interp_elements);
  interp_ML.interp_vec(num_interp_elements, interp_x_list.begin(),
                       interp_x_list.end(), result.begin());

  const std::vector<long unsigned int> interp_shape = {interp_grid.size()};
  plot_square(result, interp_shape);

  return 0;
}
