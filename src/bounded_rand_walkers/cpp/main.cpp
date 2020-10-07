#include "common.hpp"
#include "linterp.hpp"
#include "pdfs.hpp"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "rejection_sampling.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xaxis_iterator.hpp"
#include "xtensor/xaxis_slice_iterator.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xview.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <map>
#include <math.h>
#include <nlopt.h>
#include <nlopt.hpp>
#include <numeric>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"

namespace py = pybind11;

vect_dvect bounds_to_vect(xt::pyarray<double> &vertices) {
  /* Vertices are given in as a pyarray. Convert these to a vector of vectors.
   */
  vect_dvect vert_vect;
  dvect vertex;
  for (size_t i = 0; i < vertices.shape()[0]; ++i) {
    // for (size_t j=0; j<2; ++j) {vertex.push_back(xt::view(vertices, i, j));}
    for (size_t j = 0; j < 2; ++j) {
      vertex.push_back(vertices(i, j));
    }
    vert_vect.push_back(vertex);
    vertex.clear();
  }
  return vert_vect;
}

typedef Int2Type<1> one_d;
typedef Int2Type<2> two_d;

SamplerBase *get_sampler(one_d d, size_t blocks,
                         double pdf(const dvect &, dvect &, void *),
                         pdf_data *data_ptr) {
  print("Initialising 1D sampler");
  return new Sampler1D(blocks, pdf, data_ptr);
}

SamplerBase *get_sampler(two_d d, size_t blocks,
                         double pdf(const dvect &, dvect &, void *),
                         pdf_data *data_ptr) {
  print("Initialising 2D sampler");
  return new Sampler2D(blocks, pdf, data_ptr);
}

std::vector<xt::pyarray<double>>
generate_data(xt::pyarray<double> &bounds_vertices, size_t samples = 1000,
              std::string pdf_name = "gauss", dvect centre = {0.0, 0.0},
              double width = 0.2, double decay_rate = 1, double exponent = 1,
              double binsize = 0.1, size_t blocks = 50, int seed = -1) {
  // blocks: Used for the rejection sampler.
  // seed: To initialise the random number generator. Giving -1 (default)
  // yields a different initialisation upon every call.
  //
  // Returns: [positions, steps], both of which are numpy arrays.

  if (samples < 10) {
    std::string msg = "Need samples >=10.";
    throw std::invalid_argument(msg);
  }

  set_seed(seed);

  // Configuration START
  const size_t dims = 2;
  vect_dvect bounds =
      bounds_to_vect(bounds_vertices); // only needed for `dims=2`

  double (*pdf)(const dvect &, dvect &, void *) = get_pdf(pdf_name);

  // pdf params are specified here
  // these params vary for each pdf!!
  pdf_data data;
  struct pdf_data *data_ptr = &data;
  data.centre = centre;
  data.width = width;
  data.decay_rate = decay_rate;
  data.exponent = exponent;
  data.binsize = binsize;
  // Configuration END

  // this is used to select the proper sampler
  Int2Type<dims> d_int_type;
  // store the positions
  dxarray positions = xt::empty<double>({samples + 1, dims});
  // store the steps (will have 1 less entry than position)
  dxarray steps = xt::empty<double>({samples, dims});
  // get initial position
  dvect coord; // store initial position
  bool in_bounds = false;

  SamplerBase *sampler = get_sampler(d_int_type, blocks, pdf, data_ptr);
  print("Finished Initialising");

  do {
    coord = {}; // empty initially
    // initialise coord randomly
    for (size_t i(0); i < dims; ++i) {
      coord.push_back(random_real(-1., 1.));
    }
    if (dims == 1) {
      if ((coord[0] < 1) && (coord[0] > 0)) {
        in_bounds = true;
      }
    } else if (dims == 2) {
      in_bounds = in_shape(coord, bounds);
    }
  } while (!in_bounds);

  // Assign the initial position.
  xt::row(positions, 0) = xt::adapt(coord);

  // get start time
  double start = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();

  for (size_t i(0); i < samples; ++i) {
    // carry out a sampling step until the new position lies within the
    // bounds
    dvect new_coord;
    dvect step;
    in_bounds = false;
    do {
      // copy into a new coord that is modified in the while loop
      new_coord = coord;
      step = sampler->sample(coord, pdf);
      // add step to new_coord to get new position
      for (size_t j(0); j < dims; ++j) {
        new_coord[j] += step[j];
      }
      if (dims == 1) {
        if ((new_coord[0] < 1) && (new_coord[0] > 0)) {
          in_bounds = true;
        }
      } else if (dims == 2) {
        in_bounds = in_shape(new_coord, bounds);
      }
    } while (!in_bounds);
    coord = new_coord;
    // now record this information
    xt::row(positions, i + 1) = xt::adapt(coord);
    xt::row(steps, i) = xt::adapt(step);

    // timing information
    if (((i > 2) && (i == (size_t)1e3)) ||
        (((i + 1) % ((size_t)(samples / 10))) == 0)) {
      // get current time
      double current = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
      // elapsed in seconds
      double elapsed = (current - start) / 1000.;
      double specific_elapsed = elapsed / i;
      double remaining = (samples - i) * specific_elapsed;
      printf("iteration: %0.2e, elapsed: %0.3e %s, remaining: %0.3e %s, "
             "specific time: %0.5e %s\n",
             (double)i, elapsed, "s", remaining, "s", specific_elapsed, "s");
    }
  }

  std::vector<xt::pyarray<double>> out{positions, steps};
  return out;
}

xt::pyarray<double> return_pdf(unsigned long int N) {
  dxarray y = xt::empty<double>({N});

  dvect pos;
  dvect dummy{};
  dxarray x = xt::linspace<double>(0.0, 3.0, N);

  pdf_data data;
  struct pdf_data *data_ptr = &data;
  data.centre = dvect{0.0, 0.0};
  data.width = 2.;

  for (size_t i(0); i < N; i++) {
    pos = {
        x[i],
    };
    y[i] = (funky(pos, dummy, data_ptr));
  }
  return y;
}

xt::pyarray<double> funky2(xt::pyarray<double> &x) {
  dvect dummy{};

  // Create struct that holds the pdf parameters.
  pdf_data data;
  struct pdf_data *data_ptr = &data;
  data.width = 2.;
  data.centre = dvect{0., 0.};

  // size_t d = x.dimension();
  // std::cout << "Dimensions:" << d << std::endl;

  // auto&& s = x.shape();
  // std::cout << "Shape:" << xt::adapt(s) << std::endl;

  // unsigned long int N = (d == 2) ? x.shape()[0] : 1;
  unsigned long int N = x.shape()[0];
  dxarray y = xt::empty<double>({N});

  auto iter = axis_begin(x, 0);
  auto end = axis_end(x, 0);

  unsigned long int i = 0;

  auto x_i = *iter;

  while (iter != end) {
    // std::cout << "Row:" << i << std::endl;
    x_i = *iter++;
    // std::cout << x_i << std::endl;

    // Convert pyarray to vector.
    // dvect x_vect(x_i.begin(), x_i.end());

    // for (auto el: x_vect)
    //     std::cout << el << " ";
    // std::cout << std::endl;

    // y[i] = funky(x_vect, dummy, data_ptr);
    y[i] = funky(dvect(x_i.begin(), x_i.end()), dummy, data_ptr);
    i++;
  }

  return y;
}

double call_func(std::string pdf_name, double p) {
  pdf_data data;
  struct pdf_data *data_ptr = &data;
  data.width = 2.;
  data.centre = dvect{0., 0.};

  dvect x{p};
  dvect dummy{};

  return get_pdf(pdf_name)(x, dummy, data_ptr);
}

std::vector<xt::pyarray<double>> interp_1d_test() {
  /* `f_values` contains the original function values in [0, 1], and `result`
   * should contain the interpolated values, also in [0, 1], but with 10x the
   * data points.
   */
  srand(time(NULL));
  float max_rand = RAND_MAX;
  const int l = 10;

  dvect grid = vect_linspace(0.0, 1.0, l);

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
  // plot_square(f_values, shape);

  // construct the interpolator. the last two arguments are pointers to the
  // underlying data
  InterpMultilinear<1, double> interp_ML(grid_iter_list.begin(),
                                         grid_sizes.begin(), f_values.data(),
                                         f_values.data() + num_elements);

  // interpolate one value
  array<double, 1> args = {0.5};
  printf("%f -> %f\n", args[0], interp_ML.interp(args.begin()));

  dvect interp_grid = vect_linspace(0.0, 1.0, l * 10);

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
  // plot_square(result, interp_shape);

  std::vector<xt::pyarray<double>> out{xt::adapt(f_values), xt::adapt(result)};
  return out;
}

double myfunc(unsigned n, const double *x, double *grad, void *my_func_data) {
  return pow(x[0], 2.) + pow(x[1], 2.);
}

double myconstraint(unsigned n, const double *x, double *grad, void *data) {
  return std::abs(x[0] - 0.3);
}

void nlopt_example() {
  double lb[2] = {-0.5, 0.1}; /* lower bounds */
  double ub[2] = {0.2, 0.2};  /* upper bounds */
  nlopt_opt opt;
  opt = nlopt_create(NLOPT_LN_COBYLA, 2); /* algorithm and dimensionality */
  nlopt_set_lower_bounds(opt, lb);
  nlopt_set_upper_bounds(opt, ub);
  nlopt_set_min_objective(opt, myfunc, NULL);
  // nlopt_add_inequality_constraint(opt, myconstraint, NULL, 1e-8);
  // nlopt_add_inequality_constraint(opt, myconstraint, NULL, 1e-8);
  nlopt_set_xtol_rel(opt, 1e-4);
  double x[2] = {0., 0.12}; /* `*`some` `initial` `guess`*` */
  double minf; /* `*`the` `minimum` `objective` `value,` `upon` `return`*` */
  int result = nlopt_optimize(opt, x, &minf);
  if (result < 0) {
    printf("nlopt failed!\n");
    print(result);
  } else {
    printf("found minimum at f(%g,%g) = %0.10g\n", x[0], x[1], minf);
  }
  nlopt_destroy(opt);
}

void nlopt_tophat_example() {
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
  auto output = find_maximum_pos(arbitrary, x, lb, ub, data_ptr);
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
  auto output = find_maximum_pos(arbitrary, x, lb, ub, data_ptr);
  print(output);

  /*
  // Additional trials to see how discontinuous functions are handled
  x = {0.15, 0.0};
  auto output2 = find_maximum_pos(tophat, x, lb, ub);
  print(output2);

  x = {0.35, 0.0};
  auto output3 = find_maximum_pos(tophat, x, lb, ub);
  print(output3);
  */
}

std::vector<xt::pyarray<double>> vectortest() {
  /* Similar to `interp_1d_test`.
   */
  srand(time(NULL));
  float max_rand = RAND_MAX;
  const int D = 1;
  const int l = 10;
  dvect grid = vect_linspace(0.0, 1.0, l);
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
  // plot_square(f_values, shape);

  // construct the interpolator. the last two arguments are pointers to the
  // underlying data
  InterpMultilinear<D, double> interp_ML(grid_iter_list.begin(),
                                         grid_sizes.begin(), f_values.data(),
                                         f_values.data() + num_elements);

  // interpolate one value
  array<double, 1> args = {0.5};
  printf("%f -> %f\n", args[0], interp_ML.interp(args.begin()));

  dvect interp_grid = vect_linspace(0.0, 1.0, l * 10);
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
  // plot_square(result, interp_shape);
  std::vector<xt::pyarray<double>> out{xt::adapt(f_values), xt::adapt(result)};
  return out;
}

// Python Module and Docstrings

PYBIND11_MODULE(_bounded_rand_walkers_cpp, m) {
  xt::import_numpy();

  m.doc() = R"pbdoc(
        Bounded random walker simulation.
    )pbdoc";

  using namespace pybind11::literals;

  dvect default_centre = {0.0, 0.0};
  double default_width = 0.2;
  dvect default_start_pos = {0.0, 0.0};

  m.def("generate_data", generate_data, "Generate random walker data.",
        "bounds_vertices"_a, "samples"_a = 1000, "pdf_name"_a = "gauss",
        "centre"_a = default_centre, "width"_a = default_width,
        "decay_rate"_a = 1, "exponent"_a = 1, "binsize"_a = 0.1,
        "blocks"_a = 50, "seed"_a = -1);

  m.def("interp_1d_test", interp_1d_test, "1D interpolation test.");

  m.def("nlopt_tophat_example", nlopt_tophat_example, "Tophat peak finding.");

  m.def("vectortest", vectortest, "1D interpolation test.");

  m.def("testing_1d", &testing_1d, "1D sampling test.", "N"_a = 1000,
        "pdf_name"_a = "gauss", "start_pos"_a = 0, "centre"_a = default_centre,
        "width"_a = default_width, "decay_rate"_a = 1, "exponent"_a = 1,
        "binsize"_a = 0.1, "blocks"_a = 2000);

  m.def("testing_2d", testing_2d, "2D sampling test.", "N"_a = 1000,
        "pdf_name"_a = "gauss", "start_pos"_a = default_start_pos,
        "centre"_a = default_centre, "width"_a = default_width,
        "decay_rate"_a = 1, "exponent"_a = 1, "binsize"_a = 0.1,
        "blocks"_a = 100);

  m.def("call_func", call_func, "Call specified pdf.");

  m.def("gauss", s_gauss, "Gaussian pdf.", "x"_a, "width"_a = default_width,
        "centre"_a = default_centre);

  m.def("funky", s_funky, "Funky pdf.", "x"_a, "centre"_a = default_centre,
        "width"_a = default_width);

  m.def("funky2", s_funky2, "Funky pdf with 2x scaling.", "x"_a,
        "centre"_a = default_centre, "width"_a = default_width);
}
