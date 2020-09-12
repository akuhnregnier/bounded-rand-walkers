#ifndef REJECTION_SAMPLING_H
#define REJECTION_SAMPLING_H

#include "boost/multi_array.hpp"
#include "common.h"
#include "linterp.h"
#include "pdfs.h"
#include "polygon_inclusion.h"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xcontainer.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xfunction.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xslice.hpp"
#include "xtensor/xview.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <map>
#include <math.h>
#include <memory>
#include <nlopt.hpp>
#include <random>
#include <string>
#include <sys/time.h>
#include <typeinfo>
#include <vector>

double find_maximum(nlopt::vfunc func, dvect &x, dvect &lb, dvect &ub,
                    void *my_func_data) {
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
  double maxf; /* `*`the` `minimum` `objective` `value,` `upon` `return`*` */
  int result = opt.optimize(x, maxf);
  if (result < 0) {
    printf("nlopt failed!\n");
    print(result);
  } else {
    if (x.size() == 1) {
      if (VERBOSE) {
        printf("found maximum at f(%g) = %0.10g\n", x[0], maxf);
      }
    } else if (x.size() == 2) {
      if (VERBOSE) {
        printf("found maximum at f(%g,%g) = %0.10g\n", x[0], x[1], maxf);
      }
    }
  }
  // nlopt_destroy(opt);
  dxarray max_pos;
  max_pos = xt::zeros<double>({
      x.size(),
  });
  for (long unsigned int i = 0; i < x.size(); ++i) {
    max_pos[i] = x[i];
  }
  return maxf;
}

struct interp {
  InterpMultilinear<1, double> *interp_ML_ptr;

  interp(dvect &grid, dvect &f_values) {
    // std::vector<double> grid = linspace(0.0, 1.0, l);
    std::vector<std::vector<double>::iterator> grid_iter_list;
    grid_iter_list.push_back(grid.begin());
    array<int, 1> grid_sizes;
    grid_sizes[0] = grid.size();
    int num_elements = grid_sizes[0];
    // construct the interpolator. the last two arguments are pointers to the
    // underlying data now assign this to the pointer so it can be used later
    interp_ML_ptr = new InterpMultilinear<1, double>(
        grid_iter_list.begin(), grid_sizes.begin(), f_values.data(),
        f_values.data() + num_elements);
  }

  template <class array_like> double operator()(const array_like &args) {
    // interpolate one value
    return interp_ML_ptr->interp(args.begin());
  }
  ~interp() { delete interp_ML_ptr; }
};

class SamplerBase {
public:
  virtual dvect sample(const dvect &position,
                       double pdf(const dvect &, dvect &,
                                  void *)) = 0; // pure virtual function
};

class Sampler1D : public SamplerBase {
public:
  dxarray bounds{-1, 1};
  dxarray pdf_values;
  dxarray edges;
  dxarray centres;
  dxarray max_box_values;
  dxarray discrete_cdf;
  struct pdf_params;
  dvect dummy{};
  std::vector<interp> interpolators;
  std::vector<interp> inv_interpolators;
  pdf_data *func_data_store;

  Sampler1D(int blocks, double pdf(const dvect &, dvect &, void *),
            pdf_data *func_data) {
    print("4 arg init called");
    func_data_store = func_data;
    dvect centre_vect = func_data_store->centre;
    std::vector<std::size_t> centre_shape = {
        centre_vect.size(),
    };

    dxarray centre = xt::adapt(centre_vect, centre_shape);

    pdf_values = xt::zeros<double>({
        blocks,
    });
    max_box_values = xt::zeros<double>({
        blocks,
    });

    edges = xt::linspace<double>(bounds[0], bounds[1], blocks + 1);
    centres = (xt::view(edges, xt::range(1, blocks + 1)) +
               xt::view(edges, xt::range(0, blocks))) /
              2.;

    // now evaluate the pdf at each of the positions given in the
    // centres_lists, filling in the resulting values into
    // pdf_values
    dvect coord;
    dvect lower_edge;
    dvect upper_edge;
    double max_value;
    double found_max;
    vector<dvect> starts;
    for (int i = 0; i < blocks; ++i) {
      coord = {
          centres[i],
      };
      pdf_values[i] = pdf(coord, dummy, func_data);
      lower_edge = {
          edges[i],
      };
      upper_edge = {
          edges[i + 1],
      };
      max_value = 0;
      starts.push_back(lower_edge);
      starts.push_back(upper_edge);
      starts.push_back(coord);
      if (VERBOSE) {
        print("finding maxes");
      }
      for (long unsigned int j = 0; j < starts.size(); ++j) {
        if (VERBOSE) {
          print(j);
          print(starts.size());
        }
        found_max =
            find_maximum(pdf, starts[j], lower_edge, upper_edge, func_data);
        if (found_max > max_value) {
          max_value = found_max;
        }
      }
      if (max_value < pdf_values[i]) {
        throw 10;
      }
      max_box_values[i] = max_value;
      starts.clear();
    }

    // now use these max values to define the required interpolators
    discrete_cdf = cumsum(max_box_values);
    double max_discrete_cdf =
        *std::max_element(discrete_cdf.begin(), discrete_cdf.end());
    discrete_cdf /= max_discrete_cdf; // Rescale to [0, 1]
    // form `probs` vector by appending a `0` to the beginning of
    // `discrete_cdf`, in order to define probabilities at each bin
    // edge, such that the interpolation can get the probability in
    // between all of these edges then.

    // discrete_cdf.insert(discrete_cdf.begin(), 0.);
    dxarray zero_element = {
        0.,
    };
    discrete_cdf = xt::concatenate(xtuple(zero_element, discrete_cdf), 0);
    // print("discrete cdf");
    // print_1d(discrete_cdf);
    // print("vector form");
    // print_1d(discrete_cdf_vect);
    auto discrete_cdf_vect = transform_to_vect(discrete_cdf);
    auto edges_vect = transform_to_vect(edges);
    interpolators.emplace_back(discrete_cdf_vect, edges_vect);
    inv_interpolators.emplace_back(edges_vect, discrete_cdf_vect);
  }

  dvect sample(const dvect &position,
               double pdf(const dvect &, dvect &, void *)) {
    dvect coord;
    double prob2;
    double ratio;
    do {
      if (VERBOSE) {
        print("calling again");
      }
      dvect min_step = {-position[0]};
      dvect max_step = {1 - position[0]};
      // clip the steps according to the space boundaries
      if (min_step[0] < bounds[0]) {
        min_step[0] = bounds[0];
      }
      if (max_step[0] > bounds[1]) {
        max_step[0] = bounds[1];
      }
      auto min_prob = inv_interpolators[0](min_step);
      auto max_prob = inv_interpolators[0](max_step);

      // std::random_device rd;  //Will be used to obtain a seed for the random
      // number engine std::mt19937 gen(rd()); //Standard
      // mersenne_twister_engine seeded with rd()
      // std::uniform_real_distribution<double> dis(min_prob, max_prob); //
      // uniform, unbiased dvect prob = {dis(gen)};

      dvect prob = {random_real(min_prob, max_prob)};

      // print("prob");
      // print(prob);
      coord = {interpolators[0](prob)};
      double pdf_val = pdf(coord, dummy, func_data_store);
      if (pdf_val < 1e-6) {
        ratio = 0.;
        continue;
      }

      int max_val_index = 0;
      for (long unsigned int i = 0; i < edges.size(); ++i) {
        if (coord[0] < edges[i]) {
          max_val_index = i - 1;
          break;
        }
      }
      auto max_box_val = max_box_values[max_val_index];
      ratio = pdf_val / max_box_val;

      // std::uniform_real_distribution<double> dis2(0., 1.); // uniform,
      // unbiased prob2 = dis2(gen);

      prob2 = random_real(0., 1.);
    } while (prob2 >= ratio); // ie. return if (prob < ratio)

    return coord;
  }
};

class Sampler2D : public SamplerBase {
public:
  dxarray bounds{{-2, -2}, // min values
                 {2, 2}};  // max values
  dxarray pdf_values;
  dxarray edges;
  dxarray centres;
  dxarray max_box_values;
  dxarray discrete_cdf;
  dxarray first_discrete_cdf;
  dxarray second_discrete_cdf;
  struct pdf_params;
  dvect dummy{};
  std::vector<interp> interpolators;
  std::vector<interp> inv_interpolators;
  pdf_data *func_data_store;

  Sampler2D(int blocks, double pdf(const dvect &, dvect &, void *),
            pdf_data *func_data) {
    if (VERBOSE) {
      print("4 arg init called");
    }
    // make sure that the vectors do not have to re-allocate, which does not
    // work with the 'interp' struct apparently
    interpolators.reserve(blocks + 1);
    inv_interpolators.reserve(blocks + 1);

    func_data_store = func_data;
    dvect centre_vect = func_data_store->centre;
    std::vector<std::size_t> centre_shape = {
        centre_vect.size(),
    };

    dxarray centre = xt::adapt(centre_vect, centre_shape);

    pdf_values = xt::zeros<double>({blocks, blocks});
    max_box_values = xt::zeros<double>({blocks, blocks});

    // these edges and centres are the same in each dimension,
    // since the boundaries are square.
    edges =
        xt::linspace<double>(xt::index_view(bounds, {{0, 0}})[0],
                             xt::index_view(bounds, {{1, 0}})[0], blocks + 1);
    centres = (xt::view(edges, xt::range(1, blocks + 1)) +
               xt::view(edges, xt::range(0, blocks))) /
              2.;

    // now evaluate the pdf at each of the positions given in the
    // centres_lists, filling in the resulting values into
    // pdf_values
    dvect coord;
    dvect lower_edge;
    dvect upper_edge;
    dvect trial_coord;
    double max_value;
    double found_max;
    vector<dvect> starts;
    auto coord_grid = xt::meshgrid(centres, centres);
    auto x_coords = std::get<0>(coord_grid);
    auto y_coords = std::get<1>(coord_grid);
    for (long unsigned int i = 0; i != blocks; ++i) {
      for (long unsigned int j = 0; j != blocks; ++j) {
        coord = {xt::index_view(x_coords, {{i, j}})[0],
                 xt::index_view(y_coords, {{i, j}})[0]};
        if (VERBOSE) {
          print("coord");
          print_1d(coord);
          print(pdf(coord, dummy, func_data));
        }
        xt::index_view(pdf_values, {{i, j}}) = pdf(coord, dummy, func_data);

        lower_edge = {edges[i], edges[j]};
        upper_edge = {edges[i + 1], edges[j + 1]};
        starts.push_back(lower_edge);
        starts.push_back(upper_edge);
        starts.push_back(coord);
        trial_coord = {edges[i], edges[j + 1]};
        starts.push_back(trial_coord);
        trial_coord = {edges[i + 1], edges[j]};
        starts.push_back(trial_coord);
        max_value = 0;
        if (VERBOSE) {
          print("finding maxes");
        }
        for (long unsigned int j = 0; j < starts.size(); ++j) {
          if (VERBOSE) {
            print(j);
            print(starts.size());
          }
          found_max =
              find_maximum(pdf, starts[j], lower_edge, upper_edge, func_data);
          if (found_max > max_value) {
            max_value = found_max;
          }
        }
        if (max_value < pdf_values[i]) {
          throw 10;
        }
        xt::index_view(max_box_values, {{i, j}}) = max_value;
        starts.clear();
      }
    }

    auto first_summed_values = xt::sum(max_box_values, {
                                                           1,
                                                       });

    // now use these max values to define the required interpolators
    first_discrete_cdf = cumsum(first_summed_values);
    double max_discrete_cdf =
        *std::max_element(first_discrete_cdf.begin(), first_discrete_cdf.end());
    first_discrete_cdf /= max_discrete_cdf; // Rescale to [0, 1]
    // form `probs` vector by appending a `0` to the beginning of
    // `discrete_cdf`, in order to define probabilities at each bin
    // edge, such that the interpolation can get the probability in
    // between all of these edges then.

    // discrete_cdf.insert(discrete_cdf.begin(), 0.);
    dxarray zero_element = {
        0.,
    };
    first_discrete_cdf =
        xt::concatenate(xtuple(zero_element, first_discrete_cdf), 0);
    // print("discrete cdf");
    // print_1d(discrete_cdf);
    // print("vector form");
    // print_1d(discrete_cdf_vect);
    auto first_discrete_cdf_vect = transform_to_vect(first_discrete_cdf);
    auto edges_vect = transform_to_vect(edges);
    interpolators.emplace_back(first_discrete_cdf_vect, edges_vect);
    inv_interpolators.emplace_back(edges_vect, first_discrete_cdf_vect);

    // as opposed to the 1D case, we now need to add more interpolators for the
    // 2nd dimension

    second_discrete_cdf = xt::zeros<double>({
        blocks + 1,
    });
    // leave a column of 0s on the very 'left' (beginning) for the same
    // reason as the additional 0 element was added to the
    // first_discrete_cdf above.
    for (long unsigned int row(0); row < blocks; ++row) {
      // reset to 0s just to be sure. 1 extra zero as explained above.
      second_discrete_cdf = xt::zeros<double>({
          blocks + 1,
      });
      auto row_data = xt::view(max_box_values, row, xt::all());
      xt::view(second_discrete_cdf, xt::range(1, xt::placeholders::_)) =
          cumsum(row_data);

      double max_second_discrete_cdf = *std::max_element(
          second_discrete_cdf.begin(), second_discrete_cdf.end());
      if (VERBOSE) {
        print("max");
        print(max_second_discrete_cdf);
      }
      if (max_second_discrete_cdf > 1e-4) {
        second_discrete_cdf /= max_second_discrete_cdf; // Rescale to [0, 1]
      }
      auto second_discrete_cdf_vect = transform_to_vect(second_discrete_cdf);
      // can re-use the `edges_vect` from above, since the edges are symmetric
      // in both directions!
      //
      interpolators.emplace_back(second_discrete_cdf_vect, edges_vect);
      inv_interpolators.emplace_back(edges_vect, second_discrete_cdf_vect);

      // The `interpolators` vector now contains the interpolators
      // for both dimensions - in this case, the first element ([0])
      // contains the interpolator for the first dimension, and the
      // remaining 'blocks' elements contain the interpolators for
      // the 2nd dimension.
    }
  }

  dvect sample(const dvect &pos, double pdf(const dvect &, dvect &, void *)) {
    dvect coord;
    double prob2;
    double ratio;
    std::vector<std::size_t> position_shape = {2, 1};
    dxarray position = xt::adapt(pos, position_shape);
    position.reshape({2, 1});
    dxarray bound_starts = {-1, 1};
    bound_starts.reshape({1, 2});
    dxarray axes_step_bounds = xt::eval(bound_starts - position);

    // print("broadcasting stuff");
    // print(bound_starts);
    // print(position);
    // print(axes_step_bounds);
    //
    // clip the steps according to the space boundaries
    // np.clip(a, min, max) - > xt::clip(a, min, max)
    //
    // print(axes_step_bounds);
    // print(bounds);
    //
    // Is clipping even necessary? If positions stay within [-1, 1],
    // then by default, the interval [-2, 2] will not be exceeded.
    // axes_step_bounds = xt::clip(axes_step_bounds, xt::amin(bounds, {0}),
    // xt::amax(bounds, {0}));

    do {
      if (VERBOSE) {
        print("calling 2D again - position is:");
        print(position);
      }

      dvect min_step = {xt::index_view(axes_step_bounds, {{0, 0}})[0]};
      dvect max_step = {xt::index_view(axes_step_bounds, {{0, 1}})[0]};
      auto min_prob = inv_interpolators[0](min_step);
      auto max_prob = inv_interpolators[0](max_step);

      dvect first_prob = {random_real(min_prob, max_prob)};

      // print("prob");
      // print(prob);
      coord = {interpolators[0](first_prob)};

      if (VERBOSE) {
        print("1st dimension pdf coord param:");
        print_1d(coord);
        printf(
            "min step %g, max step %g, min prob %g, max prob %g, 1st prob %g\n",
            min_step[0], max_step[0], min_prob, max_prob, first_prob[0]);
      }

      size_t first_max_val_index = 0;
      for (size_t i = 0; i < edges.size(); ++i) {
        if (coord[0] < edges[i]) {
          if (VERBOSE) {
            printf("edges: ");
            print_1d(edges);
            printf("get index, comp. %g < %g\n", coord[0], edges[i]);
          }
          first_max_val_index = i - 1;
          break; // break out of the for loop as the desired index has been
                 // found
        }
      }

      // repeat the generation above for the second dimension, using the
      // `first_max_val_index` in doing so, append to `coord` using push_back
      // such that the required 2 coordinates are generated

      // 1st row this time - 2nd dimension
      min_step = {xt::index_view(axes_step_bounds, {{1, 0}})[0]};
      max_step = {xt::index_view(axes_step_bounds, {{1, 1}})[0]};
      // use the index and the offset +1 (for the first interpolator)
      // to locate the correct interpolator for the 2nd dimension
      min_prob = inv_interpolators[1 + first_max_val_index](min_step);
      max_prob = inv_interpolators[1 + first_max_val_index](max_step);
      dvect second_prob = {random_real(min_prob, max_prob)};
      coord.push_back(interpolators[1 + first_max_val_index](second_prob));

      size_t second_max_val_index = 0;
      for (size_t i = 0; i < edges.size(); ++i) {
        if (coord[1] < edges[i]) {
          second_max_val_index = i - 1;
          break; // break out of the for loop as the desired index has been
                 // found
        }
      }

      double pdf_val = pdf(coord, dummy, func_data_store);
      if (pdf_val < 1e-6) {
        ratio = -1.; // impossible, so skip this iteration of the while loop
        if (VERBOSE) {
          print("pdf coord param:");
          print_1d(coord);
          printf("first max val index %ld\n", first_max_val_index);
          printf("min step %g, max step %g, min prob %g, max prob %g, 2nd prob "
                 "%g\n",
                 min_step[0], max_step[0], min_prob, max_prob, second_prob[0]);
          printf("pdf val = %g, therefore re-doing\n", pdf_val);
        }
        continue;
      }

      auto max_box_val = xt::index_view(
          max_box_values, {{first_max_val_index, second_max_val_index}})[0];
      ratio = pdf_val / max_box_val;

      // std::uniform_real_distribution<double> dis2(0., 1.); // uniform,
      // unbiased prob2 = dis2(gen);

      prob2 = random_real(0., 1.);
      if (VERBOSE) {
        printf("%g >= %g\n", prob2, ratio);
      }
    } while (prob2 >= ratio); // ie. return if (prob < ratio)

    return coord;
  }
};

template <class x_type> inline dvect transform_to_vect(const x_type &arr) {
  dvect arr_vect(arr.size());
  for (long unsigned int it = 0; it < arr.size(); ++it) {
    arr_vect[it] = arr[it];
  }
  return arr_vect;
}

template <class arr_type> auto cumsum(const arr_type &arr) {
  auto cum_sum = arr;
  for (long unsigned int i = 1; i < arr.size(); ++i) {
    cum_sum[i] += cum_sum[i - 1];
  }
  return cum_sum;
}

void testing_1d() {
  dvect dummy{};

  dvect arr = {0., 1., 2., 2.5};
  dvect arr2 = cumsum(arr);
  print("arr");
  for (auto const &value : arr) {
    print(value);
  }
  print("arr2");
  for (auto const &value : arr2) {
    print(value);
  }
  print("Max of arr");
  print(*std::max_element(arr.begin(), arr.end()));

  int l = 5;
  std::vector<double> grid = linspace(0.0, 1.0, l);
  std::vector<double> f_values = linspace(1.0, 5.0, l);

  interp interp_s(grid, f_values);

  std::vector<double> args = {0.53};
  print(interp_s(args));

  std::vector<interp> struct_instances;
  struct_instances.emplace_back(grid, f_values);

  args = {0.59};
  print(struct_instances[0](args));

  pdf_data data;
  struct pdf_data *data_ptr = &data;
  data.centre = dvect{0.3, 0.0};
  data.width = 0.2;

  print("pdf test");
  dvect coord{0.3, 1.2};
  print("size");
  // print(coord)
  print(coord.size());
  print(tophat(coord, dummy, data_ptr));

  // double start =
  // std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  // int EVALS = 100;
  // double result;
  // for (int i=0; i<EVALS; ++i) {
  //     result = tophat(coord, dummy, data_ptr);
  // }
  // double end =
  // std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  // double time = (end - start) / 1000.;
  // print("time taken:");
  // print(time);
  // print("per evaluation");
  // print(time/((double)EVALS));

  Sampler1D sampler(2000, tophat, data_ptr);

  // print("edges");
  // print(sampler.edges);
  // print(sampler.edges.size());
  // print(sampler.centres);
  // print(sampler.centres.size());
  // print(sampler.pdf_values);
  // print(sampler.max_box_values);

  print("interp testing");
  print_1d(args);
  print(sampler.interpolators[0](args));
  args = {0.67};
  print_1d(args);
  print(sampler.interpolators[0](args));

  print("sampling");
  double start_s = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  long unsigned int L = 1000;
  dvect sample_results(L);
  for (long unsigned int i = 0; i < L; ++i) {
    sample_results[i] = sampler.sample(dvect{0.3}, tophat)[0];
  }
  double end_s = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();
  double time_s = (end_s - start_s) / 1000.;
  print("time taken:");
  print(time_s);
  print("per evaluation");
  print(time_s / ((double)L));

  const std::vector<long unsigned int> shape{
      L,
  };
  plot_hist(sample_results, shape);

  /*
  xt::xarray<double> arr;
  arr = xt::linspace<double>(-1., 1., 5);
  std::cout << arr*2 << std::endl;


  print(tophat.centre);
  print(tophat.width);
  print(tophat.pdf(0.1));
  print(tophat.pdf(0.5));
  xt::xarray<double> results;
  // results = xt::zeros<double>({arr.size(),});
  results = xt::linspace<double>(1., 2., arr.size());
  print("starting loop");
  print(results);
  for(auto it=arr.begin(); it!=arr.end(); ++it){
      print("it");
      print(*it);
      print("diff");
      print(it - arr.begin());
      results[it - arr.begin()] = tophat.pdf(*it);
  }
  print("results");
  print(results);

  // const std::vector<long unsigned int> shape {arr.size(),};
  // plot_square(results, shape);

  */
}

void testing_2d() {
  int blocks = 4;
  dxarray bounds{{-2, -2}, {2, 2}};
  auto test = xt::index_view(bounds, {{0, 0}});
  print(test);
  auto edges =
      xt::linspace<double>(xt::index_view(bounds, {{0, 0}})[0],
                           xt::index_view(bounds, {{1, 0}})[0], blocks + 1);
  print(edges);
  print(edges.size());
  auto centres = (xt::view(edges, xt::range(1, blocks + 1)) +
                  xt::view(edges, xt::range(0, blocks))) /
                 2.;
  print(centres);
  print_1d(centres.shape());
  auto coord_grid = xt::meshgrid(centres, centres);
  print_1d(std::get<0>(coord_grid).shape());
  print_1d(std::get<0>(coord_grid));
  print_1d(std::get<1>(coord_grid));
  auto x_coords = std::get<1>(coord_grid);

  // print("iteration");
  // for(auto it=x_coords.begin(); it!=x_coords.end(); ++it) {
  // // for (int i=0; i<x_coords.size(); ++i) {
  //     print(*it);
  // }
  print(bounds.size());

  //////////////////////////////////////////////////

  pdf_data data;
  struct pdf_data *data_ptr = &data;
  data.centre = dvect{0.6, 0.0};
  data.width = 0.2;

  double (*pdf)(const dvect &, dvect &, void *) = arbitrary;

  print("Initialising 2D sampler");
  Sampler2D sampler(8, pdf, data_ptr);
  print("Finished Initialising");

  print("sampler internals");
  print(sampler.edges);
  print(sampler.edges.size());
  print(sampler.centres);
  print(sampler.centres.size());
  print(sampler.pdf_values);
  print(sampler.max_box_values);

  return;

  std::vector<double> args = {0.53, 0.2};
  print("interp testing");
  print_1d(args);
  print(sampler.interpolators[0](args));
  args = {0.67, 0.};
  print_1d(args);
  print(sampler.interpolators[0](args));

  print(sampler.second_discrete_cdf);

  dxarray bounds2 = {{-2, -2}, {2, 2}};
  dxarray axes_step_bounds = {{-1 + 0.1, 1 - 0.2}, {-1 + 0.1, 1 - 0.2}};

  auto clipr = xt::clip(axes_step_bounds, xt::amin(bounds2, {0}),
                        xt::amax(bounds2, {0}));

  print("clipping");
  print(clipr);
  print(xt::amin(bounds2, {0}));

  print_1d(sampler.sample(args, pdf));

  args = {0., 0.};

  dvect sample_results;
  dvect sample;
  size_t samples = 1000;

  print("sampling 2D");
  double start_s = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  for (size_t i = 0; i < samples; ++i) {
    sample = sampler.sample(args, pdf);
    sample_results.push_back(sample[0]);
    sample_results.push_back(sample[1]);
  }
  double end_s = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();
  double time_s = (end_s - start_s) / 1000.;
  print("time taken:");
  print(time_s);
  print("per evaluation");
  print(time_s / ((double)samples));

  const std::vector<size_t> s_shape{samples, 2};
  plot_hist(sample_results, s_shape);
}

void testing_2d_redux() {
  // pdf params
  pdf_data data;
  struct pdf_data *data_ptr = &data;
  data.centre = dvect{0.0, 0.0};
  data.width = 0.2;

  // random walker position
  dvect pos = {0., 0.};

  double (*pdf)(const dvect &, dvect &, void *) = gauss;

  print("Initialising 2D sampler");
  Sampler2D sampler(70, pdf, data_ptr);
  print("Finished Initialising");

  dvect sample_results;
  dvect sample;
  size_t samples = 500000;

  print("sampling 2D");
  double start_s = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  for (size_t i = 0; i < samples; ++i) {
    sample = sampler.sample(pos, pdf);
    sample_results.push_back(sample[0]);
    sample_results.push_back(sample[1]);
  }
  double end_s = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();
  double time_s = (end_s - start_s) / 1000.;
  print("time taken:");
  print(time_s);
  print("per evaluation");
  print(time_s / ((double)samples));

  const std::vector<size_t> s_shape{samples, 2};
  plot_hist(sample_results, s_shape);
}

#endif
