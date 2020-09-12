#include "rejection_sampling.h"
#include <sstream>
#include <string>

auto get_triangle_bounds() {
  // bounds with x coords in the first column and y coords in the second
  // has to be given in clockwise order!
  vect_dvect triangle_vect;
  triangle_vect.push_back(dvect{0, 0});
  triangle_vect.push_back(dvect{0, 1});
  triangle_vect.push_back(dvect{1, 0});
  triangle_vect.push_back(dvect{0, 0});
  return triangle_vect;
}

auto get_weird_bounds() {
  // bounds with x coords in the first column and y coords in the second
  // has to be given in clockwise order!
  vect_dvect weird_bounds_vect;
  weird_bounds_vect.push_back({0.1, 0.3});
  weird_bounds_vect.push_back({0.25, 0.98});
  weird_bounds_vect.push_back({0.9, 0.9});
  weird_bounds_vect.push_back({0.7, 0.4});
  weird_bounds_vect.push_back({0.4, 0.05});
  weird_bounds_vect.push_back({0.1, 0.3});
  return weird_bounds_vect;
}

auto get_circle_bounds() {
  // bounds with x coords in the first column and y coords in the second
  // has to be given in clockwise order!
  vect_dvect circle_bounds_vect;
  circle_bounds_vect.push_back({1.00000000e+00, 0.00000000e+00});
  circle_bounds_vect.push_back({9.87688341e-01, 1.56434465e-01});
  circle_bounds_vect.push_back({9.51056516e-01, 3.09016994e-01});
  circle_bounds_vect.push_back({8.91006524e-01, 4.53990500e-01});
  circle_bounds_vect.push_back({8.09016994e-01, 5.87785252e-01});
  circle_bounds_vect.push_back({7.07106781e-01, 7.07106781e-01});
  circle_bounds_vect.push_back({5.87785252e-01, 8.09016994e-01});
  circle_bounds_vect.push_back({4.53990500e-01, 8.91006524e-01});
  circle_bounds_vect.push_back({3.09016994e-01, 9.51056516e-01});
  circle_bounds_vect.push_back({1.56434465e-01, 9.87688341e-01});
  circle_bounds_vect.push_back({6.12323400e-17, 1.00000000e+00});
  circle_bounds_vect.push_back({-1.56434465e-01, 9.87688341e-01});
  circle_bounds_vect.push_back({-3.09016994e-01, 9.51056516e-01});
  circle_bounds_vect.push_back({-4.53990500e-01, 8.91006524e-01});
  circle_bounds_vect.push_back({-5.87785252e-01, 8.09016994e-01});
  circle_bounds_vect.push_back({-7.07106781e-01, 7.07106781e-01});
  circle_bounds_vect.push_back({-8.09016994e-01, 5.87785252e-01});
  circle_bounds_vect.push_back({-8.91006524e-01, 4.53990500e-01});
  circle_bounds_vect.push_back({-9.51056516e-01, 3.09016994e-01});
  circle_bounds_vect.push_back({-9.87688341e-01, 1.56434465e-01});
  circle_bounds_vect.push_back({-1.00000000e+00, 1.22464680e-16});
  circle_bounds_vect.push_back({-9.87688341e-01, -1.56434465e-01});
  circle_bounds_vect.push_back({-9.51056516e-01, -3.09016994e-01});
  circle_bounds_vect.push_back({-8.91006524e-01, -4.53990500e-01});
  circle_bounds_vect.push_back({-8.09016994e-01, -5.87785252e-01});
  circle_bounds_vect.push_back({-7.07106781e-01, -7.07106781e-01});
  circle_bounds_vect.push_back({-5.87785252e-01, -8.09016994e-01});
  circle_bounds_vect.push_back({-4.53990500e-01, -8.91006524e-01});
  circle_bounds_vect.push_back({-3.09016994e-01, -9.51056516e-01});
  circle_bounds_vect.push_back({-1.56434465e-01, -9.87688341e-01});
  circle_bounds_vect.push_back({-1.83697020e-16, -1.00000000e+00});
  circle_bounds_vect.push_back({1.56434465e-01, -9.87688341e-01});
  circle_bounds_vect.push_back({3.09016994e-01, -9.51056516e-01});
  circle_bounds_vect.push_back({4.53990500e-01, -8.91006524e-01});
  circle_bounds_vect.push_back({5.87785252e-01, -8.09016994e-01});
  circle_bounds_vect.push_back({7.07106781e-01, -7.07106781e-01});
  circle_bounds_vect.push_back({8.09016994e-01, -5.87785252e-01});
  circle_bounds_vect.push_back({8.91006524e-01, -4.53990500e-01});
  circle_bounds_vect.push_back({9.51056516e-01, -3.09016994e-01});
  circle_bounds_vect.push_back({9.87688341e-01, -1.56434465e-01});
  return circle_bounds_vect;
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

int main() {
  set_seed();
  pdf_data data;
  struct pdf_data *data_ptr = &data;

  // Configuration START
  size_t samples = (size_t)1e5;
  const size_t dims = 2;
  vect_dvect bounds = get_circle_bounds(); // only needed for `dims=2`
  std::string bounds_name = "circle";
  // used for the rejection sampler
  size_t blocks = 60;
  // set pdf to use here!!
  double (*pdf)(const dvect &, dvect &, void *) = funky;
  // adjust this to match the pdf above for the final filenames!!
  std::string pdf_string = "funky";
  // pdf params are specified here
  // these params vary for each pdf!!
  data.centre = dvect{0.0, 0.0};
  data.width = 2.;
  // Configuration END

  // this is used to select the proper sampler
  Int2Type<dims> d_int_type;
  // store the positions
  vect_dvect positions;
  positions.reserve(samples + 1);
  // store the steps (will have 1 less entry than position)
  vect_dvect steps;
  steps.reserve(samples);
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

  positions.push_back(coord);

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
    positions.push_back(coord);
    steps.push_back(step);

    // timing information
    if ((i > 2) && (i == (size_t)1e3) ||
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

  // flatten the positions
  dvect position_results;
  position_results.reserve((samples + 1) * dims);
  for (auto &values : positions) {
    for (auto &coord : values) {
      position_results.push_back(coord);
    }
  }
  // flatten the steps
  dvect steps_results;
  steps_results.reserve(samples * dims);
  for (auto &values : steps) {
    for (auto &step_axis : values) {
      steps_results.push_back(step_axis);
    }
  }
  const std::vector<size_t> position_shape{samples + 1, dims};
  const std::vector<size_t> step_shape{samples, dims};

  // plot positions and steps
  // plot_hist(position_results, position_shape);
  // plot_hist(steps_results, step_shape);

  std::stringstream ss;
  ss.precision(3);
  // generate end of filename
  ss << std::scientific;
  ss << "_samples_" << (double)samples << "_dims_" << dims << "_bounds_"
     << bounds_name << "_pdf_" << pdf_string << "_centre_"
     << format_1d(data.centre) << "_width_" << data.width << ".npy";
  std::string filename_suffix = ss.str();
  ss.str("");
  ss.clear();

  // save positions
  std::string pos_filename = "positions" + filename_suffix;
  // if this file already exists, prepend an integer to it
  std::string dir_name = "data/";
  int i = 0;
  while (file_exists(dir_name + std::to_string(i) + pos_filename)) {
    ++i;
  }
  pos_filename = dir_name + std::to_string(i) + pos_filename;
  std::cout << "saving positions to:" << pos_filename << std::endl;
  save_np(position_results, position_shape, pos_filename);

  // save steps
  std::string step_filename = "steps" + filename_suffix;
  // re-use the same integer as before
  step_filename = dir_name + std::to_string(i) + step_filename;
  std::cout << "saving steps to:" << step_filename << std::endl;
  save_np(steps_results, step_shape, step_filename);

  return 0;
}
