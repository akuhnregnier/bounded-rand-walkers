#include "cnpy.h"
#include "common.h"
#include "pdfs.h"
#include <iostream>
#include <string>
#include <vector>

int main() {
  const int l = 10000;
  const std::vector<long unsigned int> shape = {l};
  dvect y{};
  dvect pos;
  dvect dummy{};
  dvect x = linspace(0.0, 3.0, l);

  pdf_data data;
  struct pdf_data *data_ptr = &data;
  data.centre = dvect{0.0, 0.0};
  data.width = 2.;

  for (size_t i(0); i < l; i++) {
    pos = {
        x[i],
    };
    y.push_back(funky(pos, dummy, data_ptr));
  }
  plot_square(y, shape);
  return 0;
}
