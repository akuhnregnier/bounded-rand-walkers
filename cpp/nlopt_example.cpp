#include <math.h>
#include <nlopt.h>
#include <iostream>
#include <typeinfo>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <ctime>
#include <map>
#include "cnpy.h"
#include <cmath>
#include "boost/multi_array.hpp"
#include "linterp.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"


template <class print_type>
void print(print_type to_print){
    std::cout << to_print << std::endl;
}


double myfunc(unsigned n, const double *x, double *grad, void *my_func_data)
{
    return pow(x[0], 2.) + pow(x[1], 2.);
}


double myconstraint(unsigned n, const double *x, double *grad, void *data)
{
    return std::abs(x[0] - 0.3);
} 

int main(){
    double lb[2] = { -0.5, 0.1 }; /* lower bounds */
    double ub[2] = { 0.2, 0.2 }; /* upper bounds */
    nlopt_opt opt;
    opt = nlopt_create(NLOPT_LN_COBYLA, 2); /* algorithm and dimensionality */
    nlopt_set_lower_bounds(opt, lb);
    nlopt_set_upper_bounds(opt, ub);
    nlopt_set_min_objective(opt, myfunc, NULL);
    // nlopt_add_inequality_constraint(opt, myconstraint, NULL, 1e-8);
    // nlopt_add_inequality_constraint(opt, myconstraint, NULL, 1e-8);
    nlopt_set_xtol_rel(opt, 1e-4);
    double x[2] = { 0., 0.12 };  /* `*`some` `initial` `guess`*` */
    double minf; /* `*`the` `minimum` `objective` `value,` `upon` `return`*` */
    int result = nlopt_optimize(opt, x, &minf);
    if (result < 0) {
        printf("nlopt failed!\n");
        print(result);
    }
    else {
        printf("found minimum at f(%g,%g) = %0.10g\n", x[0], x[1], minf);
    }
    nlopt_destroy(opt);

    return 0;
}
