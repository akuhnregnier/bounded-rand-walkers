#include <boost/shared_ptr.hpp>

#include <roboptim/core/twice-differentiable-function.hh>
#include <roboptim/core/io.hh>
#include <roboptim/core/solver.hh>
#include <roboptim/core/solver-factory.hh>

using namespace roboptim;

struct F : public TwiceDifferentiableFunction
{
  F () : TwiceDifferentiableFunction (4, 1, "x₀ * x₃ * (x₀ + x₁ + x₂) + x₂")
  {
  }

  void
  impl_compute (result_ref result, const_argument_ref x) const
  {
    result[0] = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
  }

  void
  impl_gradient (gradient_ref grad, const_argument_ref x, size_type) const
  {
    grad << x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]),
            x[0] * x[3],
            x[0] * x[3] + 1,
            x[0] * (x[0] + x[1] + x[2]);
  }

  void
  impl_hessian (hessian_ref h, const_argument_ref x, size_type) const
  {
    h << 2 * x[3],               x[3], x[3], 2 * x[0] + x[1] + x[2],
         x[3],                   0.,   0.,   x[0],
         x[3],                   0.,   0.,   x[1],
         2 * x[0] + x[1] + x[2], x[0], x[0], 0.;
  }
};
