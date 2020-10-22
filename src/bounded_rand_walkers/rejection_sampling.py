#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adaptive rejection sampling from a given distribution."""
import logging

import numpy as np
from scipy import optimize
from scipy.interpolate import RegularGridInterpolator

from .utils import get_centres


class Sampler(object):
    def __init__(self, pdf, dimensions, blocks=100, bounds=None, seed=-1):
        """Initialise sampler.

        Parameters
        ----------
        pdf : callable
            Pdf to sample from.
        dimensions : {1, 2}
            Used to define array of bounds, eg. np.array([[-1], [1]]) in 1D (from 0
            to 1) or np.array([[-2, -2], [2, 2]]) for a 2D square boundary with
            corners at (0, 0) and (1, 1).
        blocks : int
            Number of blocks to use for the adaptive sampling algorithm.
        bounds : array
            Can be used to override the automatic bounds as mentioned for
            `dimensions`.
        seed : int
            Random number generator seed. If -1, no seed will be used and results will
            vary from run to run. Pass `seed` > -1 for repeatable results.

        """
        self.logger = logging.getLogger(__name__)
        self.logger.debug("starting init")
        self.rng = np.random.default_rng(
            seed if seed is not None and seed > -1 else None
        )
        self.pdf = pdf
        if bounds is None:
            if dimensions == 1:
                self.bounds = np.array([-1, 1]).reshape(2, 1)
            elif dimensions == 2:
                self.bounds = np.array([[-2, -2], [2, 2]])
        else:
            self.bounds = bounds
        self.dims = dimensions

        # Sample the input pdf at `blocks` positions.
        self.pdf_values = np.zeros([blocks] * self.dims, dtype=np.float64)
        self.centres_list = []  # the block (bin) centres in each dimension
        self.edges_list = []  # the block (bin) edges in each dimension
        for lower, upper in self.bounds.T:
            # lower and upper bounds for each dimension
            sample_edges = np.linspace(lower, upper, blocks + 1)
            self.edges_list.append(sample_edges)
            sample_centres = get_centres(sample_edges)
            self.centres_list.append(sample_centres)
        # Now evaluate the pdf at each of the positions given in the centres_lists,
        # filling in the resulting values into self.pdf_values.
        for indices in np.ndindex(*self.pdf_values.shape):
            values = [
                centres[index] for centres, index in zip(self.centres_list, indices)
            ]
            self.pdf_values[indices] = self.pdf(np.array(values))

        # Within each grid domain, find the maximum pdf value, which will form the
        # basis for getting the 'g' function below.
        # Get an envelope of the pdf, `g` using `self.blocks` uniform pdfs that across
        # their specific intervals always have a value that is at least that of the
        # pdf in the same interval.
        self.logger.debug("starting fn max finding")
        self.max_box_values = np.zeros_like(self.pdf_values)

        def inv_pdf(x):
            return -self.pdf(np.array(x))

        for indices in np.ndindex(*self.pdf_values.shape):
            min_edges = [edges[index] for edges, index in zip(self.edges_list, indices)]
            max_edges = [
                edges[index + 1] for edges, index in zip(self.edges_list, indices)
            ]
            centres = [
                centres[index] for centres, index in zip(self.centres_list, indices)
            ]

            edges_array = np.array((min_edges, max_edges))
            max_args = []
            max_values = []
            target_value = self.pdf_values[indices]
            for x0_indices in np.ndindex(edges_array.squeeze().shape):
                x0 = [edges_array[j, k] for k, j in enumerate(x0_indices)]
                # Now perform the function minimization from `min_edges` to
                # `max_edges`, in order to find the maximum in that range. This is
                # achieved by minimizing -1 * pdf.
                self.logger.debug("calling at:{:}".format(x0))
                x_max = optimize.fmin_tnc(
                    func=inv_pdf,
                    x0=x0,
                    bounds=[(l, u) for l, u in zip(min_edges, max_edges)],
                    approx_grad=True,
                    disp=0,
                )[0]
                max_args.append(x_max)
                max_values.append(self.pdf(np.array(x_max)))
                if max_values[-1] > target_value:
                    break
            max_value = np.max(max_values)
            if np.isclose(max_value, 0):
                # The minimisation has been completed successfully.
                self.logger.debug("calling centre:{:}".format(centres))
                x_max = optimize.fmin_tnc(
                    func=inv_pdf,
                    x0=centres,
                    bounds=[(l, u) for l, u in zip(min_edges, max_edges)],
                    approx_grad=True,
                    disp=0,
                )[0]
            self.logger.debug(
                "max value:{:} at {:}".format(self.pdf(np.array(x_max)), x_max)
            )
            self.max_box_values[indices] = self.pdf(np.array(x_max))

        self.max_box_values += 1e-7
        # Prevent minuscule differences from getting buried in the floating point
        # limit.

        diffs = self.max_box_values - self.pdf_values
        invalid_indices = np.where(diffs < 0)
        self.max_box_values[invalid_indices] = np.max(self.max_box_values)
        diffs2 = self.max_box_values - self.pdf_values
        assert np.min(diffs2) >= 0, "g(x) needs to be > f(x)"
        # Trim the boundaries by reducing the bounds such that only non-zero parts of
        # the pdf within the sampling region. This is important for a very narrow
        # tophat distribution, for example.
        non_zero_mask = self.max_box_values > 1e-6
        non_zero_indices = np.where(non_zero_mask)

        bounds = np.zeros((2, dimensions), dtype=np.float64)
        for i, axes_indices in enumerate(non_zero_indices):
            # `axes_indices` contains the indices for one axis, which
            # contribute to the bounds array in one column.
            min_i, max_i = np.min(axes_indices), np.max(axes_indices)
            bounds[0, i] = self.edges_list[i][min_i]
            bounds[1, i] = self.edges_list[i][max_i + 1]
        if not np.all(non_zero_mask) and not np.all(self.bounds == bounds):
            # If there are some zero elements left, AND if the bounds have changed
            # compared to the input bounds - last condition is necessary to avoid an
            # endless loop in the 2D case.
            self.logger.debug(
                "calling init again with reduced bounds:\n{:}".format(bounds)
            )
            self.__init__(pdf, dimensions, blocks, bounds)

        self.logger.debug("starting lin_interp")
        self.lin_interp_cdf()

    def lin_interp_cdf(self):
        """Get linear interpolation for every block in terms of the inverted CDF, ie.
        an interpolation of probability vs position. This should be done separately
        for each dimension.

        These interpolators will be used in order to get from a randomly sampled
        probability to the corresponding position, which is then distributed according
        the to discrete pdf given by max_box_values. A comparison with the actual pdf
        at that point along with another randomly sampled number then completes to
        sampling process.

        What was described above produces samples across the entire region described
        by the given `bounds`. This is akin to sampling step sizes for a square
        domain if the walker is at the centre of the box. In order to further restrict
        the possible outputs of the sampling, it would be required to restrict the
        possible values returned by the linear interpolators. This can be done by
        defining inverse interpolators, that, given the bounds of the step sizes, give
        the bounds of the probabilities, which can then be used to restrict the
        probabilities with which the positions are sampled.

        """
        self.interpolators = []
        self.inverse_interpolators = []
        nr_axes = self.dims
        # Discrete cdf along each axis.
        first_discrete_cdf = np.cumsum(
            np.sum(
                self.max_box_values, axis=(tuple([i for i in range(nr_axes) if i != 0]))
            )
        )
        # Now we want the inverse of this to get the position along the first axis.
        # Also rescale this such that the maximum value is 1.
        first_discrete_cdf /= np.max(first_discrete_cdf)  # rescale to [0, 1]
        first_edges = self.edges_list[0]
        # Now we know the cdf and coordinate of each bin centre we need to linearly
        # interpolate such that the probabilities are on the x-axis, and the positions
        # on the y-axis.

        first_probs = np.hstack(
            (np.array([0]).reshape(1, 1), first_discrete_cdf.reshape(1, -1))
        )
        self.first_interpolator = RegularGridInterpolator(
            # Add the first 0 explicitly - this corresponds to the lowest coordinate
            # possible, ie. the first edge.
            first_probs.reshape(1, -1),
            first_edges.reshape(-1),
        )
        self.first_inv_interpolator = RegularGridInterpolator(
            first_edges.reshape(1, -1), first_probs.reshape(-1)
        )
        # The first interpolator in this list will be used in order to find the second
        # interpolator to use and so on, one for each dimension.
        self.interpolators.append(self.first_interpolator)
        self.inverse_interpolators.append(self.first_inv_interpolator)
        if nr_axes == 2:
            second_discrete_cdf = np.cumsum(
                np.sum(
                    self.max_box_values,
                    axis=(tuple([i for i in range(nr_axes) if i not in (0, 1)])),
                ),
                axis=1
                # Such that the cumulative sum increases in the y-direction and one
                # cumulative sum is carried out per entry in the x-axis.
            )
            filled_discrete_cdf = np.zeros(
                (second_discrete_cdf.shape[0], second_discrete_cdf.shape[1] + 1),
                dtype=np.float64,
            )
            filled_discrete_cdf[:, 1:] = second_discrete_cdf
            edges = self.edges_list[1]
            self.second_interpolators = []
            self.second_inv_interpolators = []
            for discrete_cdf_series in filled_discrete_cdf:
                discrete_cdf_series /= np.max(discrete_cdf_series)
                self.second_interpolators.append(
                    RegularGridInterpolator(
                        discrete_cdf_series.reshape(1, -1), edges.reshape(-1)
                    )
                )
                self.second_inv_interpolators.append(
                    RegularGridInterpolator(
                        edges.reshape(1, -1), discrete_cdf_series.reshape(-1)
                    )
                )
            self.interpolators.append(self.second_interpolators)
            self.inverse_interpolators.append(self.second_inv_interpolators)
        if nr_axes > 2:
            raise NotImplementedError("Higher Dimensions not Implemented")

    def sample(self, position):
        """Generate a sample from the pdf given a position.

        Parameters
        ----------
        position : array
            Position with respect to which to sample from the pdf.

        Returns
        -------
        sampled : array
            Sample.

        """
        output = []
        centre_indices = []
        if self.dims == 1:
            axes_step_bounds = np.array((-position, 1 - position)).reshape(1, 2)
        elif self.dims == 2:
            position = position.reshape(2, 1)
            axes_step_bounds = np.array([-1, 1]).reshape(1, 2) - position
        axes_step_bounds = np.clip(
            axes_step_bounds, np.min(self.bounds, axis=0), np.max(self.bounds, axis=0)
        )

        for (i, (interpolators, edges, inv_interpolators, step_bounds)) in enumerate(
            zip(
                self.interpolators,
                self.edges_list,
                self.inverse_interpolators,
                axes_step_bounds,
            )
        ):
            if i == 0:
                # Only a single interpolator.
                interpolator = interpolators
                inv_interpolator = inv_interpolators
            else:
                interpolator = interpolators[centre_indices[-1]]
                inv_interpolator = inv_interpolators[centre_indices[-1]]

            # Use the inv_interpolator in order to get the probability bounds which
            # will only return valid step sizes using the interpolator.
            min_prob, max_prob = inv_interpolator(step_bounds)

            prob = self.rng.uniform(min_prob, max_prob)

            coord = interpolator([prob])
            output.append(coord)

            interpolator_index = np.where(coord >= edges)[0][-1]
            centre_indices.append(interpolator_index)

        # Check that the probabilities in `probs` are indeed lower than those returned
        # by the original pdf.
        pdf_val = self.pdf(np.array(output))
        max_box_val = self.max_box_values[tuple(centre_indices)]
        ratio = pdf_val / max_box_val
        prob = self.rng.uniform(0, 1)

        # This comparison causes the output to be an approximation to the true pdf, as
        # opposed to simply the max_box_values representation of the pdf.
        if prob < ratio:
            return np.array(output)
        else:
            self.logger.debug("{:} more than {:}, calling again".format(prob, ratio))
            return self.sample(position=position)
