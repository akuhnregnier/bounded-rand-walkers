# -*- coding: utf-8 -*-
import numpy as np
import pytest
from numpy.testing import assert_allclose

import bounded_rand_walkers.test_sampling as brw_test_sampling
from bounded_rand_walkers.functions import Freehand, Freehand2, Gaussian


@pytest.mark.parametrize(
    "test_func", (brw_test_sampling.test_1d, brw_test_sampling.test_2d)
)
@pytest.mark.parametrize(
    "cpp_pdf_name,pdf_params,python_pdf_func",
    [
        ("gauss", dict(width=0.2), Gaussian),
        ("gauss", dict(width=0.4), Gaussian),
        ("gauss", dict(width=0.5), Gaussian),
        ("freehand", dict(width=2.0), Freehand),
        ("freehand2", dict(width=2.0), Freehand2),
    ],
)
def test_sampling(test_func, cpp_pdf_name, pdf_params, python_pdf_func):
    python_pdf = python_pdf_func(**pdf_params)

    mses = {"cpp": [], "py": []}

    for seed, N in enumerate((int(2e3), int(7e3))):
        sample_data = test_func(
            cpp_pdf_name=cpp_pdf_name,
            python_pdf=python_pdf,
            bins=10,
            N=N,
            seed=seed,
            return_data=True,
            **pdf_params,
        )
        analytical = sample_data["analytical"]

        for sampled in (sample_data[key] for key in ("sampled_cpp", "sampled_py")):
            assert_allclose(sampled, analytical, rtol=1e-1, atol=3e-1)

        for cat in ("cpp", "py"):
            mses[cat].append(
                np.sqrt(np.mean(np.square(analytical - sample_data[f"sampled_{cat}"])))
            )

    for mse_vals in mses.values():
        assert np.all(np.diff(mse_vals) < 0)


def test_cpp_pdf_name():
    for test_func in (brw_test_sampling.test_1d, brw_test_sampling.test_2d):
        with pytest.raises(ValueError):
            test_func(cpp_pdf_name="undefined")
