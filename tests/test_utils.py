# -*- coding: utf-8 -*-
import numpy as np

from bounded_rand_walkers.utils import match_ref


def test_match_ref():
    x = np.linspace(0, 1, 10)
    y = np.random.RandomState(0).random(10) * 0.5

    ref_x = np.linspace(0, 1, 10)[:8]
    ref_y = (
        np.random.RandomState(0).random(10)[:8]
        + np.random.RandomState(1).random(8) * 0.1
    )

    diff_before = np.mean(np.abs(y[:8] - ref_y))

    scaled = match_ref(x, y, ref_x, ref_y, interpolate=False)
    assert scaled.shape == y.shape

    diff_after_match = np.mean(np.abs(scaled[:8] - ref_y))

    assert diff_after_match / diff_before < 0.05

    scaled_interp = match_ref(x + 0.01, y, ref_x, ref_y, interpolate=True)
    assert scaled_interp.shape == y.shape

    diff_after_interp_match = np.mean(np.abs(scaled_interp[:8] - ref_y))

    assert diff_after_interp_match / diff_before < 0.05
