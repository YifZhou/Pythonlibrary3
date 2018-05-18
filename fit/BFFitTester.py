#! /usr/bin/env python
"""Brute Force Fitting result tester calculate a series value around
the best fit value to test whether globally optimized solution was
found

Test function should have the form of
func(params0, *args, **kwargs)
"""
import numpy as np


def BFFitTester(func, params0, i_params, pSample, args=None, kwargs=None):
    """Calculate the optimizing function around the best-fit values

    :param func: function being tested
    :param params0: best-fit values found by optimizers (scipy minimize/mcmc/lmfit, etc.)
    :param i_params: to test the ith parameter
    :param pSample: sample point of the target parameter
    :param args: extra arguments required by func
    :param kwargs: extra keyword arguments required by func
    :returns: func_values, the return value of the optimization function at the testing point
    :rtype: numpy array

    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    params_test = params0.copy()
    func_values = np.zeros_like(pSample)
    for i, p_i in enumerate(pSample):
        params_test[i_params] = p_i
        func_values[i] = func(params_test, *args, **kwargs)
    return func_values, pSample
