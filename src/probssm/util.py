import functools
import json
import logging
import os
import warnings

import numpy as np
import scipy.special


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def unions1d(*arrays):
    return functools.reduce(np.union1d, (a for a in arrays if a is not None))


def sloped_sigmoid(x, slope, x_offset=0.0, y_offset=0.0):
    return scipy.special.expit(slope * (x - x_offset)) + y_offset


def d_sloped_sigmoid(x, slope, x_offset=0.0, y_offset=0.0):
    s = sloped_sigmoid(x, slope, x_offset, y_offset)
    return slope * (s * (1.0 - s))


def split_train_test(all_idcs, list_lambda_predicates_test):
    test_mask = np.logical_or.reduce([p(all_idcs) for p in list_lambda_predicates_test])

    train_mask = np.logical_not(test_mask)

    train_idcs = all_idcs[train_mask]
    test_idcs = all_idcs[test_mask]

    return train_idcs, test_idcs


def test_jacobian(
    dim,
    jacobian,
    function,
    h=1e-6,
):
    # Compute numeric approximation to Jacobian using central differences
    numeric_jacobian = np.zeros_like(jacobian)

    for i in range(dim):
        eps = np.zeros(dim)
        eps[i] = h

        numeric_jacobian[..., i] = (
            function(eps.squeeze()) - function(-eps.squeeze())
        ) / (2.0 * h)

    # Compare
    if not np.allclose(jacobian, numeric_jacobian):
        warnings.warn("Jacobian not OK")
        logging.debug(
            f"Jacobian not OK. Errors: \n{np.abs(jacobian - numeric_jacobian)}, max = {np.max(np.abs(jacobian - numeric_jacobian))}"
        )
    else:
        logging.debug("Jacobian OK")


def args_to_json(path, args, kwargs):
    with open(path, mode="x") as f:

        def _default(o):
            if isinstance(o, (int, float, str)):
                return o

            if isinstance(o, os.PathLike):
                return str(o)

            if isinstance(o, (list, tuple)):
                if all(
                    entry is None or isinstance(entry, (int, float, str)) for entry in o
                ):
                    return o

            return repr(o)

        json.dump(
            dict(**args.__dict__.copy(), **kwargs),
            f,
            indent=4,
            default=_default,
        )
