"""
Utilities for use in the dtcscc algorithms

@author : Spencer Lyon
@date : 2016-01-21 16:04

"""
import numpy as np

from dolo.algos.dtcscc.perturbations import approximate_controls
# from dolo.numeric.timeseries import asymptotic_variance
# from dolo.numeric.discretization import quantization_nodes
from dolo.numeric.discretization import gauss_hermite_nodes
from dolo.numeric.interpolation.smolyak import SmolyakGrid
from dolo.numeric.interpolation.splines import MultivariateSplines


# from dolo.numeric.interpolation.multilinear import MultilinearInterpolator

# -------------------------------- #
# Global solution method utilities #
# -------------------------------- #


def _get_initial_dr(model, initial_dr, pert_order):
    if initial_dr is None:
        if pert_order == 1:
            return approximate_controls(model)

        if pert_order > 1:
            raise Exception("Perturbation order > 1 not supported (yet).")

    return initial_dr


def _get_approximation_space(model, bounds, initial_dr, verbose):
    if bounds is not None:
        pass

    elif model.options and 'approximation_space' in model.options:

        if verbose:
            print('Using bounds specified by model')

        approx = model.options['approximation_space']
        a = approx['a']
        b = approx['b']

        bounds = np.row_stack([a, b])
        bounds = np.array(bounds, dtype=float)

    else:
        raise ValueError("TODO: asymptotic_variance missing??")
        if verbose:
            print('Using asymptotic bounds given by first order solution.')

        # this will work only if initial_dr is a Taylor expansion
        Q = asymptotic_variance(initial_dr.A.real,
                                initial_dr.B.real,
                                initial_dr.sigma,
                                T=T)

        devs = np.sqrt(np.diag(Q))
        bounds = np.row_stack([
            initial_dr.S_bar - devs * n_s,
            initial_dr.S_bar + devs * n_s,
        ])

    return bounds


def _get_interp(bounds, interp_orders, interp_type, smolyak_order):
    if interp_orders is None:
        interp_orders = [5] * bounds.shape[1]

    if interp_type == 'smolyak':
        return SmolyakGrid(bounds[0, :], bounds[1, :], smolyak_order)
    elif interp_type == 'spline':
        return MultivariateSplines(bounds[0, :], bounds[1, :], interp_orders)
    else:
        msg = "Unknown `interp_type` {}. Possible values are smolyak or spline"
        raise ValueError(msg.format(interp_type))


def _get_integration(model, integration, integration_orders):
    if integration == 'optimal_quantization':
        raise ValueError("TODO: optimal_quantization disabled??")
        # [epsilons, weights] = quantization_nodes(N_e, sigma)
    elif integration == 'gauss-hermite':
        sigma = model.covariances
        if not integration_orders:
            integration_orders = [3] * sigma.shape[0]
        [epsilons, weights] = gauss_hermite_nodes(integration_orders, sigma)

    return epsilons, weights
