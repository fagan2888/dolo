import time

import numpy as np

from dolo.numeric.optimize.ncpsolve import ncpsolve
from dolo.numeric.optimize.newton import (SerialDifferentiableFunction,
                                          serial_newton)

from dolo.algos.dtcscc.util import (_get_initial_dr,
                                    _get_approximation_space,
                                    _get_interp,
                                    _get_integration)


def time_iteration(model,  bounds=None, verbose=False, initial_dr=None,
                   pert_order=1, with_complementarities=True,
                   interp_type='smolyak', smolyak_order=3, interp_orders=None,
                   maxit=500, tol=1e-8, inner_maxit=10,
                   integration='gauss-hermite', integration_orders=None,
                   T=200, n_s=3, hook=None):
    '''
    Finds a global solution for ``model`` using backward time-iteration.

    This algorithm iterates on the residuals of the arbitrage equations

    Parameters
    ----------
    model : NumericModel
        "fg" or "fga" model to be solved
    bounds : ndarray
        boundaries for approximations. First row contains minimum values.
        Second row contains maximum values.
    verbose : boolean
        if True, display iterations
    initial_dr : decision rule
        initial guess for the decision rule
    pert_order : {1}
        if no initial guess is supplied, the perturbation solution at order
        ``pert_order`` is used as initial guess
    with_complementarities : boolean (True)
        if False, complementarity conditions are ignored
    interp_type : {`smolyak`, `spline`}
        type of interpolation to use for future controls
    smolyak_orders : int
        parameter ``l`` for Smolyak interpolation
    interp_orders : 1d array-like
        list of integers specifying the number of nodes in each dimension if
        ``interp_type="spline" ``

    Returns
    -------
    decision rule :
        approximated solution
    '''

    def vprint(t):
        if verbose:
            print(t)

    parms = model.calibration['parameters']
    sigma = model.covariances

    initial_dr = _get_initial_dr(model, initial_dr, pert_order)

    if interp_type == 'perturbations':
        return initial_dr

    bounds = _get_approximation_space(model, bounds, initial_dr, verbose)
    dr = _get_interp(bounds, interp_orders, interp_type, smolyak_order)
    epsilons, weights = _get_integration(model, integration,
                                         integration_orders)

    vprint('Starting time iteration')

    # TODO: transpose

    grid = dr.grid

    xinit = initial_dr(grid)
    xinit = xinit.real  # just in case...

    f = model.functions['arbitrage']
    g = model.functions['transition']

    # define objective function (residuals of arbitrage equations)
    def fun(x):
        return step_residual(grid, x, dr, f, g, parms, epsilons, weights)

    ##
    t1 = time.time()
    err = 1
    x0 = xinit
    it = 0

    verbit = True if verbose == 'full' else False

    if with_complementarities:
        lbfun = model.functions['controls_lb']
        ubfun = model.functions['controls_ub']
        lb = lbfun(grid, parms)
        ub = ubfun(grid, parms)
    else:
        lb = None
        ub = None

    if verbose:
        headline = '|{0:^4} | {1:10} | {2:8} | {3:8} | {4:3} |'
        headline = headline.format('N', ' Error', 'Gain', 'Time', 'nit')
        stars = '-'*len(headline)
        print(stars)
        print(headline)
        print(stars)

        # format string for within loop
        fmt_str = '|{0:4} | {1:10.3e} | {2:8.3f} | {3:8.3f} | {4:3} |'

    err_0 = 1

    while err > tol and it < maxit:
        # update counters
        t_start = time.time()
        it += 1

        # update interpolation coefficients (NOTE: filters through `fun`)
        dr.set_values(x0)

        # Derivative of objective function
        sdfun = SerialDifferentiableFunction(fun)

        # Apply solver with current decision rule for controls
        if with_complementarities:
            [x, nit] = ncpsolve(sdfun, lb, ub, x0, verbose=verbit,
                                maxit=inner_maxit)
        else:
            [x, nit] = serial_newton(sdfun, x0, verbose=verbit)

        # update error and print if `verbose`
        err = abs(x-x0).max()
        err_SA = err/err_0
        err_0 = err
        t_finish = time.time()
        elapsed = t_finish - t_start
        if verbose:
            print(fmt_str.format(it, err, err_SA, elapsed, nit))

        # Update control vector
        x0[:] = x  # x0 = x0 + (x-x0)

        # call user supplied hook, if any
        if hook:
            hook(dr, it, err)

        # warn and bail if we get inf
        if False in np.isfinite(x0):
            print('iteration {} failed : non finite value')
            return [x0, x]

    if it == maxit:
        import warnings
        warnings.warn(UserWarning("Maximum number of iterations reached"))

    # compute final fime and do final printout if `verbose`
    t2 = time.time()
    if verbose:
        print(stars)
        print('Elapsed: {} seconds.'.format(t2 - t1))
        print(stars)

    return dr


def step_residual(s, x, dr, f, g, parms, epsilons, weights):
    """
    Comptue the residuals of the arbitrage equaitons.

    Recall that the arbitrage equations have the form

        0 = E_t [f(...)]

    This function computes and returns the right hand side.
    """

    # TODO: transpose
    n_draws = epsilons.shape[0]
    [N, n_x] = x.shape
    ss = np.tile(s, (n_draws, 1))
    xx = np.tile(x, (n_draws, 1))
    ee = np.repeat(epsilons, N, axis=0)

    # evaluate transition (g) to update state
    ssnext = g(ss, xx, ee, parms)
    xxnext = dr(ssnext)  # evaluate decision rule (dr) to update controls

    # evaluate arbitrage/Euler equations (f) to compute values
    val = f(ss, xx, ee, ssnext, xxnext, parms)

    # apply quadrature to compute implicit expectation in arbitrage equations
    res = np.zeros((N, n_x))
    for i in range(n_draws):
        res += weights[i] * val[N*i:N*(i+1), :]

    return res


def test_residuals(s, dr, f, g, parms, epsilons, weights):

    n_draws = epsilons.shape[1]

    n_g = s.shape[1]
    x = dr(s)
    n_x = x.shape[0]

    ss = np.tile(s, (1, n_draws))
    xx = np.tile(x, (1, n_draws))
    ee = np.repeat(epsilons, n_g, axis=1)

    ssnext = g(ss, xx, ee, parms)
    xxnext = dr(ssnext)
    val = f(ss, xx, ee, ssnext, xxnext, parms)

    errors = np.zeros((n_x, n_g))
    for i in range(n_draws):
        errors += weights[i] * val[:, n_g*i:n_g*(i+1)]

    squared_errors = np.power(errors, 2)
    std_errors = np.sqrt(np.sum(squared_errors, axis=0)/len(squared_errors))

    return std_errors
