# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import abc
import inspect
import operator
import warnings

from functools import reduce

import numpy as np

from astropy.utils.exceptions import AstropyUserWarning

DEFAULT_ACC = 1.0e-10
DEFAULT_MAXITER = 200


class MPFitter(object):
    """Use the MPFit implementation of Levenberg-Marquardt least-squares minimization."""
    
    def __init__(self, quiet=False, debug=False, nprint=1):
        self.fit_info = {
            'status' : None,
            'fnorm' : None,
            'covar' : None,
            'message' : None,
            'nfev': None,
            'niter': None,
            'perror': None
        }
        super(MPFitter, self).__init__()
        self.quiet = quiet
        self.debug = debug
        self.nprint = nprint
    
    
    def make_parinfo(self, model):
        """Construct a parinfo dictionary from a model."""
        parinfo = []
        for idx, name in list(enumerate(model.param_names)):
            pardict = {
                'value' : model.parameters[idx],
                'fixed' : model.fixed[name],
                'parname' : name,
            }
            bounds = model.bounds[name]
            if bounds != (None, None):
                pardict['limited'] = (bounds[0] is not None, bounds[1] is not None)
                pardict['limits'] = np.asanyarray(bounds, dtype=np.float)
            parinfo.append(pardict)
        return parinfo
    
    # Taken directly from astropy.modeling.fitting.LevMarLSQFitter
    # Adjusted to handle MPFit's status return value.
    def objective_function(self, fps, fjac=None, **kwargs):
        """
        Function to minimize.

        Parameters
        ----------
        fps : list
            parameters returned by the fitter
        fjac : None or list
            parameters for which to compute the jacobian
        args : list
            [model, [weights], [input coordinates]]
        """
        status = 0
        model = kwargs['model']
        weights = args['weights']
        _fitter_to_model_params(model, fps)
        meas = args['err']
        r = [status]
        if weights is None:
            residuals = np.ravel(model(*args[2 : -1]) - meas)
            r.append(residuals)
        else:
            residuals = np.ravel(weights * (model(*args[2 : -1]) - meas))
            r.append(residuals)
        if fjac is not None:
            fderiv = np.array(self._wrap_deriv(fps, model, weights, *args[2 : -1]))
            r.append(fderiv)
        return r
        
    def __call__(self, model, x, y, z=None, weights=None,
                 maxiter=DEFAULT_MAXITER, ftol=DEFAULT_ACC, xtol=DEFAULT_ACC,
                 gtol=DEFAULT_ACC, factor=100.0, iterfunct='default',
                 iterkw={}, nocovar=False, rescale=0, autoderivative=True, diag=None,
                 epsfcn=None):
        """
        Fit data to this model.

        Parameters
        ----------
        model : `~astropy.modeling.FittableModel`
            model to fit to x, y, z
        x : array
           input coordinates
        y : array
           input coordinates
        z : array (optional)
           input coordinates
        weights : array (optional)
           weights
        maxiter : int
            maximum number of iterations
        acc : float
            Relative error desired in the approximate solution
        epsilon : float
            A suitable step length for the forward-difference
            approximation of the Jacobian (if model.fjac=None). If
            epsfcn is less than the machine precision, it is
            assumed that the relative errors in the functions are
            of the order of the machine precision.
        estimate_jacobian : bool
            If False (default) and if the model has a fit_deriv method,
            it will be used. Otherwise the Jacobian will be estimated.
            If True, the Jacobian will be estimated in any case.

        Returns
        -------
        model_copy : `~astropy.modeling.FittableModel`
            a copy of the input model with parameters set by the fitter
        """

        from .extern.mpfit import MPFit

        model_copy = _validate_model(model, self.supported_constraints)
        farg = (model_copy, weights, )
        keys = ('model', 'weights', )
        inputs =  _convert_input(x, y, z)
        farg = farg + inputs
        if len(inputs) == 3:
            keys = keys + ('x', 'y', 'err')
        elif len(inputs) == 2:
            keys = keys + ('x', 'err')
        functkw = dict(zip(keys, farg))
        
        if model_copy.fit_deriv is None and not autoderivative:
            warnings.warn("Can't compute automatic derivatives, "
                          "no model.fit_deriv is defined, using autoderivative=True"
                          " instead",
                          AstropyUserWarning)
            autoderivative = True
        
        init_values = model.parameters
        
        parinfo = make_parinfo(model)
        
        
        fitresult = MPFit(
            self.objective_function, init_values, functkw=functkw, maxiter=maxiter, 
            epsfcn=epsfcn, xtol=acc, gtol=acc, ftol=acc, factor=factor, parinfo=parinfo,
            nprint=self.nprint, iterfunct=iterfunct, iterkw={}, nocovar=int(nocovar),
            rescale=int(rescale), autoderivative=int(autoderivative), diag=diag,
            quiet=int(self.quiet))
        
        _fitter_to_model_params(model_copy, fitresult.params)
        
        self.fit_info['status'] = fitresult.status
        self.fit_info['fnorm'] = fitresult.fnorm
        self.fit_info['covar'] = fitresult.covar
        self.fit_info['nfev'] = fitresult.nfev
        self.fit_info['niter'] = fitresult.niter
        self.fit_info['perror'] = fitresult.perror
        
        
        if fitresult.status not in [1, 2, 3, 4]:
            warnings.warn("The fit may be unsuccessful; check "
                          "fit_info['message'] for more information.",
                          AstropyUserWarning)
            self.fit_info['message'] = fitresult.errmsg
        elif fitresult.status == 1:
            self.fit_info['message'] = "Both actual and predicted relative reductions in the sum of squares are at most ftol."
        elif fitresult.status == 2:
            self.fit_info['message'] = "Relative error between two consecutive iterates is at most xtol."
        elif fitresult.status == 3:
            self.fit_info['message'] = "Conditions for status = 1 and status = 2 both hold."
        elif fitresult.status == 4:
            self.fit_info['message'] = "The cosine of the angle between fvec and any column of the jacobian is at most gtol in absolute value."

        # now try to compute the true covariance matrix
        if (len(y) > len(init_values)) and cov_x is not None:
            cov = self.fit_info['covar']
            pcor = cov * 0.0
            for i in range(n):
               for j in range(n):
                  pcor[i,j] = cov[i,j]/np.sqrt(cov[i,i]*cov[j,j])
            self.fit_info['param_cor'] = pcor
        else:
            self.fit_info['param_cor'] = None
        
        return model_copy

    @staticmethod
    def _wrap_deriv(params, model, weights, x, y, z=None):
        """
        Wraps the method calculating the Jacobian of the function to account
        for model constraints.

        `scipy.optimize.leastsq` expects the function derivative to have the
        above signature (parlist, (argtuple)). In order to accomodate model
        constraints, instead of using p directly, we set the parameter list in
        this function.
        """
        if any(model.fixed.values()) or any(model.tied.values()):

            if z is None:
                full_deriv = np.array(model.fit_deriv(x, *model.parameters))
            else:
                full_deriv = np.array(model.fit_deriv(x, y, *model.parameters))

            pars = [getattr(model, name) for name in model.param_names]
            fixed = [par.fixed for par in pars]
            tied = [par.tied for par in pars]
            tied = list(np.where([par.tied is not False for par in pars],
                                 True, tied))
            fix_and_tie = np.logical_or(fixed, tied)
            ind = np.logical_not(fix_and_tie)

            if not model.col_fit_deriv:
                full_deriv = np.asarray(full_deriv).T
                residues = np.asarray(full_deriv[np.nonzero(ind)])
            else:
                residues = full_deriv[np.nonzero(ind)]

            return [np.ravel(_) for _ in residues]
        else:
            if z is None:
                return model.fit_deriv(x, *params)
            else:
                return [np.ravel(_) for _ in model.fit_deriv(x, y, *params)]
    