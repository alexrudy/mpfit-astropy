
from numpy import linalg
from numpy.testing.utils import assert_allclose, assert_almost_equal

from astropy.tests.helper import pytest
from astropy.modeling.fitting import *
from astropy.utils import NumpyRNGContext
from astropy.modeling import models
from astropy.modeling.tests.utils import ignore_non_integer_warning

from ..fitter import *

try:
    from scipy import optimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

fitters = [SimplexLSQFitter, SLSQPLSQFitter, LevMarLSQFitter]
_RANDOM_SEED = 0x1337

@pytest.mark.skipif('not HAS_SCIPY')
class TestMPFitter(object):
    """Test the MPFit tools"""
    
    def test_estimated_vs_analytic_deriv(self):
        """
        Runs `MPFitter` with estimated and analytic derivatives of a
        `Gaussian1D`.
        """

        fitter = MPFitter()
        model = fitter(self.gauss, self.xdata, self.ydata, autoderivative=False)
        g1e = models.Gaussian1D(100, 5.0, stddev=1)
        efitter = MPFitter()
        emodel = efitter(g1e, self.xdata, self.ydata, autoderivative=True)
        assert_allclose(model.parameters, emodel.parameters, rtol=10 ** (-3))

    def test_with_optimize(self):
        """
        Tests results from `LevMarLSQFitter` against `scipy.optimize.leastsq`.
        """

        fitter = MPFitter()
        model = fitter(self.gauss, self.xdata, self.ydata,
                       autoderivative=True)

        def func(p, x):
            return p[0] * np.exp(-0.5 / p[2] ** 2 * (x - p[1]) ** 2)

        def errfunc(p, x, y):
            return func(p, x) - y

        result = optimize.leastsq(errfunc, self.initial_values,
                                  args=(self.xdata, self.ydata))
        assert_allclose(model.parameters, result[0], rtol=10 ** (-3))

    def test_with_weights(self):
        """
        Tests results from `LevMarLSQFitter` with weights.
        """
        # part 1: weights are equal to 1
        fitter = MPFitter()
        model = fitter(self.gauss, self.xdata, self.ydata,
                       autoderivative=True)
        withw = fitter(self.gauss, self.xdata, self.ydata,
                       autoderivative=True, weights=np.ones_like(self.xdata))

        assert_allclose(model.parameters, withw.parameters, rtol=10 ** (-4))

        # part 2: weights are 0 or 1 (effectively, they are a mask)
        weights = np.zeros_like(self.xdata)
        weights[::2] = 1.
        mask = weights >= 1.

        model = fitter(self.gauss, self.xdata[mask], self.ydata[mask],
                       autoderivative=True)
        withw = fitter(self.gauss, self.xdata, self.ydata,
                       autoderivative=True, weights=weights)

        assert_allclose(model.parameters, withw.parameters, rtol=10 ** (-4))


    @pytest.mark.parametrize('fitter_class', fitters)
    def test_fitter_against_MPFit(self, fitter_class):
        """Tests results from non-linear fitters against `LevMarLSQFitter`."""

        mpfit = MPFitter()
        fitter = fitter_class()
        with ignore_non_integer_warning():
            new_model = fitter(self.gauss, self.xdata, self.ydata)
        model = mpfit(self.gauss, self.xdata, self.ydata)
        assert_allclose(model.parameters, new_model.parameters,
                        rtol=10 ** (-4))

    def test_LSQ_SLSQP_with_constraints(self):
        """
        Runs `LevMarLSQFitter` and `SLSQPLSQFitter` on a model with
        constraints.
        """

        g1 = models.Gaussian1D(100, 5, stddev=1)
        g1.mean.fixed = True
        mpfit = MPFitter()
        fitter = LevMarLSQFitter()
        fslsqp = SLSQPLSQFitter()
        with ignore_non_integer_warning():
            slsqp_model = fslsqp(g1, self.xdata, self.ydata)
        model = fitter(g1, self.xdata, self.ydata)
        mpmodel = mpfit(g1, self.xdata, self.ydata)
        assert_allclose(mpmodel.parameters, slsqp_model.parameters,
                        rtol=10 ** (-4))
        assert_allclose(mpmodel.parameters, model.parameters,
                        rtol=10 ** (-4))


    def test_param_cov(self):
        """
        Tests that the 'param_cov' fit_info entry gets the right answer for
        *linear* least squares, where the answer is exact
        """

        a = 2
        b = 100

        with NumpyRNGContext(_RANDOM_SEED):
            x = np.linspace(0, 1, 100)
            # y scatter is amplitude ~1 to make sure covarience is
            # non-negligible
            y = x*a + b + np.random.randn(len(x))

        #first compute the ordinary least squares covariance matrix
        X = np.matrix(np.vstack([x, np.ones(len(x))]).T)
        beta = np.linalg.inv(X.T * X) * X.T * np.matrix(y).T
        s2 = np.sum((y - (X * beta).A.ravel())**2) / (len(y) - len(beta))
        olscov = np.linalg.inv(X.T * X) * s2

        #now do the non-linear least squares fit
        mod = models.Linear1D(a, b)
        fitter = MPFitter()

        fmod = fitter(mod, x, y)

        assert_allclose(fmod.parameters, beta.A.ravel())
        assert_allclose(olscov, fitter.fit_info['covar'])
    