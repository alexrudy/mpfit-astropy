# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
mpfit-astropy provides an astropy.modelling.Fitter class which uses the powerful machenery in MPFit.

Perform Levenberg-Marquardt least-squares minimization, based on MINPACK-1.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from example_mod import *
