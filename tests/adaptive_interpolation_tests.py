from nose.tools import *
import adaptive_interpolation.adaptive_interpolation as adapt_i

def mono7(x):
    return sum([x**i for i in range(8)])

# test initializing the interpolant
def test_legendre():
    #interp = Interpolant(mono7, 1e-10)
    pass


def test_chebyshev():
    pass


def test_eval_coeff():
    pass


def test_get_nodes():
    # my_interpolant = Interpolant()
    pass

