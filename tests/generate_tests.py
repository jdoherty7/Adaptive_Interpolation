from nose.tools import *
import adaptive_interpolation.generate as generate

def test_generation():
    pass


def test_code_execution():
    func1 = lambda x: np.sin(x**1.1)

    a, b = -10, 10
    domain_size = 500
    x = np.linspace(a, b, domain_size, dtype=np.float64)
    max_val = la.norm(func1(x), np.inf)

    app_c = adapt_i.make_interpolant(a, b, func1, 10, 1e-5, "chebyshev")
    app_l = adapt_i.make_interpolant(a, b, func1, 10, 1e-5, "legendre")
    app_m = adapt_i.make_interpolant(a, b, func1, 10, 1e-5, "monomial")

    codes = []
    for i in range(2):
        for j in range(2):
            # make sure chebyshev code evaluates to correct values
            adapt_i.generate_code(app_c, i, j, domain_size)
            app_c_est = app_c.evaluate(x)
            run_c_est = adapt_i.run_code(x, app_c, j)
            assert la.norm(app_c_est-run_c_est, np.inf)/max_val < 1e-12

            # make sure legendre code evaluates to correct values
            adapt_i.generate_code(app_l, i, j, domain_size)
            app_l_est = app_l.evaluate(x)
            run_l_est = adapt_i.run_code(x, app_l, j)
            assert la.norm(app_l_est-run_l_est, np.inf)/max_val < 1e-12

            # make sure monomial code evaluates to correct values
            adapt_i.generate_code(app_m, i, j, domain_size)
            app_m_est = app_m.evaluate(x)
            run_m_est = adapt_i.run_code(x, app_m, j)
            assert la.norm(app_m_est-run_m_est, np.inf)/max_val < 1e-12
