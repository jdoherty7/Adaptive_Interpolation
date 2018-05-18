"""
New adaptive interpolation with better adaptive method
"""

from __future__ import division

import copy
import numpy as np
import numpy.linalg as la
import scipy.special as spec
import scipy.optimize as optimize


class Tree:
    def __init__(self, root=0):
        self.root = root
        self.size = 0
        self.max_level = 0

    def visualize(self):
        pass

    def adapt(self):
        pass



class Node:
    def __init__(self, parent, left=0, right=0):
        self.parent = parent
        self.left = left
        self.right = right
        self.level = -1
        self.level = self.get_level()
        self.data = 0
        if left != 0:
            left.parent = self
        if right != 0:
            left.parent = self

    def get_level(self):
        if self.level == -1:
            if (self.parent == 0):
                self.level = 0
            else:
                self.level = self.parent.level + 1
        return self.level


class Interpolant(object):
    # defining parameters of an adaptive method
    def __init__(self, f, order, error, interpolant_choice, 
                 dtype, guaranteed_accurate=True, optimizations=[]):
        dt = int(dtype)
        # use recursions till node interval is order*machine precision - some tol const
        # max_recur is max number of recursion levels allowed for adaption
        # 34 reaches a spacing of 10**-15
        if dt <= 32:
            self.dtype = np.float32
            self.max_recur = 24
        elif dt <= 64:
            self.dtype = np.float64
            self.max_recur = 53 - 10
        elif dt <= 80:
            self.dtype = np.longdouble
            self.max_recur = 64 - 10
        else:
            raise Exception("Incorrect data type specified")
        if "calc intervals" in optimizations:
            # 14 to store in int32
            # 25 to store in int64
            self.max_recur = 14

        if interpolant_choice not in ['chebyshev', 'legendre', 'monomial']:
            string_err = "{0} is not a valid \
                          interpolant.\n".format(interpolant_choice)
            string_err+= "legendre, chebyshev, and monomial are the choices."
            raise ValueError(string_err)
        # function pass, must be vectorized
        self.function = f
        self.lower_bound = 0
        self.upper_bound = 0
        # max order allwed to create interpolation
        self.max_order = order
        # string specifying basis choice
        self.basis = interpolant_choice
        self.tree = Tree(Node(0))
        self.tree.size+=1
        self.allowed_error = error
        self.guaranteed_accurate = guaranteed_accurate
        self.leaves = []
        # for testing better methods
        self.optimizations=optimizations


    # function to evaluate Legendre polynomials of a number, x, up to order n
    def legendre(self, n, x):
        if n == 0:
            return np.array([1.], dtype=self.dtype)
        elif n == 1:
            return np.array([1., x], dtype=self.dtype)
        elif n > 1:
            L = [self.dtype(1.), self.dtype(x)]
            for i in range(2, int(n+1)):
                first_term = self.dtype(2*i-1)*self.dtype(x)*L[i-1]
                second_term = self.dtype(i-1)*L[i-2]
                L.append((first_term + second_term)*(1./n))
            return np.array(L, dtype=self.dtype)

    # function to evaluate chebyshev polynomials of a value x up to order n
    def chebyshev(self, n, x):
        if n == 0:
            return np.array([1.], dtype=self.dtype)
        elif n == 1:
            return np.array([1., x], dtype=self.dtype)
        elif n > 1:
            C = [self.dtype(1.), self.dtype(x)]
            for i in range(2, int(n+1)):
                C.append(self.dtype(2)*self.dtype(x)*C[i-1] - C[i-2])
            return np.array(C, dtype=self.dtype)

    # transformation for orthogonal functions, from [a, b] -> [-1, 1]
    def transform(self, x, a, b):
        scale = (x - a)/(b - a)
        return 2*scale - 1

    # given an order an a number, x. the polynomials of order 0 to n
    # are returned, evaluated for the given number.
    def basis_function(self, x, n, basis, a, b):
        xscaled = (2*(x - a)/(b - a)) - 1
        if (basis == 'legendre'):
            #return spec.eval_legendre(n, x)
            return self.legendre(n, xscaled)
        elif (basis == 'chebyshev'):
            #return spec.eval_chebyt(n, x)
            return self.chebyshev(n, xscaled)
        else:
            #return np.polyval(np.ones(n), x)
            return np.array([x**i for i in range(int(n)+1)], dtype=self.dtype)

    # given a list of coefficients, evaluate what the interpolant's value
    # will be for the given x value(s). Assumes that x is an array
    # coeff is coefficients of a basis (string) of a given order (integer)
    def eval_coeff(self, coeff, x, basis, order, a, b):
        my_vals = []
        if type(x) == type([]) or type(x) == type(np.array([0])):
            for x0 in x:
                xs = self.basis_function(x0, order, basis, a, b)
                val = np.dot(coeff, xs)
                my_vals.append(val)
            return np.array(my_vals, dtype=self.dtype)
        else:
            xs = self.basis_function(x, order, basis, a, b)
            return np.dot(coeff, xs)

    # gets n chebyshev nodes from a to b
    def get_cheb(self, a, b, n):
        if n == 1:
            return np.array([(a+b)/2.], dtype=self.dtype)
        k = np.array(range(1, int(n) + 1)[::-1], dtype=self.dtype)
        nodes = np.cos((2.*k - 2.)*np.pi/(2.*int(n-1)))
        # change range from -1 to 1 to a to b
        return (b-a)*.5*(nodes + 1.) + a

    # find interpolated coefficients given a basis for
    # evaluation and nodes to evaluate the function at.
    def interpolate(self, nodes, basis, a, b):
        length = len(nodes)
        V = np.empty(shape=(length, length), dtype=self.dtype)
        for i in range(length):
            V[i, :] = self.basis_function(nodes[i], length-1, basis, a, b)
        # try to solve for coefficients, if there is a singular matrix
        # or some other error then return None to indicate an error
        try: return la.solve(V, self.function(nodes))
        except: return None

    # finds error using the max val as the max on the entire interval, not the current
    # below is the max number of points that can be evaluated exactly
    # (self.upper_bound - self.lower_bound)*(2**(self.max_recur+1))
    def find_error(self, coeff, a, b, order):
        # get number of points for each interval
        n      = min(5e3*(b-a) + 10, 5e3)
        lb, ub = self.lower_bound, self.upper_bound
        num_nodes = 5e3*(ub - lb) + 10

        # get full interval and subinterval
        full_x = np.linspace(lb, ub, num_nodes, dtype=self.dtype)
        x      = np.linspace(a, b, n, dtype=self.dtype)

        # evaluate absolute infinity norm on subinterval
        # and infinity norm of function on full interval
        approx = self.eval_coeff(coeff, x, self.basis, order, a, b)
        actual = self.function(x)
        max_abs_err = la.norm(approx - actual, np.inf)
        max_val_full_int = la.norm(self.function(full_x), np.inf)

        # calculate relative error on the subinterval
        return max_abs_err/max_val_full_int

    # adaptive method finding an interpolant for a function
    # this uses a specified order and basis function
    def adapt(self, a, b, node):
        #print(a, b)
        self.tree.max_level = max(self.tree.max_level, node.level)
        # prevent from refining the interval too greatly
        if node.level > self.max_recur:
            string_err0 = "Recursed too far. Try changing the order of\n"
            string_err0+= "the interpolant used, raise the allowed error,\n"
            string_err0+= "or set accurate=False.\n"
            if self.guaranteed_accurate:
                raise ValueError(string_err0)
            else:
                return
        # get nodes to evaluate interpolant with
        nodes = self.get_cheb(a, b, self.max_order+1)
        # get coefficients of interpolant defined on the nodes
        # guaranteed to never give a singular matrix
        coeff = self.interpolate(nodes, self.basis, a, b)
        if coeff is None:
            string_err1 = "Singular matrix obtained on bounds [{0} {1}]\n".format(a, b)
            string_err1+= "If using monomials try using an orthogonal polynomial.\n"
            string_err1+= "Otherwise, try a different order interpolant, lower the\n"
            string_err1+= "allowed error, or set accurate=False\n"
            if self.guaranteed_accurate:
                raise ValueError(string_err1)
            else:
                return
        # calculate the maximum relative error on the interval
        # using these coefficients
        this_error = self.find_error(coeff, a, b, self.max_order)
        # append the coefficients and the range they are valid on to this
        # array also the basis function and order of in this range
        node.data = [(a+b)/2., coeff, [a, b], this_error]
        # if error is larger than maximum allowed relative error
        # then refine the interval
        if this_error > self.allowed_error:
            # adapt on the left subinterval then the right subinterval
            self.tree.size += 2
            node.left = Node(node)
            node.right = Node(node)
            self.adapt(a, (a+b)/2., node.left)
            self.adapt((a+b)/2., b, node.right)


    ########################################################
    #                                                      #
    # Section Containing Functions for Remez interpolation #
    #                                                      #
    ########################################################

    # find interpolated coefficients given a basis for
    # evaluation and nodes to evaluate the function at.
    # n is order
    def solve_remez_system(self, nodes, order, a, b):
        n = int(order)
        length = n + 2
        V = np.zeros((length, length))
        for i in range(length):
            V[i, :-1] = self.basis_function(nodes[i], n, self.basis, a, b)
            V[i, -1] = (-1)**(i+1)
        try: return la.solve(V, self.function(nodes))
        except: return None

    # update node choices based on places with maximum error near
    # the current node choices, leave endpoints as is
    # if order 0 is used the nodes are not changed
    def update_nodes_incorrect(self, nodes, coeff, n, a, b):
        # see FUNCTION APPROXIMATION AND THE REMEZ ALGORITHM to fix this exchange step
        # should find roots and then find the max error in between those roots
        if nodes.shape[0] > 2:
            err = lambda x: np.abs(self.eval_coeff(coeff, x, self.basis, n, a, b)
                                 - self.function(x))
            new_nodes = np.zeros(len(nodes), dtype=self.dtype)
            new_nodes[0] = nodes[0]
            new_nodes[-1] = nodes[-1]
            for i in range(1, len(nodes)-1):
                c, d = (new_nodes[i-1] + nodes[i])/2, (nodes[i] + nodes[i+1])/2
                x = np.linspace(c, d, 1e3, dtype=self.dtype)
                new_nodes[i] = x[np.argmax(err(x))]
            # shouldnt this be: new_nodes = locmax(err(x))
            # assert new_nodes.shape[0] == n
            # locmax is unclear if there are high frequency terms.
            return new_nodes
        else:
            return nodes

    def find_roots(self, err, nodes,c,d,coeff):
        roots = np.zeros(len(nodes)-1, dtype=self.dtype)
        for i in range(len(roots)):
            a, b = nodes[i], nodes[i+1]
            if (b - a)/(2) < np.finfo(self.dtype).eps*b:
                print(c,d)
                roots[i] = (a + b)/2
            else:
                roots[i] = optimize.brentq(err, a, b)
        return roots

    # update node choices based on places with maximum error near
    # the current node choices, leave endpoints as is
    # if order 0 is used the nodes are not changed
    def update_nodes(self, nodes, coeff, n, a, b):
        # Error of the interpolation
        err = lambda x: self.eval_coeff(coeff, x, self.basis, n, a, b) \
                      - self.function(x)
        new_nodes = np.zeros(len(nodes), dtype=self.dtype)
        # Roots of the Error function. Should be N+1 by Equioscillation Theorem
        roots = self.find_roots(err, nodes, a, b, coeff)
        # New nodes are the points that have the maximum absolute value of error
        # within the intervals between each of the roots.
        for i in range(len(nodes)):
            c = a if i == 0          else roots[i-1]
            d = b if i == len(roots) else roots[i]
            neg_abs = lambda x: -np.abs(err(x))
            new_nodes[i] = optimize.fminbound(neg_abs, c, d)
        return new_nodes


    def check_eq_alt(self, array, error):
        tolerance = 10*np.finfo(self.dtype).eps
        equal = (np.max(np.abs(array)) - np.min(np.abs(array))) <= tolerance
        last_sign = np.sign(array[0])
        alternate = True
        for i in range(1,len(array)):
            alternate = alternate and (last_sign == -np.sign(array[i]))
            last_sign = np.sign(array[i])
        return equal and alternate


    def remez(self, a, b, n):
        remez_nodes = self.get_cheb(a, b, n+2)
        #x = np.linspace(a, b, min(5e3, (b-a)/self.allowed_error), dtype=self.dtype)
        for _ in range(40):
            solution = self.solve_remez_system(remez_nodes, n, a, b)
            if solution is None: return solution # singular matrix
            coeff = solution[:-1]
            error = np.abs(solution[-1])
            if "remez incorrect" in self.optimizations:
                M = self.update_nodes_incorrect(remez_nodes, coeff, n, a, b)
            else:
                try:
                    M = self.update_nodes(remez_nodes, coeff, n, a, b)
                except:
                    break
            err = lambda x: self.eval_coeff(coeff, x, self.basis, n,
                                            a, b) - self.function(x)
            remez_nodes = M
            if self.check_eq_alt(err(remez_nodes), error): break
        #print(err(M))
        #print(b-a, error, self.check_eq_alt(err(M), error))
        #print(la.norm(self.get_cheb(a, b, n+2)-remez_nodes, np.inf)/(b-a))
        #print(M)
        return coeff, remez_nodes


    # adaptive method utilizing the remez algorithm for interpolation
    def remez_adapt(self, a, b, node):
        #print(a, b, "Remez")
        #print((b-a)/(self.max_order+2))
        self.tree.max_level = max(self.tree.max_level, node.level)
        if node.level > self.max_recur:
            string_err0 = "Recursed too far. Try changing the order of\n"
            string_err0+= "the interpolant used, raise the allowed error,\n"
            string_err0+= "or set accurate=False.\n"
            if self.guaranteed_accurate:
                raise ValueError(string_err0)
            else:
                return
        # get coeff on interval utilizing the remez algorithm
        ret = self.remez(a, b, self.max_order)
        if ret is None:
            if self.guaranteed_accurate:
                string_err1 = "Singular matrix obtained on bounds [{0} {1}]\n".format(a, b)
                string_err1+= "If using monomials try using an orthogonal polynomial.\n"
                string_err1+= "Otherwise, try a different order interpolant, lower the\n"
                string_err1+= "allowed error, or set accurate=False\n"
                raise ValueError(string_err1)
            else:
                return
        coeff, M = ret[0], ret[1]
        this_error = self.find_error(coeff, a, b, self.max_order)
        node.data = [(a+b)/2., coeff, [a, b], this_error]
        #print("Error", np.log10(this_error), (b-a)/(self.max_order+2), node.level)
        if this_error > self.allowed_error:
            # adapt on the left subinterval then the right subinterval
            self.tree.size += 2
            node.left = Node(node)
            node.right = Node(node)
            self.remez_adapt(a, (a+b)/2., node.left)
            self.remez_adapt((a+b)/2., b, node.right)


    # Method to run the adaptive method initially
    def run_adapt(self, lower_bound, upper_bound, adapt_type):
        if upper_bound <= lower_bound:
            raise Exception("Upper bound must be greater than lower bound.")
        self.lower_bound = self.dtype(lower_bound)
        self.upper_bound = self.dtype(upper_bound)
        if adapt_type.lower() == "variable":
            self.variable_order_adapt(self.lower_bound, self.upper_bound, self.tree.root)
        elif adapt_type.lower() == "remez":
            self.remez_adapt(self.lower_bound, self.upper_bound, self.tree.root)
        else:
            self.adapt(self.lower_bound, self.upper_bound, self.tree.root)


        # Estimated Recursion Depth, From Taylors Remainder Theorem
        # assuming smooth and continous and n+1 derivative exists
        if 0:
            nodes = self.get_cheb(lower_bound, upper_bound, self.max_order+2)
            coeff = self.interpolate(nodes, "monomials", lower_bound, upper_bound)
            coeff[-1] = coeff[-1]
            import scipy.misc as sm
            import scipy.special as spec
            dfn = abs(sm.factorial(self.max_order+1)*coeff[-1])
            print("dfn+1: ", dfn, coeff[-1], np.log2(dfn)/(self.max_order+1))
            if 0:
                f = lambda x: self.eval_coeff(coeff, x, "monomials", self.max_order+1, lower_bound, upper_bound)
                import matplotlib.pyplot as plt
                plt.figure()
                x = np.linspace(lower_bound, upper_bound, 1000)
                dfn = la.norm(spec.jvp(0, x, self.max_order+1), np.inf)
                print(dfn)
                plt.plot(x, f(x))
                plt.plot(x, spec.jvp(0, x, self.max_order+1))
                plt.plot(x, self.function(x))
                plt.show()

            depth = -np.log2(self.allowed_error)/(self.max_order+1)
            depth+=  np.log2(upper_bound - lower_bound)
            depth+=  np.log2(dfn)/(self.max_order+1)

            print("Estimated Depth:   ", depth)
            print("Actual Tree Depth: ", self.tree.max_level)


            def test_cheb_err():
                import numpy.linalg as la
                from numpy.polynomial import chebyshev as cheb
                import matplotlib.pyplot as plt
                x = 0*np.linspace(lower_bound, upper_bound, 5e4)# + np.finfo(np.float64).eps
                xs = 2*(x/(upper_bound-lower_bound)) -1- lower_bound
                for n in range(3, 20):
                    nodes = self.get_cheb(lower_bound, upper_bound, n+1)
                    coeff = self.interpolate(nodes, "chebyshev", lower_bound, upper_bound)
                    f = lambda x: self.eval_coeff(coeff, x, "chebyshev", n,
                                   lower_bound, upper_bound)
                    if 0:
                        plt.figure()
                        plt.plot(x, cheb.chebval(xs, coeff))
                        plt.plot(x, f(x), 'g')
                        plt.plot(x, self.function(x), 'r')

                        plt.show()
                    dx = (2**8)*np.finfo(np.float64).eps
                    print(n, la.norm(f(x) - f(x + dx), np.inf)/dx)
                    #print(n, la.norm(f(x) - cheb.chebval(xs, coeff), np.inf))
                    #print(n, la.norm(f(x), np.inf), la.norm(cheb.chebval(xs, coeff), np.inf))
            test_cheb_err()


        # add a condition to check if tree is good enough already?
        optimal = self.tree.size == 2**(self.tree.max_level+1) - 1
        if "balance" in self.optimizations and not optimal:
            leaves = self.get_leaves(self.tree.root)
            #print(leaves, len(leaves))
            if "combine" in self.optimizations:
                leaves = self.combine_leaves(leaves)

            print("Original Height: ", self.tree.max_level)
            self.tree = self.create_new_tree(leaves)
            print("Balanced Height: ", self.tree.max_level)
            print('\n\n\n')
            l = self.get_leaves(self.tree.root)
            #print(l)
            #print(len(l))
        """
        import scipy.sparse as sp
        N = 2*(self.max_order+1)
        bounds = np.arange(-(self.max_order+1)//2,(self.max_order+1)//2)
        print(bounds)
        diags = []
        for i in bounds:
            diags.append((self.max_order+1-abs(i))*np.ones(N - abs(i)))
        D = sp.diags(diags, bounds)
        print(D.todense())
        x = np.linspace(lower_bound, upper_bound, N)
        depth+= la.norm(D @ self.function(x), np.inf)
        """



    # Possible future pruning functions
    def get_leaves(self, node, leaves=[]):
        left, right = 0, 0
        if type(node.left) != int:
            left = node.left.data[0]
        if type(node.right) != int:
            right = node.right.data[0]
        print(node.data, left, right)
        if node.left == 0 and node.right == 0:
            leaves.append(node)
        else:
            self.get_leaves(node.left, leaves)
            self.get_leaves(node.right, leaves)
        return leaves

    def combine_leaves(self, leaves):
        i = 0
        while i < len(leaves)-1:
            new_node = self.replace(leaves[i], leaves[i+1])
            if new_node == False:
                i+=1
            else:
                # found better interpolant
                del leaves[i+1]
                del leaves[i]
                leaves.insert(i, new_node)
        return leaves


    def replace(self, node1, node2):
        a1, b2 = node1.data[2][0], node2.data[2][1]
        nodes = self.get_cheb(a1, b2, self.max_order+1)
        coeff = self.interpolate(nodes, self.basis, a1, b2)
        if coeff is None:
            raise ValueError("Singular Matrix while combining leaves?") 
        error = self.find_error(coeff, a1, b2, self.max_order)
        if error < self.allowed_error:
            node = Node(0)
            node.data = [(a1 + b2)/2., coeff, [a1, b2], error]
            return node
        return False


    def create_new_tree(self, leaves):
        level = copy.deepcopy(leaves)
        next_level = []
        size = len(leaves)
        while len(level) > 1:
            rev = leaves[-1].get_level() < leaves[0].get_level()
            if rev:
                level.reverse()
            length = len(level)//2
            for i in range(length):
                # this should set children's parent correctly
                left = level.pop(0)
                right = level.pop(0)
                if rev:
                    parent = Node(0, right, left)
                else:
                    parent = Node(0, left, right)
                parent.data = [(left.data[0] + right.data[0])/2]
                next_level.append(parent)
            size += length
            # add any remaining leaves so they are on the next level
            assert len(level) <= 1
            for i in range(len(level)):
                next_level.append(level.pop(0))
            if rev:
                next_level.reverse()
            level = next_level
            next_level = []
        ################
        new_root = level[0]
        new_tree = Tree(new_root)
        new_tree.size = size
        new_tree.max_level = max(leaves[-1].get_level(), leaves[0].get_level())
        return new_tree

