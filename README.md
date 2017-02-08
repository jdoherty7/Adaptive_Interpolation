# Adaptive Interpolation Project
This is a pythonic method for creating an approximation for a function
on a given domain with a specified error. It is done by adaptively interpolating
said function on the domain until the allowed error is reached. C code is then
generated and returned which evaluates said interpolant. This code can then be
run in the library using a method that uses pyopencl to run the code in parallel.


# Quick and Easy Demonstration

To see a demonstration of the code in action download and install the package.

Run: python performance_tests.py

This will run the script. Should run a function that displays the interpolant
and actual function in a plot as well as a plot of the absolute errors between
the two. The parameters given are a domain between 0 and 5, the 0th bessel
function, a 30th order interpolant, and an allowed relative error of 1e-14.

It then displays times run for the current run and errors in the functions.
The most important thing here is the relative error that is found, defined
as Inf_norm(abs_errors)/Inf_norm(actual_values).

The times are displayed more so that as parameters are tweaked the user can
see how times may change with such tweakings. Furthermore, if the user
increases the size of the x being evaulated a better picture of the 
asymptotic performance of the interpolant comes into view.





## adaptive_interpolation
Calls on all of the scripts below to run an adpative interpolation method.
The user can choose the interpolant being used, the nodes being used, the
maximum relative error allowed, the interval to interpolate on, the function
to approximate, and the order of the interpolant.

## adapt
Runs the actual adaptive method. Returns the coefficients of the interpolant 
and the ranges on which these coefficients are valid.

## approximator
A class that allows the coefficients and ranges in adapt2 to be utilized in
a practical way. Mainly, this allows a fairly quick evaluation of a large
array of numbers with said interpolant.

## generate
Creates a string that can be executed as C code. Also contains a method
to run the C code using pyopencl.

