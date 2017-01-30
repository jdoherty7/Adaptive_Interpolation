# Adaptive Interpolation Project
This is a pythonic method for creating an approximation for a function
on a given domain with a specified error. It is done by adaptively interpolating
said function on the domain until the allowed error is reached. C code is then
generated and returned which evaluates said interpolant. This code can then be
run in the library using a method that uses pyopencl to run the code in parallel.


## main
Calls on all of the scripts below to run an adpative interpolation method.
The user can choose the interpolant being used, the nodes being used, the
maximum relative error allowed, the interval to interpolate on, the function
to approximate, and the order of the interpolant.

## adapt
Runs the actual method. Returns the coefficients of the interpolant and the
ranges on which these coefficients are valid.

## approximator
A class that allows the coefficients and ranges in adapt2 to be utilized in
a practical way. Mainly, this allows a fairly quick evaluation of a large
array of numbers with said interpolant.

## generate
Creates a string that can be executed as C code. Also contains a method
to run the C code using pyopencl.

