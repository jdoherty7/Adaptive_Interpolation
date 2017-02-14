# Adaptive Interpolation Project (BETA)
This is a pythonic method for creating an approximation for a function
on a given domain with a specified error. It is done by adaptively interpolating
said function on the domain until the allowed error is reached. C code is then
generated and returned which evaluates said interpolant. This code can then be
run in the library using a method that uses pyopencl to run the code in parallel.


# Quick and Easy Demonstration

To see a demonstration of the code in action download this git repository.

Change your working directory to adaptive_interpolation, which is the folder
containing the core scripts for this project. Then, while in said directory
run the performance_test.py script. (ie. python performance_test.py )

This script will run a function that displays the interpolant
and actual function in a plot as well as a plot of the absolute errors between
the two. The parameters given are an interval of 0 to 5, the 0th bessel
function, a 30th order interpolant, and an allowed relative error of 1e-14.

It then displays times run for the current run and errors in the functions.
The most important thing here is the relative error that is found, defined
as Inf_norm(abs_errors)/Inf_norm(actual_values).



## adaptive_interpolation

This contains all the functions available for the user to call. These functions
are used to make an interpolant, then generate code based on that interpolant,
then run the code generated using pyopencl.


### make_x_interpolant

Calls on all of the scripts below to run an adpative interpolation method.
The user can choose the interpolant being used, the nodes being used, the
maximum relative error allowed, the interval to interpolate on, the function
to approximate, and the order of the interpolant.

### generate_code

### run_generated_code

## adapt
Script containing the interpolant class which runs the actual adaptive method.
This interpolant class is capable of running a fixed order adaptive interpolation
as well as a variable order interplation. A Remez based interpolation will also be
available soon.


## approximator
A class that allows the I and ranges in adapt2 to be utilized in
a practical way. Mainly, this allows a fairly quick evaluation of a large
array of numbers with said interpolant.

## generate
Creates a string that can be executed as C code. Also contains a method
to run the C code using pyopencl.

