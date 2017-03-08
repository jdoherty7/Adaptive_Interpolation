=====================================
Adaptive Interpolation Project (BETA)
=====================================

This is a pythonic method for creating an approximation for a function
on a given domain with a specified error. It is done by adaptively interpolating
said function on the domain until the allowed error is reached. C code is then
generated and returned which evaluates said interpolant. This code can then be
run in the library using a method that uses pyopencl to run the code in parallel.

----------------------------
Quick and Easy Demonstration
----------------------------

To see a demonstration of this code first clone this git repository to
your machine using:

``
git clone https://www.github.com/jdoherty7/Adaptive_Interpolation
``

Then run the demonstration by entering the cloned directory and running:

``
python demo.py
```

The code should begin interpolating three functions. One is a bessel function,
one is a wavy sin function, and another is a piecewise function. These are 
generated sequentially. THE MATPLOTLIB PLOTS MUST BE CLOSED TO GENERATE THE NEXT
INTERPOLANT. The plots display the estimated and actual
values as well as the absolute errors between the estimated values and the
actual values. The allowed relative error is shown as a red line.

NOTE: When prompted for a processor by pyopencl please press ENTER,
without typting anything to continue.

