"""
Create C Code that will evaluate the approximator
class much more quickly.

"""

import numpy as np

#input is an approximator class
#output is C code.

#function for power so as not to use #include<math.h>
power = """
float power(float x, float n) {
    int number = 1;
    for (int i=0; i<n; ++i) {
        number *= x
    }
    return(number)
}
"""

#first do simple thing
#make if statements for all ranges and evaluate them for just monomials
#doing this with a string.
def generate_srting(ap):
    string = ""
    string += "float power(float, float)"
    string += "int main() {"
    string += "float value;"
    for i in range(len(ap.ranges)):
        string += "if (({0} <= x) && (x <= {1})) ".format(ap.ranges[i][0], ap.ranges[i][1])
        string += "{"
        string += "value = "
        for j in range(ap.order):
            string += "{0}*power(x, {1})".format(ap.coeff[i][j], j)
            if j != ap.order:
                string += " + "
        string += ";"
        string += "}"
    string += "return value;"
    string += "}"
    string += power
    return string
