#include <fstream>
#include <iostream>
#include <string>
#include "objs/simple_ispc.h"
using namespace ispc;
using namespace std;

//extern void eval(float tree, float x, float y);

void file_to_array(const char *file_name, float *array) {
    ifstream myfile(file_name);
    string mystring;
    int i = 0;
    while (getline(myfile, mystring, ',')) {
        if (!cin) {
            cout << "failure " << i << endl;
        }
        //cout << i << " " << mystring << endl;
        array[i] = stof(mystring);
        i++;
    }
    myfile.close();
}



int main() {
    int opt = 0;
    int midsize, leftsize, rightsize, a_size, b_size, coeffsize, tsize, fsize;
    float *mid, *left, *right, *interval_a, *interval_b, *coeff, *tree, *f;

    ifstream myfile("sizes.txt");
    string mystring;

    getline(myfile, mystring, ',');
    int xsize = stoi(mystring);
    
    getline(myfile, mystring, ',');
    int ysize = stoi(mystring);

    if (opt == 1) {
        getline(myfile, mystring, ',');
        midsize = stoi(mystring);
        getline(myfile, mystring, ',');
        leftsize = stoi(mystring);
        getline(myfile, mystring, ',');
        rightsize = stoi(mystring);
        getline(myfile, mystring, ',');
        a_size = stoi(mystring);
        getline(myfile, mystring, ',');
        b_size = stoi(mystring);
        getline(myfile, mystring, ',');
        coeffsize = stoi(mystring);
    }
    else {
        getline(myfile, mystring, ',');
        tsize = stoi(mystring);
    }

    // this should be changed to another method if 
    // it is allowed to have the map withouth the arrays
    // or the arrays without the map
    if (opt == 1) {
        getline(myfile, mystring, ',');
        fsize = stoi(mystring);
    }
    myfile.close();


    float *x = new float[xsize];
    float *y = new float[ysize];

    if (opt == 1) {
        mid = new float[midsize];
        left = new float[leftsize];
        right = new float[rightsize];
        interval_a = new float[a_size];
        interval_b = new float[b_size];
        coeff = new float[coeffsize];
    }
    else {
        tree = new float[tsize];
    }

    if (opt == 1) {
        f = new float[fsize];
    }


    for (int i=0; i<xsize; i++) {
        x[i] = i*(20.0)/(float)(xsize);
    }
    for (int i=0; i<ysize; i++) {
        y[i] = (float)0.0;
    }

    //cout << "before file to array" << endl;
    if (opt == 1) {
        file_to_array("mid.txt", mid);
        file_to_array("left.txt", left);
        file_to_array("right.txt", right);
        file_to_array("interval_a.txt", interval_a);
        file_to_array("interval_b.txt", interval_b);
        file_to_array("coeff.txt", coeff);
    }
    else {
        file_to_array("tree.txt", tree);
    }
    if (opt == 1) {
        file_to_array("f.txt", f);
    }

    // evaluation, this file will need to be generated... or at least this part.
    if (opt == 1) {
       //eval(mid, left, right, interval_a, interval_b, coeff, f, x, y);
    }
    else {
       eval(tree, x, y);
    }



    if (opt == 1) {
        delete [] mid;
        delete [] left;
        delete [] right;
        delete [] interval_a;
        delete [] interval_b;
        delete [] coeff;
    }
    else {
        delete [] tree;
    }

    if (opt == 1) {
        delete [] f;
    }


    delete [] x;
    delete [] y;


    return 0;

}


