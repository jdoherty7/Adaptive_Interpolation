export void eval( float  *tree,  float *x,  float *y) {
    for (int n=0; n<16; n++) {
        int index = 0;
        float T0, T1, Tn, a, b, s, x_scaled;
        for (int i=1; i<2; i++){
            index = tree[index] > x[n] ? (int)tree[index+5] : (int)tree[index+6];
        } 
        a = tree[index+3];
        b = tree[index+4];
        x_scaled = (2./(b - a))*(x[n] - a) - 1.0;
        T0 = 1.0;
        T1 = x_scaled;
        s = tree[index+1]*T0;
        s = s + tree[index+2]*T1;
        y[n] = s;
    }
}
