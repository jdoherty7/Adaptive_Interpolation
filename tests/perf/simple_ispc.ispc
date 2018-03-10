export void eval(const uniform float tree[], uniform float x[], uniform float y[]){

varying float T0, T1, Tn, a, b, s, x_scaled, xn;
	for (uniform int nbase=0; nbase<67108864; nbase+=programCount) {
	varying int n = nbase + programIndex;
	varying int index = 0;
	xn = x[n];
	for (uniform int i=1; i<8; i++){
		index = tree[index] > xn ? (int)tree[index+9] : (int)tree[index+10];
	} 
	y[n] = (float)index;
	}
	
}
