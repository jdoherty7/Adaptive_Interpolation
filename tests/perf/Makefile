CXX=g++ -m64
CXXFLAGS=-std=c++11 -Iobjs/ -O3 -Wall
ISPC=~/Desktop/ispc-v1.9.1-linux/ispc
ISPCFLAGS=-g -O0 --arch=x86-64 --target=avx2-i32x8 --opt=disable-fma --woff



default: simple

.PHONY: dirs clean
.PRECIOUS: objs/simple.h

dirs:
	/bin/mkdir -p objs/

clean:
	/bin/rm -rf objs *~ simple

simple: dirs  objs/simple.o objs/simple_ispc.o
	$(CXX) $(CXXFLAGS) -o $@ objs/simple.o objs/simple_ispc.o


objs/simple.o: simple.cpp objs/simple_ispc.h 
	$(CXX) $(CXXFLAGS) -c -o $@ $<

objs/%_ispc.h objs/%_ispc.o: %_ispc.ispc
	$(ISPC) $(ISPCFLAGS) $< -o objs/$*_ispc.o -h objs/$*_ispc.h


