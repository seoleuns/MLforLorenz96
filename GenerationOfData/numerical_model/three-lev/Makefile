PYFC = f2py
FC = gfortran

all: numodels params

numodels: params.f90 lorenz96.f90
	$(PYFC) --overwrite-signature -h numodels.pyf $^ numodels.f90 -m $@
	$(PYFC) -c numodels.pyf $^ numodels.f90
	mv $@.*.so numodels.so

params:
	$(PYFC) -c -m params params.f90
	mv $@.*.so params.so

.PHONY: clean
clean:
	rm -f *.o *.mod *.so *.pyf *.c
