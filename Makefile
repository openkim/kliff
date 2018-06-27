build:
	python setup.py build
develop:
	python setup.py develop
inplace:
	python setup.py build_ext --inplace
install:
	python setup.py install
clean:
	python setup.py clean
clean_all:
	rm -rf build/ dist/ *.egg-info/ *.so */*.so */*.pyc\
  tests/__pycache__/

