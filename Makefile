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
	rm -rf build/ dist/ \
	*.egg-info/ */*.so */*/*.so  */*/*/*.so \
   */__pycache__/ */*/__pycache__/ */*/*/__pycache__/ \
   kliff.log  */kliff.log  */*/kliff.log  */*/*/kliff.log

