CXX = g++
LDFLAGS =
LDLIBS = -lm
TF_INC = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
CXXFLAGS = -std=c++11 -g -Wall -O2 -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I $(TF_INC)

SOURCES = int_pot_op.cc

OBJECTS = $(SOURCES:.cc=.o)

# create shared lib
int_pot_op.so:$(OBJECTS)
	$(CXX) -shared $(LDFLAGS) $(OBJECTS) -o $@ $(LDLIBS)

# compile source
.o:
	$(CXX) $(CXXFLAGS) $< -o $@


.PHONY: clean
clean:
	rm *.o *.so
