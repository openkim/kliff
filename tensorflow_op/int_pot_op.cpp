#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include<iostream>

using namespace tensorflow;

// op interface
REGISTER_OP("IntPot")
  .Attr("T: {float, double}")
  .Input("coords: T")    // atomic coords of contriburing atoms
  .Input("zeta: T")    // generalized coords
  .Input("dzetadr: T")   // derivative of generalized coords w.r.t. atomic coords
  .Output("gen_coords: T")   // same as zeta
  .Output("dgen_datomic_coords: T")    // same as dzetadr

  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    // set output shape
    c->set_output(0, c->input(1));
    c->set_output(1, c->input(2));
    return Status::OK();
  });


// op implementation
template <typename T>
class IntPotOp : public OpKernel {
  public:
    explicit IntPotOp(OpKernelConstruction* context) : OpKernel(context) { }

    void Compute(OpKernelContext* context) override {

      // Grab the input tensor
      //const Tensor& coords_tensor = context->input(0);
      const Tensor& zeta_tensor = context->input(1);
      const Tensor& dzetadr_tensor = context->input(2);
      // transform to Eigen tensor
      //auto coords = coords_tensor.flat<T>();
      auto zeta = zeta_tensor.flat<T>();
      auto dzetadr = dzetadr_tensor.flat<T>();

      // we have to allocate memory for output, although it is the the same
      // as the input in our case
      Tensor* gen_coords_tensor = NULL;
      Tensor* dgen_datomic_coords_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, zeta_tensor.shape(),
            &gen_coords_tensor));
      OP_REQUIRES_OK(context, context->allocate_output(1, dzetadr_tensor.shape(),
            &dgen_datomic_coords_tensor));

      // transform to Eigen tensor
      auto gen_coords = gen_coords_tensor->flat<T>();
      auto dgen_datomic_coords = dgen_datomic_coords_tensor->flat<T>();

      // copy data from input to output
      int N = zeta.size();
      for (int i = 0; i < N; i++) {
        gen_coords(i) = zeta(i);
      }
      N = dzetadr.size();
      for (int i = 0; i < N; i++) {
        dgen_datomic_coords(i) = dzetadr(i);
      }


//TODO delete
/*
      auto data1 = gen_coords_tensor->matrix<T>();
      std::cout<<"flag: gen_coords"<<std::endl;
      for (int i=0; i<data1.dimension(0); i++) {
        for (int j=0; j<data1.dimension(1); j++) {
          std::cout<<data1(i,j)<<" ";
        }
        std::cout<<std::endl;
      }
      std::cout<<std::endl;

      auto data2 = dgen_datomic_coords_tensor->tensor<T, 3>();
      std::cout<<"flag: dgen_coords"<<std::endl;
      for (int i=0; i<data2.dimension(0); i++) {
        for (int j=0; j<data2.dimension(1); j++) {
          for (int k=0; k<data2.dimension(2); k++) {
            std::cout<<data2(i,j,k)<<" ";
          }
          std::cout<<std::endl;
        }
        std::cout<<std::endl;
      }
      std::cout<<std::endl;
*/

    }
};


// register kernel
#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
    Name("IntPot").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
    IntPotOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL


