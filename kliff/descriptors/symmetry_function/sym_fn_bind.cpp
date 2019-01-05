#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include "sym_fn.h"

namespace py = pybind11;


PYBIND11_MODULE(sf, m) {
  m.doc() = "Symmetry function descriptor for ANN potential.";

  py::class_<Descriptor>(m, "Descriptor")
    .def(py::init<>())

    .def("get_num_descriptors", &Descriptor::get_num_descriptors)

    .def("set_cutoff",
      [](Descriptor &d, char* name, py::array_t<double> rcuts) {
        d.set_cutoff(name, rcuts.shape(0), rcuts.mutable_data(0));
        return;
      },
      py::arg("name"),
      py::arg("rcuts").noconvert()
    )

    .def("add_descriptor",
      [](Descriptor &d, char* name, py::array_t<double> values) {
        auto rows = values.shape(0);
        auto cols = values.shape(1);
        d.add_descriptor(name, values.mutable_data(0), rows, cols);
        return;
      },
      py::arg("name"),
      py::arg("values").noconvert()
    )

    .def("get_gen_coords",
      [](Descriptor &d, py::array_t<double> coords, py::array_t<int> particleSpecies,
         py::array_t<int> neighlist, py::array_t<int> numneigh,
         py::array_t<int> image, int Natoms, int Ncontrib, int Ndescriptor) {

        // create empty vectors to hold return data
        std::vector<double> gen_coords(Ncontrib*Ndescriptor, 0.0);

        d.get_generalized_coords(coords.mutable_data(0),
            particleSpecies.mutable_data(0), neighlist.mutable_data(0),
            numneigh.mutable_data(0), image.mutable_data(0),
            Natoms, Ncontrib, Ndescriptor,
            gen_coords.data(), nullptr);

        // pack gen_coords into a buffer that numpy array can understand
        auto gen_coords_2D = py::array (py::buffer_info (
          gen_coords.data(),   // data pointer
          sizeof(double),  // size of one element
          py::format_descriptor<double>::format(),  //Python struct-style format descriptor
          2,  // dimension
          {Ncontrib, Ndescriptor},  // size of each dimension
          {sizeof(double)*Ndescriptor, sizeof(double)}  // stride (in bytes) for each dimension
        ));

        return gen_coords_2D;
      },
      py::arg("coords").noconvert(),
      py::arg("particleSpecies").noconvert(),
      py::arg("neighlist").noconvert(),
      py::arg("numneigh").noconvert(),
      py::arg("image").noconvert(),
      py::arg("Natoms"),
      py::arg("Ncontrib"),
      py::arg("Ndescriptor")
    )

    .def("get_gen_coords_and_deri",
      [](Descriptor &d, py::array_t<double> coords, py::array_t<int> particleSpecies,
         py::array_t<int> neighlist, py::array_t<int> numneigh,
         py::array_t<int> image, int Natoms, int Ncontrib, int Ndescriptor) {

        // create empty vectors to hold return data
        std::vector<double> gen_coords(Ncontrib*Ndescriptor, 0.0);
        std::vector<double> d_gen_coords(Ncontrib*Ndescriptor*3*Ncontrib, 0.0);

        d.get_generalized_coords(coords.mutable_data(0),
            particleSpecies.mutable_data(0), neighlist.mutable_data(0),
            numneigh.mutable_data(0), image.mutable_data(0),
            Natoms, Ncontrib, Ndescriptor,
            gen_coords.data(), d_gen_coords.data());

        // pack gen_coords into a buffer that numpy array can understand
        auto gen_coords_2D = py::array (py::buffer_info (
          gen_coords.data(),   // data pointer
          sizeof(double),  // size of one element
          py::format_descriptor<double>::format(),  //Python struct-style format descriptor
          2,  // dimension
          {Ncontrib, Ndescriptor},  // size of each dimension
          {sizeof(double)*Ndescriptor, sizeof(double)}  // stride (in bytes) for each dimension
        ));

        // pack dgen_coords into a buffer that numpy array can understand
        auto d_gen_coords_3D = py::array (py::buffer_info (
          d_gen_coords.data(),
          sizeof(double),
          py::format_descriptor<double>::format(),
          3,
          {Ncontrib, Ndescriptor, 3*Ncontrib},
          {sizeof(double)*Ndescriptor*3*Ncontrib, sizeof(double)*3*Ncontrib, sizeof(double)}
        ));

        py::tuple t(2);
        t[0] = gen_coords_2D;
        t[1] = d_gen_coords_3D;
        return t;
      },
      py::arg("coords").noconvert(),
      py::arg("particleSpecies").noconvert(),
      py::arg("neighlist").noconvert(),
      py::arg("numneigh").noconvert(),
      py::arg("image").noconvert(),
      py::arg("Natoms"),
      py::arg("Ncontrib"),
      py::arg("Ndescriptor"),
      "Return (gen_coords, d_gen_coords)"
    );


}

