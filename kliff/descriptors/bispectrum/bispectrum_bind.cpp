#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include "bispectrum.h"

namespace py = pybind11;


PYBIND11_MODULE(bs, m) {
  m.doc() = "Bispectrum descriptor.";

  py::class_<Bispectrum>(m, "Bispectrum")
    .def(py::init<double, int, int, int, double, int, int>())

    .def("set_cutoff",
      [](Bispectrum &d, char* name, py::array_t<double> rcuts, double rcutfac) {
        d.set_cutoff(name, rcuts.shape(0), rcuts.data(0), rcutfac);
      },
      py::arg("name"),
      py::arg("rcuts").noconvert(),
      py::arg("rcutfac")
    )

    .def("set_weight",
      [](Bispectrum &d, py::array_t<double> weight) {
        d.set_weight(weight.size(), weight.data(0));
      },
      py::arg("weight").noconvert()
    )

    .def("set_radius",
      [](Bispectrum &d, py::array_t<double> radius) {
        d.set_radius(radius.size(), radius.data(0));
      },
      py::arg("radius").noconvert()
    )

    .def("compute_B",
      [](Bispectrum &d, py::array_t<double> coords, py::array_t<int> species,
         py::array_t<int> neighlist, py::array_t<int> numneigh,
         py::array_t<int> image,
         int Natoms, int Ncontrib, int Ndescriptor) {

        // create empty vectors to hold return data
        std::vector<double> zeta(Ncontrib*Ndescriptor, 0.0);
        std::vector<double> dzeta_dr(Ncontrib*Ndescriptor*Ncontrib*3, 0.0);

        d.compute_B(coords.data(0), species.data(0),
            neighlist.data(0), numneigh.data(0), image.data(0),
            Natoms, Ncontrib, zeta.data(), dzeta_dr.data());

        // pack zeta into a buffer that numpy array can understand
        auto zeta_2D = py::array (py::buffer_info (
          zeta.data(),   // data pointer
          sizeof(double),  // size of one element
          py::format_descriptor<double>::format(),  //Python struct-style format descriptor
          2,  // dimension
          {Ncontrib, Ndescriptor},  // size of each dimension
          {sizeof(double)*Ndescriptor, sizeof(double)}  // stride (in bytes) for each dimension
        ));

        // pack dzeta into a buffer that numpy array can understand
        auto dzeta_dr_4D = py::array (py::buffer_info (
          dzeta_dr.data(),
          sizeof(double),
          py::format_descriptor<double>::format(),
          4,
          {Ncontrib, Ndescriptor, Ncontrib, 3},
          {sizeof(double)*Ndescriptor*Ncontrib*3, sizeof(double)*Ncontrib*3,
            sizeof(double)*3, sizeof(double)}
        ));

        py::tuple t(2);
        t[0] = zeta_2D;
        t[1] = dzeta_dr_4D;
        return t;
      },
      py::arg("coords").noconvert(),
      py::arg("species").noconvert(),
      py::arg("neighlist").noconvert(),
      py::arg("numneigh").noconvert(),
      py::arg("image").noconvert(),
      py::arg("Natoms"),
      py::arg("Ncontrib"),
      py::arg("Ndescriptor"),
      "Return (zeta, dzeta_dr)"
    )




//    .def("get_num_descriptors", &Descriptor::get_num_descriptors)
//
//    .def("set_cutoff",
//      [](Descriptor &d, char* name, py::array_t<double> rcuts) {
//        d.set_cutoff(name, rcuts.shape(0), rcuts.mutable_data(0));
//        return;
//      },
//      py::arg("name"),
//      py::arg("rcuts").noconvert()
//    )
//
//    .def("add_descriptor",
//      [](Descriptor &d, char* name, py::array_t<double> values) {
//        auto rows = values.shape(0);
//        auto cols = values.shape(1);
//        d.add_descriptor(name, values.mutable_data(0), rows, cols);
//        return;
//      },
//      py::arg("name"),
//      py::arg("values").noconvert()
//    )
//
//    .def("get_zeta",
//      [](Descriptor &d, py::array_t<double> coords, py::array_t<int> species,
//         py::array_t<int> neighlist, py::array_t<int> numneigh,
//         py::array_t<int> image, int Natoms, int Ncontrib, int Ndescriptor) {
//
//        // create empty vectors to hold return data
//        std::vector<double> zeta(Ncontrib*Ndescriptor, 0.0);
//
//        d.get_generalized_coords(coords.mutable_data(0),
//            species.mutable_data(0), neighlist.mutable_data(0),
//            numneigh.mutable_data(0), image.mutable_data(0),
//            Natoms, Ncontrib, Ndescriptor,
//            zeta.data(), nullptr);
//
//        // pack zeta into a buffer that numpy array can understand
//        auto zeta_2D = py::array (py::buffer_info (
//          zeta.data(),   // data pointer
//          sizeof(double),  // size of one element
//          py::format_descriptor<double>::format(),  //Python struct-style format descriptor
//          2,  // dimension
//          {Ncontrib, Ndescriptor},  // size of each dimension
//          {sizeof(double)*Ndescriptor, sizeof(double)}  // stride (in bytes) for each dimension
//        ));
//
//        return zeta_2D;
//      },
//      py::arg("coords").noconvert(),
//      py::arg("species").noconvert(),
//      py::arg("neighlist").noconvert(),
//      py::arg("numneigh").noconvert(),
//      py::arg("image").noconvert(),
//      py::arg("Natoms"),
//      py::arg("Ncontrib"),
//      py::arg("Ndescriptor")
//    )
//

    ;
}

