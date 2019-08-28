#include "sym_fn.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(sf, m)
{
  m.doc() = "Symmetry function descriptor for ANN potential.";

  py::class_<Descriptor>(m, "Descriptor")
      .def(py::init<>())

      .def("get_num_descriptors", &Descriptor::get_num_descriptors)

      .def(
          "set_cutoff",
          [](Descriptor & d, char * name, py::array_t<double> rcuts) {
            d.set_cutoff(name, rcuts.shape(0), rcuts.data(0));
            return;
          },
          py::arg("name"),
          py::arg("rcuts").noconvert())

      .def(
          "add_descriptor",
          [](Descriptor & d, char * name, py::array_t<double> values) {
            auto rows = values.shape(0);
            auto cols = values.shape(1);
            d.add_descriptor(name, values.data(0), rows, cols);
            return;
          },
          py::arg("name"),
          py::arg("values").noconvert())


      .def(
          "generate_one_atom",
          [](Descriptor & d,
             int i,
             py::array_t<double> coords,
             py::array_t<int> particleSpecies,
             py::array_t<int> neighlist,
             bool grad) {
            // create empty vectors to hold return data
            int Ndescriptor = d.get_num_descriptors();
            int numnei = neighlist.shape(0);
            std::vector<double> zeta(Ndescriptor, 0.0);
            std::vector<double> grad_zeta(Ndescriptor * 3 * (numnei + 1), 0.0);

            d.generate_one_atom(i,
                                coords.data(0),
                                particleSpecies.data(0),
                                neighlist.data(0),
                                numnei,
                                zeta.data(),
                                grad_zeta.data(),
                                grad);

            // pack zeta into a buffer that numpy array can understand
            auto zeta_py = py::array(py::buffer_info(
                zeta.data(),  // data pointer
                sizeof(double),  // size of one element
                py::format_descriptor<double>::format(),  // Python struct-style
                // format descriptor
                1,  // dimension
                {Ndescriptor},  // size of each dimension
                {sizeof(double)}  // stride (in bytes) for each dimension
                ));

            // pack dzeta into a buffer that numpy array can understand
            auto grad_zeta_py = py::array(py::buffer_info(
                grad_zeta.data(),
                sizeof(double),
                py::format_descriptor<double>::format(),
                2,
                {Ndescriptor, 3 * (numnei + 1)},
                {sizeof(double) * 3 * (numnei + 1), sizeof(double)}));

            py::none n;  // None

            py::tuple t(2);
            t[0] = zeta_py;
            if (grad) { t[1] = grad_zeta_py; }
            else
            {
              t[1] = n;
            }
            return t;
          },
          py::arg("i"),
          py::arg("coords").noconvert(),
          py::arg("particleSpecies").noconvert(),
          py::arg("neighlist").noconvert(),
          py::arg("grad"),
          "Return (zeta, grad_zeta)");
}
