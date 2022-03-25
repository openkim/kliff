// Author: Mingjian Wen (wenxx151@gmail.com)

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "neighbor_list.h"

#define MY_WARNING(message)                                           \
  {                                                                   \
    std::cerr << "* Error (Neighbor List) : \"" << message            \
              << "\" : " << __LINE__ << ":" << __FILE__ << std::endl; \
  }

namespace py = pybind11;


namespace
{
struct PyNeighListDestroy
{
  void operator()(NeighList * neighList) const { nbl_clean(&neighList); }
};
}  // namespace


PYBIND11_MODULE(neighlist, module)
{
  module.doc() = "Python binding to neighbor list.";

  py::class_<NeighList, std::unique_ptr<NeighList, PyNeighListDestroy> >(
      module, "NeighList", py::module_local())
      .def(py::init([]() {
    NeighList * neighList = new NeighList;
    return std::unique_ptr<NeighList, PyNeighListDestroy>(std::move(neighList));
      }))
      .def("build",
           [](NeighList &self,
              py::array_t<double> coords,
              double const influence_distance,
              py::array_t<double> cutoffs,
              py::array_t<int> need_neigh) {
    int const natoms_1 = static_cast<int>(coords.size() / 3);
    int const natoms_2 = static_cast<int>(need_neigh.size());

    if (natoms_1 != natoms_2)
    {
      MY_WARNING("\"coords\" size and \"need_neigh\" size do not match!");
    }

    int const natoms = natoms_1 <= natoms_2 ? natoms_1 : natoms_2;
    double const * coords_data = coords.data();
    int const number_of_cutoffs = static_cast<int>(cutoffs.size());
    double const * cutoffs_data = cutoffs.data();
    int const * need_neigh_data = need_neigh.data();

    int error = nbl_build(&self,
                          natoms,
                          coords_data,
                          influence_distance,
                          number_of_cutoffs,
                          cutoffs_data,
                          need_neigh_data);
    if (error == 1)
    {
      throw std::runtime_error("Cell size too large! (partilces fly away) or\n"
                               "Collision of atoms happened!");
    }
      }, "Build the neighbor list.",
         py::arg("coords").noconvert(),
         py::arg("influence_distance"),
         py::arg("cutoffs").noconvert(),
         py::arg("need_neigh").noconvert())
      .def("get_neigh",
           [](NeighList &self,
              py::array_t<double> cutoffs,
              int const neighbor_list_index,
              int const particle_number) {
    int const number_of_cutoffs = static_cast<int>(cutoffs.size());
    double const * cutoffs_data = cutoffs.data();
    int number_of_neighbors = 0;
    int const * neigh_of_atom;

    void const * const data_object
        = reinterpret_cast<void const * const>(&self);

    int error = nbl_get_neigh(data_object,
                              number_of_cutoffs,
                              cutoffs_data,
                              neighbor_list_index,
                              particle_number,
                              &number_of_neighbors,
                              &neigh_of_atom);
    if (error == 1)
    {
      if (neighbor_list_index >= self.numberOfNeighborLists)
      {
        throw std::runtime_error("neighbor_list_index = "
                                 + std::to_string(neighbor_list_index)
                                 + " >= self.numberOfNeighborLists = "
                                 + std::to_string(self.numberOfNeighborLists));
      }
      else if (cutoffs_data[neighbor_list_index]
               > self.lists[neighbor_list_index].cutoff)
      {
        throw std::runtime_error(
            "cutoffs_data[neighbor_list_index] = "
            + std::to_string(cutoffs_data[neighbor_list_index])
            + " > self.lists[neighbor_list_index].cutoff = "
            + std::to_string(self.lists[neighbor_list_index].cutoff));
      }
      else
      {
        throw std::runtime_error(
            "particle_number = " + std::to_string(particle_number) + " < 0!");
      }
    }

    // pack as a numpy array
    auto neighbors_of_particle = py::array(py::buffer_info(
        const_cast<int *>(neigh_of_atom),  // data pointer
        sizeof(int),  // size of one element
        py::format_descriptor<int>::format(),  // Python struct-style
                                               // format descriptor
        1,  // dimension
        {number_of_neighbors},  // size of each dimension
        {sizeof(int)}  // stride of each dimension
        ));

    py::tuple re(2);
    re[0] = number_of_neighbors;
    re[1] = neighbors_of_particle;
    return re;
  }, R"pbdoc(
     Get the number of neighbors and neighbors of particle.

     Returns:
         int, 1darray: number_of_neighbors, neighbors_of_particle
     )pbdoc",
     py::arg("cutoffs").noconvert(),
     py::arg("neighbor_list_index"),
     py::arg("particle_number"));

  module.def("create", []() {
    NeighList * neighList = new NeighList;
    return std::unique_ptr<NeighList, PyNeighListDestroy>(std::move(neighList));
  }, R"pbdoc(
     Create a new NeighList object.

     Returns:
         NeighList: neighList
     )pbdoc"
  );

  // cannot bind `nbl_get_neigh_kim` directly, since it has pointer arguments
  // so we return a pointer to this function
  module.def("get_neigh_kim", []() {
    // the allowed return pointer type by pybind11 is: void const *
    // so cast the function pointer to it, and we need to cast back when
    // using it
    return (void const *) &nbl_get_neigh;
  });

  module.def("create_paddings",
             [](double const influence_distance,
                py::array_t<double> cell,
                py::array_t<int> pbc,
                py::array_t<double> coords,
                py::array_t<int> species) {
    int const natoms_1 = static_cast<int>(coords.size() / 3);
    int const natoms_2 = static_cast<int>(species.size());

    if (natoms_1 != natoms_2)
    {
      MY_WARNING("\"coords\" size and \"need_neigh\" size do not match!");
    }

    int const natoms = natoms_1 <= natoms_2 ? natoms_1 : natoms_2;
    double const * cell_data = cell.data();
    int const * pbc_data = pbc.data();
    double const * coords_data = coords.data();
    int const * species_data = species.data();

    int number_of_pads;
    std::vector<double> pad_coords;
    std::vector<int> pad_species;
    std::vector<int> pad_image;

    int error = nbl_create_paddings(natoms,
                                    influence_distance,
                                    cell_data,
                                    pbc_data,
                                    coords_data,
                                    species_data,
                                    number_of_pads,
                                    pad_coords,
                                    pad_species,
                                    pad_image);
    if (error == 1)
    {
      throw std::runtime_error(
          "In inverting the cell matrix, the determinant is 0!");
    }

    // pack as a 2D numpy array
    auto coordinates_of_paddings
        = py::array(py::buffer_info(pad_coords.data(),
                                    sizeof(double),
                                    py::format_descriptor<double>::format(),
                                    2,
                                    {number_of_pads, 3},
                                    {sizeof(double) * 3, sizeof(double)}));

    // pack as a numpy array
    auto species_code_of_paddings
        = py::array(py::buffer_info(pad_species.data(),
                                    sizeof(int),
                                    py::format_descriptor<int>::format(),
                                    1,
                                    {number_of_pads},
                                    {sizeof(int)}));

    // pack as a numpy array
    auto master_particle_of_paddings
        = py::array(py::buffer_info(pad_image.data(),
                                    sizeof(int),
                                    py::format_descriptor<int>::format(),
                                    1,
                                    {number_of_pads},
                                    {sizeof(int)}));

    py::tuple re(3);
    re[0] = coordinates_of_paddings;
    re[1] = species_code_of_paddings;
    re[2] = master_particle_of_paddings;
    return re;
  }, R"pbdoc(
     Create padding.

     Returns:
         2darray, 1darray, 1darray: coordinates_of_paddings,
             species_code_of_paddings, master_particle_of_paddings
     )pbdoc",
     py::arg("influence_distance"),
     py::arg("cell").noconvert(),
     py::arg("pbc").noconvert(),
     py::arg("coords").noconvert(),
     py::arg("species").noconvert());
}
