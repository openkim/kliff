/*
 * Contains functions to generate complete graph from a given set of
 * coordinates and species. The generated graph uses the neighbor list
 * code from the kliff.nl module internally.

 * Author: Amit Gupta
 * Email: gupta839@umn.edu

 * License: LGPL v2.1, This is distributed under the original KLIFF license.
 */

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "neighbor_list.h"
#include "periodic_table.hpp"

namespace py = pybind11;

struct GraphData
{
  /* Data structure to hold the graph data. This is used to pass data
   * between python and C++. It mirrors the data structure used by the
   * Pytorch Geometric library. By default it uses double precision for
   * floats and int64_t for integers.
   */
  GraphData() {};  // default constructor
  int n_layers;  // number of layers in the graph, i.e. influence distance/
                 // cutoff
  int n_nodes;  // number of nodes/atoms in the graph
  int idx;  // index of the configuration in the dataset
  std::vector<py::array_t<int64_t> > edge_index;  // nx2 edge index of the graph
  py::array_t<double> coords;  // coordinates of the atoms
  double energy;  // energy of the system that can be used for training, field
                  // to be assigned in python
  py::array_t<double> forces;  // forces of the system that can be used for
                               // training, field to be assigned in python
  py::array_t<int64_t> images;  // periodic images of the atoms
  py::array_t<int64_t> species;  // species index of the atoms
  py::array_t<int64_t> z;  // atomic number of the atoms
  py::array_t<double> cell;  // cell of the system
  py::array_t<int64_t>
      contributions;  // contributing of the atoms to the energy
  py::array_t<double> shifts;  // shifts of the atoms, for MIC graphs
};

// Quick and dirty
// TODO try and see ways to reduce memory allocation
// For some reason it looks like original neighbor lists
// were following singleton pattern, but it not clear why.
// Mostly because the time taken to initialize the NeigList object
// Shall be minimal as opposed to calculations. And in any case data
// is copied to python as return_value_policy was not defined explicitly.
// <*It might be useful in getting distributed graphs.*> As once calculated
// neighbors will be reused. But as on python side the here will be a live
// reference, not sure if "nodelete" is needed.

/**
 * Convert a vector of edge-set layers into a 2 × N edge-index array.
 *
 * Each element of `unrolled_graph` is a set of (src, dst) tuples that
 * describes all edges in one graph layer.  For every layer *i* the routine
 * allocates a contiguous `int64_t` array of size 2 × n_edges and stores the
 * pointer in `graph_edge_indices_out[i]`.
 **/
void graph_set_to_graph_array(
    std::vector<std::set<std::tuple<int64_t, int64_t> > > & unrolled_graph,
    int64_t ** graph_edge_indices_out)
{
  int i = 0;
  for (auto const edge_index_set : unrolled_graph)
  {
    int j = 0;
    int graph_size = static_cast<int>(edge_index_set.size());
    graph_edge_indices_out[i] = new int64_t[graph_size * 2];
    for (auto bond_pair : edge_index_set)
    {
      graph_edge_indices_out[i][j] = std::get<0>(bond_pair);
      graph_edge_indices_out[i][j + graph_size] = std::get<1>(bond_pair);
      j++;
    }
    i++;
  }
}

/**
 * Build a staged graph for parallel applications from atomic coordinates.
 *
 * The routine generates up to `n_graph_layers` neighbor shells within a
 * given `cutoff`, respecting periodic boundary conditions (`pbc`) defined
 * by the simulation `cell`.  Node species are inferred from `element_list`.
 * The function returns a fully populated `GraphData` object that can be
 * passed directly to downstream graph-based workflows.
 *
 * @param n_graph_layers  Number of staged graph layers to include.
 * @param cutoff          Pair-distance cutoff (in the same units as
 * `coords_in`).
 * @param element_list    List mapping atomic symbols to species indices.
 * @param coords_in       Nx3 array of Cartesian coordinates (float64,
 * C-contiguous).
 * @param cell            3x3 simulation cell matrix (float64).
 * @param pbc             Length-3 vector of periodic boundary flags (0 or 1).
 *
 * TODO: Move to smart ptrs/std::vectors
 *       Simplifies graph construction like TorchML driver
 *
 * @return Populated GraphData instance containing edges, coordinates,
 *         cell, species, and other topology information for all layers.
 */
GraphData get_staged_graph(
    int n_graph_layers,
    double cutoff,
    std::vector<std::string> & element_list,
    py::array_t<double, py::array::c_style | py::array::forcecast> & coords_in,
    py::array_t<double> & cell,
    py::array_t<int> & pbc)
{
  int n_atoms = element_list.size();
  double infl_dist = cutoff * n_graph_layers;
  std::vector<int> species_code;
  species_code.reserve(n_atoms);
  for (auto elem : element_list) { species_code.push_back(get_z(elem)); }

  int n_pad;
  std::vector<double> pad_coords;
  std::vector<int> pad_species;
  std::vector<int> pad_image;

  NeighList * nl;
  nbl_initialize(&nl);
  nbl_create_paddings(n_atoms,
                      infl_dist,
                      cell.data(),
                      pbc.data(),
                      coords_in.data(),
                      species_code.data(),
                      n_pad,
                      pad_coords,
                      pad_species,
                      pad_image);
  int n_coords = n_atoms * 3;
  int padded_coord_size = n_coords + n_pad * 3;
  double * padded_coords = new double[padded_coord_size];
  int * need_neighbors = new int[n_atoms + n_pad];

  auto r = coords_in.unchecked<2>();
  int pos_ij = 0;
  for (int i = 0; i < n_atoms; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      pos_ij = i * 3 + j;
      padded_coords[pos_ij] = r(i, j);
    }
    need_neighbors[i] = 1;
  }
  for (int i = 0; i < n_pad; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      pos_ij = (n_atoms + i) * 3 + j;
      padded_coords[pos_ij] = pad_coords[i * 3 + j];
    }
    need_neighbors[n_atoms + i] = 1;
  }

  // cast to const
  double * const & const_padded_coords = padded_coords;
  int * const & const_need_neighbors = need_neighbors;

  nbl_build(nl,
            n_atoms + n_pad,
            const_padded_coords,
            infl_dist,
            1,
            &cutoff,
            const_need_neighbors);

  // Build complete graph
  // TODO distributed graph generation (basically unroll the loop)
  int number_of_neighbors;
  int const * neighbors;
  int neighbor_list_index = 0;

  std::tuple<int64_t, int64_t> bond_pair, rev_bond_pair;
  std::vector<std::set<std::tuple<int64_t, int64_t> > > unrolled_graph(
      n_graph_layers);
  std::vector<int> next_list, prev_list;

  for (int atom_i = 0; atom_i < n_atoms; atom_i++)
  {
    prev_list.push_back(atom_i);
    for (int i = 0; i < n_graph_layers; i++)
    {
      // std::set<std::tuple<int, int> > conv_layer;
      if (!prev_list.empty())
      {
        // this condition is needed for edge cases where the selected
        // atom has no neighbors I dont think it will be ever
        // encountered in real problems so not sure if this is
        // necessary. TODO See if I can remove it safely.
        do {
          int curr_atom = prev_list.back();
          prev_list.pop_back();

          nbl_get_neigh(nl,
                        1,
                        &cutoff,
                        neighbor_list_index,
                        curr_atom,
                        &number_of_neighbors,
                        &neighbors);

          for (int j = 0; j < number_of_neighbors; j++)
          {
            bond_pair = std::make_tuple(curr_atom, neighbors[j]);
            rev_bond_pair = std::make_tuple(neighbors[j], curr_atom);
            unrolled_graph[i].insert(bond_pair);
            unrolled_graph[i].insert(rev_bond_pair);
            next_list.push_back(neighbors[j]);
          }
          // neighbor list pointer just points to nl object list, so
          // not needed to be freed
        } while (!prev_list.empty());
        prev_list.swap(next_list);
      }
    }
    prev_list.clear();
  }
  int64_t ** graph_edge_indices = new int64_t *[n_graph_layers];
  graph_set_to_graph_array(unrolled_graph, graph_edge_indices);

  GraphData gs;
  gs.n_layers = n_graph_layers;

  for (int i = 0; i < n_graph_layers; i++)
  {
    auto edge_index_i = py::array_t<int64_t>(
        {2, static_cast<int>(unrolled_graph[i].size())}, graph_edge_indices[i]);
    // py::array_t<int64_t>({2, unrolled_graph[i].size()},
    // graph_edge_indices[i]);
    gs.edge_index.push_back(std::move(edge_index_i));
  }

  gs.coords = py::array_t<double>({n_atoms + n_pad, 3}, padded_coords);
  species_code.reserve(pad_species.size());
  for (auto padding_species : pad_species)
  {
    species_code.push_back(padding_species);
  }
  // cast vector  to int_64
  auto species_code_64
      = std::vector<int64_t>(species_code.begin(), species_code.end());
  gs.z = py::array_t<int64_t>(species_code_64.size(), species_code_64.data());


  // species map to increasing index, from 0 to n_species
  // Count number of unique elements in species_code, and assign them 0 to n
  // numbers
  std::set<int64_t> unique_species(species_code.begin(), species_code.end());
  std::map<int64_t, int64_t> species_map;
  int64_t species_index = 0;
  for (auto species_ : unique_species)
  {
    species_map[species_] = species_index;
    species_index++;
  }
  // map species_code to species_map
  std::vector<int64_t> species_code_mapped;
  species_code_mapped.reserve(species_code.size());
  for (auto species_ : species_code)
  {
    species_code_mapped.push_back(species_map[species_]);
  }

  gs.species = py::array_t<int64_t>(species_code_mapped.size(),
                                    species_code_mapped.data());

  // Full Image vector for easier post processing (torch scatter sum)
  std::vector<int64_t> pad_image_full;
  pad_image_full.reserve(n_atoms + pad_image.size());
  for (int64_t i = 0; i < n_atoms; i++) { pad_image_full.push_back(i); }
  pad_image_full.insert(
      pad_image_full.end(), pad_image.begin(), pad_image.end());
  // implicit int -> int64_t, TODO check for potential issues
  gs.images
      = py::array_t<int64_t>(pad_image_full.size(), pad_image_full.data());

  for (int i = 0; i < n_atoms; i++) { need_neighbors[i] = 0; }
  // TODO: check if this is needed
  // convert need_neighbors to 64 bit integer vector
  std::vector<int64_t> need_neighbors_64(need_neighbors,
                                         need_neighbors + n_atoms + n_pad);
  gs.contributions
      = py::array_t<int64_t>(n_atoms + n_pad, need_neighbors_64.data());

  gs.n_nodes = n_atoms + n_pad;

  gs.cell = py::array_t<double>(cell.size(), cell.data());
  gs.shifts = py::array_t<double>({n_atoms + n_pad, 3});
  std::fill_n(
      static_cast<double *>(gs.shifts.mutable_data()), gs.shifts.size(), 0.0);

  delete[] padded_coords;
  delete[] need_neighbors;
  for (int i = 0; i < n_graph_layers; i++) { delete[] graph_edge_indices[i]; }
  delete[] graph_edge_indices;
  nbl_clean(&nl);
  return gs;
}

/**
 * Build a minimum image convention based graphs from atomic coordinates.
 *
 * The routine generates neighbor shells within a
 * given `cutoff`, respecting periodic boundary conditions (`pbc`) defined
 * by the simulation `cell`.  Node species are inferred from `element_list`.
 * The function returns a fully populated `GraphData` object that can be
 * passed directly to downstream graph-based workflows.
 *
 * @param cutoff          Pair-distance cutoff (in the same units as
 * `coords_in`).
 * @param element_list    List mapping atomic symbols to species indices.
 * @param coords_in       Nx3 array of Cartesian coordinates (float64,
 * C-contiguous).
 * @param cell            3x3 simulation cell matrix (float64).
 * @param pbc             Length-3 vector of periodic boundary flags (0 or 1).
 *
 * TODO: Move to smart ptrs/std::vectors, refactor and remove unrolled_graph
 *
 * @return Populated GraphData instance containing edges, coordinates,
 *         cell, species, and other topology information for all layers.
 */
GraphData get_mic_graph(double cutoff,
                        const std::vector<std::string> & element_list,
                        py::array_t<double> coords_in,
                        py::array_t<double> cell,
                        py::array_t<int> pbc)
{
  int n_atoms = static_cast<int>(element_list.size());
  std::vector<int> species_code;
  species_code.reserve(n_atoms);
  for (auto elem : element_list) { species_code.push_back(get_z(elem)); }

  int n_pad;
  std::vector<double> pad_coords;
  std::vector<int> pad_species;
  std::vector<int> pad_image;

  NeighList * nl;
  nbl_initialize(&nl);
  nbl_create_paddings(n_atoms,
                      cutoff,
                      cell.data(),
                      pbc.data(),
                      coords_in.data(),
                      species_code.data(),
                      n_pad,
                      pad_coords,
                      pad_species,
                      pad_image);

  auto padded_coords = std::make_unique<double[]>((n_atoms + n_pad) * 3);
  auto padded_species = std::make_unique<int[]>(n_atoms + n_pad);
  auto need_neigh = std::make_unique<int[]>(n_atoms + n_pad);
  auto padded_images = std::make_unique<int[]>(n_atoms + n_pad);

  std::copy_n(coords_in.data(), n_atoms * 3, padded_coords.get());
  std::copy_n(species_code.data(), n_atoms, padded_species.get());
  std::fill_n(need_neigh.get(), n_atoms, 1);
  std::iota(padded_images.get(), padded_images.get() + n_atoms, 0);

  for (int i = 0; i < n_pad; ++i)
  {
    std::copy_n(&pad_coords[i * 3], 3, &padded_coords[(n_atoms + i) * 3]);
    padded_species[n_atoms + i] = pad_species[i];
    need_neigh[n_atoms + i] = 0;
    padded_images[n_atoms + i] = pad_image[i];
  }

  nbl_build(nl,
            n_atoms + n_pad,
            padded_coords.get(),
            cutoff,
            1,
            &cutoff,
            need_neigh.get());

  // quick hack for reusing existing functions
  std::vector<std::set<std::tuple<int64_t, int64_t> > > unrolled_graph(1);

  for (int i = 0; i < n_atoms; ++i)
  {
    int number_of_neighbors;
    const int * neighbors;
    nbl_get_neigh(nl, 1, &cutoff, 0, i, &number_of_neighbors, &neighbors);
    for (int j = 0; j < number_of_neighbors; ++j)
    {
      unrolled_graph[0].insert({i, neighbors[j]});
      unrolled_graph[0].insert({neighbors[j], i});
    }
  }

  int64_t ** graph_edge_indices = new int64_t *[1];
  graph_set_to_graph_array(unrolled_graph, graph_edge_indices);

  int n_edges = static_cast<int>(unrolled_graph[0].size());
  auto shifts = py::array_t<double>({n_edges, 3});
  auto shifts_ptr = shifts.mutable_data();

  for (int i = 0; i < n_edges; ++i)
  {
    std::array<int64_t, 2> edge
        = {graph_edge_indices[0][i], graph_edge_indices[0][i + n_edges]};
    for (int j = 0; j < 3; ++j)
    {
      shifts_ptr[i * 3 + j] = padded_coords[edge[1] * 3 + j]
                              - padded_coords[edge[0] * 3 + j]
                              - padded_coords[padded_images[edge[1]] * 3 + j]
                              + padded_coords[padded_images[edge[0]] * 3 + j];
    }
    graph_edge_indices[0][i] = padded_images[edge[0]];
    graph_edge_indices[0][i + n_edges] = padded_images[edge[1]];
  }

  std::set<int64_t> unique_species(species_code.begin(), species_code.end());
  std::map<int64_t, int64_t> species_map;
  int64_t species_index = 0;
  for (auto species : unique_species)
  {
    species_map[species] = species_index++;
  }
  // cast vector  to int_64
  auto species_code_64
      = std::vector<int64_t>(species_code.begin(), species_code.end());

  auto species_code_mapped = py::array_t<int64_t>(n_atoms);
  auto species_code_mapped_ptr = species_code_mapped.mutable_data();
  for (int i = 0; i < n_atoms; ++i)
  {
    species_code_mapped_ptr[i] = species_map[species_code[i]];
  }

  std::vector<int64_t> contributions(n_atoms, 1);

  GraphData graph_data;
  graph_data.n_layers = 1;
  graph_data.n_nodes = n_atoms;
  graph_data.edge_index.push_back(
      py::array_t<int64_t>({2, n_edges}, graph_edge_indices[0]));
  graph_data.coords = coords_in;
  graph_data.cell = cell;
  graph_data.z = py::array_t<int64_t>(n_atoms, species_code_64.data());
  graph_data.species = species_code_mapped;
  graph_data.contributions
      = py::array_t<int64_t>(n_atoms, contributions.data());
  graph_data.shifts = shifts;


  delete[] graph_edge_indices[0];
  delete[] graph_edge_indices;

  nbl_clean(&nl);

  return graph_data;
}


PYBIND11_MODULE(graph_module, m)
{
  /**
   * @class GraphData
   * @brief Structure representing graph data compatible with PyTorch Geometric.
   *
   * GraphData stores graph-related information such as node coordinates, edges,
   * atomic numbers, energies, and forces, facilitating easy data transfer
   * between Python preprocessing and C++ computational routines.
   * Mainly meant for KLIFF graph transformations
   */
  m.doc() = "High-performance graph utilities for KLIFF";

  py::class_<GraphData>(m, "GraphData")
      .def(py::init<>(), "Default constructor")
      .def_property(
          "edge_index",
          [](const GraphData & self)
              -> const std::vector<py::array_t<int64_t> > & {
            return self.edge_index;
          },
          [](GraphData & self, const std::vector<py::array_t<int64_t> > & vec) {
            for (const auto & arr : vec)
            {
              if (arr.ndim() != 2 || arr.shape(1) != 2
                  || arr.dtype().kind() != 'i' || arr.itemsize() != 8)
              {
                throw std::runtime_error(
                    "edge_index must be a list of Nx2 int64 arrays");
              }
            }
            self.edge_index = vec;
          },
          "List of Nx2 int64 arrays representing graph edges with runtime "
          "validation.")
      .def_property(
          "coords",
          [](const GraphData & self) -> const py::array_t<double> & {
            return self.coords;
          },
          [](GraphData & self, const py::array_t<double> & arr) {
            if (arr.ndim() != 2 || arr.shape(1) != 3
                || arr.dtype().kind() != 'f' || arr.itemsize() != 8)
            {
              throw std::runtime_error(
                  "coords must be an Nx3 array of float64 values");
            }
            self.coords = arr;
          },
          "Nx3 array of atomic coordinates with runtime validation.")
      .def_readwrite("n_layers",
                     &GraphData::n_layers,
                     "Number of graph layers (influence cutoff).")
      .def_readwrite("n_nodes",
                     &GraphData::n_nodes,
                     "Number of nodes (atoms) in the graph.")
      .def_readwrite(
          "energy", &GraphData::energy, "System energy used for training.")
      .def_readwrite(
          "forces", &GraphData::forces, "Nx3 array of atomic forces.")
      .def_readwrite(
          "idx", &GraphData::idx, "Index of the graph in the dataset.")
      .def_readwrite(
          "images", &GraphData::images, "Periodic boundary images for atoms.")
      .def_readwrite(
          "species", &GraphData::species, "Species indices for atoms.")
      .def_readwrite("z", &GraphData::z, "Atomic numbers of atoms.")
      .def_readwrite(
          "cell", &GraphData::cell, "3x3 array defining the simulation cell.")
      .def_readwrite("contributions",
                     &GraphData::contributions,
                     "Atomic contributions to energy.")
      .def_readwrite(
          "shifts",
          &GraphData::shifts,
          "Shifts vectors to be subtracted from pbc coordinate vectors");
  /**
   * @brief Generates a staged graph for given configurations.
   *
   * Constructs a graph based on provided atomic coordinates, cell
   * parameters, and periodic boundary conditions. Useful for preprocessing
   * configurations for graph-based machine learning or molecular
   * simulations.
   *
   * Staged graph enable parallel graph convolutions.
   *
   * @param n_graph_layers Number of graph layers or influence distance.
   * @param cutoff Cutoff radius for interactions.
   * @param element_list List of atomic species identifiers.
   * @param coords_in Input coordinates of atoms.
   * @param cell Simulation cell dimensions (3x3).
   * @param pbc Periodic boundary conditions (boolean array of length 3).
   *
   * @return GraphData structure populated with the complete graph
   * information.
   */
  m.def("get_staged_graph",
        &get_staged_graph,
        py::arg("n_graph_layers"),
        py::arg("cutoff"),
        py::arg("element_list"),
        py::arg("coords_in"),
        py::arg("cell"),
        py::arg("pbc"),
        "Generates staged graph for given atomic configuration and "
        "parameters.");

  /**
   * @brief Generates a mic graph for given configurations.
   *
   * Constructs a graph based on provided atomic coordinates, cell parameters,
   * and periodic boundary conditions. Useful for preprocessing configurations
   * for graph-based machine learning or molecular simulations.
   *
   * Conventional graphs for popular ML libraries
   *
   * @param cutoff Cutoff radius for interactions.
   * @param element_list List of atomic species identifiers.
   * @param coords_in Input coordinates of atoms.
   * @param cell Simulation cell dimensions (3x3).
   * @param pbc Periodic boundary conditions (boolean array of length 3).
   *
   * @return GraphData structure populated with the complete graph information.
   */
  m.def("get_mic_graph",
        &get_mic_graph,
        py::arg("cutoff"),
        py::arg("element_list"),
        py::arg("coords_in"),
        py::arg("cell"),
        py::arg("pbc"),
        "Generates mic/conventional graph for given atomic configuration and "
        "parameters.");
}
