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
#include <string>
#include <vector>
#include <set>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "neighbor_list.h"
#include "periodic_table.hpp"

namespace py = pybind11;

struct GraphData{
    /* Data structure to hold the graph data. This is used to pass data
     * between python and C++. It mirrors the data structure used by the
     * Pytorch Geometric library. By default it uses double precision for
     * floats and int64_t for integers.
     */
    GraphData() {}; // default constructor
    int n_layers; // number of layers in the graph, i.e. influence distance/ cutoff
    int n_nodes;  // number of nodes/atoms in the graph
    int idx; // index of the configuration in the dataset
    std::vector<py::array_t<int64_t> > edge_index; // nx2 edge index of the graph
    py::array_t<double> coords; // coordinates of the atoms
    double energy; // energy of the system that can be used for training, field to be assigned in python
    py::array_t<double> forces; // forces of the system that can be used for training, field to be assigned in python
    py::array_t<int64_t> images; // periodic images of the atoms
    py::array_t<int64_t> species; // species index of the atoms
    py::array_t<int64_t> z; // atomic number of the atoms
    py::array_t<double> cell; // cell of the system
    py::array_t<int64_t> contributions; // contributing of the atoms to the energy
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

/* This function converts the graph from a set of tuples to a 2D array
 * of size 2 x n_edges.
 */
void graph_set_to_graph_array(
    std::vector<std::set<std::tuple<int64_t, int64_t> > > & unrolled_graph,
    int64_t ** graph_edge_indices_out)
    {
    int i = 0;
    for (auto const edge_index_set : unrolled_graph){
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

/* This function generates the complete graph from a given set of
 * coordinates and species and returns a GraphData object.
 */
GraphData get_complete_graph(
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
                        rev_bond_pair
                            = std::make_tuple(neighbors[j], curr_atom);
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
        auto edge_index_i = py::array_t<int64_t>({2, static_cast<int>(unrolled_graph[i].size())}, graph_edge_indices[i]);
            //py::array_t<int64_t>({2, unrolled_graph[i].size()}, graph_edge_indices[i]);
        gs.edge_index.push_back(std::move(edge_index_i));
    }

    gs.coords = py::array_t<double>({n_atoms + n_pad, 3}, padded_coords);
    species_code.reserve(pad_species.size());
    for (auto padding_species : pad_species)
    {
        species_code.push_back(padding_species);
    }
    // cast vector  to int_64
    auto species_code_64 = std::vector<int64_t>(species_code.begin(), species_code.end());
    gs.z = py::array_t<int64_t>(species_code_64.size(), species_code_64.data());


    // species map to increasing index, from 0 to n_species
    // Count number of unique elements in species_code, and assign them 0 to n numbers
    std::set<int64_t> unique_species(species_code.begin(), species_code.end());
    std::map<int64_t, int64_t> species_map;
    int64_t species_index = 0;
    for (auto species_ : unique_species){
        species_map[species_] = species_index;
        species_index++;
    }
    // map species_code to species_map
    std::vector<int64_t> species_code_mapped;
    species_code_mapped.reserve(species_code.size());
    for (auto species_ : species_code){
        species_code_mapped.push_back(species_map[species_]);
    }

    gs.species = py::array_t<int64_t>(species_code_mapped.size(), species_code_mapped.data());

    // Full Image vector for easier post processing (torch scatter sum)
    std::vector<int64_t> pad_image_full;
    pad_image_full.reserve(n_atoms + pad_image.size());
    for (int64_t i = 0; i < n_atoms; i++) { pad_image_full.push_back(i); }
    pad_image_full.insert(
        pad_image_full.end(), pad_image.begin(), pad_image.end());
    // implicit int -> int64_t, TODO check for potential issues
    gs.images = py::array_t<int64_t>(pad_image_full.size(), pad_image_full.data());

    for (int i = 0; i < n_atoms; i++) { need_neighbors[i] = 0; }
    // TODO: check if this is needed
    // convert need_neighbors to 64 bit integer vector
    std::vector<int64_t> need_neighbors_64(need_neighbors, need_neighbors + n_atoms + n_pad);
    gs.contributions
        = py::array_t<int64_t>(n_atoms + n_pad, need_neighbors_64.data());

    gs.n_nodes = n_atoms + n_pad;

    gs.cell = py::array_t<double>(cell.size(), cell.data());

    delete[] padded_coords;
    delete[] need_neighbors;
    for (int i = 0; i < n_graph_layers; i++) { delete[] graph_edge_indices[i]; }
    delete[] graph_edge_indices;
    nbl_clean(&nl);
    return gs;
}


PYBIND11_MODULE(graph_module, m)
{
    py::class_<GraphData>(m, "GraphData")
        .def(py::init<>())
        .def_readwrite("edge_index", &GraphData::edge_index)
        .def_readwrite("coords", &GraphData::coords)
        .def_readwrite("n_layers", &GraphData::n_layers)
        .def_readwrite("n_nodes", &GraphData::n_nodes)
        .def_readwrite("energy", &GraphData::energy)
        .def_readwrite("forces", &GraphData::forces)
        .def_readwrite("idx", &GraphData::idx)
        .def_readwrite("images", &GraphData::images)
        .def_readwrite("species", &GraphData::species)
        .def_readwrite("z", &GraphData::z)
        .def_readwrite("cell", &GraphData::cell)
        .def_readwrite("contributions", &GraphData::contributions);
    m.def("get_complete_graph",
            &get_complete_graph,
            py::arg("n_graph_layers"),
            py::arg("cutoff"),
            py::arg("element_list"),
            py::arg("coords_in"),
            py::arg("cell"),
            py::arg("pbc"),
            "gets complete graphs of configurations");
}
