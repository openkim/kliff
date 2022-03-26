// Author: Mingjian Wen (wenxx151@gmail.com)

#include "neighbor_list.h"
#include "helper.hpp"

#include <cmath>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

// WARNING: Do not use std::numeric_limits<double>::epsilon() (or even multiply
// it by 10) as TOL. It is too small and can cause numerical error. Believe me.
#define TOL 1.0e-10


void nbl_clean_content(NeighList * const nl)
{
  if (nl)
  {
    if (nl->lists)
    {
      for (int i = 0; i < nl->numberOfNeighborLists; i++)
      {
        NeighListOne * cnl = &(nl->lists[i]);
        if (cnl->Nneighbors) delete[] cnl->Nneighbors;
        if (cnl->neighborList) delete[] cnl->neighborList;
        if (cnl->beginIndex) delete[] cnl->beginIndex;
        cnl->numberOfParticles = 0;
        cnl->cutoff = 0.0;
        cnl->Nneighbors = nullptr;
        cnl->neighborList = nullptr;
        cnl->beginIndex = nullptr;
      }
      delete[] nl->lists;
    }
    nl->lists = nullptr;
    nl->numberOfNeighborLists = 0;
  }
}


void nbl_allocate_memory(NeighList * const nl,
                         int const numberOfCutoffs,
                         int const numberOfParticles)
{
  if (nl)
  {
    nl->lists = new NeighListOne[numberOfCutoffs];
    nl->numberOfNeighborLists = numberOfCutoffs;
    for (int i = 0; i < numberOfCutoffs; i++)
    {
      NeighListOne * cnl = &(nl->lists[i]);
      cnl->Nneighbors = new int[numberOfParticles];
      cnl->beginIndex = new int[numberOfParticles];
    }
  }
}


void nbl_initialize(NeighList ** const nl) { *nl = new NeighList; }


void nbl_clean(NeighList ** const nl)
{
  if (*nl)
  {
    nbl_clean_content(*nl);

    delete (*nl);
  }

  // nullify pointer
  (*nl) = nullptr;
}


int nbl_build(NeighList * const nl,
              int const numberOfParticles,
              double const * coordinates,
              double const influenceDistance,
              int const numberOfCutoffs,
              double const * cutoffs,
              int const * needNeighbors)
{
  // find max and min extend of coordinates
  double min[3];
  double max[3];

  // init max and min of coordinates to that of the first atom
  min[0] = coordinates[0];
  min[1] = coordinates[1];
  min[2] = coordinates[2];
  // +1 to prevent max==min
  max[0] = coordinates[0] + 1.0;
  max[1] = coordinates[1] + 1.0;
  max[2] = coordinates[2] + 1.0;

  for (int i = 0, l = 0; i < numberOfParticles; i++)
  {
    if (max[0] < coordinates[l]) { max[0] = coordinates[l]; }
    if (min[0] > coordinates[l]) { min[0] = coordinates[l]; }
    ++l;
    if (max[1] < coordinates[l]) { max[1] = coordinates[l]; }
    if (min[1] > coordinates[l]) { min[1] = coordinates[l]; }
    ++l;
    if (max[2] < coordinates[l]) { max[2] = coordinates[l]; }
    if (min[2] > coordinates[l]) { min[2] = coordinates[l]; }
    ++l;
  }

  // make the cell box
  int size[3];

  size[0] = static_cast<int>((max[0] - min[0]) / influenceDistance);
  size[1] = static_cast<int>((max[1] - min[1]) / influenceDistance);
  size[2] = static_cast<int>((max[2] - min[2]) / influenceDistance);

  if (size[0] <= 0) size[0] = 1;
  if (size[1] <= 0) size[1] = 1;
  if (size[2] <= 0) size[2] = 1;

  int size_total = size[0] * size[1] * size[2];
  if (size_total > 1000000000)
  {
    MY_WARNING("Cell size too large. Check if you have partilces fly away.");
    return 1;
  }

  // assign atoms into cells
  std::vector<std::vector<int> > cells(size_total);
  for (int i = 0; i < numberOfParticles; i++)
  {
    int index[3];

    coords_to_index(&coordinates[3 * i], size, max, min, index);

    int const idx
        = index[0] + index[1] * size[0] + index[2] * size[0] * size[1];

    cells[idx].push_back(i);
  }

  // create neighbors

  // free previous neigh content and then create new
  nbl_clean_content(nl);
  nbl_allocate_memory(nl, numberOfCutoffs, numberOfParticles);

  std::vector<double> cutsqs(numberOfCutoffs);
  for (int i = 0; i < numberOfCutoffs; i++)
  {
    cutsqs[i] = cutoffs[i] * cutoffs[i];
  }

  // temporary neigh container
  std::vector<std::vector<int> > tmp_neigh(numberOfCutoffs);
  std::vector<int> total(numberOfCutoffs, 0);
  std::vector<int> num_neigh(numberOfCutoffs);

  for (int i = 0; i < numberOfParticles; i++)
  {
    for (int k = 0; k < numberOfCutoffs; k++) { num_neigh[k] = 0; }

    if (needNeighbors[i])
    {
      double const coordinates_i_x = coordinates[3 * i];
      double const coordinates_i_y = coordinates[3 * i + 1];
      double const coordinates_i_z = coordinates[3 * i + 2];

      int index[3];
      coords_to_index(&coordinates[3 * i], size, max, min, index);

      // loop over neighborling cells and the cell atom i resides
      for (int ii = std::max(0, index[0] - 1);
           ii <= std::min(index[0] + 1, size[0] - 1);
           ii++)
      {
        for (int jj = std::max(0, index[1] - 1);
             jj <= std::min(index[1] + 1, size[1] - 1);
             jj++)
        {
          for (int kk = std::max(0, index[2] - 1);
               kk <= std::min(index[2] + 1, size[2] - 1);
               kk++)
          {
            int const idx = ii + jj * size[0] + kk * size[0] * size[1];

            for (std::size_t m = 0; m < cells[idx].size(); m++)
            {
              int n = cells[idx][m];
              if (n != i)
              {
                double const dx = coordinates[3 * n] - coordinates_i_x;
                double const dy = coordinates[3 * n + 1] - coordinates_i_y;
                double const dz = coordinates[3 * n + 2] - coordinates_i_z;
                double const rsq = dx * dx + dy * dy + dz * dz;

                if (rsq < TOL)
                {
                  std::ostringstream stringStream;
                  stringStream << "Collision of atoms " << i + 1 << " and "
                               << n + 1 << ". ";
                  stringStream << "Their distance is " << std::sqrt(rsq) << "."
                               << std::endl;
                  std::string my_str = stringStream.str();
                  MY_WARNING(my_str);
                  return 1;
                }
                for (int k = 0; k < numberOfCutoffs; k++)
                {
                  if (rsq < cutsqs[k])
                  {
                    tmp_neigh[k].push_back(n);
                    num_neigh[k]++;
                  }
                }
              }
            }
          }
        }
      }
    }

    for (int k = 0; k < numberOfCutoffs; k++)
    {
      nl->lists[k].Nneighbors[i] = num_neigh[k];
      nl->lists[k].beginIndex[i] = total[k];
      total[k] += num_neigh[k];
    }
  }

  for (int k = 0; k < numberOfCutoffs; k++)
  {
    nl->lists[k].numberOfParticles = numberOfParticles;
    nl->lists[k].cutoff = cutoffs[k];
    nl->lists[k].neighborList = new int[total[k]];
    std::memcpy(
        nl->lists[k].neighborList, tmp_neigh[k].data(), sizeof(int) * total[k]);
  }

  return 0;
}


int nbl_get_neigh(void const * const dataObject,
                  int const numberOfCutoffs,
                  double const * const cutoffs,
                  int const neighborListIndex,
                  int const particleNumber,
                  int * const numberOfNeighbors,
                  int const ** const neighborsOfParticle)
{
  NeighList * nl = (NeighList *) dataObject;

  if (neighborListIndex >= nl->numberOfNeighborLists) { return 1; }

  NeighListOne * cnl = &(nl->lists[neighborListIndex]);

  if (cutoffs[neighborListIndex] > cnl->cutoff + TOL) { return 1; }

  // invalid id
  int numberOfParticles = cnl->numberOfParticles;

  if ((particleNumber >= numberOfParticles) || (particleNumber < 0))
  {
    MY_WARNING("Invalid part ID in nbl_get_neigh");
    return 1;
  }

  // number of neighbors
  *numberOfNeighbors = cnl->Nneighbors[particleNumber];

  // neighbor list starting point
  int idx = cnl->beginIndex[particleNumber];

  *neighborsOfParticle = cnl->neighborList + idx;

  return 0;
}


int nbl_create_paddings(int const numberOfParticles,
                        double const cutoff,
                        double const * cell,
                        int const * PBC,
                        double const * coordinates,
                        int const * speciesCode,
                        int & numberOfPaddings,
                        std::vector<double> & coordinatesOfPaddings,
                        std::vector<int> & speciesCodeOfPaddings,
                        std::vector<int> & masterOfPaddings)
{
  // transform coordinates into fractional coordinates
  double tcell[9];
  double fcell[9];

  transpose(cell, tcell);

  int error = inverse(tcell, fcell);
  if (error) { return error; }

  double frac_coords[3 * numberOfParticles];

  double min[3] = {1e10, 1e10, 1e10};
  double max[3] = {-1e10, -1e10, -1e10};

  for (int i = 0; i < numberOfParticles; i++)
  {
    const double * atom_coords = coordinates + (3 * i);

    double x = dot(fcell, atom_coords);
    double y = dot(fcell + 3, atom_coords);
    double z = dot(fcell + 6, atom_coords);

    frac_coords[3 * i + 0] = x;
    frac_coords[3 * i + 1] = y;
    frac_coords[3 * i + 2] = z;

    if (x < min[0]) { min[0] = x; }
    if (y < min[1]) { min[1] = y; }
    if (z < min[2]) { min[2] = z; }
    if (x > max[0]) { max[0] = x; }
    if (y > max[1]) { max[1] = y; }
    if (z > max[2]) { max[2] = z; }
  }

  // add some extra value to deal with edge case
  min[0] -= TOL;
  min[1] -= TOL;
  min[2] -= TOL;
  max[0] += TOL;
  max[1] += TOL;
  max[2] += TOL;

  // volume of cell
  double xprod[3];

  cross(cell + 3, cell + 6, xprod);

  double volume = std::abs(dot(cell, xprod));

  // distance between parallelpiped cell faces
  double dist[3];

  cross(cell + 3, cell + 6, xprod);
  dist[0] = volume / norm(xprod);

  cross(cell + 6, cell + 0, xprod);
  dist[1] = volume / norm(xprod);

  cross(cell, cell + 3, xprod);
  dist[2] = volume / norm(xprod);

  // number of cells in each direction
  double const ratio[3]
      = {cutoff / dist[0], cutoff / dist[1], cutoff / dist[2]};

  int const size[3] = {static_cast<int>(std::ceil(ratio[0])),
                       static_cast<int>(std::ceil(ratio[1])),
                       static_cast<int>(std::ceil(ratio[2]))};

  double const size_ratio_diff_x = static_cast<double>(size[0]) - ratio[0];
  double const size_ratio_diff_y = static_cast<double>(size[1]) - ratio[1];
  double const size_ratio_diff_z = static_cast<double>(size[2]) - ratio[2];


  // creating padding atoms
  for (int i = -size[0]; i <= size[0]; i++)
  {
    for (int j = -size[1]; j <= size[1]; j++)
    {
      for (int k = -size[2]; k <= size[2]; k++)
      {
        // skip contributing atoms
        if (i == 0 && j == 0 && k == 0) { continue; }

        // apply BC
        if (PBC[0] == 0 && i != 0) { continue; }
        if (PBC[1] == 0 && j != 0) { continue; }
        if (PBC[2] == 0 && k != 0) { continue; }

        for (int at = 0; at < numberOfParticles; at++)
        {
          double const x = frac_coords[3 * at + 0];
          double const y = frac_coords[3 * at + 1];
          double const z = frac_coords[3 * at + 2];

          // select the necessary atoms to repeate for the most outside bins
          // the follwing few lines can be easily understood when assuming
          // size=1
          if (i == -size[0] && x - min[0] < size_ratio_diff_x) { continue; }
          if (i == size[0] && max[0] - x < size_ratio_diff_x) { continue; }
          if (j == -size[1] && y - min[1] < size_ratio_diff_y) { continue; }
          if (j == size[1] && max[1] - y < size_ratio_diff_y) { continue; }
          if (k == -size[2] && z - min[2] < size_ratio_diff_z) { continue; }
          if (k == size[2] && max[2] - z < size_ratio_diff_z) { continue; }

          // fractional coordinates of padding atom at
          double atom_coords[3] = {i + x, j + y, k + z};

          // absolute coordinates of padding atoms
          coordinatesOfPaddings.push_back(dot(tcell, atom_coords));
          coordinatesOfPaddings.push_back(dot(tcell + 3, atom_coords));
          coordinatesOfPaddings.push_back(dot(tcell + 6, atom_coords));

          // padding speciesCode code and image
          speciesCodeOfPaddings.push_back(speciesCode[at]);
          masterOfPaddings.push_back(at);
        }
      }
    }
  }

  numberOfPaddings = static_cast<int>(masterOfPaddings.size());

  return 0;
}
