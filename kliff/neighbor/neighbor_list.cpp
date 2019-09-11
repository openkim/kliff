// Author: Mingjian Wen (wenxx151@umn.edu)

#include "neighbor_list.h"
#include "helper.hpp"
#include <cmath>
#include <cstring>
#include <sstream>
#include <vector>

#define DIM 3
#define TOL 1.0e-10


void nbl_clean_content(NeighList * const nl)
{
  if (nl != NULL)
  {
    for (int i = 0; i < nl->numberOfNeighborLists; i++)
    {
      NeighListOne * cnl = &(nl->lists[i]);
      delete[] cnl->Nneighbors;
      delete[] cnl->neighborList;
      delete[] cnl->beginIndex;
      cnl->numberOfParticles = 0;
      cnl->cutoff = 0.0;
      cnl->Nneighbors = NULL;
      cnl->neighborList = NULL;
      cnl->beginIndex = NULL;
    }
    delete[] nl->lists;
    nl->lists = NULL;
    nl->numberOfNeighborLists = 0;
  }
}


void nbl_allocate_memory(NeighList * const nl,
                         int const numberOfCutoffs,
                         int const numberOfParticles)
{
  nl->lists = new NeighListOne[numberOfCutoffs];
  nl->numberOfNeighborLists = numberOfCutoffs;
  for (int i = 0; i < numberOfCutoffs; i++)
  {
    NeighListOne * cnl = &(nl->lists[i]);
    cnl->numberOfParticles = 0;
    cnl->cutoff = 0.0;
    cnl->Nneighbors = new int[numberOfParticles];
    cnl->neighborList = NULL;
    cnl->beginIndex = new int[numberOfParticles];
  }
}


void nbl_initialize(NeighList ** const nl)
{
  *nl = new NeighList;
  (*nl)->numberOfNeighborLists = 0;
  (*nl)->lists = NULL;
}


void nbl_clean(NeighList ** const nl)
{
  nbl_clean_content(*nl);

  delete (*nl);

  // nullify pointer
  (*nl) = NULL;
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
  double min[DIM];
  double max[DIM];

  // init max and min of coordinates to that of the first atom
  for (int k = 0; k < DIM; k++)
  {
    min[k] = coordinates[k];
    max[k] = coordinates[k] + 1.0;  // +1 to prevent max==min for 1D and 2D case
  }
  for (int i = 0; i < numberOfParticles; i++)
  {
    for (int j = 0; j < DIM; j++)
    {
      if (max[j] < coordinates[DIM * i + j])
      { max[j] = coordinates[DIM * i + j]; }
      if (min[j] > coordinates[DIM * i + j])
      { min[j] = coordinates[DIM * i + j]; }
    }
  }

  // make the cell box
  int size_total = 1;
  int size[DIM];
  for (int i = 0; i < DIM; i++)
  {
    size[i] = static_cast<int>((max[i] - min[i]) / influenceDistance);
    size[i] = size[i] <= 0 ? 1 : size[i];
    size_total *= size[i];
  }
  if (size_total > 1000000000)
  {
    MY_WARNING("Cell size too large. Check if you have partilces fly away.");
    return 1;
  }

  // assign atoms into cells
  std::vector<std::vector<int> > cells(size_total);
  for (int i = 0; i < numberOfParticles; i++)
  {
    int index[DIM];
    coords_to_index(&coordinates[DIM * i], size, max, min, index);
    int idx = index[0] + index[1] * size[0] + index[2] * size[0] * size[1];
    cells[idx].push_back(i);
  }

  // create neighbors

  // free previous neigh content and then create new
  nbl_clean_content(nl);
  nbl_allocate_memory(nl, numberOfCutoffs, numberOfParticles);

  double * cutsqs = new double[numberOfCutoffs];
  for (int i = 0; i < numberOfCutoffs; i++)
  { cutsqs[i] = cutoffs[i] * cutoffs[i]; }

  // temporary neigh container
  std::vector<int> * tmp_neigh = new std::vector<int>[numberOfCutoffs];
  int * total = new int[numberOfCutoffs];
  int * num_neigh = new int[numberOfCutoffs];
  for (int k = 0; k < numberOfCutoffs; k++) { total[k] = 0; }

  for (int i = 0; i < numberOfParticles; i++)
  {
    for (int k = 0; k < numberOfCutoffs; k++) { num_neigh[k] = 0; }

    if (needNeighbors[i])
    {
      int index[DIM];
      coords_to_index(&coordinates[DIM * i], size, max, min, index);

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
            int idx = ii + jj * size[0] + kk * size[0] * size[1];

            for (size_t m = 0; m < cells[idx].size(); m++)
            {
              int n = cells[idx][m];
              if (n != i)
              {
                double rsq = 0.0;

                for (int k = 0; k < DIM; k++)
                {
                  double del
                      = coordinates[DIM * n + k] - coordinates[DIM * i + k];
                  rsq += del * del;
                }
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

  delete[] cutsqs;
  delete[] tmp_neigh;
  delete[] num_neigh;
  delete[] total;

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
  int error = 1;
  NeighList * nl = (NeighList *) dataObject;

  if (neighborListIndex >= nl->numberOfNeighborLists) { return error; }
  NeighListOne * cnl = &(nl->lists[neighborListIndex]);

  if (cutoffs[neighborListIndex] > cnl->cutoff + TOL) { return error; }

  // invalid id
  int numberOfParticles = cnl->numberOfParticles;
  if ((particleNumber >= numberOfParticles) || (particleNumber < 0))
  {
    MY_WARNING("Invalid part ID in nbl_get_neigh");
    return error;
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

  double frac_coords[DIM * numberOfParticles];
  double min[DIM] = {1e10, 1e10, 1e10};
  double max[DIM] = {-1e10, -1e10, -1e10};
  for (int i = 0; i < numberOfParticles; i++)
  {
    const double * atom_coords = coordinates + (DIM * i);
    double x = dot(fcell, atom_coords);
    double y = dot(fcell + 3, atom_coords);
    double z = dot(fcell + 6, atom_coords);
    frac_coords[DIM * i + 0] = x;
    frac_coords[DIM * i + 1] = y;
    frac_coords[DIM * i + 2] = z;
    if (x < min[0]) { min[0] = x; }
    if (y < min[1]) { min[1] = y; }
    if (z < min[2]) { min[2] = z; }
    if (x > max[0]) { max[0] = x; }
    if (y > max[1]) { max[1] = y; }
    if (z > max[2]) { max[2] = z; }
  }

  // add some extra value to deal with edge case
  for (int i = 0; i < DIM; i++)
  {
    min[i] -= 1e-10;
    max[i] += 1e-10;
  }

  // volume of cell
  double xprod[DIM];
  cross(cell + 3, cell + 6, xprod);
  double volume = std::abs(dot(cell, xprod));

  // distance between parallelpiped cell faces
  double dist[DIM];
  cross(cell + 3, cell + 6, xprod);
  dist[0] = volume / norm(xprod);
  cross(cell + 6, cell + 0, xprod);
  dist[1] = volume / norm(xprod);
  cross(cell, cell + 3, xprod);
  dist[2] = volume / norm(xprod);

  // number of cells in each direction
  double ratio[DIM];
  double size[DIM];
  for (int i = 0; i < DIM; i++)
  {
    ratio[i] = cutoff / dist[i];
    size[i] = static_cast<int>(std::ceil(ratio[i]));
  }

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
          double x = frac_coords[DIM * at + 0];
          double y = frac_coords[DIM * at + 1];
          double z = frac_coords[DIM * at + 2];

          // select the necessary atoms to repeate for the most outside bins
          // the follwing few lines can be easily understood when assuming
          // size=1
          if (i == -size[0]
              && x - min[0] < static_cast<double>(size[0]) - ratio[0])
          { continue; }
          if (i == size[0]
              && max[0] - x < static_cast<double>(size[0]) - ratio[0])
          { continue; }
          if (j == -size[1]
              && y - min[1] < static_cast<double>(size[1]) - ratio[1])
          { continue; }
          if (j == size[1]
              && max[1] - y < static_cast<double>(size[1]) - ratio[1])
          { continue; }
          if (k == -size[2]
              && z - min[2] < static_cast<double>(size[2]) - ratio[2])
          { continue; }
          if (k == size[2]
              && max[2] - z < static_cast<double>(size[2]) - ratio[2])
          { continue; }

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

  numberOfPaddings = masterOfPaddings.size();

  return 0;
}
