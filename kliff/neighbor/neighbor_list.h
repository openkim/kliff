// Author: Mingjian Wen (wenxx151@umn.edu)

#ifndef NEIGHBOR_LIST_H_
#define NEIGHBOR_LIST_H_

#include <vector>

typedef struct
{
  int numberOfParticles;
  double cutoff;
  int * Nneighbors;
  int * neighborList;
  int * beginIndex;
} NeighListOne;

// neighbor list structure
typedef struct
{
  int numberOfNeighborLists;
  NeighListOne * lists;
} NeighList;


void nbl_initialize(NeighList ** const nl);

int nbl_create_paddings(int const numberOfParticles,
                        double const cutoff,
                        double const * cell,
                        int const * PBC,
                        double const * coordinates,
                        int const * speciesCode,
                        int & numberOfPaddings,
                        std::vector<double> & coordinatesOfPaddings,
                        std::vector<int> & speciesCodeOfPaddings,
                        std::vector<int> & masterOfPaddings);

int nbl_build(NeighList * const nl,
              int const numberOfParticles,
              double const * coordinates,
              double const influenceDistance,
              int const numberOfCutoffs,
              double const * cutoffs,
              int const * needNeighbors);

int nbl_get_neigh(void const * const nl,
                  int const numberOfCutoffs,
                  double const * const cutoffs,
                  int const neighborListIndex,
                  int const particleNumber,
                  int * const numberOfNeighbors,
                  int const ** const neighborsOfParticle);

void nbl_clean(NeighList ** const nl);


#endif
