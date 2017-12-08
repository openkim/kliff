#include <vector>
#include <iostream>
#include <algorithm>
#include "layers.h"

#define DIM 3

typedef double vectorOfSizeDIM[DIM];


// create layers
// To assign atoms into different layers. If `ruct_layer < 0', it will be
// determined internally within the code by finding the max of the min of pair
// distance between eatch atom and its neighbors. This is a bit more expensive
// since it runs through the neighborlist once.
//

int create_layers(const int Natoms, const double* coordinates,
    const int* neighlist, const int* numneigh, const double rcut_layer,
    std::vector<int>& in_layer)
{
  // convenient access to coords
  const vectorOfSizeDIM* coords = (vectorOfSizeDIM*) coordinates;


  // neighbor list
  std::vector<int> first_neigh(Natoms);  // first neigh index of each atom in neighbor list
  first_neigh[0] = 0;
  for (int i=1; i<Natoms; i++) {
    first_neigh[i] = first_neigh[i-1] + numneigh[i-1];
  }

  // cutoff used to find atoms in the same layer

  double cutsq_layer;
  if (rcut_layer > 0) {
    cutsq_layer = rcut_layer * rcut_layer;
  }
  else {   // max of min of pair distance is rcut_layer

    std::vector<double> min_rsq(Natoms, 1e10);

    for (int i=0; i<Natoms; i++) {
      int num_neigh = numneigh[i];
      const int* ilist = neighlist + first_neigh[i];

      for (int jj=0; jj<num_neigh; ++jj) {
        int j = ilist[jj];

        // squared distance of rij
        double delx = coords[j][0]-coords[i][0];
        double dely = coords[j][1]-coords[i][1];
        double delz = coords[j][2]-coords[i][2];
        double rsq = delx*delx + dely*dely + delz*delz;

        if (rsq < min_rsq[i]) {
          min_rsq[i] = rsq;
        }
      }
    }

    double max_min_rsq = *(std::max_element(min_rsq.begin(), min_rsq.end()));
    cutsq_layer = max_min_rsq * 1.01;  // *1.01 to get over edge case
  }
  // we can set it manually to speed up
  // cutsq_layer = (0.72*3.35)*(0.72*3.35);


  // layers
  int nremain; // number of atoms not included in any layer
  int nlayers;

  // init vars
  nlayers = 1;
  nremain = Natoms;
  in_layer.assign(Natoms, -1); // -1 means atoms not in any layer

  // create all layers
  while(true) {

    // current layer contains which atoms (init to -1 indicating no atom)
    std::vector<int> layer(nremain, -1);

    // find an atom not incldued in any layer and start with it
    int currentLayer = nlayers - 1;
    for (int k=0; k<Natoms; k++) {
      if (in_layer[k] == -1) {
        in_layer[k] = currentLayer;
        layer[0] = k; // first atom in current layer
        break;
      }
    }

    int nin = 1; // number of atoms in current layer
    int ii = 0;  // index of atoms in current layer

    while(true) { // find all atoms in currentLayer

      int i = layer[ii];

      // get neighbors of atom i
      int num_neigh = numneigh[i];
      const int* ilist = neighlist + first_neigh[i];

      // loop over the neighbors of atom i
      for (int jj=0; jj<num_neigh; ++jj) {

        int j = ilist[jj];

        // squared distance of rij
        double delx = coords[j][0]-coords[i][0];
        double dely = coords[j][1]-coords[i][1];
        double delz = coords[j][2]-coords[i][2];
        double rsq = delx*delx + dely*dely + delz*delz;

        // should be included in current layer
        if (rsq < cutsq_layer) {

          if (in_layer[j] == -1) { // has not been included in some layer
            nin += 1;
            layer[nin-1] = j;
            in_layer[j] = currentLayer;
          }
          else {
            // in a layer but not the current layer, should not happen provided the
            // choice of cutsq_layer is appropriate
            if(in_layer[j] != currentLayer){
              std::cerr <<"ERROR: attempting to include atom " <<j <<"in layer "
                  <<currentLayer <<", but it is already in layer " <<in_layer[j]
                  <<"." <<std::endl;
              return -1;
            }
          }
        }

      } // loop on jj


      // get to the next atom in current layer
      ii++;
      if (ii == nin) break;

    } // finding atoms in one layer

    nremain -= nin;
    if (nremain == 0) break;
    nlayers += 1;
  } // finding atoms in all layers


  //TODO delete debug
/*  std::cout<<"Cutoff for layer "<<sqrt(cutsq_layer)<<std::endl;
  std::cout <<"Number of layers: " <<nlayers <<std::endl;
  std::cout <<"#atom id     layer"<<std::endl;
  for (int i=0; i<Natoms; i++) {
    std::cout <<i <<"         "<<in_layer[i]<<std::endl;
  }
*/

  return 0;
}


