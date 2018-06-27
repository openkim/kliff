#ifndef LAYERS_H_
#define LAYERS_H_

int create_layers(const int Natoms, const double* coordinates,
    const int* neighlist, const int* numneigh, const double rcut_layer,
    std::vector<int>& in_layer);

#endif // LAYERS_H_
