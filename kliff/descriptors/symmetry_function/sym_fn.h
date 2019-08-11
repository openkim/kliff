#ifndef DESCRIPTOR_H_
#define DESCRIPTOR_H_

#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "helper.hpp"

#define MY_PI 3.1415926535897932
#define DIM 3

// Symmetry functions taken from:

typedef double (*CutoffFunction)(double r, double rcut);
typedef double (*dCutoffFunction)(double r, double rcut);

class Descriptor
{
 public:
  Descriptor();
  ~Descriptor();

  inline void set_species(std::vector<std::string> & species)
  {
    species_ = species;
  };
  inline void get_species(std::vector<std::string> & species)
  {
    species = species_;
  };
  inline int get_num_species() { return species_.size(); };

  void
  set_cutoff(char const * name, int const Nspecies, double const * rcut_2D);
  inline double get_cutoff(int iCode, int jCode)
  {
    return rcut_2D_[iCode][jCode];
  };

  void add_descriptor(char const * name,
                      double const * values,
                      int const row,
                      int const col);
  int read_parameter_file(FILE * const filePointer);
  int get_num_descriptors();
  void set_feature_mean_and_std(bool normalize,
                                int const size,
                                double const * means,
                                double const * stds);
  inline void get_feature_mean_and_std(int i, double & mean, double & std)
  {
    mean = feature_mean_[i];
    std = feature_std_[i];
  };
  inline bool need_normalize() { return normalize_; };

  void generate_one_atom(int const i,
                         double const * coordinates,
                         int const * particleSpeciesCode,
                         int const * neighlist,
                         int const numnei,
                         double * const desc,
                         double * const grad_desc,
                         bool grad);

 private:
  std::vector<std::string> species_;
  double ** rcut_2D_;

  std::vector<std::string> name_;  // name of each descriptor
  std::vector<int> starting_index_;  // starting index of each descriptor
                                     // in generalized coords
  std::vector<double **> params_;  // params of each descriptor
  std::vector<int>
      num_param_sets_;  // number of parameter sets of each descriptor
  std::vector<int> num_params_;  // size of parameters of each descriptor
  bool has_three_body_;

  bool normalize_;  // whether to normalize the data
  std::vector<double> feature_mean_;
  std::vector<double> feature_std_;

  // symmetry functions
  void sym_g1(double r, double rcut, double & phi);
  void sym_g2(double eta, double Rs, double r, double rcut, double & phi);
  void sym_g3(double kappa, double r, double rcut, double & phi);
  void sym_g4(double zeta,
              double lambda,
              double eta,
              const double * r,
              const double * rcut,
              double & phi);
  void sym_g5(double zeta,
              double lambda,
              double eta,
              const double * r,
              const double * rcut,
              double & phi);

  void sym_d_g1(double r, double rcut, double & phi, double & dphi);
  void sym_d_g2(double eta,
                double Rs,
                double r,
                double rcut,
                double & phi,
                double & dphi);
  void
  sym_d_g3(double kappa, double r, double rcut, double & phi, double & dphi);
  void sym_d_g4(double zeta,
                double lambda,
                double eta,
                const double * r,
                const double * rcut,
                double & phi,
                double * const dphi);
  void sym_d_g5(double zeta,
                double lambda,
                double eta,
                const double * r,
                const double * rcut,
                double & phi,
                double * const dphi);

  // for debug purpose
  void echo_input()
  {
    std::cout << "=====================================" << std::endl;
    for (size_t i = 0; i < name_.size(); i++)
    {
      int rows = num_param_sets_.at(i);
      int cols = num_params_.at(i);
      std::cout << "name: " << name_.at(i) << ", rows: " << rows
                << ", cols: " << cols << std::endl;
      for (int m = 0; m < rows; m++)
      {
        for (int n = 0; n < cols; n++)
        { std::cout << params_.at(i)[m][n] << " "; }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }

    // centering and normalization
    std::cout << "centering and normalizing params" << std::endl;
    std::cout << "means:" << std::endl;
    for (size_t i = 0; i < feature_mean_.size(); i++)
    { std::cout << feature_mean_.at(i) << std::endl; }
    std::cout << "stds:" << std::endl;
    for (size_t i = 0; i < feature_std_.size(); i++)
    { std::cout << feature_std_.at(i) << std::endl; }
  }

  CutoffFunction cutoff_func_;
  dCutoffFunction d_cutoff_func_;
};

// cutoffs
inline double cut_cos(double r, double rcut)
{
  if (r < rcut)
    return 0.5 * (cos(MY_PI * r / rcut) + 1);
  else
    return 0.0;
}

inline double d_cut_cos(double r, double rcut)
{
  if (r < rcut)
    return -0.5 * MY_PI / rcut * sin(MY_PI * r / rcut);
  else
    return 0.0;
}

#endif  // DESCRIPTOR_H_
