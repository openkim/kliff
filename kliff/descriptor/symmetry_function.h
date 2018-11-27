#ifndef DESCRIPTOR_H_
#define DESCRIPTOR_H_

#include <cmath>
#include <string>
#include <cstring>
#include <vector>
#include <iostream>

#define MY_PI 3.1415926535897932

// Symmetry functions taken from:


typedef double (*CutoffFunction)(double r, double rcut);
typedef double (*dCutoffFunction)(double r, double rcut);

class Descriptor
{
  public:
    Descriptor();
		~Descriptor();

		void set_cutoff(const char* name, const int Nspecies, const double* rcuts);
		void add_descriptor(const char* name, const double* values,
        const int row, const int col);
    int get_num_descriptors();
    void get_generalized_coords(const double* coords, const int* species_code,
        const int* neighlist, const int* numneigh, const int* image,
        const int Natoms, const int Ncontrib, const int Ndescriptor,
        double* const gen_coords, double* const d_gen_coords = nullptr);


	private:
		std::vector<std::string> name_;    // name of each descriptor
		std::vector<int> starting_index_;  // starting index of each descriptor
                                      // in generalized coords
		std::vector<double**> params_;     // params of each descriptor
		std::vector<int> num_param_sets_;  // number of parameter sets of each descriptor
		std::vector<int> num_params_;      // size of parameters of each descriptor
    bool has_three_body_;
    double** rcuts_;
    double Nspecies_;

		// symmetry functions
    void sym_g1(double r, double rcut, double &phi);
    void sym_g2(double eta, double Rs, double r, double rcut, double &phi);
    void sym_g3(double kappa, double r, double rcut, double &phi);
    void sym_g4(double zeta, double lambda, double eta,
        const double* r, const double* rcut, double &phi);
    void sym_g5(double zeta, double lambda, double eta,
        const double* r, const double* rcut, double &phi);

    void sym_d_g1(double r, double rcut, double &phi, double &dphi);
    void sym_d_g2(double eta, double Rs, double r, double rcut, double &phi,
        double &dphi);
    void sym_d_g3(double kappa, double r, double rcut, double &phi, double &dphi);
    void sym_d_g4(double zeta, double lambda, double eta,
        const double* r, const double* rcut, double &phi, double* const dphi);
    void sym_d_g5(double zeta, double lambda, double eta,
        const double* r, const double* rcut, double &phi, double* const dphi);


//TODO delete; for debug purpose
    void echo_input() {
      std::cout<<"====================================="<<std::endl;
      for (size_t i=0; i<name_.size(); i++) {
        int rows = num_param_sets_.at(i);
        int cols = num_params_.at(i);
        std::cout<<"name: "<<name_.at(i)<<", rows: "<<rows<<", cols: "<<cols<<std::endl;
        for (int m=0; m<rows; m++) {
          for (int n=0; n<cols; n++) {
            std::cout<<params_.at(i)[m][n]<< " ";
          }
          std::cout<<std::endl;
        }
        std::cout<<std::endl;
      }
    }


		CutoffFunction cutoff_;
		dCutoffFunction d_cutoff_;
};


// cutoffs
inline double cut_cos(double r, double rcut) {
	if (r < rcut)
		return 0.5 * (cos(MY_PI*r/rcut) + 1);
	else
		return 0.0;
}

inline double d_cut_cos(double r, double rcut) {
	if (r < rcut)
		return -0.5*MY_PI/rcut * sin(MY_PI*r/rcut);
	else
		return 0.0;
}


//TODO correct it
inline double cut_exp(double r, double rcut) {
	if (r < rcut)
		return 1;
	else
		return 0.0;
}

inline double d_cut_exp(double r, double rcut) {
	if (r < rcut)
		return 0.0;
	else
		return 0.0;
}


// helper
void AllocateAndInitialize2DArray(double**& arrayPtr, int const extentZero,
    int const extentOne);

void Deallocate2DArray(double**& arrayPtr);


#endif // DESCRIPTOR_H_


