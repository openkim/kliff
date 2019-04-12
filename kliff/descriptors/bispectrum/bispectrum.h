#ifndef BISPECTRUM_H_
#define BISPECTRUM_H_

#include <complex>

#define MY_PI 3.1415926535897932
#define DIM 3

typedef double VectorOfSizeDIM[DIM];

struct BISPECTRUM_LOOPINDICES {
  int j1, j2, j;
};

class Bispectrum {

public:
  Bispectrum(double, int, int, int, double, int, int);

  ~Bispectrum();
  void build_indexlist();
  void init();
  double memory_usage();

  int ncoeff;

  // compute bispectrum for a set of atoms
  void compute_B(double const* coordinates, int const* particleSpecies,
          int const* neighlist, int const* numneigh, int const* image,
          int const Natoms, int const Ncontrib,
          double* const zeta, double* const dzetadr);

  // cutoff
  void set_cutoff(const char* name, const int Nspecies, const double* rcuts_in);
  // element weight
  void set_weight(const int Nspecies, const double* weight_in);
  // element radius
  void set_radius(const int Nspecies, const double* radius_in);



  // functions for bispectrum coefficients

  void compute_ui(int);
  void compute_zi();
  void compute_bi();
  void copy_bi2bvec();

  // functions for derivatives

  void compute_duidrj(double*, double, double);
  void compute_dbidrj();
  void compute_dbidrj_nonsymm();
  void copy_dbi2dbvec();
  double compute_sfac(double, double);
  double compute_dsfac(double, double);

  //per bispectrum class instance for OMP use

  double* bvec, ** dbvec;
  double** rij;
  int* inside;
  double* wj;
  double* rcutij;
  int nmax;

  void grow_rij(int);

  int twojmax, diagonalstyle;
  double*** uarraytot_r, *** uarraytot_i;
  double***** zarray_r, ***** zarray_i;
  double*** uarraytot_r_b, *** uarraytot_i_b;
  double***** zarray_r_b, ***** zarray_i_b;
  double*** uarray_r, *** uarray_i;

private:

  // cutoff
  double** rcuts;
  // element weight
  double* wjelem;


  double rmin0, rfac0;

  //use indexlist instead of loops, constructor generates these
  BISPECTRUM_LOOPINDICES* idxj;
  int idxj_max;
  // data for bispectrum coefficients

  double***** cgarray;
  double** rootpqarray;
  double*** barray;

  // derivatives of data

  double**** duarray_r, **** duarray_i;
  double**** dbarray;

  static const int nmaxfactorial = 167;
  static const double nfac_table[];
  double factorial(int);

  void create_twojmax_arrays();
  void destroy_twojmax_arrays();
  void init_clebsch_gordan();
  void init_rootpqarray();
  void jtostr(char*, int);
  void mtostr(char*, int, int);
  void print_clebsch_gordan(FILE*);
  void zero_uarraytot();
  void addself_uarraytot(double);
  void add_uarraytot(double, double, double);
  void compute_uarray(double, double, double,
                      double, double);
  double deltacg(int, int, int);
  int compute_ncoeff();
  void compute_duarray(double, double, double,
                       double, double, double, double, double);

  // if number of atoms are small use per atom arrays
  // for twojmax arrays, rij, inside, bvec
  // this will increase the memory footprint considerably,
  // but allows parallel filling and reuse of these arrays
  int use_shared_arrays;

  // Sets the style for the switching function
  // 0 = none
  // 1 = cosine
  int switch_flag;

  // Self-weight
  double wself;

  int bzero_flag; // 1 if bzero subtracted from barray
  double *bzero;  // array of B values for isolated atoms
};


#endif /* BISPECTRUM_H_ */

