#ifndef KLIFF_BISPECTRUM_HPP_
#define KLIFF_BISPECTRUM_HPP_

#include "helper.hpp"

/*! \class BISPECTRUM_LOOPINDICES
 * \brief The structure for the Bispectrum loop indices
 *
 */
struct BISPECTRUM_LOOPINDICES
{
  int j1;
  int j2;
  int j;
};

/*!
 * \brief

 * This implementation is based on the method outlined
 * in Bartok[1], using formulae from VMK[2].
 *
 * For the Clebsch-Gordan coefficients, we convert the VMK half-integral
 * labels a, b, c, alpha, beta, gamma to array offsets j1, j2, j, m1, m2, m
 * using the following relations:
 *
 * j1 = 2*a
 * j2 = 2*b
 * j =  2*c
 *
 * m1 = alpha+a      2*alpha = 2*m1 - j1
 * m2 = beta+b    or 2*beta = 2*m2 - j2
 * m =  gamma+c      2*gamma = 2*m - j
 *
 * in this way:
 *
 * -a <= alpha <= a
 * -b <= beta <= b
 * -c <= gamma <= c
 *
 * becomes:
 *
 * 0 <= m1 <= j1
 * 0 <= m2 <= j2
 * 0 <= m <= j
 *
 * and the requirement that
 * a+b+c be integral implies that
 * j1+j2+j must be even.
 * The requirement that:
 *
 * gamma = alpha+beta
 *
 * becomes:
 *
 * 2*m - j = 2*m1 - j1 + 2*m2 - j2
 *
 * Similarly, for the Wigner U-functions U(J,m,m') we
 * convert the half-integral labels J,m,m' to
 * array offsets j,ma,mb:
 *
 * j = 2*J
 * ma = J+m
 * mb = J+m'
 *
 * so that:
 *
 * 0 <= j <= 2*Jmax
 * 0 <= ma, mb <= j.
 *
 * For the bispectrum components B(J1,J2,J) we convert to:
 *
 * j1 = 2*J1
 * j2 = 2*J2
 * j = 2*J
 *
 * and the requirement:
 *
 * |J1-J2| <= J <= J1+J2, for j1+j2+j integral
 *
 * becomes:
 *
 * |j1-j2| <= j <= j1+j2, for j1+j2+j even integer
 *
 * or
 *
 * j = |j1-j2|, |j1-j2|+2,...,j1+j2-2,j1+j2
 *
 * [1] Albert Bartok-Partay, "Gaussian Approximation..."
 * Doctoral Thesis, Cambrindge University, (2009)

 * [2] D. A. Varshalovich, A. N. Moskalev, and V. K. Khersonskii,
 * "Quantum Theory of Angular Momentum," World Scientific (1988)
 *
 */
class Bispectrum
{
 public:
  /*!
   * \brief Construct a new Bispectrum object
   *
   * \param rfac0_in
   * \param twojmax_in
   * \param diagonalstyle_in
   * \param use_shared_arrays_in
   * \param rmin0_in
   * \param switch_flag_in
   * \param bzero_flag_in
   */
  Bispectrum(double const rfac0_in,
             int const twojmax_in,
             int const diagonalstyle_in,
             int const use_shared_arrays_in,
             double const rmin0_in,
             int const switch_flag_in,
             int const bzero_flag_in);

  /*!
   * \brief Destroy the Bispectrum object
   *
   */
  ~Bispectrum();

  /*!
   * \brief Craete an index list based on the input index style.
   *
   *
   */
  void build_indexlist();

  /*!
   * \brief
   *
   */
  void init();

  /*!
   * \brief Compute memory usage of arrays
   *
   * \return double
   */
  double memory_usage();

  /*!
   * \brief Computes bispectrum for a set of atoms
   *
   * For example eq(5) of ``Gaussian Approximation Potentials: The Accuracy of
   * Quantum Mechanics, without the Electrons``, by Gabor Csany
   *
   * \param coordinates
   * \param particleSpecies
   * \param neighlist
   * \param numneigh
   * \param image
   * \param Natoms
   * \param Ncontrib
   * \param zeta
   * \param dzetadr
   */
  void compute_B(double const * coordinates,
                 int const * particleSpecies,
                 int const * neighlist,
                 int const * numneigh,
                 int const * image,
                 int const Natoms,
                 int const Ncontrib,
                 double * const zeta,
                 double * const dzetadr);

  /*!
   * \brief Set the cutoff
   *
   * \param name
   * \param Nspecies
   * \param rcuts_in
   */
  void
  set_cutoff(char * name, std::size_t const Nspecies, double const * rcuts_in);

  /*!
   * \brief Set the element weight
   *
   * \param Nspecies
   * \param weight_in
   */
  void set_weight(int const Nspecies, double const * weight_in);

  /*!
   * \brief Set the element radius
   *
   * \param Nspecies
   * \param radius_in
   */
  void set_radius(int const Nspecies, double const * radius_in);

  /*!
   * \brief functions for bispectrum coefficients
   *
   * Compute Ui by summing over neighbors j
   *
   * \param jnum Number of neighbors j
   */
  void compute_ui(int const jnum);

  /*!
   * \brief
   *
   * Compute Zi by summing over products of Ui
   */
  void compute_zi();

  /*!
   * \brief
   *
   * Compute Bi by summing `conj(Ui)*Zi`
   */
  void compute_bi();

  /*!
   * \brief Copy Bi array to a vector
   *
   */
  void copy_bi2bvec();

  /*!
   * \brief Calculate derivative of Ui w.r.t. atom j
   *
   * \param rij_in
   * \param wj_in
   * \param rcut_in
   */
  void compute_duidrj(double const * rij_in,
                      double const wj_in,
                      double const rcut_in);

  /*!
   * \brief Calculate derivative of Bi w.r.t. atom j
   * variant using indexlist for j1,j2,j
   * variant using symmetry relation
   *
   */
  void compute_dbidrj();

  /*!
   * \brief Calculate derivative of Bi w.r.t. atom j
   * variant using indexlist for j1,j2,j
   * variant not using symmetry relation
   *
   */
  void compute_dbidrj_nonsymm();

  /*!
   * \brief Copy Bi derivatives into a vector
   *
   */
  void copy_dbi2dbvec();

  /*!
   * \brief
   *
   * \param r
   * \param rcut_in
   *
   * \return double
   */
  double compute_sfac(double const r, double const rcut_in);

  /*!
   * \brief
   *
   * \param r
   * \param rcut_in
   *
   * \return double
   */
  double compute_dsfac(double const r, double const rcut_in);

  /*!
   * \brief
   *
   * \param newnmax New maximum number of neigbors
   */
  void grow_rij(int const newnmax);

 private:
  /*!  */

  /*!
   * \brief Function to find the factorial of a positive integer number
   *
   * \param n A positive integer number greater than 0 and less than
   * nmaxfactorial \sa @nmaxfactorial
   *
   * \return double Factorial of a positive integer number
   */
  inline double factorial(int const n);

  /*!
   * \brief Create a twojmax arrays object
   *
   */
  void create_twojmax_arrays();

  /*!
   * \brief Assign Clebsch-Gordan coefficients using
   * the quasi-binomial formula VMK 8.2.1(3)
   *
   */
  void init_clebsch_gordan();

  /*!
   * \brief Pre-compute table of `sqrt[p/m2]`, `p, q = 1,twojmax`
   * the `p = 0, q = 0` entries are allocated and skipped for convenience.
   *
   */
  void init_rootpqarray();

  /*!
   * \brief Composes a string = `j/2`
   *
   * \param str_out A string
   * \param j
   */
  inline void jtostr(char * str_out, int const j);

  /*!
   * \brief Composes a string = `m - j/2`
   *
   * \param str_out A string
   * \param j
   * \param m
   */
  inline void mtostr(char * str_out, int const j, int const m);

  /*!
   * \brief Write the list values of Clebsch-Gordan coefficients using notation
   * of VMK Table 8.11
   *
   * \param file File name
   */
  void print_clebsch_gordan(FILE * file);

  /*!
   * \brief
   *
   */
  void zero_uarraytot();

  /*!
   * \brief
   *
   * \param wself_in Input self weight
   */
  void addself_uarraytot(double const wself_in);

  /*!
   * \brief Add Wigner U-functions for one neighbor to the total
   *
   * \param r
   * \param wj_in
   * \param rcut_in
   */
  void add_uarraytot(double const r, double const wj_in, double const rcut_in);

  /*!
   * \brief Compute Wigner U-functions for one neighbor
   *
   * \param x X-coordinate
   * \param y Y-coordinate
   * \param z Z-coordinate
   * \param z0
   * \param r
   */
  void compute_uarray(double const x,
                      double const y,
                      double const z,
                      double const z0,
                      double const r);

  /*!
   * \brief The delta function given by VMK Eq. 8.2(1)
   *
   * \param j1
   * \param j2
   * \param j
   *
   * \return double
   */
  inline double deltacg(int const j1, int const j2, int const j);

  /*!
   * \brief
   *
   */
  int compute_ncoeff();

  /*!
   * \brief Compute derivatives of Wigner U-functions for one neighbor
   * \sa compute_uarray
   *
   * \param x X-coordinate
   * \param y Y-coordinate
   * \param z Z-coordinate
   * \param z0
   * \param r
   * \param dz0dr
   * \param wj_in
   * \param rcut_in
   */
  void compute_duarray(double const x,
                       double const y,
                       double const z,
                       double const z0,
                       double const r,
                       double const dz0dr,
                       double const wj_in,
                       double const rcut_in);

 public:
  /*! */
  int ncoeff;

  // per bispectrum class instance for OMP use

  /*! */
  std::vector<double> bvec;

  /*! */
  Array2D<double> dbvec;

  /*! */
  Array2D<double> rij;

  /*! */
  std::vector<int> inside;

  /*! */
  std::vector<double> wj;

  /*! */
  std::vector<double> rcutij;

  /*! */
  int nmax;

  /*! */
  int twojmax;

  /*! */
  int diagonalstyle;

  /*! */
  Array3D<double> uarraytot_r;
  Array3D<double> uarraytot_i;

  /*! */
  Array5D<double> zarray_r;
  Array5D<double> zarray_i;

  /*! */
  Array3D<double> uarraytot_r_b;
  Array3D<double> uarraytot_i_b;

  /*! */
  Array5D<double> zarray_r_b;
  Array5D<double> zarray_i_b;

  /*! */
  Array3D<double> uarray_r;
  Array3D<double> uarray_i;

 private:
  /*! Cutoff radius */
  Array2D<double> rcuts;

  /*! Element weight */
  std::vector<double> wjelem;

  /*!  */
  double rmin0;

  /*!  */
  double rfac0;

  /*! use indexlist instead of loops, constructor generates these*/
  std::vector<BISPECTRUM_LOOPINDICES> idxj;

  /*!  */
  int idxj_max;

  // data for bispectrum coefficients

  /*!  */
  Array5D<double> cgarray;

  /*!  */
  Array2D<double> rootpqarray;

  /*!  */
  Array3D<double> barray;

  // derivatives of data

  /*!  */
  Array4D<double> duarray_r;
  Array4D<double> duarray_i;

  /*!  */
  Array4D<double> dbarray;

  /*! Maximum value of an integer for which factorial can be calculated. */
  static const int nmaxfactorial = 167;

  /*! Factorial n table, `size Bispectrum::nmaxfactorial+1` */
  static double const nfac_table[];

  /*!
   * \brief
   * if number of atoms are small use per atom arrays
   * for twojmax arrays, rij, inside, bvec
   * this will increase the memory footprint considerably,
   * but allows parallel filling and reuse of these arrays
   */
  int use_shared_arrays;

  /*!
   * \brief Sets the style for the switching function
   * 0 = none
   * 1 = cosine
   */
  int switch_flag;

  /*! Self-weight */
  double wself;

  /*! 1 if bzero subtracted from barray */
  int bzero_flag;

  /*! array of B values for isolated atoms */
  std::vector<double> bzero;
};

#endif  // KLIFF_BISPECTRUM_HPP_
