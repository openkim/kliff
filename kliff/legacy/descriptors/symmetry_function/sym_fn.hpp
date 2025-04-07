#ifndef KLIFF_DESCRIPTOR_HPP_
#define KLIFF_DESCRIPTOR_HPP_

#include "helper.hpp"

#include <numeric>

#include <cmath>
#include <cstring>
#include <map>
#include <string>

#ifdef DIM
#undef DIM
#endif

#define DIM 3

typedef double VectorOfSizeDIM[DIM];

/*!
 * \brief Cutoff function
 *
 * \param r The distance between atoms \c i and \c j
 * \param rcut The cutoff radius
 *
 * \return double The cutoff function value at \c r
 */
inline double cut_cos(double const r, double const rcut);

/*!
 * \brief Derivative of cutoff function
 *
 * \param r The distance between atoms \c i and \c j
 * \param rcut The cutoff radius
 *
 * \return double The derivative of cutoff function value at \c r
 */
inline double d_cut_cos(double const r, double const rcut);

/*!
 * \brief A polymorphic function wrapper type for the symmetry functions
 */
using CutoffFunction = double (*)(double, double);

/*!
 * \brief A polymorphic function wrapper type for the symmetry functions
 */
using dCutoffFunction = double (*)(double, double);

/*! \class Descriptor
 *
 * \brief The descriptor to count for atom-centered symmetry functions for
 * constructing high-dimensional neural network potentials
 *
 *
 *
 */
class Descriptor
{
 public:
  /*!
   * \brief Construct a new Descriptor object
   *
   */
  Descriptor();

  /*!
   * \brief Destroy the Descriptor object
   *
   */
  ~Descriptor();

  /*!
   * \brief Set the species
   *
   * \param species
   */
  inline void set_species(std::vector<std::string> & species);

  /*!
   * \brief Get the species
   *
   * \param species
   */
  inline void get_species(std::vector<std::string> & species);

  /*!
   * \brief Get the num species
   *
   * \return int
   */
  inline int get_num_species();

  /*!
   * \brief Set the cutoff
   *
   * \param name
   * \param Nspecies
   * \param rcut_2D
   */
  void set_cutoff(char const * name,
                  std::size_t const Nspecies,
                  double const * rcut_2D);

  /*!
   * \brief Get the cutoff object
   *
   * \param iCode
   * \param jCode
   * \return double
   */
  inline double get_cutoff(int const iCode, int const jCode);

  /*!
   * \brief
   *
   * \param name
   * \param values
   * \param row
   * \param col
   */
  void add_descriptor(char const * name,
                      double const * values,
                      int const row,
                      int const col);

  /*!
   * \brief
   *
   * \param filePointer
   * \return int
   */
  int read_parameter_file(FILE * const filePointer);

  /*!
   * \brief Get the num descriptors
   *
   * \return int
   */
  int get_num_descriptors();

  /*!
   * \brief Set the feature mean and std deviation
   *
   * \param normalize
   * \param size
   * \param means
   * \param stds
   */
  void set_feature_mean_and_std(bool const normalize,
                                int const size,
                                double const * means,
                                double const * stds);

  /*!
   * \brief Get the feature mean and std deviation
   *
   * \param i
   * \param mean
   * \param std
   */
  inline void
  get_feature_mean_and_std(int const i, double & mean, double & std);

  /*!
   * \brief
   *
   * \return true
   * \return false
   */
  inline bool need_normalize();

  /*!
   * \brief Compute the descriptor values and their derivatives w.r.t. the
   * coordinates of atom i and its neighbors.
   *
   * \param i Index of atom \c i
   * \param coordinates Coordinates of all the atoms
   * \param particleSpeciesCode Index number (code) of the particle species
   * \param neighlist Neighborlist of atom \c i
   * \param numnei Number of neighbors of atom \c i
   * \param desc Descriptor
   * \param grad_desc Gradient of the descriptor
   * \param grad Flag which indicates for computing and returning both \c desc,
   * and \c grad_desc
   *
   * \note
   * `grad_desc` should be of length numDesc*(numnei+1)*DIM. The last DIM
   * associated with each descriptor whose last store the derivate of
   * descriptors w.r.t. the coordinates of atom i.
   */
  void generate_one_atom(int const i,
                         double const * coordinates,
                         int const * particleSpeciesCode,
                         int const * neighlist,
                         int const numnei,
                         double * const desc,
                         double * const grad_desc,
                         bool const grad);

 private:
  // Symmetry functions: Jorg Behler, J. Chem. Phys. 134, 074106, 2011.

  /*!
   * \brief Radial symmetry function \c g1 suitable for describing the
   * radial environment of atom \c i
   *
   * \c g1 is the sum of the cutoff functions with respect to
   * all neighboring atoms
   *
   * \param r The distance between atoms \c i and \c j
   * \param rcut The cutoff radius
   * \param phi Radial symmetry function \c g1 value
   */
  void sym_g1(double const r, double const rcut, double & phi);

  void sym_d_g1(double const r, double const rcut, double & phi, double & dphi);

  /*!
   * \brief Radial symmetry function \c g2 suitable for describing the
   * radial environment of atom \c i
   *
   * \c g2 is the sum of Gaussians multiplied by cutoff functions. The shifted
   * \g2 is suitable to describe a spherical shell around the reference atom.
   *
   * \param eta Defines the width of the Gaussians
   * \param Rs Shifts the center of the Gaussians to a certain radial distance
   * \param r The distance between atoms \c i and \c j
   * \param rcut The cutoff radius
   * \param phi Radial symmetry function \c g2 value
   */
  void sym_g2(double const eta,
              double const Rs,
              double const r,
              double const rcut,
              double & phi);

  void sym_d_g2(double const eta,
                double const Rs,
                double const r,
                double const rcut,
                double & phi,
                double & dphi);

  /*!
   * \brief Radial symmetry function \c g3 suitable for describing the
   * radial environment of atom \c i
   *
   * \c g3 is a damped cosine function with a period length adjusted
   * by parameter \c kappa
   *
   * \param kappa Adjusting the period length of the damped cosine function
   * \param r The distance between atoms \c i and \c j
   * \param rcut The cutoff radius
   * \param phi Radial symmetry function \c g3 value
   *
   * \note
   * Care must be taken when using the \c g3 function. Neighboring atoms can
   * cancel each other’s due to the existence of positive and negative function
   * values. It is recommended to use \c g3 in combination with other symmetry
   * functions.
   */
  void
  sym_g3(double const kappa, double const r, double const rcut, double & phi);

  void sym_d_g3(double const kappa,
                double const r,
                double const rcut,
                double & phi,
                double & dphi);

  /*!
   * \brief Summations of cosine functions of the angles centered at atom \c i.
   *
   * The angular part must be symmetric with respect to \c 180 angle
   *
   * \param zeta Provides the angular resolution. High values yield a narrower
   * range of nonzero symmetry function values \param lambda Shifting the maxima
   * of the cosine function (can have the values +1 or -1) \param eta
   * Controlling the radial resolution of the radial function. \param r The
   * distance between atoms \c i and \c j \param rcut The cutoff radius \param
   * phi Function \c g4 value
   */
  void sym_g4(double const zeta,
              double const lambda,
              double const eta,
              double const * r,
              double const * rcut,
              double & phi);

  void sym_d_g4(double const zeta,
                double const lambda,
                double const eta,
                double const * r,
                double const * rcut,
                double & phi,
                double * const dphi);

  /*!
   * \brief Summations of cosine functions of the angles centered at atom \c i.
   *
   * The angular part must be symmetric with respect to \c 180 angle.
   * In function \c g5 there is no constraint on the distance between atoms
   * resulting in a larger number of terms in the summation.
   *
   * \param zeta Provides the angular resolution. High values yield a narrower
   * range of nonzero symmetry function values \param lambda Shifting the maxima
   * of the cosine function (can have the values +1 or -1) \param eta
   * Controlling the radial resolution of the radial function. \param r The
   * distance between atoms \c i and \c j \param rcut The cutoff radius \param
   * phi Function \c g4 value
   */
  void sym_g5(double const zeta,
              double const lambda,
              double const eta,
              double const * r,
              double const * rcut,
              double & phi);

  void sym_d_g5(double const zeta,
                double const lambda,
                double const eta,
                double const * r,
                double const * rcut,
                double & phi,
                double * const dphi);

  /*! Prining for debugging purpose */
  void echo_input();

  /*! Function pointer */
  CutoffFunction cutoff_func_;

  /*! Function pointer */
  dCutoffFunction d_cutoff_func_;

 private:
  /*! Vector of species */
  std::vector<std::string> species_;

  /*! Vector containing the name of each descriptor */
  std::vector<std::string> name_;

  /*! Starting index of each descriptor in generalized coords */
  std::vector<int> starting_index_;

  /*! The cutoff radius */
  Array2D<double> rcut_2D_;

  /*! Parameters of each descriptor */
  std::vector<Array2D<double> > params_;

  /*! Number of parameter sets of each descriptor */
  std::vector<int> num_param_sets_;

  /*! size of parameters of each descriptor */
  std::vector<int> num_params_;

  /*! */
  std::vector<double> feature_mean_;

  /*! Vector of mean */
  std::vector<double> feature_std_;

  /*! Flag for three body interactions */
  bool has_three_body_;

  /*! Flag to indicate if to normalize the data or not */
  bool normalize_;
};

#undef DIM

#endif  // KLIFF_DESCRIPTOR_HPP_
