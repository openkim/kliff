#include "sym_fn.hpp"

#include <cstring>

#ifdef MY_PI
#undef MY_PI
#endif

#define MY_PI 3.1415926535897932

#ifdef DIM
#undef DIM
#endif

#define DIM 3

#ifdef MAXLINE
#undef MAXLINE
#endif

#define MAXLINE 20480

#ifdef LOG_ERROR
#undef LOG_ERROR
#endif

#define LOG_ERROR(msg)                                           \
  {                                                              \
    std::ostringstream ss;                                       \
    ss << msg;                                                   \
    std::string _Messagef_(FormatMessageFileLineFunctionMessage( \
        "Error ", __FILE__, __LINE__, __FUNCTION__, ss.str()));  \
    std::cerr << _Messagef_;                                     \
  }


inline double cut_cos(double const r, double const rcut)
{
  return (r < rcut) ? 0.5 * (std::cos(MY_PI * r / rcut) + 1.0) : 0.0;
}

inline double d_cut_cos(double const r, double const rcut)
{
  return (r < rcut) ? -0.5 * MY_PI / rcut * std::sin(MY_PI * r / rcut) : 0.0;
}

Descriptor::Descriptor() : has_three_body_(false), normalize_(true)
{
  // Support cos cutoff function currently, but can be easily extended
  cutoff_func_ = &cut_cos;

  d_cutoff_func_ = &d_cut_cos;
}

Descriptor::~Descriptor() {}

inline void Descriptor::set_species(std::vector<std::string> & species)
{
  species_.resize(species.size());
  std::copy(species.begin(), species.end(), species_.begin());
};

inline void Descriptor::get_species(std::vector<std::string> & species)
{
  species.resize(species_.size());
  std::copy(species_.begin(), species_.end(), species.begin());
};

inline int Descriptor::get_num_species() { return species_.size(); }

void Descriptor::set_cutoff(char const * name,
                            std::size_t const Nspecies,
                            double const * rcut_2D)
{
  // to avoid unused warning
  (void) name;

  rcut_2D_.resize(Nspecies, Nspecies, rcut_2D);
}

inline double Descriptor::get_cutoff(int const iCode, int const jCode)
{
  return rcut_2D_(iCode, jCode);
};

void Descriptor::add_descriptor(char const * name,
                                double const * values,
                                int const row,
                                int const col)
{
  name_.push_back(name);

  Array2D<double> params(row, col, values);
  params_.push_back(std::move(params));

  auto sum = std::accumulate(num_param_sets_.begin(), num_param_sets_.end(), 0);
  starting_index_.push_back(sum);

  num_param_sets_.push_back(row);
  num_params_.push_back(col);

  if (strcmp(name, "g4") == 0 || strcmp(name, "g5") == 0)
  { has_three_body_ = true; }
}

int Descriptor::read_parameter_file(FILE * const filePointer)
{
  char nextLine[MAXLINE];
  int endOfFileFlag = 0;

  char name[128];

  // species and cutoff
  getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
  if (std::sscanf(nextLine, "%s", name) != 1)
  {
    LOG_ERROR("Unable to read cutoff type from line:\n"
              + std::string(nextLine));
    return true;
  }

  lowerCase(name);
  if (std::strcmp(name, "cos") != 0)
  {
    LOG_ERROR("Currently, only cutoff type `cos` is supported, but given %s.\n"
              + std::string(name));
    return true;
  }

  // Number of scpecies
  int numSpecies;

  getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
  if (std::sscanf(nextLine, "%d", &numSpecies) != 1)
  {
    LOG_ERROR("Unable to read number of species from line:\n"
              + std::string(nextLine));
    return true;
  }

  // cutoff and species
  double cutoff;

  char spec1[32];
  char spec2[32];

  rcut_2D_.resize(numSpecies, numSpecies);

  // Clear the container
  species_.clear();

  // keep track of known species
  std::map<std::string, int> speciesMap;

  int iIndex, jIndex;

  // species code integer and code starts from 0
  int const numUniqueSpeciesPairs = (numSpecies + 1) * numSpecies / 2;
  for (int i = 0, index = 0; i < numUniqueSpeciesPairs; ++i)
  {
    getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
    if (std::sscanf(nextLine, "%s %s %lg", spec1, spec2, &cutoff) != 3)
    {
      LOG_ERROR("Unable to read species name & cutoff from line:\n"
                + std::string(nextLine));
      return true;
    }

    // check for new species
    std::string s1(spec1);

    std::map<std::string, int>::const_iterator iIter = speciesMap.find(s1);
    if (iIter == speciesMap.end())
    {
      speciesMap[s1] = index;
      species_.push_back(s1);
      iIndex = index;
      ++index;
    }
    else
    {
      iIndex = speciesMap[s1];
    }

    std::string s2(spec2);
    std::map<std::string, int>::const_iterator jIter = speciesMap.find(s2);
    if (jIter == speciesMap.end())
    {
      speciesMap[s2] = index;
      species_.push_back(s2);
      jIndex = index;
      ++index;
    }
    else
    {
      jIndex = speciesMap[s2];
    }

    // store cutoff values
    rcut_2D_(iIndex, jIndex) = rcut_2D_(jIndex, iIndex) = cutoff;
  }

  // descriptor
  int numDescTypes;
  int numParams;
  int numParamSets;

  // number of descriptor types
  getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
  if (std::sscanf(nextLine, "%d", &numDescTypes) != 1)
  {
    LOG_ERROR("Unable to read number of descriptor types from line:\n"
              + std::string(nextLine));
    return true;
  }

  // descriptor
  for (int i = 0; i < numDescTypes; ++i)
  {
    // descriptor name and parameter dimensions
    getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
    if (std::sscanf(nextLine, "%s", name) != 1)
    {
      LOG_ERROR("Unable to read the descriptor name from line:\n"
                + std::string(nextLine));
      return true;
    }

    // change to lower case name
    lowerCase(name);
    if (std::strcmp(name, "g1") == 0) { add_descriptor(name, nullptr, 1, 0); }
    else
    {
      // re-read name, and read number of param sets and number of params
      if (std::sscanf(nextLine, "%s %d %d", name, &numParamSets, &numParams)
          != 3)
      {
        LOG_ERROR("Unable to read the descriptor name * number of sets & "
                  "parameters from line:\n"
                  + std::string(nextLine));
        return true;
      }

      // check size of params is correct w.r.t its name
      // change name to lower case
      lowerCase(name);
      if (std::strcmp(name, "g2") == 0)
      {
        if (numParams != 2)
        {
          LOG_ERROR("The number of params for descriptor G2 is incorrect, "
                    "expecting 2, but given "
                    + std::to_string(numParams));
          return true;
        }
      }
      else if (std::strcmp(name, "g3") == 0)
      {
        if (numParams != 1)
        {
          LOG_ERROR("The number of params for descriptor G3 is incorrect, "
                    "expecting 1, but given "
                    + std::to_string(numParams));
          return true;
        }
      }
      else if (std::strcmp(name, "g4") == 0)
      {
        if (numParams != 3)
        {
          LOG_ERROR("The number of params for descriptor G4 is incorrect, "
                    "expecting 3, but given "
                    + std::to_string(numParams));
          return true;
        }
      }
      else if (std::strcmp(name, "g5") == 0)
      {
        if (numParams != 3)
        {
          LOG_ERROR("The number of params for descriptor G5 is incorrect, "
                    "expecting 3, but given "
                    + std::to_string(numParams));
          return true;
        }
      }
      else
      {
        LOG_ERROR("Unsupported descriptor `" + std::string(name)
                  + "' from line:\n" + nextLine);
        return true;
      }

      {
        // read descriptor params
        Array2D<double> descParams(numParamSets, numParams);

        for (int j = 0; j < numParamSets; ++j)
        {
          getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);

          if (getXdouble(nextLine, numParams, descParams.data_1D(j).data()))
          {
            LOG_ERROR("Unable to read descriptor parameters from line:\n"
                      + std::string(nextLine));
            return true;
          }
        }

        // copy data to Descriptor
        add_descriptor(name, descParams.data(), numParamSets, numParams);
      }
    }
  }

  // centering and normalizing params
  // flag, whether we use this feature
  getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
  if (std::sscanf(nextLine, "%*s %s", name) != 1)
  {
    LOG_ERROR("Unable to read normalization flag from line:\n"
              + std::string(nextLine));
    return true;
  }

  lowerCase(name);
  bool normalize = std::strcmp(name, "true") == 0;

  if (normalize)
  {
    int size;

    // size of the data, this should be equal to numDescs
    getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
    if (std::sscanf(nextLine, "%d", &size) != 1)
    {
      LOG_ERROR("Unable to read the size of centering & normalizing data from "
                "the line:\n"
                + std::string(nextLine));
      return true;
    }
    else
    {
      // Get the number of descriptors
      auto numDescs = get_num_descriptors();

      if (size != numDescs)
      {
        LOG_ERROR("Size of centering & normalizing data inconsistent with the "
                  "number of descriptors. \n Size = "
                  + std::to_string(size)
                  + " & num_descriptors= " + std::to_string(numDescs));
        return true;
      }
    }

    // read means
    std::vector<double> means(size, static_cast<double>(0));

    for (int i = 0; i < size; ++i)
    {
      getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
      if (std::sscanf(nextLine, "%lg", &means[i]) != 1)
      {
        LOG_ERROR("Unable to read `means' from line:\n"
                  + std::string(nextLine));
        return true;
      }
    }

    // read standard deviations
    std::vector<double> stds(size, static_cast<double>(0));

    for (int i = 0; i < size; ++i)
    {
      getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
      if (std::sscanf(nextLine, "%lg", &stds[i]) != 1)
      {
        LOG_ERROR("Unable to read `stds' from line:\n" + std::string(nextLine));
        return true;
      }
    }

    // store info into descriptor class
    set_feature_mean_and_std(normalize, size, means.data(), stds.data());
  }
  else
  {
    normalize_ = false;
  }

  // everything is OK
  return false;
}

int Descriptor::get_num_descriptors()
{
  return std::accumulate(num_param_sets_.begin(), num_param_sets_.end(), 0);
}

void Descriptor::set_feature_mean_and_std(bool const normalize,
                                          int const size,
                                          double const * means,
                                          double const * stds)
{
  normalize_ = normalize;
  for (int i = 0; i < size; ++i)
  {
    feature_mean_.push_back(means[i]);
    feature_std_.push_back(stds[i]);
  }
}

inline void
Descriptor::get_feature_mean_and_std(int const i, double & mean, double & std)
{
  mean = feature_mean_[i];
  std = feature_std_[i];
};

inline bool Descriptor::need_normalize() { return normalize_; };

void Descriptor::generate_one_atom(int const i,
                                   double const * coords,
                                   int const * particleSpeciesCodes,
                                   int const * neighlist,
                                   int const numnei,
                                   double * const desc,
                                   double * const grad_desc,
                                   bool const grad)
{
  // prepare data
  VectorOfSizeDIM * coordinates = (VectorOfSizeDIM *) coords;

  int const iSpecies = particleSpeciesCodes[i];

  // Setup loop over neighbors of current particle
  for (int jj = 0; jj < numnei; ++jj)
  {
    // adjust index of particle neighbor
    int const j = neighlist[jj];
    int const jSpecies = particleSpeciesCodes[j];

    // cutoff between ij
    double rcutij = rcut_2D_(iSpecies, jSpecies);

    // Compute rij
    double rij[DIM];
    for (int dim = 0; dim < DIM; ++dim)
    { rij[dim] = coordinates[j][dim] - coordinates[i][dim]; }

    double const rijsq = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
    double const rijmag = std::sqrt(rijsq);

    // if particles i and j not interact
    if (rijmag > rcutij) { continue; }

    // Loop over descriptors
    // two-body descriptors
    for (std::size_t p = 0; p < name_.size(); ++p)
    {
      if (name_[p] != "g1" && name_[p] != "g2" && name_[p] != "g3")
      { continue; }

      int idx = starting_index_[p];

      // Loop over same descriptor but different parameter set
      for (int q = 0; q < num_param_sets_[p]; ++q)
      {
        double gc = 0.0;
        double dgcdr_two = 0.0;

        if (name_[p] == "g1")
        {
          grad ? sym_d_g1(rijmag, rcutij, gc, dgcdr_two)
               : sym_g1(rijmag, rcutij, gc);
        }
        else if (name_[p] == "g2")
        {
          double eta = params_[p](q, 0);
          double Rs = params_[p](q, 1);

          grad ? sym_d_g2(eta, Rs, rijmag, rcutij, gc, dgcdr_two)
               : sym_g2(eta, Rs, rijmag, rcutij, gc);
        }
        else if (name_[p] == "g3")
        {
          double kappa = params_[p](q, 0);

          grad ? sym_d_g3(kappa, rijmag, rcutij, gc, dgcdr_two)
               : sym_g3(kappa, rijmag, rcutij, gc);
        }

        desc[idx] += gc;

        if (grad)
        {
          int const page = idx * DIM * (numnei + 1);

          for (int dim = 0; dim < DIM; ++dim)
          {
            double const pair = dgcdr_two * rij[dim] / rijmag;

            grad_desc[page + numnei * DIM + dim] -= pair;
            grad_desc[page + jj * DIM + dim] += pair;
          }
        }

        ++idx;
      }
    }

    // three-body descriptors
    if (has_three_body_ == false) { continue; }

    // Loop over kk
    for (int kk = jj + 1; kk < numnei; ++kk)
    {
      // Adjust index of particle neighbor
      int const k = neighlist[kk];
      int const kSpecies = particleSpeciesCodes[k];

      // cutoff between ik and jk
      double const rcutik = rcut_2D_[iSpecies][kSpecies];
      double const rcutjk = rcut_2D_[jSpecies][kSpecies];

      // Compute rik, rjk and their squares
      double rik[DIM];
      double rjk[DIM];

      for (int dim = 0; dim < DIM; ++dim)
      {
        rik[dim] = coordinates[k][dim] - coordinates[i][dim];
        rjk[dim] = coordinates[k][dim] - coordinates[j][dim];
      }

      double const riksq = rik[0] * rik[0] + rik[1] * rik[1] + rik[2] * rik[2];
      double const rjksq = rjk[0] * rjk[0] + rjk[1] * rjk[1] + rjk[2] * rjk[2];

      double const rikmag = std::sqrt(riksq);
      double const rjkmag = std::sqrt(rjksq);

      // Check whether three-dody not interacting
      if (rikmag > rcutik) { continue; }

      double const rvec[3] = {rijmag, rikmag, rjkmag};
      double const rcutvec[3] = {rcutij, rcutik, rcutjk};

      // Loop over descriptors
      // three-body descriptors
      for (size_t p = 0; p < name_.size(); ++p)
      {
        if (name_[p] != "g4" && name_[p] != "g5") { continue; }

        int idx = starting_index_[p];

        // Loop over same descriptor but different parameter set
        for (int q = 0; q < num_param_sets_[p]; ++q)
        {
          double gc = 0.0;
          double dgcdr_three[3] = {0.0, 0.0, 0.0};

          if (name_[p] == "g4")
          {
            double zeta = params_[p](q, 0);
            double lambda = params_[p](q, 1);
            double eta = params_[p](q, 2);

            grad ? sym_d_g4(zeta, lambda, eta, rvec, rcutvec, gc, dgcdr_three)
                 : sym_g4(zeta, lambda, eta, rvec, rcutvec, gc);
          }
          else if (name_[p] == "g5")
          {
            double zeta = params_[p](q, 0);
            double lambda = params_[p](q, 1);
            double eta = params_[p](q, 2);

            grad ? sym_d_g5(zeta, lambda, eta, rvec, rcutvec, gc, dgcdr_three)
                 : sym_g5(zeta, lambda, eta, rvec, rcutvec, gc);
          }

          desc[idx] += gc;

          if (grad)
          {
            int const page = idx * DIM * (numnei + 1);

            for (int dim = 0; dim < DIM; ++dim)
            {
              double pair_ij = dgcdr_three[0] * rij[dim] / rijmag;
              double pair_ik = dgcdr_three[1] * rik[dim] / rikmag;
              double pair_jk = dgcdr_three[2] * rjk[dim] / rjkmag;

              grad_desc[page + numnei * DIM + dim] += -pair_ij - pair_ik;
              grad_desc[page + jj * DIM + dim] += pair_ij - pair_jk;
              grad_desc[page + kk * DIM + dim] += pair_ik + pair_jk;
            }
          }

          ++idx;
        }
      }
    }
  }
}

void Descriptor::sym_g1(double const r, double const rcut, double & phi)
{
  phi = cutoff_func_(r, rcut);
}

void Descriptor::sym_d_g1(double const r,
                          double const rcut,
                          double & phi,
                          double & dphi)
{
  phi = cutoff_func_(r, rcut);
  dphi = d_cutoff_func_(r, rcut);
}

void Descriptor::sym_g2(double const eta,
                        double const Rs,
                        double const r,
                        double const rcut,
                        double & phi)
{
  phi = std::exp(-eta * (r - Rs) * (r - Rs)) * cutoff_func_(r, rcut);
}

void Descriptor::sym_d_g2(double const eta,
                          double const Rs,
                          double const r,
                          double const rcut,
                          double & phi,
                          double & dphi)
{
  if (r > rcut)
  {
    phi = 0.0;
    dphi = 0.0;
  }
  else
  {
    double const eterm = std::exp(-eta * (r - Rs) * (r - Rs));
    double const determ = -2 * eta * (r - Rs) * eterm;

    double const fc = cutoff_func_(r, rcut);
    double const dfc = d_cutoff_func_(r, rcut);

    phi = eterm * fc;
    dphi = determ * fc + eterm * dfc;
  }
}

void Descriptor::sym_g3(double const kappa,
                        double const r,
                        double const rcut,
                        double & phi)
{
  phi = std::cos(kappa * r) * cutoff_func_(r, rcut);
}

void Descriptor::sym_d_g3(double const kappa,
                          double const r,
                          double const rcut,
                          double & phi,
                          double & dphi)
{
  double const costerm = std::cos(kappa * r);
  double const dcosterm = -kappa * std::sin(kappa * r);

  double const fc = cutoff_func_(r, rcut);
  double const dfc = d_cutoff_func_(r, rcut);

  phi = costerm * fc;
  dphi = dcosterm * fc + costerm * dfc;
}

void Descriptor::sym_g4(double const zeta,
                        double const lambda,
                        double const eta,
                        double const * r,
                        double const * rcut,
                        double & phi)
{
  double const rij = r[0];
  double const rik = r[1];
  double const rjk = r[2];

  double const rcutij = rcut[0];
  double const rcutik = rcut[1];
  double const rcutjk = rcut[2];

  if (rij > rcutij || rik > rcutik || rjk > rcutjk) { phi = 0.0; }
  else
  {
    double const rijsq = rij * rij;
    double const riksq = rik * rik;
    double const rjksq = rjk * rjk;

    // i is the apex atom
    double const cos_ijk = (rijsq + riksq - rjksq) / (2 * rij * rik);

    double const base = 1 + lambda * cos_ijk;

    // prevent numerical instability (when lambda=-1 and cos_ijk=1)
    double const costerm = (base <= 0) ? 0.0 : std::pow(base, zeta);

    double const eterm = std::exp(-eta * (rijsq + riksq + rjksq));

    phi = std::pow(2, 1 - zeta) * costerm * eterm * cutoff_func_(rij, rcutij)
          * cutoff_func_(rik, rcutik) * cutoff_func_(rjk, rcutjk);
  }
}

void Descriptor::sym_d_g4(double const zeta,
                          double const lambda,
                          double const eta,
                          double const * r,
                          double const * rcut,
                          double & phi,
                          double * const dphi)
{
  double const rij = r[0];
  double const rik = r[1];
  double const rjk = r[2];

  double const rcutij = rcut[0];
  double const rcutik = rcut[1];
  double const rcutjk = rcut[2];

  if (rij > rcutij || rik > rcutik || rjk > rcutjk)
  {
    phi = 0.0;

    dphi[0] = 0.0;
    dphi[1] = 0.0;
    dphi[2] = 0.0;
  }
  else
  {
    double const rijsq = rij * rij;
    double const riksq = rik * rik;
    double const rjksq = rjk * rjk;

    // cosine term, i is the apex atom
    double const cos_ijk = (rijsq + riksq - rjksq) / (2 * rij * rik);
    double const dcos_dij = (rijsq - riksq + rjksq) / (2 * rijsq * rik);
    double const dcos_dik = (riksq - rijsq + rjksq) / (2 * rij * riksq);
    double const dcos_djk = -rjk / (rij * rik);

    double costerm;
    double dcosterm_dcos;

    double const base = 1.0 + lambda * cos_ijk;

    // prevent numerical instability (when lambda=-1 and cos_ijk=1)
    if (base <= 0)
    {
      costerm = 0.0;
      dcosterm_dcos = 0.0;
    }
    else
    {
      double const power = std::pow(base, zeta);
      double const power_minus1 = power / base;

      costerm = power;
      dcosterm_dcos = zeta * power_minus1 * lambda;
    }

    double const dcosterm_dij = dcosterm_dcos * dcos_dij;
    double const dcosterm_dik = dcosterm_dcos * dcos_dik;
    double const dcosterm_djk = dcosterm_dcos * dcos_djk;

    // exponential term
    double const eterm = std::exp(-eta * (rijsq + riksq + rjksq));

    double const determ_dij = -2 * eterm * eta * rij;
    double const determ_dik = -2 * eterm * eta * rik;
    double const determ_djk = -2 * eterm * eta * rjk;

    // power 2 term
    double const p2 = std::pow(2, 1 - zeta);

    // cutoff_func
    double const fcij = cutoff_func_(rij, rcutij);
    double const fcik = cutoff_func_(rik, rcutik);
    double const fcjk = cutoff_func_(rjk, rcutjk);

    double const fcprod = fcij * fcik * fcjk;

    double const dfcprod_dij = d_cutoff_func_(rij, rcutij) * fcik * fcjk;
    double const dfcprod_dik = d_cutoff_func_(rik, rcutik) * fcij * fcjk;
    double const dfcprod_djk = d_cutoff_func_(rjk, rcutjk) * fcij * fcik;

    // phi
    phi = p2 * costerm * eterm * fcprod;
    // dphi_dij
    dphi[0] = p2
              * (dcosterm_dij * eterm * fcprod + costerm * determ_dij * fcprod
                 + costerm * eterm * dfcprod_dij);
    // dphi_dik
    dphi[1] = p2
              * (dcosterm_dik * eterm * fcprod + costerm * determ_dik * fcprod
                 + costerm * eterm * dfcprod_dik);
    // dphi_djk
    dphi[2] = p2
              * (dcosterm_djk * eterm * fcprod + costerm * determ_djk * fcprod
                 + costerm * eterm * dfcprod_djk);
  }
}

void Descriptor::sym_g5(double const zeta,
                        double const lambda,
                        double const eta,
                        double const * r,
                        double const * rcut,
                        double & phi)
{
  double const rij = r[0];
  double const rik = r[1];

  double const rcutij = rcut[0];
  double const rcutik = rcut[1];

  if (rij > rcutij || rik > rcutik) { phi = 0.0; }
  else
  {
    double const rjk = r[2];

    double const rijsq = rij * rij;
    double const riksq = rik * rik;
    double const rjksq = rjk * rjk;

    // i is the apex atom
    double const cos_ijk = (rijsq + riksq - rjksq) / (2 * rij * rik);

    double const base = 1.0 + lambda * cos_ijk;

    // prevent numerical instability (when lambda=-1 and cos_ijk=1)

    double const costerm = (base <= 0) ? 0.0 : std::pow(base, zeta);

    double const eterm = std::exp(-eta * (rijsq + riksq));

    phi = std::pow(2, 1 - zeta) * costerm * eterm * cutoff_func_(rij, rcutij)
          * cutoff_func_(rik, rcutik);
  }
}

void Descriptor::sym_d_g5(double const zeta,
                          double const lambda,
                          double const eta,
                          double const * r,
                          double const * rcut,
                          double & phi,
                          double * const dphi)
{
  double const rij = r[0];
  double const rik = r[1];

  double const rcutij = rcut[0];
  double const rcutik = rcut[1];

  if (rij > rcutij || rik > rcutik)
  {
    phi = 0.0;

    dphi[0] = 0.0;
    dphi[1] = 0.0;
    dphi[2] = 0.0;
  }
  else
  {
    double const rjk = r[2];

    double const rijsq = rij * rij;
    double const riksq = rik * rik;
    double const rjksq = rjk * rjk;

    // cosine term, i is the apex atom
    double const cos_ijk = (rijsq + riksq - rjksq) / (2 * rij * rik);
    double const dcos_dij = (rijsq - riksq + rjksq) / (2 * rijsq * rik);
    double const dcos_dik = (riksq - rijsq + rjksq) / (2 * rij * riksq);
    double const dcos_djk = -rjk / (rij * rik);

    double costerm;
    double dcosterm_dcos;

    double const base = 1 + lambda * cos_ijk;

    // prevent numerical instability (when lambda=-1 and cos_ijk=1)
    if (base <= 0)
    {
      costerm = 0.0;
      dcosterm_dcos = 0.0;
    }
    else
    {
      costerm = std::pow(base, zeta);
      dcosterm_dcos = zeta * std::pow(base, zeta - 1) * lambda;
    }

    double const dcosterm_dij = dcosterm_dcos * dcos_dij;
    double const dcosterm_dik = dcosterm_dcos * dcos_dik;
    double const dcosterm_djk = dcosterm_dcos * dcos_djk;

    // exponential term
    double const eterm = std::exp(-eta * (rijsq + riksq));
    double const determ_dij = -2 * eterm * eta * rij;
    double const determ_dik = -2 * eterm * eta * rik;

    // power 2 term
    double const p2 = std::pow(2, 1 - zeta);

    // cutoff_func
    double const fcij = cutoff_func_(rij, rcutij);
    double const fcik = cutoff_func_(rik, rcutik);

    double const fcprod = fcij * fcik;

    double const dfcprod_dij = d_cutoff_func_(rij, rcutij) * fcik;
    double const dfcprod_dik = d_cutoff_func_(rik, rcutik) * fcij;

    // phi
    phi = p2 * costerm * eterm * fcprod;
    // dphi_dij
    dphi[0] = p2
              * (dcosterm_dij * eterm * fcprod + costerm * determ_dij * fcprod
                 + costerm * eterm * dfcprod_dij);
    // dphi_dik
    dphi[1] = p2
              * (dcosterm_dik * eterm * fcprod + costerm * determ_dik * fcprod
                 + costerm * eterm * dfcprod_dik);
    // dphi_djk
    dphi[2] = p2 * dcosterm_djk * eterm * fcprod;
  }
}

void Descriptor::echo_input()
{
  std::cout << "=====================================" << std::endl;

  for (size_t i = 0; i < name_.size(); ++i)
  {
    int const rows = num_param_sets_.at(i);
    int const cols = num_params_.at(i);

    std::cout << "name: " << name_.at(i) << ", rows: " << rows
              << ", cols: " << cols << std::endl;
    for (int m = 0; m < rows; ++m)
    {
      for (int n = 0; n < cols; ++n)
      { std::cout << params_.at(i).at(m, n) << " "; }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  // centering and normalization
  std::cout << "centering and normalizing params" << std::endl;

  std::cout << "means:" << std::endl;

  for (size_t i = 0; i < feature_mean_.size(); ++i)
  { std::cout << feature_mean_.at(i) << std::endl; }

  std::cout << "stds:" << std::endl;
  for (size_t i = 0; i < feature_std_.size(); ++i)
  { std::cout << feature_std_.at(i) << std::endl; }
}

#undef LOG_ERROR
#undef MAXLINE
#undef DIM
#undef MY_PI
