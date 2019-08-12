#include "sym_fn.h"

#define LOG_ERROR(msg) \
  (std::cerr << "ERROR (descriptor): " << (msg) << std::endl)

Descriptor::Descriptor() :
    rcut_2D_(nullptr),
    has_three_body_(false),
    normalize_(true)
{
  return;
}

Descriptor::~Descriptor()
{
  for (size_t i = 0; i < params_.size(); i++)
  { Deallocate2DArray<double>(params_.at(i)); }

  Deallocate2DArray<double>(rcut_2D_);
}

int Descriptor::read_parameter_file(FILE * const filePointer)
{
  int ier;
  int endOfFileFlag = 0;
  char nextLine[MAXLINE];
  char errorMsg[1024];
  char name[128];

  // cutoff and species
  int numSpecies;
  int numUniqueSpeciesPairs;
  char spec1[32], spec2[32];
  int iIndex, jIndex;
  double cutoff;
  double ** cutoff_2D;
  std::vector<std::string> species;

  // descriptor
  int numDescTypes;
  int numDescs;
  int numParams;
  int numParamSets;
  double ** descParams = nullptr;

  // species and cutoff
  getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
  ier = sscanf(nextLine, "%s", name);
  if (ier != 1)
  {
    sprintf(errorMsg, "unable to read cutoff type from line:\n");
    strcat(errorMsg, nextLine);
    LOG_ERROR(errorMsg);
    return true;
  }
  lowerCase(name);
  if (strcmp(name, "cos") != 0)
  {
    sprintf(errorMsg,
            "Currently, only cutoff type `cos` is supported, but given %s.\n",
            name);
    LOG_ERROR(errorMsg);
    return true;
  }

  getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
  ier = sscanf(nextLine, "%d", &numSpecies);
  if (ier != 1)
  {
    sprintf(errorMsg, "unable to read number of species from line:\n");
    strcat(errorMsg, nextLine);
    LOG_ERROR(errorMsg);
    return true;
  }
  numUniqueSpeciesPairs = (numSpecies + 1) * numSpecies / 2;
  AllocateAndInitialize2DArray<double>(cutoff_2D, numSpecies, numSpecies);

  // keep track of known species
  std::map<std::string, int> speciesMap;
  int index = 0;  // species code integer code starting from 0

  for (int i = 0; i < numUniqueSpeciesPairs; i++)
  {
    getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
    ier = sscanf(nextLine, "%s %s %lg", spec1, spec2, &cutoff);
    if (ier != 3)
    {
      sprintf(errorMsg, "unable to read cutoff from line:\n");
      strcat(errorMsg, nextLine);
      LOG_ERROR(errorMsg);
      return true;
    }

    // check for new species
    std::string s1(spec1);
    std::map<std::string, int>::const_iterator iIter = speciesMap.find(s1);
    if (iIter == speciesMap.end())
    {
      speciesMap[s1] = index;
      species.push_back(s1);
      iIndex = index;
      index++;
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
      species.push_back(s2);
      jIndex = index;
      index++;
    }
    else
    {
      jIndex = speciesMap[s2];
    }

    // store cutoff values
    cutoff_2D[iIndex][jIndex] = cutoff_2D[jIndex][iIndex] = cutoff;
  }

  // register species cutoff
  set_species(species);
  set_cutoff(name, numSpecies, cutoff_2D[0]);
  Deallocate2DArray<double>(cutoff_2D);

  // number of descriptor types
  getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
  ier = sscanf(nextLine, "%d", &numDescTypes);
  if (ier != 1)
  {
    sprintf(errorMsg, "unable to read number of descriptor types from line:\n");
    strcat(errorMsg, nextLine);
    LOG_ERROR(errorMsg);
    return true;
  }

  // descriptor
  for (int i = 0; i < numDescTypes; i++)
  {
    // descriptor name and parameter dimensions
    getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);

    // name of descriptor
    ier = sscanf(nextLine, "%s", name);
    if (ier != 1)
    {
      sprintf(errorMsg, "unable to read descriptor from line:\n");
      strcat(errorMsg, nextLine);
      LOG_ERROR(errorMsg);
      return true;
    }
    lowerCase(name);  // change to lower case name
    if (strcmp(name, "g1") == 0) { add_descriptor(name, nullptr, 1, 0); }
    else
    {
      // re-read name, and read number of param sets and number of params
      ier = sscanf(nextLine, "%s %d %d", name, &numParamSets, &numParams);
      if (ier != 3)
      {
        sprintf(errorMsg, "unable to read descriptor from line:\n");
        strcat(errorMsg, nextLine);
        LOG_ERROR(errorMsg);
        return true;
      }
      // change name to lower case
      lowerCase(name);

      // check size of params is correct w.r.t its name
      if (strcmp(name, "g2") == 0)
      {
        if (numParams != 2)
        {
          sprintf(errorMsg,
                  "number of params for descriptor G2 is incorrect, "
                  "expecting 2, but given %d.\n",
                  numParams);
          LOG_ERROR(errorMsg);
          return true;
        }
      }
      else if (strcmp(name, "g3") == 0)
      {
        if (numParams != 1)
        {
          sprintf(errorMsg,
                  "number of params for descriptor G3 is incorrect, "
                  "expecting 1, but given %d.\n",
                  numParams);
          LOG_ERROR(errorMsg);
          return true;
        }
      }
      else if (strcmp(name, "g4") == 0)
      {
        if (numParams != 3)
        {
          sprintf(errorMsg,
                  "number of params for descriptor G4 is incorrect, "
                  "expecting 3, but given %d.\n",
                  numParams);
          LOG_ERROR(errorMsg);
          return true;
        }
      }
      else if (strcmp(name, "g5") == 0)
      {
        if (numParams != 3)
        {
          sprintf(errorMsg,
                  "number of params for descriptor G5 is incorrect, "
                  "expecting 3, but given %d.\n",
                  numParams);
          LOG_ERROR(errorMsg);
          return true;
        }
      }
      else
      {
        sprintf(errorMsg, "unsupported descriptor `%s' from line:\n", name);
        strcat(errorMsg, nextLine);
        LOG_ERROR(errorMsg);
        return true;
      }

      // read descriptor params
      AllocateAndInitialize2DArray<double>(descParams, numParamSets, numParams);
      for (int j = 0; j < numParamSets; j++)
      {
        getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
        ier = getXdouble(nextLine, numParams, descParams[j]);
        if (ier)
        {
          sprintf(errorMsg,
                  "unable to read descriptor parameters from line:\n");
          strcat(errorMsg, nextLine);
          LOG_ERROR(errorMsg);
          return true;
        }
      }

      // copy data to Descriptor
      add_descriptor(name, descParams[0], numParamSets, numParams);
      Deallocate2DArray(descParams);
    }
  }
  // number of descriptors
  numDescs = get_num_descriptors();

  // centering and normalizing params
  // flag, whether we use this feature
  getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
  ier = sscanf(nextLine, "%*s %s", name);
  if (ier != 1)
  {
    sprintf(errorMsg,
            "unable to read centering and normalization info from line:\n");
    strcat(errorMsg, nextLine);
    LOG_ERROR(errorMsg);
    return true;
  }
  lowerCase(name);
  bool normalize;
  if (strcmp(name, "true") == 0) { normalize = true; }
  else
  {
    normalize = false;
  }

  int size = 0;
  double * means = nullptr;
  double * stds = nullptr;
  if (normalize)
  {
    // size of the data, this should be equal to numDescs
    getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
    ier = sscanf(nextLine, "%d", &size);
    if (ier != 1)
    {
      sprintf(errorMsg,
              "unable to read the size of centering and normalization "
              "data info from line:\n");
      strcat(errorMsg, nextLine);
      LOG_ERROR(errorMsg);
      return true;
    }
    if (size != numDescs)
    {
      sprintf(errorMsg,
              "Size of centering and normalizing data inconsistent with "
              "the number of descriptors. Size = %d, num_descriptors=%d\n",
              size,
              numDescs);
      LOG_ERROR(errorMsg);
      return true;
    }

    // read means
    AllocateAndInitialize1DArray<double>(means, size);
    for (int i = 0; i < size; i++)
    {
      getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
      ier = sscanf(nextLine, "%lg", &means[i]);
      if (ier != 1)
      {
        sprintf(errorMsg, "unable to read `means' from line:\n");
        strcat(errorMsg, nextLine);
        LOG_ERROR(errorMsg);
        return true;
      }
    }

    // read standard deviations
    AllocateAndInitialize1DArray<double>(stds, size);
    for (int i = 0; i < size; i++)
    {
      getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
      ier = sscanf(nextLine, "%lg", &stds[i]);
      if (ier != 1)
      {
        sprintf(errorMsg, "unable to read `means' from line:\n");
        strcat(errorMsg, nextLine);
        LOG_ERROR(errorMsg);
        return true;
      }
    }
  }

  // store info into descriptor class
  set_feature_mean_and_std(normalize, size, means, stds);
  Deallocate1DArray(means);
  Deallocate1DArray(stds);

  // everything is OK
  return false;
}

void Descriptor::set_cutoff(const char * name,
                            const int Nspecies,
                            const double * rcut_2D)
{
  (void) name;  // to avoid unused warning

  // Support cos cutoff function currently, but can be easily extended
  cutoff_func_ = &cut_cos;
  d_cutoff_func_ = &d_cut_cos;

  AllocateAndInitialize2DArray(rcut_2D_, Nspecies, Nspecies);
  int idx = 0;
  for (int i = 0; i < Nspecies; i++)
  {
    for (int j = 0; j < Nspecies; j++)
    {
      rcut_2D_[i][j] = rcut_2D[idx];
      idx++;
    }
  }
}

void Descriptor::add_descriptor(char const * name,
                                double const * values,
                                int const row,
                                int const col)
{
  int idx = 0;
  double ** params = 0;
  AllocateAndInitialize2DArray(params, row, col);
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      params[i][j] = values[idx];
      idx++;
    }
  }

  idx = 0;
  for (size_t i = 0; i < num_param_sets_.size(); i++)
  { idx += num_param_sets_[i]; }

  name_.push_back(name);
  params_.push_back(params);
  num_param_sets_.push_back(row);
  num_params_.push_back(col);
  starting_index_.push_back(idx);

  if (strcmp(name, "g4") == 0 || strcmp(name, "g5") == 0)
  { has_three_body_ = true; }
}

void Descriptor::set_feature_mean_and_std(bool normalize,
                                          int const size,
                                          double const * means,
                                          double const * stds)
{
  normalize_ = normalize;
  for (int i = 0; i < size; i++)
  {
    feature_mean_.push_back(means[i]);
    feature_std_.push_back(stds[i]);
  }
}

int Descriptor::get_num_descriptors()
{
  int N = 0;

  for (size_t i = 0; i < num_param_sets_.size(); i++)
  { N += num_param_sets_.at(i); }
  return N;
}

//*****************************************************************************
// Compute the descriptor values and their derivatives w.r.t. the coordinates of
// atom i and its neighbors.
// Note `grad_desc` should be of length numDesc*(numnei+1)*DIM. The last DIM
// associated with each descriptor whose last store the derivate of descriptors
// w.r.t. the coordinates of atom i.
//*****************************************************************************

void Descriptor::generate_one_atom(int const i,
                                   double const * coords,
                                   int const * particleSpeciesCodes,
                                   int const * neighlist,
                                   int const numnei,
                                   double * const desc,
                                   double * const grad_desc,
                                   bool grad)
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
    double rcutij = rcut_2D_[iSpecies][jSpecies];

    // Compute rij
    double rij[DIM];
    for (int dim = 0; dim < DIM; ++dim)
    { rij[dim] = coordinates[j][dim] - coordinates[i][dim]; }
    double const rijsq = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
    double const rijmag = sqrt(rijsq);

    // if particles i and j not interact
    if (rijmag > rcutij) { continue; }

    // two-body descriptors
    for (size_t p = 0; p < name_.size(); p++)
    {
      if (name_[p] != "g1" && name_[p] != "g2" && name_[p] != "g3")
      { continue; }
      int idx = starting_index_[p];

      for (int q = 0; q < num_param_sets_[p]; q++)
      {
        double gc = 0;
        double dgcdr_two = 0;
        if (name_[p] == "g1")
        {
          if (grad) { sym_d_g1(rijmag, rcutij, gc, dgcdr_two); }
          else
          {
            sym_g1(rijmag, rcutij, gc);
          }
        }
        else if (name_[p] == "g2")
        {
          double eta = params_[p][q][0];
          double Rs = params_[p][q][1];
          if (grad) { sym_d_g2(eta, Rs, rijmag, rcutij, gc, dgcdr_two); }
          else
          {
            sym_g2(eta, Rs, rijmag, rcutij, gc);
          }
        }
        else if (name_[p] == "g3")
        {
          double kappa = params_[p][q][0];
          if (grad) { sym_d_g3(kappa, rijmag, rcutij, gc, dgcdr_two); }
          else
          {
            sym_g3(kappa, rijmag, rcutij, gc);
          }
        }

        desc[idx] += gc;
        if (grad)
        {
          int page = idx * DIM * (numnei + 1);
          for (int dim = 0; dim < DIM; ++dim)
          {
            double pair = dgcdr_two * rij[dim] / rijmag;
            grad_desc[page + numnei * DIM + dim] -= pair;
            grad_desc[page + jj * DIM + dim] += pair;
          }
        }
        idx += 1;

      }  // loop over same descriptor but different parameter set
    }  // loop over descriptors

    // three-body descriptors
    if (has_three_body_ == false) continue;

    for (int kk = jj + 1; kk < numnei; ++kk)
    {
      // adjust index of particle neighbor
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
      double const rikmag = sqrt(riksq);
      double const rjkmag = sqrt(rjksq);

      if (rikmag > rcutik) continue;  // three-dody not interacting

      double const rvec[3] = {rijmag, rikmag, rjkmag};
      double const rcutvec[3] = {rcutij, rcutik, rcutjk};

      for (size_t p = 0; p < name_.size(); p++)
      {
        if (name_[p] != "g4" && name_[p] != "g5") { continue; }
        int idx = starting_index_[p];

        for (int q = 0; q < num_param_sets_[p]; q++)
        {
          double gc = 0;
          double dgcdr_three[3] = {0, 0, 0};
          if (name_[p] == "g4")
          {
            double zeta = params_[p][q][0];
            double lambda = params_[p][q][1];
            double eta = params_[p][q][2];
            if (grad)
            { sym_d_g4(zeta, lambda, eta, rvec, rcutvec, gc, dgcdr_three); }
            else
            {
              sym_g4(zeta, lambda, eta, rvec, rcutvec, gc);
            }
          }
          else if (name_[p] == "g5")
          {
            double zeta = params_[p][q][0];
            double lambda = params_[p][q][1];
            double eta = params_[p][q][2];
            if (grad)
            { sym_d_g5(zeta, lambda, eta, rvec, rcutvec, gc, dgcdr_three); }
            else
            {
              sym_g5(zeta, lambda, eta, rvec, rcutvec, gc);
            }
          }

          desc[idx] += gc;

          if (grad)
          {
            int page = idx * DIM * (numnei + 1);
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
          idx += 1;

        }  // loop over same descriptor but different parameter set
      }  // loop over descriptors
    }  // loop over kk
  }  // loop over jj
}

//*****************************************************************************
// Symmetry functions: Jorg Behler, J. Chem. Phys. 134, 074106, 2011.
//*****************************************************************************

void Descriptor::sym_g1(double r, double rcut, double & phi)
{
  phi = cutoff_func_(r, rcut);
}

void Descriptor::sym_d_g1(double r, double rcut, double & phi, double & dphi)
{
  phi = cutoff_func_(r, rcut);
  dphi = d_cutoff_func_(r, rcut);
}

void Descriptor::sym_g2(
    double eta, double Rs, double r, double rcut, double & phi)
{
  phi = exp(-eta * (r - Rs) * (r - Rs)) * cutoff_func_(r, rcut);
}

void Descriptor::sym_d_g2(
    double eta, double Rs, double r, double rcut, double & phi, double & dphi)
{
  if (r > rcut)
  {
    phi = 0.;
    dphi = 0.;
  }
  else
  {
    double eterm = exp(-eta * (r - Rs) * (r - Rs));
    double determ = -2 * eta * (r - Rs) * eterm;
    double fc = cutoff_func_(r, rcut);
    double dfc = d_cutoff_func_(r, rcut);
    phi = eterm * fc;
    dphi = determ * fc + eterm * dfc;
  }
}

void Descriptor::sym_g3(double kappa, double r, double rcut, double & phi)
{
  phi = cos(kappa * r) * cutoff_func_(r, rcut);
}

void Descriptor::sym_d_g3(
    double kappa, double r, double rcut, double & phi, double & dphi)
{
  double costerm = cos(kappa * r);
  double dcosterm = -kappa * sin(kappa * r);
  double fc = cutoff_func_(r, rcut);
  double dfc = d_cutoff_func_(r, rcut);

  phi = costerm * fc;
  dphi = dcosterm * fc + costerm * dfc;
}

void Descriptor::sym_g4(double zeta,
                        double lambda,
                        double eta,
                        const double * r,
                        const double * rcut,
                        double & phi)
{
  double rij = r[0];
  double rik = r[1];
  double rjk = r[2];
  double rcutij = rcut[0];
  double rcutik = rcut[1];
  double rcutjk = rcut[2];
  double rijsq = rij * rij;
  double riksq = rik * rik;
  double rjksq = rjk * rjk;

  if (rij > rcutij || rik > rcutik || rjk > rcutjk) { phi = 0.0; }
  else
  {
    // i is the apex atom
    double cos_ijk = (rijsq + riksq - rjksq) / (2 * rij * rik);

    double costerm;
    double base = 1 + lambda * cos_ijk;
    // prevent numerical instability (when lambda=-1 and cos_ijk=1)
    if (base <= 0) { costerm = 0; }
    else
    {
      costerm = pow(base, zeta);
    }

    double eterm = exp(-eta * (rijsq + riksq + rjksq));

    phi = pow(2, 1 - zeta) * costerm * eterm * cutoff_func_(rij, rcutij)
          * cutoff_func_(rik, rcutik) * cutoff_func_(rjk, rcutjk);
  }
}

void Descriptor::sym_d_g4(double zeta,
                          double lambda,
                          double eta,
                          const double * r,
                          const double * rcut,
                          double & phi,
                          double * const dphi)
{
  double rij = r[0];
  double rik = r[1];
  double rjk = r[2];
  double rcutij = rcut[0];
  double rcutik = rcut[1];
  double rcutjk = rcut[2];
  double rijsq = rij * rij;
  double riksq = rik * rik;
  double rjksq = rjk * rjk;

  if (rij > rcutij || rik > rcutik || rjk > rcutjk)
  {
    phi = 0.0;
    dphi[0] = 0.0;
    dphi[1] = 0.0;
    dphi[2] = 0.0;
  }
  else
  {
    // cosine term, i is the apex atom
    double cos_ijk = (rijsq + riksq - rjksq) / (2 * rij * rik);
    double dcos_dij = (rijsq - riksq + rjksq) / (2 * rijsq * rik);
    double dcos_dik = (riksq - rijsq + rjksq) / (2 * rij * riksq);
    double dcos_djk = -rjk / (rij * rik);

    double costerm;
    double dcosterm_dcos;
    double base = 1 + lambda * cos_ijk;
    // prevent numerical instability (when lambda=-1 and cos_ijk=1)
    if (base <= 0)
    {
      costerm = 0.0;
      dcosterm_dcos = 0.0;
    }
    else
    {
      double power = pow(base, zeta);
      double power_minus1 = power / base;
      costerm = power;
      dcosterm_dcos = zeta * power_minus1 * lambda;
    }
    double dcosterm_dij = dcosterm_dcos * dcos_dij;
    double dcosterm_dik = dcosterm_dcos * dcos_dik;
    double dcosterm_djk = dcosterm_dcos * dcos_djk;

    // exponential term
    double eterm = exp(-eta * (rijsq + riksq + rjksq));
    double determ_dij = -2 * eterm * eta * rij;
    double determ_dik = -2 * eterm * eta * rik;
    double determ_djk = -2 * eterm * eta * rjk;

    // power 2 term
    double p2 = pow(2, 1 - zeta);

    // cutoff_func
    double fcij = cutoff_func_(rij, rcutij);
    double fcik = cutoff_func_(rik, rcutik);
    double fcjk = cutoff_func_(rjk, rcutjk);
    double fcprod = fcij * fcik * fcjk;
    double dfcprod_dij = d_cutoff_func_(rij, rcutij) * fcik * fcjk;
    double dfcprod_dik = d_cutoff_func_(rik, rcutik) * fcij * fcjk;
    double dfcprod_djk = d_cutoff_func_(rjk, rcutjk) * fcij * fcik;

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

void Descriptor::sym_g5(double zeta,
                        double lambda,
                        double eta,
                        const double * r,
                        const double * rcut,
                        double & phi)
{
  double rij = r[0];
  double rik = r[1];
  double rjk = r[2];
  double rcutij = rcut[0];
  double rcutik = rcut[1];
  double rijsq = rij * rij;
  double riksq = rik * rik;
  double rjksq = rjk * rjk;

  if (rij > rcutij || rik > rcutik) { phi = 0.0; }
  else
  {
    // i is the apex atom
    double cos_ijk = (rijsq + riksq - rjksq) / (2 * rij * rik);

    double costerm;
    double base = 1 + lambda * cos_ijk;
    // prevent numerical instability (when lambda=-1 and cos_ijk=1)
    if (base <= 0) { costerm = 0; }
    else
    {
      costerm = pow(base, zeta);
    }

    double eterm = exp(-eta * (rijsq + riksq));

    phi = pow(2, 1 - zeta) * costerm * eterm * cutoff_func_(rij, rcutij)
          * cutoff_func_(rik, rcutik);
  }
}

void Descriptor::sym_d_g5(double zeta,
                          double lambda,
                          double eta,
                          const double * r,
                          const double * rcut,
                          double & phi,
                          double * const dphi)
{
  double rij = r[0];
  double rik = r[1];
  double rjk = r[2];
  double rcutij = rcut[0];
  double rcutik = rcut[1];
  double rijsq = rij * rij;
  double riksq = rik * rik;
  double rjksq = rjk * rjk;

  if (rij > rcutij || rik > rcutik)
  {
    phi = 0.0;
    dphi[0] = 0.0;
    dphi[1] = 0.0;
    dphi[2] = 0.0;
  }
  else
  {
    // cosine term, i is the apex atom
    double cos_ijk = (rijsq + riksq - rjksq) / (2 * rij * rik);
    double dcos_dij = (rijsq - riksq + rjksq) / (2 * rijsq * rik);
    double dcos_dik = (riksq - rijsq + rjksq) / (2 * rij * riksq);
    double dcos_djk = -rjk / (rij * rik);

    double costerm;
    double dcosterm_dcos;
    double base = 1 + lambda * cos_ijk;
    // prevent numerical instability (when lambda=-1 and cos_ijk=1)
    if (base <= 0)
    {
      costerm = 0.0;
      dcosterm_dcos = 0.0;
    }
    else
    {
      costerm = pow(base, zeta);
      dcosterm_dcos = zeta * pow(base, zeta - 1) * lambda;
    }
    double dcosterm_dij = dcosterm_dcos * dcos_dij;
    double dcosterm_dik = dcosterm_dcos * dcos_dik;
    double dcosterm_djk = dcosterm_dcos * dcos_djk;

    // exponential term
    double eterm = exp(-eta * (rijsq + riksq));
    double determ_dij = -2 * eterm * eta * rij;
    double determ_dik = -2 * eterm * eta * rik;

    // power 2 term
    double p2 = pow(2, 1 - zeta);

    // cutoff_func
    double fcij = cutoff_func_(rij, rcutij);
    double fcik = cutoff_func_(rik, rcutik);
    double fcprod = fcij * fcik;
    double dfcprod_dij = d_cutoff_func_(rij, rcutij) * fcik;
    double dfcprod_dik = d_cutoff_func_(rik, rcutik) * fcij;

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
