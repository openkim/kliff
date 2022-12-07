// Author: Mingjian Wen (wenxx151@gmail.com)

#ifndef HELPER_HPP_
#define HELPER_HPP_

#include <cmath>
#include <cstdlib>
#include <iostream>

#define SMALL 1.0e-10

#define MY_WARNING(message)                                           \
  {                                                                   \
    std::cerr << "* Error (Neighbor List) : \"" << message            \
              << "\" : " << __LINE__ << ":" << __FILE__ << std::endl; \
  }

// norm of a 3-element vector
inline double norm(double const * a)
{
  return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}


// dot product of two 3-element vectors
inline double dot(double const * a, double const * b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}


// cross product of two 3-element vectors
inline void cross(double const * a, double const * b, double * const axb)
{
  axb[0] = a[1] * b[2] - a[2] * b[1];
  axb[1] = a[2] * b[0] - a[0] * b[2];
  axb[2] = a[0] * b[1] - a[1] * b[0];
}


// determinant of a 3 by 3 matrix
inline double det(double const * mat)
{
  return mat[0] * mat[4] * mat[8] - mat[0] * mat[5] * mat[7]
         - mat[1] * mat[3] * mat[8] + mat[1] * mat[5] * mat[6]
         + mat[2] * mat[3] * mat[7] - mat[2] * mat[4] * mat[6];
}


// determinant of a 2 by 2 matrix
inline double det2(double a11, double a12, double a21, double a22)
{
  return (a11 * a22) - (a12 * a21);
}


// transpose of a 3 by 3 matrix
inline void transpose(double const * mat, double * const trans)
{
  trans[0] = mat[0];
  trans[1] = mat[3];
  trans[2] = mat[6];
  trans[3] = mat[1];
  trans[4] = mat[4];
  trans[5] = mat[7];
  trans[6] = mat[2];
  trans[7] = mat[5];
  trans[8] = mat[8];
}


// inverse of a 3 by 3 matrix
inline int inverse(double const * mat, double * const inv)
{
  inv[0] = det2(mat[4], mat[5], mat[7], mat[8]);
  inv[1] = det2(mat[2], mat[1], mat[8], mat[7]);
  inv[2] = det2(mat[1], mat[2], mat[4], mat[5]);
  inv[3] = det2(mat[5], mat[3], mat[8], mat[6]);
  inv[4] = det2(mat[0], mat[2], mat[6], mat[8]);
  inv[5] = det2(mat[2], mat[0], mat[5], mat[3]);
  inv[6] = det2(mat[3], mat[4], mat[6], mat[7]);
  inv[7] = det2(mat[1], mat[0], mat[7], mat[6]);
  inv[8] = det2(mat[0], mat[1], mat[3], mat[4]);

  double const dd = det(mat);
  if (std::abs(dd) < SMALL)
  {
    MY_WARNING("Cannot invert cell matrix. Determinant is 0.");
    return 1;
  }
  for (int i = 0; i < 9; i++) { inv[i] /= dd; }
  return 0;
}


inline void coords_to_index(double const * x,
                            int const * size,
                            double const * max,
                            double const * min,
                            int * const index)
{
  index[0] = static_cast<int>(((x[0] - min[0]) / (max[0] - min[0])) * size[0]);
  index[0] = std::min(index[0], size[0] - 1);  // edge case when x[0] = max[0]
  index[1] = static_cast<int>(((x[1] - min[1]) / (max[1] - min[1])) * size[1]);
  index[1] = std::min(index[1], size[1] - 1);  // edge case when x[1] = max[1]
  index[2] = static_cast<int>(((x[2] - min[2]) / (max[2] - min[2])) * size[2]);
  index[2] = std::min(index[2], size[2] - 1);  // edge case when x[2] = max[2]
}

#endif  // HELPER_HPP_
