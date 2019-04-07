#ifndef HELPER_H_
#define HELPER_H_

#include <cmath>
#include <cstddef>

// typedefs
typedef double   VectorOfSizeDIM[3];
typedef double   VectorOfSizeSix[6];


// 1D Array
//******************************************************************************
template<class T>
void AllocateAndInitialize1DArray(T*& arrayPtr, int const extent)
{
  arrayPtr = new T[extent];
  for (int i = 0; i < extent; ++i) {
    arrayPtr[i] = 0.0;
  }
}


// deallocate memory
template<class T>
void Deallocate1DArray(T*& arrayPtr)
{
  delete [] arrayPtr;
  // nullify pointer
  arrayPtr = NULL;
}


// 2D Array
//******************************************************************************
// allocate memory and set pointers
template<class T>
void AllocateAndInitialize2DArray(T**& arrayPtr, int const extentZero,
    int const extentOne)
{
  arrayPtr = new T*[extentZero];
  arrayPtr[0] = new T[extentZero * extentOne];
  for (int i = 1; i < extentZero; ++i) {
    arrayPtr[i] = arrayPtr[i - 1] + extentOne;
  }

  // initialize
  for (int i = 0; i < extentZero; ++i) {
    for (int j = 0; j < extentOne; ++j) {
      arrayPtr[i][j] = 0.0;
    }
  }
}


// deallocate memory
template<class T>
void Deallocate2DArray(T**& arrayPtr)
{
  if (arrayPtr != NULL) {
    delete [] arrayPtr[0];
  }
  delete [] arrayPtr;

  // nullify pointer
  arrayPtr = NULL;
}


// 3D Array
//******************************************************************************
// allocate memory and set pointers
template<class T>
void AllocateAndInitialize3DArray(T***& arrayPtr, int const extentZero,
    int const extentOne, int const extentTwo)
{
  arrayPtr = new T * *[extentZero];
  arrayPtr[0] = new T*[extentZero * extentOne];
  arrayPtr[0][0] = new T[extentZero * extentOne * extentTwo];

  for (int i = 1; i < extentZero; ++i) {
    arrayPtr[i] = arrayPtr[i - 1] + extentOne;
    arrayPtr[i][0] = arrayPtr[i - 1][0] + extentOne * extentTwo;
  }

  for (int i = 0; i < extentZero; ++i) {
    for (int j = 1; j < extentOne; ++j) {
      arrayPtr[i][j] = arrayPtr[i][j - 1] + extentTwo;
    }
  }

  // initialize
  for (int i = 0; i < extentZero; ++i) {
    for (int j = 0; j < extentOne; ++j) {
      for (int k = 0; k < extentTwo; ++k) {
        arrayPtr[i][j][k] = 0.0;
      }
    }
  }
}


// deallocate memory
template<class T>
void Deallocate3DArray(T***& arrayPtr)
{
  if (arrayPtr != NULL) {
    if (arrayPtr[0] != NULL) {
      delete [] arrayPtr[0][0];
    }
    delete [] arrayPtr[0];
  }
  delete [] arrayPtr;

  // nullify pointer
  arrayPtr = NULL;
}


#endif // HELPER_H_
