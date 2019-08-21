#ifndef KLIFF_HELPER_HPP_
#define KLIFF_HELPER_HPP_

#include <vector>

#define DIM 3

/*!
 * \brief 1-dimensional array type
 *
 * \tparam T data type. Defaults to double
 */
template <class T = double>
using Array1D = std::vector<T>;

/*!
 * \brief 2-dimensional array type
 *
 * \tparam T data type. Defaults to double
 */
template <class T = double>
using Array2D = std::vector<Array1D<T>>;

/*!
 * \brief 3-dimensional array type
 *
 * \tparam T data type. Defaults to double
 */
template <class T = double>
using Array3D = std::vector<Array2D<T>>;

/*!
 * \brief 4-dimensional array type
 *
 * \tparam T data type. Defaults to double
 */
template <class T = double>
using Array4D = std::vector<Array3D<T>>;

/*!
 * \brief 5-dimensional array type
 *
 * \tparam T data type. Defaults to double
 */
template <class T = double>
using Array5D = std::vector<Array4D<T>>;

/*!
 * \brief N-dimensional array type
 *
 * \tparam N dimensionality of the array
 * \tparam T data type. Defaults to double
 */
template<unsigned int N, class T = double>
struct ArrayND
{
    typedef std::vector<typename ArrayND<N - 1, T>::type> type;
};

template<class T>
struct ArrayND<1, T>
{
    typedef Array1D<T> type;
};

/*!
 * \brief Resize the 1-dimensional array and initialize it to zero.
 *
 * \tparam T data type. Defaults to double
 *
 * \param array 1-dimensional array
 * \param extent Size of the array
 */
template <class T = double>
inline void AllocateAndInitialize1DArray(Array1D<T> &array, int const extent)
{
    array.resize(extent, static_cast<T>(0));
}

/*!
 * \brief Resize the 1-dimensional array and initialize it using the input array.
 *
 * \tparam T data type. Defaults to double
 *
 * \param array 1-dimensional array
 * \param extent Size of the array
 * \param array_in The 1-dimensional input array for initialization
 */
template <class T = double>
inline void AllocateAndInitialize1DArray(Array1D<T> &array, int const extent, Array1D<T> const &array_in)
{
    array.resize(extent);
    for (int i = 0; i < extent; i++)
        array[i] = array_in[i];
}

/*!
 * \brief Resize the 1-dimensional array and initialize it using the contiguous input array.
 *
 * \tparam T data type. Defaults to double
 *
 * \param array 1-dimensional array
 * \param extent Size of the array
 * \param array_in The contiguous input array for initialization
 */
template <class T = double>
inline void AllocateAndInitialize1DArray(Array1D<T> &array, int const extent, T const *array_in)
{
    array.resize(extent);
    for (int i = 0; i < extent; i++)
        array[i] = array_in[i];
}

/*!
 * \brief  Resize the 2-dimensional array and initialize it to zero.
 *
 * \tparam T data type. Defaults to double
 *
 * \param array 2-dimensional array
 * \param extentZero Size of the first dimension
 * \param extentOne Size of the second dimension
 */
template <class T = double>
inline void AllocateAndInitialize2DArray(Array2D<T> &array, int const extentZero, int const extentOne)
{
    array.resize(extentZero, Array1D<T>(extentOne, static_cast<T>(0)));
}

/*!
 * \brief  Resize the 2-dimensional array and initialize it using the input array.
 *
 * \tparam T data type. Defaults to double
 *
 * \param array 2-dimensional array
 * \param extentZero Size of the first dimension
 * \param extentOne Size of the second dimension
 * \param array_in The input 2-dimensional array for initialization
 */
template <class T = double>
inline void AllocateAndInitialize2DArray(Array2D<T> &array, int const extentZero, int const extentOne, Array2D<T> const &array_in)
{
    array.resize(extentZero, Array1D<T>(extentOne));
    for (int i = 0; i < extentZero; i++)
        for (int j = 0; j < extentOne; j++)
            array[i][j] = array_in[i][j];
}

/*!
 * \brief  Resize the 2-dimensional array and initialize it using the contiguous input array.
 *
 * \tparam T data type. Defaults to double
 *
 * \param array 2-dimensional array
 * \param extentZero Size of the first dimension
 * \param extentOne Size of the second dimension
 * \param array_in The contiguous input array for initialization
 */
template <class T = double>
inline void AllocateAndInitialize2DArray(Array2D<T> &array, int const extentZero, int const extentOne, T const *array_in)
{
    array.resize(extentZero, Array1D<T>(extentOne));
    for (int i = 0, c = 0; i < extentZero; i++)
        for (int j = 0; j < extentOne; j++)
            array[i][j] = array_in[c++];
}

/*!
 * \brief  Resize the 3-dimensional array and initialize it to zero.
 *
 * \tparam T data type. Defaults to double
 *
 * \param array 3-dimensional array
 * \param extentZero Size of the first dimension
 * \param extentOne Size of the second dimension
 * \param extentTwo Size of the third dimension
 */
template <class T = double>
inline void AllocateAndInitialize3DArray(Array3D<T> &array, int const extentZero, int const extentOne, int const extentTwo)
{
    array.resize(extentZero, Array2D<T>(extentOne, Array1D<T>(extentTwo, static_cast<T>(0))));
}


/*!
 * \brief  Resize the 3-dimensional array and initialize it using the input array.
 *
 * \tparam T data type. Defaults to double
 *
 * \param array 3-dimensional array
 * \param extentZero Size of the first dimension
 * \param extentOne Size of the second dimension
 * \param extentTwo Size of the third dimension
 * \param array_in The input 3-dimensional array for initialization
 */
template <class T = double>
inline void AllocateAndInitialize3DArray(Array3D<T> &array, int const extentZero, int const extentOne, int const extentTwo, Array3D<T> const &array_in)
{
    array.resize(extentZero, Array2D<T>(extentOne, Array1D<T>(extentTwo)));
    for (int i = 0; i < extentZero; i++)
        for (int j = 0; j < extentOne; j++)
            for (int k = 0; k < extentTwo; k++)
                array[i][j][k] = array_in[i][j][k];
}

/*!
 * \brief  Resize the 3-dimensional array and initialize it using the contiguous input array.
 *
 * \tparam T data type. Defaults to double
 *
 * \param array 3-dimensional array
 * \param extentZero Size of the first dimension
 * \param extentOne Size of the second dimension
 * \param extentTwo Size of the third dimension
 * \param array_in The contiguous input array for initialization
 */
template <class T = double>
inline void AllocateAndInitialize3DArray(Array3D<T> &array, int const extentZero, int const extentOne, int const extentTwo, T const *array_in)
{
    array.resize(extentZero, Array2D<T>(extentOne, Array1D<T>(extentTwo)));
    for (int i = 0, c = 0; i < extentZero; i++)
        for (int j = 0; j < extentOne; j++)
            for (int k = 0; k < extentTwo; k++)
                array[i][j][k] = array_in[c++];
}

/*!
 * \brief  Resize the 4-dimensional array and initialize it to zero.
 *
 * \tparam T data type. Defaults to double
 *
 * \param array 4-dimensional array
 * \param extentZero Size of the first dimension
 * \param extentOne Size of the second dimension
 * \param extentTwo Size of the third dimension
 * \param extentThree Size of the forth dimension
 */
template <class T = double>
inline void AllocateAndInitialize4DArray(Array4D<T> &array, int const extentZero, int const extentOne, int const extentTwo, int const extentThree)
{
    array.resize(extentZero, Array3D<T>(extentOne, Array2D<T>(extentTwo, Array1D<T>(extentThree, static_cast<T>(0)))));
}

/*!
 * \brief  Resize the 4-dimensional array and initialize it using the input array.
 *
 * \tparam T data type. Defaults to double
 *
 * \param array 4-dimensional array
 * \param extentZero Size of the first dimension
 * \param extentOne Size of the second dimension
 * \param extentTwo Size of the third dimension
 * \param extentThree Size of the forth dimension
 * \param array_in The input 4-dimensional array for initialization
 */
template <class T = double>
inline void AllocateAndInitialize4DArray(Array4D<T> &array, int const extentZero, int const extentOne, int const extentTwo, int const extentThree, Array4D<T> const &array_in)
{
    array.resize(extentZero, Array3D<T>(extentOne, Array2D<T>(extentTwo, Array1D<T>(extentThree))));
    for (int i = 0; i < extentZero; i++)
        for (int j = 0; j < extentOne; j++)
            for (int k = 0; k < extentTwo; k++)
                for (int l = 0; l < extentThree; l++)
                    array[i][j][k][l] = array_in[i][j][k][l];
}

/*!
 * \brief  Resize the 4-dimensional array and initialize it using the contiguous input array.
 *
 * \tparam T data type. Defaults to double
 *
 * \param array 4-dimensional array
 * \param extentZero Size of the first dimension
 * \param extentOne Size of the second dimension
 * \param extentTwo Size of the third dimension
 * \param extentThree Size of the forth dimension
 * \param array_in The contiguous input array for initialization
 */
template <class T = double>
inline void AllocateAndInitialize4DArray(Array4D<T> &array, int const extentZero, int const extentOne, int const extentTwo, int const extentThree, T const *array_in)
{
    array.resize(extentZero, Array3D<T>(extentOne, Array2D<T>(extentTwo, Array1D<T>(extentThree))));
    for (int i = 0, c = 0; i < extentZero; i++)
        for (int j = 0; j < extentOne; j++)
            for (int k = 0; k < extentTwo; k++)
                for (int l = 0; l < extentThree; l++)
                    array[i][j][k][l] = array_in[c++];
}

/*!
 * \brief  Resize the 5-dimensional array and initialize it to zero.
 *
 * \tparam T data type. Defaults to double
 *
 * \param array 5-dimensional array
 * \param extentZero Size of the first dimension
 * \param extentOne Size of the second dimension
 * \param extentTwo Size of the third dimension
 * \param extentThree Size of the forth dimension
 * \param extentFour Size of the fifth dimension
 */
template <class T = double>
inline void AllocateAndInitialize5DArray(Array5D<T> &array, int const extentZero, int const extentOne, int const extentTwo, int const extentThree, int const extentFour)
{
    array.resize(extentZero, Array4D<T>(extentOne, Array3D<T>(extentTwo, Array2D<T>(extentThree, Array1D<T>(extentFour, static_cast<T>(0))))));
}

/*!
 * \brief  Resize the 5-dimensional array and initialize it using the input array.
 *
 * \tparam T data type. Defaults to double
 *
 * \param array 5-dimensional array
 * \param extentZero Size of the first dimension
 * \param extentOne Size of the second dimension
 * \param extentTwo Size of the third dimension
 * \param extentThree Size of the forth dimension
 * \param extentFour Size of the fifth dimension
 * \param array_in The input 5-dimensional array for initialization
 */
template <class T = double>
inline void AllocateAndInitialize5DArray(Array5D<T> &array, int const extentZero, int const extentOne, int const extentTwo, int const extentThree, int const extentFour, Array5D<T> const &array_in)
{
    array.resize(extentZero, Array4D<T>(extentOne, Array3D<T>(extentTwo, Array2D<T>(extentThree, Array1D<T>(extentFour)))));
    for (int i = 0; i < extentZero; i++)
        for (int j = 0; j < extentOne; j++)
            for (int k = 0; k < extentTwo; k++)
                for (int l = 0; l < extentThree; l++)
                    for (int m = 0; m < extentFour; m++)
                        array[i][j][k][l][m] = array_in[i][j][k][l][m];
}

/*!
 * \brief  Resize the 5-dimensional array and initialize it using the contiguous input array.
 *
 * \tparam T data type. Defaults to double
 *
 * \param array 5-dimensional array
 * \param extentZero Size of the first dimension
 * \param extentOne Size of the second dimension
 * \param extentTwo Size of the third dimension
 * \param extentThree Size of the forth dimension
 * \param extentFour Size of the fifth dimension
 * \param array_in The contiguous input array for initialization
 */
template <class T = double>
inline void AllocateAndInitialize5DArray(Array5D<T> &array, int const extentZero, int const extentOne, int const extentTwo, int const extentThree, int const extentFour, T const *array_in)
{
    array.resize(extentZero, Array4D<T>(extentOne, Array3D<T>(extentTwo, Array2D<T>(extentThree, Array1D<T>(extentFour)))));
    for (int i = 0, c = 0; i < extentZero; i++)
        for (int j = 0; j < extentOne; j++)
            for (int k = 0; k < extentTwo; k++)
                for (int l = 0; l < extentThree; l++)
                    for (int m = 0; m < extentFour; m++)
                        array[i][j][k][l][m] = array_in[c++];
}

#endif // KLIFF_HELPER_HPP_
