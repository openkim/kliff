#include "helper.hpp"

std::string
FormatMessageFileLineFunctionMessage(std::string const & message1,
                                     std::string const & fileName,
                                     long lineNumber,
                                     std::string const & functionName,
                                     std::string const & message2)
{
  std::ostringstream ss;
  ss << "\n";
  ss << message1 << ":" << fileName << ":" << lineNumber << ":@("
     << functionName << ")\n";
  ss << message2 << "\n\n";
  return ss.str();
}
template<>
_Array_Basic<std::string>::_Array_Basic(std::size_t const count) :
    m(count, "\0")
{
}

template<>
inline void Array2D<std::string>::resize(int const extentZero,
                                         int const extentOne)
{
  _extentZero = extentZero;
  _extentOne = extentOne;
  std::size_t const _n = _extentZero * _extentOne;
  this->m.resize(_n, "\0");
}

template<>
inline void Array3D<std::string>::resize(int const extentZero,
                                         int const extentOne,
                                         int const extentTwo)
{
  _extentZero = extentZero;
  _extentOne = extentOne;
  _extentTwo = extentTwo;
  std::size_t const _n = _extentZero * _extentOne * _extentTwo;
  this->m.resize(_n, "\0");
}

template<>
inline void Array4D<std::string>::resize(int const extentZero,
                                         int const extentOne,
                                         int const extentTwo,
                                         int const extentThree)
{
  _extentZero = extentZero;
  _extentOne = extentOne;
  _extentTwo = extentTwo;
  _extentThree = extentThree;
  std::size_t const _n = _extentZero * _extentOne * _extentTwo * _extentThree;
  this->m.resize(_n, "\0");
}

template<>
inline void Array5D<std::string>::resize(int const extentZero,
                                         int const extentOne,
                                         int const extentTwo,
                                         int const extentThree,
                                         int const extentFour)
{
  _extentZero = extentZero;
  _extentOne = extentOne;
  _extentTwo = extentTwo;
  _extentThree = extentThree;
  _extentFour = extentFour;
  std::size_t const _n
      = _extentZero * _extentOne * _extentTwo * _extentThree * _extentFour;
  this->m.resize(_n, "\0");
}
