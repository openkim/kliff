#include "helper.hpp"

#include <cstring>

#ifdef MAXLINE
#undef MAXLINE
#endif

#define MAXLINE 20480

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

void getNextDataLine(FILE * const cstream,
                     char * nextLinePtr,
                     int const maxSize,
                     int * endOfFileFlag)
{
  do
  {
    if (std::fgets(nextLinePtr, maxSize, cstream) == NULL)
    {
      *endOfFileFlag = 1;
      break;
    }

    while (nextLinePtr[0] == ' ' || nextLinePtr[0] == '\t'
           || nextLinePtr[0] == '\n' || nextLinePtr[0] == '\r')
    { nextLinePtr++; }

  } while ((std::strncmp("#", nextLinePtr, 1) == 0)
           || (strlen(nextLinePtr) == 0));

  // remove comments starting with `#' in a line
  char * pch = std::strchr(nextLinePtr, '#');
  if (pch != NULL) { *pch = '\0'; }
}

int getXdouble(char * linePtr, int const N, double * list)
{
  char line[MAXLINE];
  std::strcpy(line, linePtr);

  char * pch;
  pch = std::strtok(line, " \t\n\r");

  int i = 0;
  while (pch != NULL)
  {
    if (std::sscanf(pch, "%lf", &list[i]) != 1) { return true; }

    pch = std::strtok(NULL, " \t\n\r");

    i++;
  }

  return (i != N);
}

int getXint(char * linePtr, int const N, int * list)
{
  char line[MAXLINE];
  std::strcpy(line, linePtr);

  char * pch;
  pch = std::strtok(line, " \t\n\r");

  int i = 0;
  while (pch != NULL)
  {
    if (std::sscanf(pch, "%d", &list[i]) != 1) { return true; }

    pch = std::strtok(NULL, " \t\n\r");

    i++;
  }

  return (i != N);
}

void lowerCase(char * linePtr)
{
  for (int i = 0; linePtr[i]; i++)
  {
    linePtr[i] = static_cast<char>(
        std::tolower(static_cast<unsigned char>(linePtr[i])));
  }
}

void lowerCase(std::string & InputLineArg)
{
  std::transform(InputLineArg.begin(),
                 InputLineArg.end(),
                 InputLineArg.begin(),
                 [](unsigned char c) {
                   unsigned char const l = std::tolower(c);
                   return (l != c) ? l : c;
                 });
}
