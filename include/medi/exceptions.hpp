#pragma once

#include <cstdio>
#include <cstdarg>


namespace medi {
  /**
   * @brief Generates an exception.
   *
   * @param ...  Arguments for a printf like output and format.
   */
  #define MEDI_EXCEPTION(...) medi::outputException( __func__, __FILE__, __LINE__, __VA_ARGS__)

  /**
   * @brief Prints the positions and the message of the exception.
   *
   * The position and function where the exceptions occurred is printed. The message will be handled
   * and formatted by a printf like function.
   *
   * @param[in] function  Name of the function from which the exception is generated.
   * @param[in]     file  File where the exception was generated.
   * @param[in]     line  Line inside the file where the exception was generated.
   * @param[in]  message  The exception message and the arguments for the formatting in the message.
   */
  inline void outputException(const char function[], const char file[], const int line, const char* message, ...) {
    fprintf(stderr, "Error in function %s (%s:%d)\nThe message is: ", function, file, line);

    va_list vl;
    va_start(vl, message);
    vfprintf(stderr, message, vl);
    va_end(vl);

    fprintf(stderr, "\n");
    exit(-1);
  }
}

