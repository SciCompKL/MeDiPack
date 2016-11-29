#pragma once

#include <mpi.h>

namespace medi {
  struct AMPI_IN_PLACE_IMPL {

      template<typename T>
      operator const T*() const {
          return reinterpret_cast<const T*>(MPI_IN_PLACE);
      }

      template<typename T>
      operator T*() const {
          return reinterpret_cast<T*>(MPI_IN_PLACE);
      }

  };

  extern const AMPI_IN_PLACE_IMPL AMPI_IN_PLACE;
}
