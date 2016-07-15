#pragma once

#include "../medipack.h"

namespace medi {

  template<typename DATATYPE>
  int TAMPI_Bcast_wrap(typename DATATYPE::Type* bufferSend, typename DATATYPE::Type* bufferRecv, int count, int root, TAMPI_Comm comm);

  template<typename DATATYPE>
  inline int TAMPI_Bcast(typename DATATYPE::Type* buffer, int count, int root, TAMPI_Comm comm) {
    return TAMPI_Bcast_wrap<DATATYPE>(static_cast<typename DATATYPE::Type*>(TAMPI_IN_PLACE), buffer, count, root, comm);
  }

  inline int MPI_Bcast_wrap(void* bufferSend, void* bufferRecv, int count, MPI_Datatype type, int root, MPI_Comm comm) {
    MEDI_UNUSED(bufferSend);
    return MPI_Bcast(bufferRecv, count, type, root, comm);
  }
}
