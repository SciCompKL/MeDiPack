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

  template<typename DATATYPE>
  int TAMPI_Reduce_global(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count, TAMPI_Op op, int root, TAMPI_Comm comm);
  template<typename SENDTYPE, typename RECVTYPE>
  int TAMPI_Gather(const typename SENDTYPE::Type* sendbuf, int sendcount, typename RECVTYPE::Type* recvbuf, int recvcount, int root, TAMPI_Comm comm);

  template<typename DATATYPE>
  inline int TAMPI_Reduce(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count, TAMPI_Op op, int root, TAMPI_Comm comm) {
    if(op.hasAdjoint) {
      return TAMPI_Reduce_global<DATATYPE>(sendbuf, recvbuf, count, op, root, comm);
    } else {
      // perform a gather and apply the operator locally
      int commSize = getCommSize(comm);

      typename DATATYPE::Type* tempbuf = new typename DATATYPE::Type[count * commSize];

      int rValue = TAMPI_Gather<DATATYPE, DATATYPE>(sendbuf, count, tempbuf, count, root, comm);

      for(int j = 1; j < commSize; ++j) {
        MPI_Reduce_local(&tempbuf[j * count], tempbuf, count, DATATYPE::Tool::MPIType, op.primalFunction);
      }

      for(int i = 0; i < count; ++i) {
        recvbuf[i] = tempbuf[i];
      }

      delete [] tempbuf;

      return rValue;
    }
  }

  inline int MPI_Reduce_global(const void* sendbuf, void* recvbuf, int count, MPI_Datatype type, MPI_Op op, int root, MPI_Comm comm) {
    return MPI_Reduce(sendbuf, recvbuf, count, type, op, root, comm);
  }
}
