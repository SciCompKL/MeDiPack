#pragma once

#include "medipack.h"

namespace medi {

  template<typename DATATYPE>
  void TAMPI_Reduce_adj(typename DATATYPE::AdjointType* sendbufAdjoints, typename DATATYPE::PassiveType* sendbufPrimals, int sendbufSize, typename DATATYPE::AdjointType* recvbufAdjoints, typename DATATYPE::PassiveType* recvbufPrimals, int recvbufSize, int count, TAMPI_Op op, int root, TAMPI_Comm comm) {
    if(root == getRank(comm)) {
      MPI_Bcast(recvbufAdjoints, recvbufSize, MPI_BYTE, root, comm);
      deleteReverseBuffer(sendbufAdjoints, sendbufPrimals, op.requiresPrimalSend);
      sendbufAdjoints = recvbufAdjoints;
      sendbufPrimals = recvbufPrimals;
      recvbufAdjoints = NULL;
      recvbufPrimals = NULL;
    } else {
      MPI_Bcast(sendbufAdjoints, sendbufSize, MPI_BYTE, root, comm);
    }
  }
}
