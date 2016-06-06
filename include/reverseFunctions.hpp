#pragma once

#include "medipack.h"

namespace medi {

  template<typename DATATYPE>
  void TAMPI_Send_adj(typename DATATYPE::AdjointType* bufAdjoints, typename DATATYPE::PassiveType* bufPrimals, int bufSize, int count, int dest, int tag, TAMPI_Comm comm) {
    MEDI_UNUSED(bufPrimals);
    MEDI_UNUSED(count);
    MPI_Recv(bufAdjoints, bufSize, MPI_BYTE, dest, tag, comm, MPI_STATUS_IGNORE);
  }

  template<typename DATATYPE>
  void TAMPI_Recv_adj(typename DATATYPE::AdjointType* bufAdjoints, typename DATATYPE::PassiveType* bufPrimals, int bufSize, int count, int src, int tag, TAMPI_Comm comm, TAMPI_Status* status) {
    MEDI_UNUSED(bufPrimals);
    MEDI_UNUSED(count);
    MEDI_UNUSED(status);
    MPI_Send(bufAdjoints, bufSize, MPI_BYTE, src, tag, comm);
  }

  template<typename DATATYPE>
  void TAMPI_Reduce_adj(typename DATATYPE::AdjointType* &sendbufAdjoints, typename DATATYPE::PassiveType* &sendbufPrimals, int sendbufSize, typename DATATYPE::AdjointType* &recvbufAdjoints, typename DATATYPE::PassiveType* &recvbufPrimals, int recvbufSize, int count, TAMPI_Op op, int root, TAMPI_Comm comm) {
    MEDI_UNUSED(count);
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
