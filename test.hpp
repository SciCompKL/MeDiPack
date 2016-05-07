#include "include/medipack.h"

namespace medi {

  template<typename DATATYPE >
  void TAMPI_Reduce_b(Handle* h) {
    TAMPI_Op op = h->op;
    int root = h->root;
    MPI_Comm comm = h->comm;

    typename DATATYPE::AdjointType* sendAdjoints = NULL;
    /*typename DATATYPE::PassiveType*/double* sendPrimals = NULL;
    int sendBuffSize = 0;
    if(root == getRank(comm)) {
      sendBuffSize = allocateReverseBuffer(sendAdjoints, sendPrimals, h->recvCount, op.requiresPrimalSend);

      DATATYPE::getAdjoints(h->recvIndices, h->recvCount, sendAdjoints);
      op.preAdjointOperation(sendAdjoints, h->recvPrimals, h->recvCount);

      if(op.requiresPrimalSend) {
        copyPrimals(sendPrimals, h->recvPrimals, h->recvCount);
      }
    }

    typename DATATYPE::AdjointType* recvAdjoints = NULL;
    /*typename DATATYPE::PassiveType*/double* recvPrimals = NULL;
    int recvBuffSize = allocateReverseBuffer(recvAdjoints, recvPrimals, h->sendCount, op.requiresPrimalSend);

    if(root == getRank(comm)) {
      MPI_Bcast(sendAdjoints, sendBuffSize, MPI_BYTE, root, comm);
      deleteReverseBuffer(recvAdjoints, recvPrimals, op.requiresPrimalSend);
      recvAdjoints = sendAdjoints;
      recvPrimals = sendPrimals;
      sendAdjoints = NULL;
      sendPrimals = NULL;
    } else {
      MPI_Bcast(recvAdjoints, recvBuffSize, MPI_BYTE, root, comm);
    }

    op.postAdjointOperation(recvAdjoints, h->sendPrimals, recvPrimals, h->sendCount);

    DATATYPE::updateAdjoints(h->sendIndices, h->sendCount, recvAdjoints);

    deleteReverseBuffer(recvAdjoints, recvPrimals, op.requiresPrimalSend);
    if(root == getRank(comm)) {
      deleteReverseBuffer(sendAdjoints, sendPrimals, op.requiresPrimalSend);
    }
  }

  template<typename DATATYPE>
  void TAMPI_Reduce(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count, TAMPI_Op op, int root, MPI_Comm comm) {
    if(!DATATYPE::Tool::IS_ActiveType) {
      MPI_Reduce(sendbuf,recvbuf,count,DATATYPE::Tool::MPIType, op.primalFunction ,root,comm);
    } else {
      Handle* h = NULL;
      if(DATATYPE::Tool::isHandleRequired()) {
        h = new Handle();
      }
      DATATYPE::Tool::startAssembly(h);

      if(op.requiresPrimal) {
        DATATYPE::getValues(sendbuf, count, h->sendPrimals);
      }

      typename DATATYPE::ModifiedType* sendbufMod = DATATYPE::prepareSendBuffer(sendbuf, count, h);
      typename DATATYPE::ModifiedType* recvbufMod = NULL;
      if(NULL != h && root == getRank(comm)) {
        recvbufMod = DATATYPE::prepareRecvBuffer(recvbuf, count, h);
      }

      if(NULL != h) {
        h->func = TAMPI_Reduce_b<DATATYPE>;
        h->op = op;
        h->root = root;
        h->comm = comm;
      }

      TAMPI_Reduce<typename DATATYPE::ModifiedNested >(sendbufMod,recvbufMod,count, *op.modifiedOp,root,comm);
      DATATYPE::Tool::addToolAction(h);

      DATATYPE::handleSendBuffer(sendbuf, sendbufMod, count, h);
      if(root == getRank(comm)) {
        DATATYPE::handleRecvBuffer(recvbuf, recvbufMod, count, h);

        if(NULL != h && op.requiresPrimal) {
          DATATYPE::getValues(recvbuf, count, h->recvPrimals);
        }
      }

      DATATYPE::Tool::stopAssembly(h);

      // do not delete h it is stored in the ad tool
    }
  }

  template<>
  void TAMPI_Reduce<EmptyDataType>(const typename EmptyDataType::Type* sendbuf, typename EmptyDataType::Type* recvbuf, int count, TAMPI_Op op, int root, MPI_Comm comm) {
    MEDI_UNUSED(sendbuf);
    MEDI_UNUSED(recvbuf);
    MEDI_UNUSED(count);
    MEDI_UNUSED(op);
    MEDI_UNUSED(root);
    MEDI_UNUSED(comm);
  }
}
