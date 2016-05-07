#include "include/medipack.h"

namespace medi {

  template<typename DATATYPE >
  void TAMPI_Reduce_b(Handle* h) {

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
      }

      TAMPI_Reduce<typename DATATYPE::ModifiedNested >(sendbufMod,recvbufMod,count, *op.modifiedOp,root,comm);
      DATATYPE::Tool::addToolAction(h);

      DATATYPE::handleSendBuffer(sendbuf, sendbufMod, count, h);
      if(root == getRank(comm)) {
        DATATYPE::handleRecvBuffer(recvbuf, recvbufMod, count, h);

        if(NULL != h && op.requiresPrimal) {
          DATATYPE::getValues(sendbuf, count, h->recvPrimals);
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
