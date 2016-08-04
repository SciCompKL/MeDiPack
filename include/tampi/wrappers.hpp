#pragma once

#include "async.hpp"
#include "../medipack.h"

namespace medi {

  template<typename DATATYPE>
  int TAMPI_Bcast_wrap(typename DATATYPE::Type* bufferSend, typename DATATYPE::Type* bufferRecv, int count, DATATYPE* datatype, int root, TAMPI_Comm comm);

  template<typename DATATYPE>
  inline int TAMPI_Bcast(typename DATATYPE::Type* buffer, int count, DATATYPE* datatype, int root, TAMPI_Comm comm) {
    return TAMPI_Bcast_wrap<DATATYPE>(static_cast<typename DATATYPE::Type*>(TAMPI_IN_PLACE), buffer, count, datatype, root, comm);
  }

  inline int MPI_Bcast_wrap(void* bufferSend, void* bufferRecv, int count, MPI_Datatype type, int root, MPI_Comm comm) {
    MEDI_UNUSED(bufferSend);
    return MPI_Bcast(bufferRecv, count, type, root, comm);
  }

  template<typename DATATYPE>
  struct TAMPI_Ireduce_local_Handle : public HandleBase {
      TAMPI_Comm comm;
      int root;
      bool all;
      int count;
      DATATYPE* datatype;
      TAMPI_Op op;
      typename DATATYPE::Type* recvbuf;
      typename DATATYPE::Type* tempbuf;

      HandleBase* origHandle;
      ContinueFunction origFunc;
  };

  template<typename DATATYPE>
  inline void performReduce(bool all, typename DATATYPE::Type* tempbuf, typename DATATYPE::Type* recvbuf, int count, DATATYPE* datatype, TAMPI_Op op, int root, TAMPI_Comm comm) {
    int commSize = getCommSize(comm);
    int commRank = getCommRank(comm);

    if(all || root == commRank) {
      datatype->performReduce(tempbuf, recvbuf, count, op, commSize);

      datatype->deleteTypeBuffer(tempbuf);
    }
  }

  template<typename DATATYPE>
  inline int TAMPI_Ireduce_local_finish(HandleBase* handle) {

    TAMPI_Ireduce_local_Handle<DATATYPE>* h = static_cast<TAMPI_Ireduce_local_Handle<DATATYPE>*>(handle);
    // first call the orignal function
    h->origFunc(h->origHandle);

    performReduce<DATATYPE>(h->all, h->tempbuf, h->recvbuf, h->count, h->datatype, h->op, h->root, h->comm);

    delete h;

    return 0;
  }

  template<typename DATATYPE>
  struct TAMPI_Ireduce_modified_Handle : public HandleBase {
      typename DATATYPE::Type* tempBuf;
      DATATYPE* datatype;
      int root;
      TAMPI_Comm comm;

      HandleBase* origHandle;
      ContinueFunction origFunc;
  };

  template<typename DATATYPE>
  inline int TAMPI_Ireduce_modified_finish(HandleBase* handle) {

    TAMPI_Ireduce_modified_Handle<DATATYPE>* h = static_cast<TAMPI_Ireduce_modified_Handle<DATATYPE>*>(handle);
    // first call the orignal function
    h->origFunc(h->origHandle);

    if(h->root != getCommRank(h->comm)) {
      h->datatype->deleteTypeBuffer(h->tempBuf);
    }

    delete h;

    return 0;
  }


  template<typename DATATYPE>
  int TAMPI_Reduce_global(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count, DATATYPE* datatype, TAMPI_Op op, int root, TAMPI_Comm comm);
  template<typename SENDTYPE, typename RECVTYPE>
  int TAMPI_Gather(const typename SENDTYPE::Type* sendbuf, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::Type* recvbuf, int recvcount, RECVTYPE* recvtype, int root, TAMPI_Comm comm);
  template<typename DATATYPE>
  int TAMPI_Allreduce_global(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count, DATATYPE* datatype, TAMPI_Op op, TAMPI_Comm comm);
  template<typename SENDTYPE, typename RECVTYPE>
  int TAMPI_Allgather(const typename SENDTYPE::Type* sendbuf, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::Type* recvbuf, int recvcount, RECVTYPE* recvtype, TAMPI_Comm comm);
  template<typename DATATYPE>
  int TAMPI_Ireduce_global(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count, DATATYPE* datatype, TAMPI_Op op, int root, TAMPI_Comm comm, TAMPI_Request* request);
  template<typename SENDTYPE, typename RECVTYPE>
  int TAMPI_Igather(const typename SENDTYPE::Type* sendbuf, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::Type* recvbuf, int recvcount, RECVTYPE* recvtype, int root, TAMPI_Comm comm, TAMPI_Request* request);
  template<typename DATATYPE>
  int TAMPI_Iallreduce_global(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count, DATATYPE* datatype, TAMPI_Op op, TAMPI_Comm comm, TAMPI_Request* request);
  template<typename SENDTYPE, typename RECVTYPE>
  int TAMPI_Iallgather(const typename SENDTYPE::Type* sendbuf, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::Type* recvbuf, int recvcount, RECVTYPE* recvtype, TAMPI_Comm comm, TAMPI_Request* request);


  template<typename DATATYPE>
  inline int TAMPI_Reduce(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count, DATATYPE* datatype, TAMPI_Op op, int root, TAMPI_Comm comm) {
    if(!datatype->getADTool().isActiveType()) {
      return TAMPI_Reduce_global<DATATYPE>(sendbuf, recvbuf, count, datatype, op, root, comm);
    } else if(op.hasAdjoint) {
      // operator has an adjoint formulation
      if(op.requiresPrimalSend) {
        // need to modify the call to an allreduce
        typename DATATYPE::Type* tempBuf = recvbuf;
        if(root != getCommRank(comm)) {
          datatype->createTypeBuffer(tempBuf, count);
        }
        int result = TAMPI_Allreduce_global<DATATYPE>(sendbuf, tempBuf, count, datatype, op, comm);
        if(root != getCommRank(comm)) {
          datatype->deleteTypeBuffer(tempBuf);
        }

        return result;
      } else {
        // just perfrom the normal call
        return TAMPI_Reduce_global<DATATYPE>(sendbuf, recvbuf, count, datatype, op, root, comm);
      }
    } else {
      // perform a gather and apply the operator locally
      int commSize = getCommSize(comm);
      int commRank = getCommRank(comm);

      typename DATATYPE::Type* tempbuf = NULL;
      if(root == commRank) {
        datatype->createTypeBuffer(tempbuf, count * commSize);
      }

      const typename DATATYPE::Type* sendbufGather = sendbuf;
      if(TAMPI_IN_PLACE == sendbuf) {
        sendbufGather = recvbuf;
      }
      int rValue = TAMPI_Gather<DATATYPE, DATATYPE>(sendbufGather, count, datatype, tempbuf, count, datatype, root, comm);

      performReduce<DATATYPE>(false, tempbuf, recvbuf, count, datatype, op, root, comm);

      return rValue;
    }
  }

  template<typename DATATYPE>
  inline int TAMPI_Ireduce(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count, DATATYPE* datatype, TAMPI_Op op, int root, TAMPI_Comm comm, TAMPI_Request* request) {
    if(!datatype->getADTool().isActiveType()) {
      return TAMPI_Ireduce_global<DATATYPE>(sendbuf, recvbuf, count, datatype, op, root, comm, request);
    } else if(op.hasAdjoint) {
      if(op.requiresPrimalSend) {
        // need to modify the call to an allreduce
        typename DATATYPE::Type* tempBuf = recvbuf;
        if(root != getCommRank(comm)) {
          datatype->createTypeBuffer(tempBuf, count);
        }
        int result = TAMPI_Iallreduce_global<DATATYPE>(sendbuf, tempBuf, count, datatype, op, comm, request);
        TAMPI_Ireduce_modified_Handle<DATATYPE>* curHandle = new TAMPI_Ireduce_modified_Handle<DATATYPE>();
        curHandle->tempBuf = tempBuf;
        curHandle->comm = comm;
        curHandle->root = root;
        curHandle->datatype = datatype;
        curHandle->origHandle = request->handle;
        curHandle->origFunc = request->func;

        // set our own handle now to the request
        request->handle = curHandle;
        request->func = (ContinueFunction)TAMPI_Ireduce_modified_finish<DATATYPE>;

        return result;
      } else {
        // just perfrom the normal call
        return TAMPI_Ireduce_global<DATATYPE>(sendbuf, recvbuf, count, datatype, op, root, comm, request);
      }
    } else {
      // perform a gather and apply the operator locally
      int commSize = getCommSize(comm);
      int commRank = getCommRank(comm);

      typename DATATYPE::Type* tempbuf = NULL;
      if(root == commRank) {
        datatype->createTypeBuffer(tempbuf, count * commSize);
      }

      const typename DATATYPE::Type* sendbufGather = sendbuf;
      if(TAMPI_IN_PLACE == sendbuf) {
        sendbufGather = recvbuf;
      }
      int rValue = TAMPI_Igather<DATATYPE, DATATYPE>(sendbufGather, count, datatype, tempbuf, count, datatype, root, comm, request);
      TAMPI_Ireduce_local_Handle<DATATYPE>* curHandle = new TAMPI_Ireduce_local_Handle<DATATYPE>();
      curHandle->comm = comm;
      curHandle->root = root;
      curHandle->all = false;
      curHandle->count = count;
      curHandle->op = op;
      curHandle->datatype = datatype;
      curHandle->recvbuf = recvbuf;
      curHandle->tempbuf = tempbuf;
      curHandle->origHandle = request->handle;
      curHandle->origFunc = request->func;

      // set our own handle now to the request
      request->handle = curHandle;
      request->func = (ContinueFunction)TAMPI_Ireduce_local_finish<DATATYPE>;

      return rValue;
    }
  }

  template<typename DATATYPE>
  inline int TAMPI_Allreduce(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count, DATATYPE* datatype, TAMPI_Op op, TAMPI_Comm comm) {
    if(op.hasAdjoint || !datatype->getADTool().isActiveType()) {
      return TAMPI_Allreduce_global<DATATYPE>(sendbuf, recvbuf, count, datatype, op, comm);
    } else {
      // perform a gather and apply the operator locally
      int commSize = getCommSize(comm);

      typename DATATYPE::Type* tempbuf = NULL;
      datatype->createTypeBuffer(tempbuf, count * commSize);

      const typename DATATYPE::Type* sendbufGather = sendbuf;
      if(TAMPI_IN_PLACE == sendbuf) {
        sendbufGather = recvbuf;
      }
      int rValue = TAMPI_Allgather<DATATYPE, DATATYPE>(sendbufGather, count, datatype, tempbuf, count, datatype, comm);

      performReduce<DATATYPE>(true, tempbuf, recvbuf, count, datatype, op, -1, comm);

      return rValue;
    }
  }

  template<typename DATATYPE>
  inline int TAMPI_Iallreduce(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count, DATATYPE* datatype, TAMPI_Op op, TAMPI_Comm comm, TAMPI_Request* request) {
    if(op.hasAdjoint || !datatype->getADTool().isActiveType()) {
      return TAMPI_Iallreduce_global<DATATYPE>(sendbuf, recvbuf, count, datatype, op, comm, request);
    } else {
      // perform a gather and apply the operator locally
      int commSize = getCommSize(comm);

      typename DATATYPE::Type* tempbuf = NULL;
      datatype->createTypeBuffer(tempbuf, count * commSize);

      const typename DATATYPE::Type* sendbufGather = sendbuf;
      if(TAMPI_IN_PLACE == sendbuf) {
        sendbufGather = recvbuf;
      }
      int rValue = TAMPI_Iallgather<DATATYPE, DATATYPE>(sendbufGather, count, datatype, tempbuf, count, datatype, comm, request);
      TAMPI_Ireduce_local_Handle<DATATYPE>* curHandle = new TAMPI_Ireduce_local_Handle<DATATYPE>();
      curHandle->comm = comm;
      curHandle->root = -1;
      curHandle->all = true;
      curHandle->count = count;
      curHandle->op = op;
      curHandle->datatype = datatype;
      curHandle->recvbuf = recvbuf;
      curHandle->tempbuf = tempbuf;
      curHandle->origHandle = request->handle;
      curHandle->origFunc = request->func;

      // set our own handle now to the request
      request->handle = curHandle;
      request->func = (ContinueFunction)TAMPI_Ireduce_local_finish<DATATYPE>;

      return rValue;
    }
  }
}
