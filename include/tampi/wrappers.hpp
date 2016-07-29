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
  struct TAMPI_Ireduce_Handle : public HandleBase {
      TAMPI_Comm comm;
      int root;
      bool all;
      int count;
      TAMPI_Op op;
      typename DATATYPE::Type* recvbuf;
      typename DATATYPE::Type* tempbuf;

      HandleBase* origHandle;
      ContinueFunction origFunc;
  };


  template<typename DATATYPE>
  inline void performReduce(bool all, typename DATATYPE::Type* tempbuf, typename DATATYPE::Type* recvbuf, int count, TAMPI_Op op, int root, TAMPI_Comm comm) {
    int commSize = getCommSize(comm);
    int commRank = getCommRank(comm);

    if(all || root == commRank) {
      for(int j = 1; j < commSize; ++j) {
        MPI_Reduce_local(&tempbuf[j * count], tempbuf, count, DATATYPE::Tool::MPIType, op.primalFunction);
      }

      for(int i = 0; i < count; ++i) {
        recvbuf[i] = tempbuf[i];
      }

      delete [] tempbuf;
    }
  }

  template<typename DATATYPE>
  inline int TAMPI_Ireduce_finish(HandleBase* handle) {

    TAMPI_Ireduce_Handle<DATATYPE>* h = static_cast<TAMPI_Ireduce_Handle<DATATYPE>*>(handle);
    // first call the orignal function
    h->origFunc(h->origHandle);

    performReduce<DATATYPE>(h->all, h->tempbuf, h->recvbuf, h->count, h->op, h->root, h->comm);

    return 0;
  }

  template<typename DATATYPE>
  int TAMPI_Reduce_global(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count, DATATYPE* datatype, TAMPI_Op op, int root, TAMPI_Comm comm);
  template<typename SENDTYPE, typename RECVTYPE>
  int TAMPI_Gather(const typename SENDTYPE::Type* sendbuf, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::Type* recvbuf, int recvcount, RECVTYPE* recvtype, int root, TAMPI_Comm comm);

  template<typename DATATYPE>
  inline int TAMPI_Reduce(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count, DATATYPE* datatype, TAMPI_Op op, int root, TAMPI_Comm comm) {
    if(op.hasAdjoint || !DATATYPE::Tool::IS_ActiveType) {
      return TAMPI_Reduce_global<DATATYPE>(sendbuf, recvbuf, count, datatype, op, root, comm);
    } else {
      // perform a gather and apply the operator locally
      int commSize = getCommSize(comm);
      int commRank = getCommRank(comm);

      typename DATATYPE::Type* tempbuf = NULL;
      if(root == commRank) {
        tempbuf = new typename DATATYPE::Type[count * commSize];
      }

      const typename DATATYPE::Type* sendbufGather = sendbuf;
      if(TAMPI_IN_PLACE == sendbuf) {
        sendbufGather = recvbuf;
      }
      int rValue = TAMPI_Gather<DATATYPE, DATATYPE>(sendbufGather, count, datatype, tempbuf, count, datatype, root, comm);

      performReduce<DATATYPE>(false, tempbuf, recvbuf, count, op, root, comm);

      return rValue;
    }
  }

  template<typename DATATYPE>
  int TAMPI_Ireduce_global(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count, DATATYPE* datatype, TAMPI_Op op, int root, TAMPI_Comm comm, TAMPI_Request* request);
  template<typename SENDTYPE, typename RECVTYPE>
  int TAMPI_Igather(const typename SENDTYPE::Type* sendbuf, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::Type* recvbuf, int recvcount, RECVTYPE* recvtype, int root, TAMPI_Comm comm, TAMPI_Request* request);

  template<typename DATATYPE>
  inline int TAMPI_Ireduce(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count, DATATYPE* datatype, TAMPI_Op op, int root, TAMPI_Comm comm, TAMPI_Request* request) {
    if(op.hasAdjoint || !DATATYPE::Tool::IS_ActiveType) {
      return TAMPI_Ireduce_global<DATATYPE>(sendbuf, recvbuf, count, datatype, op, root, comm, request);
    } else {
      // perform a gather and apply the operator locally
      int commSize = getCommSize(comm);
      int commRank = getCommRank(comm);

      typename DATATYPE::Type* tempbuf = NULL;
      if(root == commRank) {
        tempbuf = new typename DATATYPE::Type[count * commSize];
      }

      const typename DATATYPE::Type* sendbufGather = sendbuf;
      if(TAMPI_IN_PLACE == sendbuf) {
        sendbufGather = recvbuf;
      }
      int rValue = TAMPI_Igather<DATATYPE, DATATYPE>(sendbufGather, count, datatype, tempbuf, count, datatype, root, comm, request);
      TAMPI_Ireduce_Handle<DATATYPE>* curHandle = new TAMPI_Ireduce_Handle<DATATYPE>();
      curHandle->comm = comm;
      curHandle->root = root;
      curHandle->all = false;
      curHandle->count = count;
      curHandle->op = op;
      curHandle->recvbuf = recvbuf;
      curHandle->tempbuf = tempbuf;
      curHandle->origHandle = request->handle;
      curHandle->origFunc = request->func;

      // set our own handle now to the request
      request->handle = curHandle;
      request->func = (ContinueFunction)TAMPI_Ireduce_finish<DATATYPE>;

      return rValue;
    }
  }

  template<typename DATATYPE>
  int TAMPI_Allreduce_global(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count, DATATYPE* datatype, TAMPI_Op op, TAMPI_Comm comm);
  template<typename SENDTYPE, typename RECVTYPE>
  int TAMPI_Allgather(const typename SENDTYPE::Type* sendbuf, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::Type* recvbuf, int recvcount, RECVTYPE* recvtype, TAMPI_Comm comm);
  template<typename DATATYPE>
  inline int TAMPI_Allreduce(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count, DATATYPE* datatype, TAMPI_Op op, TAMPI_Comm comm) {
    if(op.hasAdjoint || !DATATYPE::Tool::IS_ActiveType) {
      return TAMPI_Allreduce_global<DATATYPE>(sendbuf, recvbuf, count, datatype, op, comm);
    } else {
      // perform a gather and apply the operator locally
      int commSize = getCommSize(comm);

      typename DATATYPE::Type* tempbuf = new typename DATATYPE::Type[count * commSize];

      const typename DATATYPE::Type* sendbufGather = sendbuf;
      if(TAMPI_IN_PLACE == sendbuf) {
        sendbufGather = recvbuf;
      }
      int rValue = TAMPI_Allgather<DATATYPE, DATATYPE>(sendbufGather, count, datatype, tempbuf, count, datatype, comm);

      performReduce<DATATYPE>(true, tempbuf, recvbuf, count, op, -1, comm);

      return rValue;
    }
  }

  template<typename DATATYPE>
  int TAMPI_Iallreduce_global(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count, DATATYPE* datatype, TAMPI_Op op, TAMPI_Comm comm, TAMPI_Request* request);
  template<typename SENDTYPE, typename RECVTYPE>
  int TAMPI_Iallgather(const typename SENDTYPE::Type* sendbuf, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::Type* recvbuf, int recvcount, RECVTYPE* recvtype, TAMPI_Comm comm, TAMPI_Request* request);
  template<typename DATATYPE>
  inline int TAMPI_Iallreduce(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count, DATATYPE* datatype, TAMPI_Op op, TAMPI_Comm comm, TAMPI_Request* request) {
    if(op.hasAdjoint || !DATATYPE::Tool::IS_ActiveType) {
      return TAMPI_Iallreduce_global<DATATYPE>(sendbuf, recvbuf, count, datatype, op, comm, request);
    } else {
      // perform a gather and apply the operator locally
      int commSize = getCommSize(comm);

      typename DATATYPE::Type* tempbuf = new typename DATATYPE::Type[count * commSize];

      const typename DATATYPE::Type* sendbufGather = sendbuf;
      if(TAMPI_IN_PLACE == sendbuf) {
        sendbufGather = recvbuf;
      }
      int rValue = TAMPI_Iallgather<DATATYPE, DATATYPE>(sendbufGather, count, datatype, tempbuf, count, datatype, comm, request);
      TAMPI_Ireduce_Handle<DATATYPE>* curHandle = new TAMPI_Ireduce_Handle<DATATYPE>();
      curHandle->comm = comm;
      curHandle->root = -1;
      curHandle->all = true;
      curHandle->count = count;
      curHandle->op = op;
      curHandle->recvbuf = recvbuf;
      curHandle->tempbuf = tempbuf;
      curHandle->origHandle = request->handle;
      curHandle->origFunc = request->func;

      // set our own handle now to the request
      request->handle = curHandle;
      request->func = (ContinueFunction)TAMPI_Ireduce_finish<DATATYPE>;

      return rValue;
    }
  }
}
