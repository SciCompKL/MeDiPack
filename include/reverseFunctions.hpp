#pragma once

#include "medipack.h"
#include "tampi/async.hpp"

namespace medi {

  template<typename DATATYPE>
  void TAMPI_Send_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, int count, DATATYPE* datatype, int dest, int tag, TAMPI_Comm comm) {
    MEDI_UNUSED(count);
    MPI_Recv(bufAdjoints, bufSize, datatype->getAdjointMpiType(), dest, tag, comm, MPI_STATUS_IGNORE);
  }

  template<typename DATATYPE>
  void TAMPI_Isend_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, int count, DATATYPE* datatype, int dest, int tag, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(count);
    MPI_Irecv(bufAdjoints, bufSize, datatype->getAdjointMpiType(), dest, tag, comm, &request->request);
  }

  template<typename DATATYPE>
  void TAMPI_Bsend_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, int count, DATATYPE* datatype, int dest, int tag, TAMPI_Comm comm) {
    MEDI_UNUSED(count);
    MPI_Recv(bufAdjoints, bufSize, datatype->getAdjointMpiType(), dest, tag, comm, MPI_STATUS_IGNORE);
  }

  template<typename DATATYPE>
  void TAMPI_Ibsend_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, int count, DATATYPE* datatype, int dest, int tag, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(count);
    MPI_Irecv(bufAdjoints, bufSize, datatype->getAdjointMpiType(), dest, tag, comm, &request->request);
  }

  template<typename DATATYPE>
  void TAMPI_Ssend_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, int count, DATATYPE* datatype, int dest, int tag, TAMPI_Comm comm) {
    MEDI_UNUSED(count);
    MPI_Recv(bufAdjoints, bufSize, datatype->getAdjointMpiType(), dest, tag, comm, MPI_STATUS_IGNORE);
  }

  template<typename DATATYPE>
  void TAMPI_Issend_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, int count, DATATYPE* datatype, int dest, int tag, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(count);
    MPI_Irecv(bufAdjoints, bufSize, datatype->getAdjointMpiType(), dest, tag, comm, &request->request);
  }

  template<typename DATATYPE>
  void TAMPI_Rsend_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, int count, DATATYPE* datatype, int dest, int tag, TAMPI_Comm comm) {
    MEDI_UNUSED(count);
    MPI_Recv(bufAdjoints, bufSize, datatype->getAdjointMpiType(), dest, tag, comm, MPI_STATUS_IGNORE);
  }

  template<typename DATATYPE>
  void TAMPI_Irsend_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, int count, DATATYPE* datatype, int dest, int tag, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(count);
    MPI_Irecv(bufAdjoints, bufSize, datatype->getAdjointMpiType(), dest, tag, comm, &request->request);
  }

  template<typename DATATYPE>
  void TAMPI_Recv_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, int count, DATATYPE* datatype, int src, int tag, TAMPI_Comm comm, TAMPI_Status* status) {
    MEDI_UNUSED(count);
    MEDI_UNUSED(status);
    MPI_Send(bufAdjoints, bufSize, datatype->getAdjointMpiType(), src, tag, comm);
  }

  template<typename DATATYPE>
  void TAMPI_Irecv_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, int count, DATATYPE* datatype, int src, int tag, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(count);
    MPI_Isend(bufAdjoints, bufSize, datatype->getAdjointMpiType(), src, tag, comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Sendrecv_adj(typename SENDTYPE::AdjointType* sendbuf, int sendbufSize, int sendcount, SENDTYPE* sendtype, int dest, int sendtag,
                     typename RECVTYPE::AdjointType* recvbuf, int recvbufSize, int recvcount, RECVTYPE* recvtype, int source, int recvtag, TAMPI_Comm comm, TAMPI_Status*  status) {

    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);
    MPI_Sendrecv(recvbuf, recvbufSize, recvtype->getAdjointMpiType(), source, recvtag, sendbuf, sendbufSize, sendtype->getAdjointMpiType(), dest, sendtag, comm, status);
  }

  template<typename DATATYPE>
  void TAMPI_Bcast_wrap_adj(typename DATATYPE::AdjointType* &sendbufAdjoints, int sendbufSize, typename DATATYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int count, DATATYPE* datatype, int root, TAMPI_Comm comm) {
    MEDI_UNUSED(count);

    MPI_Gather(recvbufAdjoints, recvbufSize, datatype->getAdjointMpiType(), sendbufAdjoints, sendbufSize, datatype->getAdjointMpiType(), root, comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Scatter_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int recvcount, RECVTYPE* recvtype, int root, TAMPI_Comm comm) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Gather(recvbufAdjoints, recvbufSize, recvtype->getAdjointMpiType(), sendbufAdjoints, sendbufSize, sendtype->getAdjointMpiType(), root, comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Iscatter_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int recvcount, RECVTYPE* recvtype, int root, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Igather(recvbufAdjoints, recvbufSize, recvtype->getAdjointMpiType(), sendbufAdjoints, sendbufSize, sendtype->getAdjointMpiType(), root, comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Scatterv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int* sendbufCounts, const int* sendbufDispl, const int* sendcount, const int* displs, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int recvcount, RECVTYPE* recvtype, int root, TAMPI_Comm comm) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Gatherv(recvbufAdjoints, recvbufSize, recvtype->getAdjointMpiType(), sendbufAdjoints, sendbufCounts, sendbufDispl, sendtype->getAdjointMpiType(), root, comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Iscatterv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int* sendbufCounts, const int* sendbufDispl, const int* sendcount, const int* displs, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int recvcount, RECVTYPE* recvtype, int root, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Igatherv(recvbufAdjoints, recvbufSize, recvtype->getAdjointMpiType(), sendbufAdjoints, sendbufCounts, sendbufDispl, sendtype->getAdjointMpiType(), root, comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Gather_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int recvcount, RECVTYPE* recvtype, int root, TAMPI_Comm comm) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Scatter(recvbufAdjoints, recvbufSize, recvtype->getAdjointMpiType(), sendbufAdjoints, sendbufSize, sendtype->getAdjointMpiType(), root, comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Igather_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int recvcount, RECVTYPE* recvtype, int root, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Iscatter(recvbufAdjoints, recvbufSize, recvtype->getAdjointMpiType(), sendbufAdjoints, sendbufSize, sendtype->getAdjointMpiType(), root, comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Gatherv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int* recvbufCounts, const int* recvbufDispls, const int* recvcounts, const int* displs, RECVTYPE* recvtype, int root, TAMPI_Comm comm) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcounts);
    MEDI_UNUSED(displs);

    MPI_Scatterv(recvbufAdjoints, recvbufCounts, recvbufDispls, recvtype->getAdjointMpiType(), sendbufAdjoints, sendbufSize, sendtype->getAdjointMpiType(), root, comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Igatherv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int* recvbufCounts, const int* recvbufDispls, const int* recvcounts, const int* displs, RECVTYPE* recvtype, int root, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcounts);
    MEDI_UNUSED(displs);

    MPI_Iscatterv(recvbufAdjoints, recvbufCounts, recvbufDispls, recvtype->getAdjointMpiType(), sendbufAdjoints, sendbufSize, sendtype->getAdjointMpiType(), root, comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Allgather_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int recvcount, RECVTYPE* recvtype, TAMPI_Comm comm) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Alltoall(recvbufAdjoints, recvbufSize, recvtype->getAdjointMpiType(), sendbufAdjoints, sendbufSize, sendtype->getAdjointMpiType(), comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Iallgather_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int recvcount, RECVTYPE* recvtype, TAMPI_Comm comm, TAMPI_Request* request ) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Ialltoall(recvbufAdjoints, recvbufSize, recvtype->getAdjointMpiType(), sendbufAdjoints, sendbufSize, sendtype->getAdjointMpiType(), comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Allgatherv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int* recvbufCounts, const int* recvbufDispls, const int* recvcounts, const int* displs, RECVTYPE* recvtype, TAMPI_Comm comm) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcounts);
    MEDI_UNUSED(displs);

    LinearDisplacements linDis(getCommSize(comm), sendbufSize);

    MPI_Alltoallv(recvbufAdjoints, recvbufCounts, recvbufDispls, recvtype->getAdjointMpiType(), sendbufAdjoints, linDis.counts, linDis.displs, sendtype->getAdjointMpiType(), comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Iallgatherv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int* recvbufCounts, const int* recvbufDispls, const int* recvcounts, const int* displs, RECVTYPE* recvtype, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcounts);
    MEDI_UNUSED(displs);

    LinearDisplacements* linDis = new LinearDisplacements(getCommSize(comm), sendbufSize);
    request->setReverseData(reinterpret_cast<void*>(linDis), LinearDisplacements::deleteFunc);

    MPI_Ialltoallv(recvbufAdjoints, recvbufCounts, recvbufDispls, recvtype->getAdjointMpiType(), sendbufAdjoints, linDis->counts, linDis->displs, sendtype->getAdjointMpiType(), comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Alltoall_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int recvcount, RECVTYPE* recvtype, TAMPI_Comm comm) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Alltoall(recvbufAdjoints, recvbufSize, recvtype->getAdjointMpiType(), sendbufAdjoints, sendbufSize, sendtype->getAdjointMpiType(), comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Ialltoall_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int recvcount, RECVTYPE* recvtype, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Ialltoall(recvbufAdjoints, recvbufSize, recvtype->getAdjointMpiType(), sendbufAdjoints, sendbufSize, sendtype->getAdjointMpiType(), comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Alltoallv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int* sendbufCounts, const int* sendbufDispls, const int* sendcounts, const int* sdispls, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int* recvbufCounts, const int* recvbufDispls, const int* recvcounts, const int* rdispls, RECVTYPE* recvtype, TAMPI_Comm comm) {
    MEDI_UNUSED(sendcounts);
    MEDI_UNUSED(sdispls);
    MEDI_UNUSED(recvcounts);
    MEDI_UNUSED(rdispls);

    MPI_Alltoallv(recvbufAdjoints, recvbufCounts, recvbufDispls, recvtype->getAdjointMpiType(), sendbufAdjoints, sendbufCounts, sendbufDispls, sendtype->getAdjointMpiType(), comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Ialltoallv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int* sendbufCounts, const int* sendbufDispls, const int* sendcounts, const int* sdispls, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int* recvbufCounts, const int* recvbufDispls, const int* recvcounts, const int* rdispls, RECVTYPE* recvtype, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(sendcounts);
    MEDI_UNUSED(sdispls);
    MEDI_UNUSED(recvcounts);
    MEDI_UNUSED(rdispls);

    MPI_Ialltoallv(recvbufAdjoints, recvbufCounts, recvbufDispls, recvtype->getAdjointMpiType(), sendbufAdjoints, sendbufCounts, sendbufDispls, sendtype->getAdjointMpiType(), comm, &request->request);
  }

  template<typename DATATYPE>
  void TAMPI_Reduce_global_adj(typename DATATYPE::AdjointType* &sendbufAdjoints, int sendbufSize, typename DATATYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int count, DATATYPE* datatype, TAMPI_Op op, int root, TAMPI_Comm comm) {
    MEDI_UNUSED(count);
    if(root == getCommRank(comm)) {
      MPI_Bcast(recvbufAdjoints, recvbufSize, datatype->getAdjointMpiType(), root, comm);
      datatype->deleteAdjointTypeBuffer(sendbufAdjoints);
      sendbufAdjoints = recvbufAdjoints;
      recvbufAdjoints = NULL;
    } else {
      MPI_Bcast(sendbufAdjoints, sendbufSize, datatype->getAdjointMpiType(), root, comm);
    }
  }

  template<typename DATATYPE>
  void TAMPI_Ireduce_global_adj(typename DATATYPE::AdjointType* &sendbufAdjoints, int sendbufSize, typename DATATYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int count, DATATYPE* datatype, TAMPI_Op op, int root, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(count);
    if(root == getCommRank(comm)) {
      MPI_Ibcast(recvbufAdjoints, recvbufSize, datatype->getAdjointMpiType(), root, comm, &request->request);
      datatype->deleteAdjointTypeBuffer(sendbufAdjoints);
      sendbufAdjoints = recvbufAdjoints;
      recvbufAdjoints = NULL;
    } else {
      MPI_Ibcast(sendbufAdjoints, sendbufSize, datatype->getAdjointMpiType(), root, comm, &request->request);
    }
  }

  template<typename DATATYPE>
  void TAMPI_Allreduce_global_adj(typename DATATYPE::AdjointType* &sendbufAdjoints, int sendbufSize, typename DATATYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int count, DATATYPE* datatype, TAMPI_Op op, TAMPI_Comm comm) {
    MEDI_UNUSED(op);
    MEDI_UNUSED(count);

    MPI_Allgather(recvbufAdjoints, recvbufSize, datatype->getAdjointMpiType(), sendbufAdjoints, sendbufSize, datatype->getAdjointMpiType(), comm);
  }

  template<typename DATATYPE>
  void TAMPI_Iallreduce_global_adj(typename DATATYPE::AdjointType* &sendbufAdjoints, int sendbufSize, typename DATATYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int count, DATATYPE* datatype, TAMPI_Op op, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(op);
    MEDI_UNUSED(count);

    MPI_Iallgather(recvbufAdjoints, recvbufSize, datatype->getAdjointMpiType(), sendbufAdjoints, sendbufSize, datatype->getAdjointMpiType(), comm, &request->request);
  }
}
