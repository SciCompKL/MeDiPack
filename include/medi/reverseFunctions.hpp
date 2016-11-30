#pragma once

#include "medipack.h"
#include "ampi/async.hpp"

namespace medi {

  template<typename DATATYPE>
  void AMPI_Send_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, int count, DATATYPE* datatype, int dest, int tag, AMPI_Comm comm) {
    MEDI_UNUSED(count);
    MPI_Recv(bufAdjoints, bufSize, datatype->getADTool().getAdjointMpiType(), dest, tag, comm, MPI_STATUS_IGNORE);
  }

  template<typename DATATYPE>
  void AMPI_Isend_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, int count, DATATYPE* datatype, int dest, int tag, AMPI_Comm comm, AMPI_Request* request) {
    MEDI_UNUSED(count);
    MPI_Irecv(bufAdjoints, bufSize, datatype->getADTool().getAdjointMpiType(), dest, tag, comm, &request->request);
  }

  template<typename DATATYPE>
  void AMPI_Bsend_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, int count, DATATYPE* datatype, int dest, int tag, AMPI_Comm comm) {
    MEDI_UNUSED(count);
    MPI_Recv(bufAdjoints, bufSize, datatype->getADTool().getAdjointMpiType(), dest, tag, comm, MPI_STATUS_IGNORE);
  }

  template<typename DATATYPE>
  void AMPI_Ibsend_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, int count, DATATYPE* datatype, int dest, int tag, AMPI_Comm comm, AMPI_Request* request) {
    MEDI_UNUSED(count);
    MPI_Irecv(bufAdjoints, bufSize, datatype->getADTool().getAdjointMpiType(), dest, tag, comm, &request->request);
  }

  template<typename DATATYPE>
  void AMPI_Ssend_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, int count, DATATYPE* datatype, int dest, int tag, AMPI_Comm comm) {
    MEDI_UNUSED(count);
    MPI_Recv(bufAdjoints, bufSize, datatype->getADTool().getAdjointMpiType(), dest, tag, comm, MPI_STATUS_IGNORE);
  }

  template<typename DATATYPE>
  void AMPI_Issend_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, int count, DATATYPE* datatype, int dest, int tag, AMPI_Comm comm, AMPI_Request* request) {
    MEDI_UNUSED(count);
    MPI_Irecv(bufAdjoints, bufSize, datatype->getADTool().getAdjointMpiType(), dest, tag, comm, &request->request);
  }

  template<typename DATATYPE>
  void AMPI_Rsend_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, int count, DATATYPE* datatype, int dest, int tag, AMPI_Comm comm) {
    MEDI_UNUSED(count);
    MPI_Recv(bufAdjoints, bufSize, datatype->getADTool().getAdjointMpiType(), dest, tag, comm, MPI_STATUS_IGNORE);
  }

  template<typename DATATYPE>
  void AMPI_Irsend_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, int count, DATATYPE* datatype, int dest, int tag, AMPI_Comm comm, AMPI_Request* request) {
    MEDI_UNUSED(count);
    MPI_Irecv(bufAdjoints, bufSize, datatype->getADTool().getAdjointMpiType(), dest, tag, comm, &request->request);
  }

  template<typename DATATYPE>
  void AMPI_Recv_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, int count, DATATYPE* datatype, int src, int tag, AMPI_Comm comm, AMPI_Status* status) {
    MEDI_UNUSED(count);
    MEDI_UNUSED(status);
    MPI_Send(bufAdjoints, bufSize, datatype->getADTool().getAdjointMpiType(), src, tag, comm);
  }

  template<typename DATATYPE>
  void AMPI_Irecv_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, int count, DATATYPE* datatype, int src, int tag, AMPI_Comm comm, AMPI_Request* request) {
    MEDI_UNUSED(count);
    MPI_Isend(bufAdjoints, bufSize, datatype->getADTool().getAdjointMpiType(), src, tag, comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Sendrecv_adj(typename SENDTYPE::AdjointType* sendbuf, int sendbufSize, int sendcount, SENDTYPE* sendtype, int dest, int sendtag,
                     typename RECVTYPE::AdjointType* recvbuf, int recvbufSize, int recvcount, RECVTYPE* recvtype, int source, int recvtag, AMPI_Comm comm, AMPI_Status*  status) {

    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);
    MPI_Sendrecv(recvbuf, recvbufSize, recvtype->getADTool().getAdjointMpiType(), source, recvtag, sendbuf, sendbufSize, sendtype->getADTool().getAdjointMpiType(), dest, sendtag, comm, status);
  }

  template<typename DATATYPE>
  void AMPI_Bcast_wrap_adj(typename DATATYPE::AdjointType* &sendbufAdjoints, int sendbufSize, typename DATATYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int count, DATATYPE* datatype, int root, AMPI_Comm comm) {
    MEDI_UNUSED(count);

    MPI_Gather(recvbufAdjoints, recvbufSize, datatype->getADTool().getAdjointMpiType(), sendbufAdjoints, sendbufSize, datatype->getADTool().getAdjointMpiType(), root, comm);
  }

  template<typename DATATYPE>
  void AMPI_Ibcast_wrap_adj(typename DATATYPE::AdjointType* &sendbufAdjoints, int sendbufSize, typename DATATYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int count, DATATYPE* datatype, int root, AMPI_Comm comm, AMPI_Request* request) {
    MEDI_UNUSED(count);

    MPI_Igather(recvbufAdjoints, recvbufSize, datatype->getADTool().getAdjointMpiType(), sendbufAdjoints, sendbufSize, datatype->getADTool().getAdjointMpiType(), root, comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Scatter_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int recvcount, RECVTYPE* recvtype, int root, AMPI_Comm comm) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Gather(recvbufAdjoints, recvbufSize, recvtype->getADTool().getAdjointMpiType(), sendbufAdjoints, sendbufSize, sendtype->getADTool().getAdjointMpiType(), root, comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Iscatter_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int recvcount, RECVTYPE* recvtype, int root, AMPI_Comm comm, AMPI_Request* request) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Igather(recvbufAdjoints, recvbufSize, recvtype->getADTool().getAdjointMpiType(), sendbufAdjoints, sendbufSize, sendtype->getADTool().getAdjointMpiType(), root, comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Scatterv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int* sendbufCounts, const int* sendbufDispl, const int* sendcount, const int* displs, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int recvcount, RECVTYPE* recvtype, int root, AMPI_Comm comm) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Gatherv(recvbufAdjoints, recvbufSize, recvtype->getADTool().getAdjointMpiType(), sendbufAdjoints, sendbufCounts, sendbufDispl, sendtype->getADTool().getAdjointMpiType(), root, comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Iscatterv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int* sendbufCounts, const int* sendbufDispl, const int* sendcount, const int* displs, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int recvcount, RECVTYPE* recvtype, int root, AMPI_Comm comm, AMPI_Request* request) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Igatherv(recvbufAdjoints, recvbufSize, recvtype->getADTool().getAdjointMpiType(), sendbufAdjoints, sendbufCounts, sendbufDispl, sendtype->getADTool().getAdjointMpiType(), root, comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Gather_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int recvcount, RECVTYPE* recvtype, int root, AMPI_Comm comm) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Scatter(recvbufAdjoints, recvbufSize, recvtype->getADTool().getAdjointMpiType(), sendbufAdjoints, sendbufSize, sendtype->getADTool().getAdjointMpiType(), root, comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Igather_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int recvcount, RECVTYPE* recvtype, int root, AMPI_Comm comm, AMPI_Request* request) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Iscatter(recvbufAdjoints, recvbufSize, recvtype->getADTool().getAdjointMpiType(), sendbufAdjoints, sendbufSize, sendtype->getADTool().getAdjointMpiType(), root, comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Gatherv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int* recvbufCounts, const int* recvbufDispls, const int* recvcounts, const int* displs, RECVTYPE* recvtype, int root, AMPI_Comm comm) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcounts);
    MEDI_UNUSED(displs);

    MPI_Scatterv(recvbufAdjoints, recvbufCounts, recvbufDispls, recvtype->getADTool().getAdjointMpiType(), sendbufAdjoints, sendbufSize, sendtype->getADTool().getAdjointMpiType(), root, comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Igatherv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int* recvbufCounts, const int* recvbufDispls, const int* recvcounts, const int* displs, RECVTYPE* recvtype, int root, AMPI_Comm comm, AMPI_Request* request) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcounts);
    MEDI_UNUSED(displs);

    MPI_Iscatterv(recvbufAdjoints, recvbufCounts, recvbufDispls, recvtype->getADTool().getAdjointMpiType(), sendbufAdjoints, sendbufSize, sendtype->getADTool().getAdjointMpiType(), root, comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Allgather_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int recvcount, RECVTYPE* recvtype, AMPI_Comm comm) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Alltoall(recvbufAdjoints, recvbufSize, recvtype->getADTool().getAdjointMpiType(), sendbufAdjoints, sendbufSize, sendtype->getADTool().getAdjointMpiType(), comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Iallgather_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int recvcount, RECVTYPE* recvtype, AMPI_Comm comm, AMPI_Request* request ) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Ialltoall(recvbufAdjoints, recvbufSize, recvtype->getADTool().getAdjointMpiType(), sendbufAdjoints, sendbufSize, sendtype->getADTool().getAdjointMpiType(), comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Allgatherv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int* recvbufCounts, const int* recvbufDispls, const int* recvcounts, const int* displs, RECVTYPE* recvtype, AMPI_Comm comm) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcounts);
    MEDI_UNUSED(displs);

    LinearDisplacements linDis(getCommSize(comm), sendbufSize);

    MPI_Alltoallv(recvbufAdjoints, recvbufCounts, recvbufDispls, recvtype->getADTool().getAdjointMpiType(), sendbufAdjoints, linDis.counts, linDis.displs, sendtype->getADTool().getAdjointMpiType(), comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Iallgatherv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int* recvbufCounts, const int* recvbufDispls, const int* recvcounts, const int* displs, RECVTYPE* recvtype, AMPI_Comm comm, AMPI_Request* request) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcounts);
    MEDI_UNUSED(displs);

    LinearDisplacements* linDis = new LinearDisplacements(getCommSize(comm), sendbufSize);
    request->setReverseData(reinterpret_cast<void*>(linDis), LinearDisplacements::deleteFunc);

    MPI_Ialltoallv(recvbufAdjoints, recvbufCounts, recvbufDispls, recvtype->getADTool().getAdjointMpiType(), sendbufAdjoints, linDis->counts, linDis->displs, sendtype->getADTool().getAdjointMpiType(), comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Alltoall_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int recvcount, RECVTYPE* recvtype, AMPI_Comm comm) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Alltoall(recvbufAdjoints, recvbufSize, recvtype->getADTool().getAdjointMpiType(), sendbufAdjoints, sendbufSize, sendtype->getADTool().getAdjointMpiType(), comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Ialltoall_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int recvcount, RECVTYPE* recvtype, AMPI_Comm comm, AMPI_Request* request) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Ialltoall(recvbufAdjoints, recvbufSize, recvtype->getADTool().getAdjointMpiType(), sendbufAdjoints, sendbufSize, sendtype->getADTool().getAdjointMpiType(), comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Alltoallv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int* sendbufCounts, const int* sendbufDispls, const int* sendcounts, const int* sdispls, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int* recvbufCounts, const int* recvbufDispls, const int* recvcounts, const int* rdispls, RECVTYPE* recvtype, AMPI_Comm comm) {
    MEDI_UNUSED(sendcounts);
    MEDI_UNUSED(sdispls);
    MEDI_UNUSED(recvcounts);
    MEDI_UNUSED(rdispls);

    MPI_Alltoallv(recvbufAdjoints, recvbufCounts, recvbufDispls, recvtype->getADTool().getAdjointMpiType(), sendbufAdjoints, sendbufCounts, sendbufDispls, sendtype->getADTool().getAdjointMpiType(), comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Ialltoallv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int* sendbufCounts, const int* sendbufDispls, const int* sendcounts, const int* sdispls, SENDTYPE* sendtype, typename RECVTYPE::AdjointType* &recvbufAdjoints, int* recvbufCounts, const int* recvbufDispls, const int* recvcounts, const int* rdispls, RECVTYPE* recvtype, AMPI_Comm comm, AMPI_Request* request) {
    MEDI_UNUSED(sendcounts);
    MEDI_UNUSED(sdispls);
    MEDI_UNUSED(recvcounts);
    MEDI_UNUSED(rdispls);

    MPI_Ialltoallv(recvbufAdjoints, recvbufCounts, recvbufDispls, recvtype->getADTool().getAdjointMpiType(), sendbufAdjoints, sendbufCounts, sendbufDispls, sendtype->getADTool().getAdjointMpiType(), comm, &request->request);
  }

  template<typename DATATYPE>
  void AMPI_Reduce_global_adj(typename DATATYPE::AdjointType* &sendbufAdjoints, int sendbufSize, typename DATATYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int count, DATATYPE* datatype, AMPI_Op op, int root, AMPI_Comm comm) {
    MEDI_UNUSED(count);
    if(root == getCommRank(comm)) {
      MPI_Bcast(recvbufAdjoints, recvbufSize, datatype->getADTool().getAdjointMpiType(), root, comm);
      datatype->getADTool().deleteAdjointTypeBuffer(sendbufAdjoints);
      sendbufAdjoints = recvbufAdjoints;
      recvbufAdjoints = NULL;
    } else {
      MPI_Bcast(sendbufAdjoints, sendbufSize, datatype->getADTool().getAdjointMpiType(), root, comm);
    }
  }

  template<typename DATATYPE>
  void AMPI_Ireduce_global_adj(typename DATATYPE::AdjointType* &sendbufAdjoints, int sendbufSize, typename DATATYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int count, DATATYPE* datatype, AMPI_Op op, int root, AMPI_Comm comm, AMPI_Request* request) {
    MEDI_UNUSED(count);
    if(root == getCommRank(comm)) {
      MPI_Ibcast(recvbufAdjoints, recvbufSize, datatype->getADTool().getAdjointMpiType(), root, comm, &request->request);
      datatype->getADTool().deleteAdjointTypeBuffer(sendbufAdjoints);
      sendbufAdjoints = recvbufAdjoints;
      recvbufAdjoints = NULL;
    } else {
      MPI_Ibcast(sendbufAdjoints, sendbufSize, datatype->getADTool().getAdjointMpiType(), root, comm, &request->request);
    }
  }

  template<typename DATATYPE>
  void AMPI_Allreduce_global_adj(typename DATATYPE::AdjointType* &sendbufAdjoints, int sendbufSize, typename DATATYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int count, DATATYPE* datatype, AMPI_Op op, AMPI_Comm comm) {
    MEDI_UNUSED(op);
    MEDI_UNUSED(count);

    MPI_Allgather(recvbufAdjoints, recvbufSize, datatype->getADTool().getAdjointMpiType(), sendbufAdjoints, sendbufSize, datatype->getADTool().getAdjointMpiType(), comm);
  }

  template<typename DATATYPE>
  void AMPI_Iallreduce_global_adj(typename DATATYPE::AdjointType* &sendbufAdjoints, int sendbufSize, typename DATATYPE::AdjointType* &recvbufAdjoints, int recvbufSize, int count, DATATYPE* datatype, AMPI_Op op, AMPI_Comm comm, AMPI_Request* request) {
    MEDI_UNUSED(op);
    MEDI_UNUSED(count);

    MPI_Iallgather(recvbufAdjoints, recvbufSize, datatype->getADTool().getAdjointMpiType(), sendbufAdjoints, sendbufSize, datatype->getADTool().getAdjointMpiType(), comm, &request->request);
  }
}
