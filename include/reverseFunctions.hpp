#pragma once

#include "medipack.h"
#include "tampi/async.hpp"

namespace medi {

  template<typename DATATYPE>
  void TAMPI_Send_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, MPI_Datatype bufType, int count, int dest, int tag, TAMPI_Comm comm) {
    MEDI_UNUSED(count);
    MPI_Recv(bufAdjoints, bufSize, bufType, dest, tag, comm, MPI_STATUS_IGNORE);
  }

  template<typename DATATYPE>
  void TAMPI_Isend_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, MPI_Datatype bufType, int count, int dest, int tag, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(count);
    MPI_Irecv(bufAdjoints, bufSize, bufType, dest, tag, comm, &request->request);
  }

  template<typename DATATYPE>
  void TAMPI_Bsend_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, MPI_Datatype bufType, int count, int dest, int tag, TAMPI_Comm comm) {
    MEDI_UNUSED(count);
    MPI_Recv(bufAdjoints, bufSize, bufType, dest, tag, comm, MPI_STATUS_IGNORE);
  }

  template<typename DATATYPE>
  void TAMPI_Ibsend_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, MPI_Datatype bufType, int count, int dest, int tag, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(count);
    MPI_Irecv(bufAdjoints, bufSize, bufType, dest, tag, comm, &request->request);
  }

  template<typename DATATYPE>
  void TAMPI_Ssend_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, MPI_Datatype bufType, int count, int dest, int tag, TAMPI_Comm comm) {
    MEDI_UNUSED(count);
    MPI_Recv(bufAdjoints, bufSize, bufType, dest, tag, comm, MPI_STATUS_IGNORE);
  }

  template<typename DATATYPE>
  void TAMPI_Issend_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, MPI_Datatype bufType, int count, int dest, int tag, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(count);
    MPI_Irecv(bufAdjoints, bufSize, bufType, dest, tag, comm, &request->request);
  }

  template<typename DATATYPE>
  void TAMPI_Rsend_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, MPI_Datatype bufType, int count, int dest, int tag, TAMPI_Comm comm) {
    MEDI_UNUSED(count);
    MPI_Recv(bufAdjoints, bufSize, bufType, dest, tag, comm, MPI_STATUS_IGNORE);
  }

  template<typename DATATYPE>
  void TAMPI_Irsend_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, MPI_Datatype bufType, int count, int dest, int tag, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(count);
    MPI_Irecv(bufAdjoints, bufSize, bufType, dest, tag, comm, &request->request);
  }

  template<typename DATATYPE>
  void TAMPI_Recv_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, MPI_Datatype bufType, int count, int src, int tag, TAMPI_Comm comm, TAMPI_Status* status) {
    MEDI_UNUSED(count);
    MEDI_UNUSED(status);
    MPI_Send(bufAdjoints, bufSize, bufType, src, tag, comm);
  }

  template<typename DATATYPE>
  void TAMPI_Irecv_adj(typename DATATYPE::AdjointType* bufAdjoints, int bufSize, MPI_Datatype bufType, int count, int src, int tag, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(count);
    MPI_Isend(bufAdjoints, bufSize, bufType, src, tag, comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Sendrecv_adj(typename SENDTYPE::AdjointType* sendbuf, int sendbufSize, MPI_Datatype sendbufType, int sendcount, int dest, int sendtag,
                     typename RECVTYPE::AdjointType* recvbuf, int recvbufSize, MPI_Datatype recvbufType, int recvcount, int source, int recvtag, TAMPI_Comm comm, TAMPI_Status*  status) {

    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);
    MPI_Sendrecv(recvbuf, recvbufSize, recvbufType, source, recvtag, sendbuf, sendbufSize, sendbufType, dest, sendtag, comm, status);
  }

  template<typename DATATYPE>
  void TAMPI_Bcast_wrap_adj(typename DATATYPE::AdjointType* &sendbufAdjoints, int sendbufSize, MPI_Datatype sendbufType, typename DATATYPE::AdjointType* &recvbufAdjoints, int recvbufSize, MPI_Datatype recvbufType, int count, int root, TAMPI_Comm comm) {
    MEDI_UNUSED(count);

    MPI_Gather(recvbufAdjoints, recvbufSize, recvbufType, sendbufAdjoints, sendbufSize, sendbufType, root, comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Scatter_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, MPI_Datatype sendbufType, int sendcount, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, MPI_Datatype recvbufType, int recvcount, int root, TAMPI_Comm comm) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Gather(recvbufAdjoints, recvbufSize, recvbufType, sendbufAdjoints, sendbufSize, sendbufType, root, comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Iscatter_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, MPI_Datatype sendbufType, int sendcount, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, MPI_Datatype recvbufType, int recvcount, int root, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Igather(recvbufAdjoints, recvbufSize, recvbufType, sendbufAdjoints, sendbufSize, sendbufType, root, comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Scatterv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int* sendbufCounts, const int* sendbufDispl, MPI_Datatype sendbufType, const int* sendcount, const int* displs, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, MPI_Datatype recvbufType, int recvcount, int root, TAMPI_Comm comm) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Gatherv(recvbufAdjoints, recvbufSize, recvbufType, sendbufAdjoints, sendbufCounts, sendbufDispl, sendbufType, root, comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Iscatterv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int* sendbufCounts, const int* sendbufDispl, MPI_Datatype sendbufType, const int* sendcount, const int* displs, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, MPI_Datatype recvbufType, int recvcount, int root, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Igatherv(recvbufAdjoints, recvbufSize, recvbufType, sendbufAdjoints, sendbufCounts, sendbufDispl, sendbufType, root, comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Gather_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, MPI_Datatype sendbufType, int sendcount, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, MPI_Datatype recvbufType, int recvcount, int root, TAMPI_Comm comm) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Scatter(recvbufAdjoints, recvbufSize, recvbufType, sendbufAdjoints, sendbufSize, sendbufType, root, comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Igather_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, MPI_Datatype sendbufType, int sendcount, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, MPI_Datatype recvbufType, int recvcount, int root, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Iscatter(recvbufAdjoints, recvbufSize, recvbufType, sendbufAdjoints, sendbufSize, sendbufType, root, comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Gatherv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, MPI_Datatype sendbufType, int sendcount, typename RECVTYPE::AdjointType* &recvbufAdjoints, int* recvbufCounts, const int* recvbufDispls, MPI_Datatype recvbufType, const int* recvcounts, const int* displs, int root, TAMPI_Comm comm) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcounts);
    MEDI_UNUSED(displs);

    MPI_Scatterv(recvbufAdjoints, recvbufCounts, recvbufDispls, recvbufType, sendbufAdjoints, sendbufSize, sendbufType, root, comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Igatherv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, MPI_Datatype sendbufType, int sendcount, typename RECVTYPE::AdjointType* &recvbufAdjoints, int* recvbufCounts, const int* recvbufDispls, MPI_Datatype recvbufType, const int* recvcounts, const int* displs, int root, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcounts);
    MEDI_UNUSED(displs);

    MPI_Iscatterv(recvbufAdjoints, recvbufCounts, recvbufDispls, recvbufType, sendbufAdjoints, sendbufSize, sendbufType, root, comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Allgather_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, MPI_Datatype sendbufType, int sendcount, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, MPI_Datatype recvbufType, int recvcount, TAMPI_Comm comm) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Alltoall(recvbufAdjoints, recvbufSize, recvbufType, sendbufAdjoints, sendbufSize, sendbufType, comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Iallgather_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, MPI_Datatype sendbufType, int sendcount, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, MPI_Datatype recvbufType, int recvcount, TAMPI_Comm comm, TAMPI_Request* request ) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Ialltoall(recvbufAdjoints, recvbufSize, recvbufType, sendbufAdjoints, sendbufSize, sendbufType, comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Allgatherv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, MPI_Datatype sendbufType, int sendcount, typename RECVTYPE::AdjointType* &recvbufAdjoints, int* recvbufCounts, const int* recvbufDispls, MPI_Datatype recvbufType, const int* recvcounts, const int* displs, TAMPI_Comm comm) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcounts);
    MEDI_UNUSED(displs);

    LinearDisplacements linDis(getCommSize(comm), sendbufSize);

    MPI_Alltoallv(recvbufAdjoints, recvbufCounts, recvbufDispls, recvbufType, sendbufAdjoints, linDis.counts, linDis.displs, sendbufType, comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Iallgatherv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, MPI_Datatype sendbufType, int sendcount, typename RECVTYPE::AdjointType* &recvbufAdjoints, int* recvbufCounts, const int* recvbufDispls, MPI_Datatype recvbufType, const int* recvcounts, const int* displs, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcounts);
    MEDI_UNUSED(displs);

    LinearDisplacements* linDis = new LinearDisplacements(getCommSize(comm), sendbufSize);
    request->setReverseData(reinterpret_cast<void*>(linDis), LinearDisplacements::deleteFunc);

    MPI_Ialltoallv(recvbufAdjoints, recvbufCounts, recvbufDispls, recvbufType, sendbufAdjoints, linDis->counts, linDis->displs, sendbufType, comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Alltoall_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, MPI_Datatype sendbufType, int sendcount, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, MPI_Datatype recvbufType, int recvcount, TAMPI_Comm comm) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Alltoall(recvbufAdjoints, recvbufSize, recvbufType, sendbufAdjoints, sendbufSize, sendbufType, comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Ialltoall_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int sendbufSize, MPI_Datatype sendbufType, int sendcount, typename RECVTYPE::AdjointType* &recvbufAdjoints, int recvbufSize, MPI_Datatype recvbufType, int recvcount, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(sendcount);
    MEDI_UNUSED(recvcount);

    MPI_Ialltoall(recvbufAdjoints, recvbufSize, recvbufType, sendbufAdjoints, sendbufSize, sendbufType, comm, &request->request);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Alltoallv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int* sendbufCounts, const int* sendbufDispls, MPI_Datatype sendbufType, const int* sendcounts, const int* sdispls, typename RECVTYPE::AdjointType* &recvbufAdjoints, int* recvbufCounts, const int* recvbufDispls, MPI_Datatype recvbufType, const int* recvcounts, const int* rdispls, TAMPI_Comm comm) {
    MEDI_UNUSED(sendcounts);
    MEDI_UNUSED(sdispls);
    MEDI_UNUSED(recvcounts);
    MEDI_UNUSED(rdispls);

    MPI_Alltoallv(recvbufAdjoints, recvbufCounts, recvbufDispls, recvbufType, sendbufAdjoints, sendbufCounts, sendbufDispls, sendbufType, comm);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  void TAMPI_Ialltoallv_adj(typename SENDTYPE::AdjointType* &sendbufAdjoints, int* sendbufCounts, const int* sendbufDispls, MPI_Datatype sendbufType, const int* sendcounts, const int* sdispls, typename RECVTYPE::AdjointType* &recvbufAdjoints, int* recvbufCounts, const int* recvbufDispls, MPI_Datatype recvbufType, const int* recvcounts, const int* rdispls, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(sendcounts);
    MEDI_UNUSED(sdispls);
    MEDI_UNUSED(recvcounts);
    MEDI_UNUSED(rdispls);

    MPI_Ialltoallv(recvbufAdjoints, recvbufCounts, recvbufDispls, recvbufType, sendbufAdjoints, sendbufCounts, sendbufDispls, sendbufType, comm, &request->request);
  }

  template<typename DATATYPE>
  void TAMPI_Reduce_global_adj(typename DATATYPE::AdjointType* &sendbufAdjoints, int sendbufSize, MPI_Datatype sendbufType, typename DATATYPE::AdjointType* &recvbufAdjoints, int recvbufSize, MPI_Datatype recvbufType, int count, TAMPI_Op op, int root, TAMPI_Comm comm) {
    MEDI_UNUSED(count);
    if(root == getCommRank(comm)) {
      MPI_Bcast(recvbufAdjoints, recvbufSize, recvbufType, root, comm);
      // TODO: add proper delete here: deleteReverseBuffer(sendbufAdjoints, sendbufPrimals, op.requiresPrimalSend);
      sendbufAdjoints = recvbufAdjoints;
      recvbufAdjoints = NULL;
    } else {
      MPI_Bcast(sendbufAdjoints, sendbufSize, sendbufType, root, comm);
    }
  }

  template<typename DATATYPE>
  void TAMPI_Ireduce_global_adj(typename DATATYPE::AdjointType* &sendbufAdjoints, int sendbufSize, MPI_Datatype sendbufType, typename DATATYPE::AdjointType* &recvbufAdjoints, int recvbufSize, MPI_Datatype recvbufType, int count, TAMPI_Op op, int root, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(count);
    if(root == getCommRank(comm)) {
      MPI_Ibcast(recvbufAdjoints, recvbufSize, recvbufType, root, comm, &request->request);
      // TODO: add proper delete here: deleteReverseBuffer(sendbufAdjoints, sendbufPrimals, op.requiresPrimalSend);
      sendbufAdjoints = recvbufAdjoints;
      recvbufAdjoints = NULL;
    } else {
      MPI_Ibcast(sendbufAdjoints, sendbufSize, sendbufType, root, comm, &request->request);
    }
  }

  template<typename DATATYPE>
  void TAMPI_Allreduce_global_adj(typename DATATYPE::AdjointType* &sendbufAdjoints, int sendbufSize, MPI_Datatype sendbufType, typename DATATYPE::AdjointType* &recvbufAdjoints, int recvbufSize, MPI_Datatype recvbufType, int count, TAMPI_Op op, TAMPI_Comm comm) {
    MEDI_UNUSED(op);
    MEDI_UNUSED(count);

    MPI_Allgather(recvbufAdjoints, recvbufSize, recvbufType, sendbufAdjoints, sendbufSize, sendbufType, comm);
  }

  template<typename DATATYPE>
  void TAMPI_Iallreduce_global_adj(typename DATATYPE::AdjointType* &sendbufAdjoints, int sendbufSize, MPI_Datatype sendbufType, typename DATATYPE::AdjointType* &recvbufAdjoints, int recvbufSize, MPI_Datatype recvbufType, int count, TAMPI_Op op, TAMPI_Comm comm, TAMPI_Request* request) {
    MEDI_UNUSED(op);
    MEDI_UNUSED(count);

    MPI_Iallgather(recvbufAdjoints, recvbufSize, recvbufType, sendbufAdjoints, sendbufSize, sendbufType, comm, &request->request);
  }
}
