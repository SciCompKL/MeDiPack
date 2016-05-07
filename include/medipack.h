#pragma once

#include <mpi.h>
#include <codi.hpp>

#define MEDI_UNUSED(name) (void)(name)
#define MEDI_CHECK_ERROR(expr) (expr)

namespace medi {
  struct Handle;
  typedef void (*ReverseFunction)(Handle* h);

  struct TAMPI_Op {
    const bool requiresPrimal;

    const MPI_Op primalFunction;
    const TAMPI_Op* modifiedOp;

    TAMPI_Op(const bool requiresPrimal, MPI_Op primalFunction, const TAMPI_Op* modifiedOp) :
      requiresPrimal(requiresPrimal),
      primalFunction(primalFunction),
      modifiedOp(modifiedOp) {}
  };


  struct Handle {
    ReverseFunction func;
    int sendCount;
    int* sendIndices;
    double* sendPrimals;

    int recvCount;
    int* recvIndices;
    double* recvPrimals;

    ~Handle() {
      if(NULL != sendIndices) { delete [] sendIndices; }
      if(NULL != sendPrimals) { delete [] sendPrimals; }
      if(NULL != recvIndices) { delete [] recvIndices; }
      if(NULL != recvPrimals) { delete [] recvPrimals; }
    }
  };

  inline int getRank(MPI_Comm comm) {
    int rank;
    MEDI_CHECK_ERROR(MPI_Comm_rank(comm, &rank));

    return rank;
  }

  struct EmptyDataType {
      typedef void Type;
  };

  template<typename T>
  struct PassiveTool {
      typedef T Type;
      typedef T ModifiedType;
      typedef T PassiveType;

      const static bool IS_ActiveType = false;
      const static bool IS_RequiresModifiedBuffer = false;

      static MPI_Datatype MPIType;
      static void init(MPI_Datatype mpiType) {
        MPIType = mpiType;
      }

      static bool isHandleRequired() { return false; }
      static inline void startAssembly(Handle* h) { MEDI_UNUSED(h); }
      static inline void addToolAction(Handle* h) { MEDI_UNUSED(h); }
      static inline void stopAssembly(Handle* h) { MEDI_UNUSED(h); }
      static inline int getIndex(const Type& value) { MEDI_UNUSED(value); return 0; }
      static inline Type getValue(const Type& value) { return value; }
      static inline void setIntoModifyBuffer(ModifiedType& modValue, const Type& value) { MEDI_UNUSED(modValue); MEDI_UNUSED(value); }
      static inline void getFromModifyBuffer(const ModifiedType& modValue, Type& value) { MEDI_UNUSED(modValue); MEDI_UNUSED(value); }
  };

  template<typename T>
  struct PassiveDataType {
      typedef PassiveTool<T> Tool;
      typedef typename Tool::Type Type;
      typedef typename Tool::PassiveType PassiveType;

      typedef typename Tool::ModifiedType ModifiedType;
      typedef EmptyDataType ModifiedNested;

      static inline ModifiedType* prepareSendBuffer(const Type* buf, int count, Handle* h) { MEDI_UNUSED(buf); MEDI_UNUSED(count); MEDI_UNUSED(h); return NULL; }
      static inline void handleSendBuffer(const Type* buf, ModifiedType* modBuf, int count, Handle* h) { MEDI_UNUSED(buf); MEDI_UNUSED(modBuf); MEDI_UNUSED(count); MEDI_UNUSED(h); }
      static inline ModifiedType* prepareRecvBuffer(Type* buf, int count, Handle* h) { MEDI_UNUSED(buf); MEDI_UNUSED(count); MEDI_UNUSED(h); return NULL; }
      static inline void handleRecvBuffer(Type* buf, ModifiedType* modBuf, int count, Handle* h) { MEDI_UNUSED(buf); MEDI_UNUSED(modBuf); MEDI_UNUSED(count); MEDI_UNUSED(h); }
      static inline void getValues(const Type* buf, int count, double* &primals) { MEDI_UNUSED(buf); MEDI_UNUSED(count); MEDI_UNUSED(primals); }
  };

  template<typename T> MPI_Datatype PassiveTool<T>::MPIType;

  template<typename ADTool>
  struct DefaultDataType {
    typedef typename ADTool::Type Type;
    typedef typename ADTool::AdjointType AdjointType;
    typedef typename ADTool::PassiveType PassiveType;

    typedef ADTool Tool;

    typedef typename ADTool::ModifiedType ModifiedType;
    typedef typename ADTool::ModifiedNested ModifiedNested;

    static inline ModifiedType* prepareSendBuffer(const Type* buf, int count, Handle* h) {
      if(ADTool::isActive()) {
        h->sendIndices = new int[count];
        for(int pos = 0; pos < count; ++pos) {
          h->sendIndices[pos] = ADTool::getIndex(buf[pos]);
        }
      }
      ModifiedType* modBuf = NULL;
      if(ADTool::IS_RequiresModifiedBuffer) {
        modBuf = new ModifiedType[count];
      } else {
        modBuf = const_cast<Type*>(buf);
      }
      for(int pos = 0; pos < count; ++pos) {
        ADTool::setIntoModifyBuffer(modBuf[pos], buf[pos]);
      }

      return modBuf;
    }

    static inline void handleSendBuffer(const Type* buf, ModifiedType* modBuf, int count, Handle* h) {
      if(ADTool::IS_RequiresModifiedBuffer) {
        delete [] modBuf;
      }
    }

    static inline ModifiedType* prepareRecvBuffer(Type* buf, int count, Handle* h) {
      (void)h;
      if(ADTool::IS_RequiresModifiedBuffer) {
        return new ModifiedType[count];
      } else {
        return buf;
      }
    }

    static inline void handleRecvBuffer(Type* buf, ModifiedType* modBuf, int count, Handle* h) {
      for(int pos = 0; pos < count; ++pos) {
        ADTool::getFromModifyBuffer(modBuf[pos], buf[pos]);
      }

      if(ADTool::isActive()) {
        h->recvIndices = new int[count];
        for(int pos = 0; pos < count; ++pos) {
          h->recvIndices[pos] = ADTool::getIndex(buf[pos]);
        }
      }
    }

    static inline void getValues(const Type* buf, int count, double* &primals) {
      primals = new PassiveType[count];
      for(int pos = 0; pos < count; ++pos) {
        primals[pos] = ADTool::getValue(buf[pos]);
      }
    }

    static inline AdjointType* getAdjoints(const int* indices, int count) {
      AdjointType* adjoints = new AdjointType[count];
      for(int pos = 0; pos < count; ++pos) {
        adjoints[pos] = ADTool::getAdjoint(indices[pos]);
      }

      return adjoints;
    }

    static inline void updateAdjoints(const int* indices, int count, const AdjointType* adjoints) {
      for(int pos = 0; pos < count; ++pos) {
        ADTool::updateAdjoint(indices[pos], adjoints[pos]);
      }

      return adjoints;
    }

    static inline void freeAdjoints(AdjointType* &adjoints) {
      delete [] adjoints;
      adjoints = NULL;
    }
  };
}
