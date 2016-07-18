#pragma once

#include <mpi.h>
#include <codi.hpp>

#define MEDI_UNUSED(name) (void)(name)
#define MEDI_CHECK_ERROR(expr) (expr)


namespace medi {

  #define TAMPI_COMM_WORLD MPI_COMM_WORLD
  #define TAMPI_Comm_rank MPI_Comm_rank
  #define TAMPI_Comm_size MPI_Comm_size
  #define TAMPI_Init MPI_Init
  #define TAMPI_STATUS_IGNORE MPI_STATUS_IGNORE
  #define TAMPI_STATUSES_IGNORE MPI_STATUSES_IGNORE
  #define TAMPI_IN_PLACE MPI_IN_PLACE
  #define TAMPI_Buffer_attach MPI_Buffer_attach
  #define TAMPI_Buffer_detach MPI_Buffer_detach

  typedef MPI_Comm TAMPI_Comm;
  typedef MPI_Status TAMPI_Status;

  struct HandleBase;
  typedef void (*ReverseFunction)(HandleBase* h);
  typedef void (*ContinueFunction)(HandleBase* h);
  typedef void (*PreAdjointOperation)(void* adjoints, void* primals, int count);
  typedef void (*PostAdjointOperation)(void* adjoints, void* primals, void* rootPrimals, int count);

  static void noPreAdjointOperation(void* adjoints, void* primals, int count) { MEDI_UNUSED(adjoints); MEDI_UNUSED(primals); MEDI_UNUSED(count); }
  static void noPostAdjointOperation(void* adjoints, void* primals, void* rootPrimals, int count) { MEDI_UNUSED(adjoints); MEDI_UNUSED(primals); MEDI_UNUSED(rootPrimals); MEDI_UNUSED(count); }

  struct TAMPI_Op {
    /*const*/ bool requiresPrimal;
    /*const*/ bool requiresPrimalSend;

    /*const*/ MPI_Op primalFunction;
    const TAMPI_Op* modifiedOp;

    /*const*/ PreAdjointOperation preAdjointOperation;
    /*const*/ PostAdjointOperation postAdjointOperation;

    TAMPI_Op() :
      requiresPrimal(false),
      requiresPrimalSend(false),
      primalFunction(MPI_SUM),
      modifiedOp(NULL),
      preAdjointOperation(noPreAdjointOperation),
      postAdjointOperation(noPostAdjointOperation) {}

    void init(const bool requiresPrimal, const bool requiresPrimalSend, MPI_Op primalFunction, const TAMPI_Op* modifiedOp, const PreAdjointOperation preAdjointOperation, const PostAdjointOperation postAdjointOperation) {
      this->requiresPrimal = requiresPrimal;
      this->requiresPrimalSend = requiresPrimalSend;
      this->primalFunction = primalFunction;
      this->modifiedOp = modifiedOp;
      this->preAdjointOperation = preAdjointOperation;
      this->postAdjointOperation = postAdjointOperation;
    }
  };

  struct LinearDisplacements {
      int* displs;
      int* counts;

      inline LinearDisplacements(int commSize, int length) {
        counts = new int[commSize];
        displs = new int[commSize];
        for(int i = 0; i < commSize; ++i) {
          counts[i] = length;
          displs[i] = i * length;
        }
      }

      inline ~LinearDisplacements() {
        delete [] displs;
        delete [] counts;
      }

      static inline void deleteFunc(void* d) {
        LinearDisplacements* data = reinterpret_cast<LinearDisplacements*>(d);

        delete data;
      }
  };

  struct HandleBase {
    ReverseFunction func;

    virtual ~HandleBase() {}
  };

  inline int getCommRank(MPI_Comm comm) {
    int rank;
    MEDI_CHECK_ERROR(MPI_Comm_rank(comm, &rank));

    return rank;
  }

  inline int getCommSize(MPI_Comm comm) {
    int size;
    MEDI_CHECK_ERROR(MPI_Comm_size(comm, &size));

    return size;
  }

  template<typename AT, typename PT>
  inline void allocateReverseBuffer(AT* &adjoints, PT* &primals, int count, int totalSize, bool allocatePrimals, int &bufferSize, MPI_Datatype& mpiType) {
    if(allocatePrimals) {
      // create both buffers as a single array and set the pointers accordingly
      bufferSize = (sizeof(AT) + sizeof(PT)) * count;
      char* buffer = new char[(sizeof(AT) + sizeof(PT)) * totalSize];
      adjoints = reinterpret_cast<AT*>(buffer);
      primals = reinterpret_cast<PT*>(buffer + sizeof(AT) * totalSize);
      mpiType = MPI_BYTE;
    } else {
      // only create the adjoint buffer
      bufferSize = count;
      adjoints = new AT[totalSize];
      primals = NULL;
      mpiType = MPI_DOUBLE;
    }
  }

  template<typename AT, typename PT>
  inline void allocateReverseBuffer(AT* &adjoints, PT* &primals, int* counts, int totalSize, bool allocatePrimals, int &bufferSize, MPI_Datatype& mpiType) {
    adjoints = new AT[totalSize];
    primals = NULL;
    mpiType = MPI_DOUBLE;
  }

  template<typename AT>
  inline void combineAdjoints(int count, AT* &adjoints, int ranks) {
    for(int curRank = 1; curRank < ranks; ++curRank) {
      for(int curPos = 0; curPos < count; ++curPos) {
        adjoints[curPos] += adjoints[count * curRank + curPos];
      }
    }
  }


  template<typename AT, typename PT>
  inline void deleteReverseBuffer(AT* &adjoints, PT* &primals, bool allocatePrimals) {
    if(NULL != adjoints) {
      if(allocatePrimals) {
        // create both buffers as a single array and set the pointers accordingly
        char* buffer = reinterpret_cast<char*>(adjoints);
        delete [] buffer;
        adjoints = NULL;
        primals = NULL;
      } else {
        // only create the adjoint buffer
        delete [] adjoints;
        adjoints = NULL;
        primals = NULL;
      }
    }
  }

  template<typename T>
  inline void copyPrimals(T* res, const T* values, int count) {
    for(int pos = 0; pos < count; ++pos) {
      res[pos] = values[pos];
    }
  }

  struct EmptyDataType {
      typedef void Type;
  };

  template<typename T>
  struct PassiveTool {
      typedef T Type;
      typedef T AdjointType;
      typedef T ModifiedType;
      typedef T PassiveType;

      const static bool IS_ActiveType = false;
      const static bool IS_RequiresModifiedBuffer = false;

      static MPI_Datatype MPIType;
      static void init(MPI_Datatype mpiType) {
        MPIType = mpiType;
      }

      static bool isHandleRequired() { return false; }
      static inline void startAssembly(HandleBase* h) { MEDI_UNUSED(h); }
      static inline void addToolAction(HandleBase* h) { MEDI_UNUSED(h); }
      static inline void stopAssembly(HandleBase* h) { MEDI_UNUSED(h); }
      static inline int getIndex(const Type& value) { MEDI_UNUSED(value); return 0; }
      static inline Type getValue(const Type& value) { return value; }
      static inline void setIntoModifyBuffer(ModifiedType& modValue, const Type& value) { MEDI_UNUSED(modValue); MEDI_UNUSED(value); }
      static inline void getFromModifyBuffer(const ModifiedType& modValue, Type& value) { MEDI_UNUSED(modValue); MEDI_UNUSED(value); }
  };

  template<typename T>
  struct PassiveDataType {
      typedef PassiveTool<T> Tool;
      typedef typename Tool::Type Type;
      typedef typename Tool::AdjointType AdjointType;
      typedef typename Tool::PassiveType PassiveType;
      typedef int IndexType;

      typedef typename Tool::ModifiedType ModifiedType;
      typedef EmptyDataType ModifiedNested;

      static inline void prepareSendBuffer(const Type* buf, int count, int ranks, ModifiedType* &bufMod, IndexType* &indices, int &indexCount) { MEDI_UNUSED(buf); MEDI_UNUSED(bufMod); MEDI_UNUSED(count); MEDI_UNUSED(ranks); MEDI_UNUSED(indices); MEDI_UNUSED(indexCount); }
      static inline void prepareInplaceSendBuffer(const Type* buf, int count, int ranks, bool all, int ownRank, ModifiedType* bufMod, IndexType* &indices, int &indexCount) { MEDI_UNUSED(buf); MEDI_UNUSED(bufMod); MEDI_UNUSED(count); MEDI_UNUSED(ranks); MEDI_UNUSED(all); MEDI_UNUSED(ownRank); MEDI_UNUSED(indices); MEDI_UNUSED(indexCount); }
      static inline void handleSendBuffer(const Type* buf, int count, int ranks, ModifiedType* bufMod, IndexType* &indices, int &indexCount) { MEDI_UNUSED(buf); MEDI_UNUSED(bufMod); MEDI_UNUSED(count); MEDI_UNUSED(ranks); MEDI_UNUSED(indices); MEDI_UNUSED(indexCount); }
      static inline void prepareRecvBuffer(Type* buf, int count, int ranks, ModifiedType* &bufMod, IndexType* &indices, int &indexCount) { MEDI_UNUSED(buf); MEDI_UNUSED(bufMod); MEDI_UNUSED(count); MEDI_UNUSED(ranks); MEDI_UNUSED(indices); MEDI_UNUSED(indexCount); }
      static inline void handleRecvBuffer(Type* buf, int count, int ranks, ModifiedType* bufMod, IndexType* &indices, int &indexCount) { MEDI_UNUSED(buf); MEDI_UNUSED(bufMod); MEDI_UNUSED(count); MEDI_UNUSED(ranks); MEDI_UNUSED(indices); MEDI_UNUSED(indexCount); }

      static inline void prepareSendBuffer(const Type* buf, const int* counts, const int* displs, int ranks, ModifiedType* &bufMod, int* &displsMod, IndexType* &indices, int* &indexCounts, int* &indexDispls, int &totalSize) {MEDI_UNUSED(buf); MEDI_UNUSED(counts); MEDI_UNUSED(displs); MEDI_UNUSED(ranks); MEDI_UNUSED(bufMod); MEDI_UNUSED(displsMod); MEDI_UNUSED(indices); MEDI_UNUSED(indexCounts); MEDI_UNUSED(indexDispls); CODI_UNUSED(totalSize);}
      static inline void prepareInplaceSendBuffer(const Type* buf, const int* counts, const int* displs, int ranks, int ownRank, ModifiedType* bufMod, int* &displsMod, IndexType* &indices, int &indexCount) {MEDI_UNUSED(buf); MEDI_UNUSED(counts); MEDI_UNUSED(displs); MEDI_UNUSED(ranks); MEDI_UNUSED(ownRank); MEDI_UNUSED(bufMod); MEDI_UNUSED(displsMod); MEDI_UNUSED(indices); MEDI_UNUSED(indexCount);}
      static inline void prepareInplaceSendBuffer(const Type* buf, const int* counts, const int* displs, int ranks, int ownRank, ModifiedType* bufMod, int* &displsMod, IndexType* &indices, int* &indexCounts, int* &indexDispls, int &totalSize) {MEDI_UNUSED(buf); MEDI_UNUSED(counts); MEDI_UNUSED(displs); MEDI_UNUSED(ranks); MEDI_UNUSED(ownRank); MEDI_UNUSED(bufMod); MEDI_UNUSED(displsMod); MEDI_UNUSED(indices); MEDI_UNUSED(indexCounts); MEDI_UNUSED(indexDispls); CODI_UNUSED(totalSize);}
      static inline void handleSendBuffer(const Type* buf, const int* counts, const int* displs, int ranks, ModifiedType* &bufMod, int* &displsMod, IndexType* &indices, int* &indexCounts, int* &indexDispls, int &totalSize) { MEDI_UNUSED(buf); MEDI_UNUSED(counts); MEDI_UNUSED(displs); MEDI_UNUSED(ranks); MEDI_UNUSED(bufMod); MEDI_UNUSED(displsMod); MEDI_UNUSED(indices); MEDI_UNUSED(indexCounts); MEDI_UNUSED(indexDispls); CODI_UNUSED(totalSize);}
      static inline void prepareRecvBuffer(const Type* buf, const int* counts, const int* displs, int ranks, ModifiedType* &bufMod, int* &displsMod, IndexType* &indices, int* &indexCounts, int* &indexDispls, int &totalSize) { MEDI_UNUSED(buf); MEDI_UNUSED(counts); MEDI_UNUSED(displs); MEDI_UNUSED(ranks); MEDI_UNUSED(bufMod); MEDI_UNUSED(displsMod); MEDI_UNUSED(indices); MEDI_UNUSED(indexCounts); MEDI_UNUSED(indexDispls); CODI_UNUSED(totalSize);}
      static inline void handleRecvBuffer(const Type* buf, const int* counts, const int* displs, int ranks, ModifiedType* &bufMod, int* &displsMod, IndexType* &indices, int* &indexCounts, int* &indexDispls, int &totalSize) { MEDI_UNUSED(buf); MEDI_UNUSED(counts); MEDI_UNUSED(displs); MEDI_UNUSED(ranks); MEDI_UNUSED(bufMod); MEDI_UNUSED(displsMod); MEDI_UNUSED(indices); MEDI_UNUSED(indexCounts); MEDI_UNUSED(indexDispls); CODI_UNUSED(totalSize);}

      static inline void getValues(const Type* buf, int count, PassiveType* &primals) { MEDI_UNUSED(buf); MEDI_UNUSED(count); MEDI_UNUSED(primals); }
      static inline void getAdjoints(const int* indices, int count, AdjointType* adjoints) { MEDI_UNUSED(indices); MEDI_UNUSED(count); MEDI_UNUSED(adjoints); }
      static inline void updateAdjoints(const int* indices, int count, const AdjointType* adjoints) { MEDI_UNUSED(indices); MEDI_UNUSED(count); MEDI_UNUSED(adjoints); }
  };

  template<typename T> MPI_Datatype PassiveTool<T>::MPIType;

  template<typename ADTool>
  struct DefaultDataType {
    typedef typename ADTool::Type Type;
    typedef typename ADTool::AdjointType AdjointType;
    typedef typename ADTool::PassiveType PassiveType;
    typedef typename ADTool::IndexType IndexType;

    typedef ADTool Tool;

    typedef typename ADTool::ModifiedType ModifiedType;
    typedef typename ADTool::ModifiedNested ModifiedNested;

    static inline void prepareSendBuffer(const Type* buf, int count, int ranks, ModifiedType* &bufMod, IndexType* &indices, int &indexCount) {
      int totalSize = count * ranks;
      if(ADTool::isActive()) {
        indexCount = count;  // we leave this here because this is the value per rank
        indices = new int[totalSize];
        for(int pos = 0; pos < totalSize; ++pos) {
          indices[pos] = ADTool::getIndex(buf[pos]);
        }
      }
      if(ADTool::IS_RequiresModifiedBuffer) {
        bufMod = new ModifiedType[totalSize];
      } else {
        bufMod = const_cast<Type*>(buf);
      }
      for(int pos = 0; pos < totalSize; ++pos) {
        ADTool::setIntoModifyBuffer(bufMod[pos], buf[pos]);
      }
    }

    static inline void prepareInplaceSendBuffer(const Type* buf, int count, int ranks, bool all, int ownRank, ModifiedType* bufMod, IndexType* &indices, int &indexCount) {
      int totalSize = count; // we only need to store the values of our rank

      // calculate our position in the buffer
      size_t offset = 0;
      if(1 != ranks) {
        offset = count * ownRank;
      }
      if(all) {
        totalSize = ranks * count;
        offset = 0;
      }

      if(ADTool::isActive()) {
        indexCount = count;  // we leave this here because this is the value per rank
        indices = new int[totalSize];
        for(int pos = 0; pos < totalSize; ++pos) {
          indices[pos] = ADTool::getIndex(buf[pos + offset]);
        }
      }
      // we do not need to create the buffer but we need to set our values into the buffer
      for(int pos = 0; pos < totalSize; ++pos) {
        ADTool::setIntoModifyBuffer(bufMod[pos + offset], buf[pos + offset]);
      }
    }

    static inline void handleSendBuffer(const Type* buf, int count, int ranks, ModifiedType* bufMod, IndexType* &indices, int &indexCount) {
      if(ADTool::IS_RequiresModifiedBuffer) {
        delete [] bufMod;
      }
    }

    static inline void prepareRecvBuffer(Type* buf, int count, int ranks, ModifiedType* &bufMod, IndexType* &indices, int &indexCount) {
      MEDI_UNUSED(indices);
      MEDI_UNUSED(indexCount);
      if(ADTool::IS_RequiresModifiedBuffer) {
        int totalSize = count * ranks;
        bufMod = new ModifiedType[totalSize];
      } else {
        bufMod = buf;
      }

      //TODO: add index deletion here
    }

    static inline void handleRecvBuffer(Type* buf, int count, int ranks, ModifiedType* bufMod, IndexType* &indices, int &indexCount) {
      int totalSize = count * ranks;
      for(int pos = 0; pos < totalSize; ++pos) {
        ADTool::getFromModifyBuffer(bufMod[pos], buf[pos]);
      }

      if(ADTool::isActive()) {
        indexCount = count; // we leave this here because the is the value per rank
        indices = new int[totalSize];
        for(int pos = 0; pos < totalSize; ++pos) {
          indices[pos] = ADTool::getIndex(buf[pos]);
        }
      }

      if(ADTool::IS_RequiresModifiedBuffer) {
        delete [] bufMod;
      }
    }

    static inline void prepareSendBuffer(const Type* buf, const int* counts, const int* displs, int ranks, ModifiedType* &bufMod, int* &displsMod, IndexType* &indices, int* &indexCounts, int* &indexDispls, int &totalSizeBuf) {
      int totalSize = 0;
      for(int i = 0; i < ranks; ++i) {
        totalSize += counts[i];
      }
      if(ADTool::isActive()) {
        indices = new int[totalSize];
        indexCounts = new int[ranks];
        indexDispls = new int[ranks];
        totalSizeBuf = totalSize;

        int curOffset = 0;
        for(int i = 0; i < ranks; ++i) {
          indexCounts[i] = counts[i];
          indexDispls[i] = curOffset;
          curOffset += counts[i];

          for(int pos = 0; pos < counts[i]; ++pos) {
            indices[pos + indexDispls[i]] = ADTool::getIndex(buf[pos + displs[i]]);
          }
        }
      }
      if(ADTool::IS_RequiresModifiedBuffer) {
        bufMod = new ModifiedType[totalSize];
        displsMod = new int[ranks];

        int curOffset = 0;
        for(int i = 0; i < ranks; ++i) {
          displsMod[i] = curOffset;
          curOffset += counts[i];

          for(int pos = 0; pos < counts[i]; ++pos) {
            ADTool::setIntoModifyBuffer(bufMod[pos + displsMod[i]], buf[pos + displs[i]]);
          }
        }
      } else {
        bufMod = const_cast<Type*>(buf);
        displsMod = const_cast<int*>(displs);
      }
    }

    static inline void prepareInplaceSendBuffer(const Type* buf, const int* counts, const int* displs, int ranks, int ownRank, ModifiedType* bufMod, int* &displsMod, IndexType* &indices, int &indexCount) {
      int ownCount = counts[ownRank];
      int ownOffset = displs[ownRank];
      if(ADTool::isActive()) {
        indices = new int[ownCount];
        indexCount = ownCount;

        // we need to get our indices
        for(int pos = 0; pos < ownCount; ++pos) {
          indices[pos] = ADTool::getIndex(buf[pos + ownOffset]);
        }
      }
      if(ADTool::IS_RequiresModifiedBuffer) {
        // buffers are crated in the prepare recv
        // we need to set our values in the modified buffer
        for(int pos = 0; pos < ownCount; ++pos) {
          ADTool::setIntoModifyBuffer(bufMod[pos + displsMod[ownRank]], buf[pos + ownOffset]);
        }
      }
    }

    static inline void prepareInplaceSendBuffer(const Type* buf, const int* counts, const int* displs, int ranks, int ownRank, ModifiedType* bufMod, int* &displsMod, IndexType* &indices, int* &indexCounts, int* &indexDispls, int &totalSizeBuf) {
      int totalSize = 0;
      for(int i = 0; i < ranks; ++i) {
        totalSize += counts[i];
      }
      if(ADTool::isActive()) {
        indices = new int[totalSize];
        indexCounts = new int[ranks];
        indexDispls = new int[ranks];
        totalSizeBuf = totalSize;

        int curOffset = 0;
        for(int i = 0; i < ranks; ++i) {
          indexCounts[i] = counts[i];
          indexDispls[i] = curOffset;
          curOffset += counts[i];

          for(int pos = 0; pos < counts[i]; ++pos) {
            indices[pos + indexDispls[i]] = ADTool::getIndex(buf[pos + displs[i]]);
          }
        }
      }
      if(ADTool::IS_RequiresModifiedBuffer) {
        // buffers are crated in the prepare recv
        // we need to set our values in the modified buffer
        for(int i = 0; i < ranks; ++i) {
          for(int pos = 0; pos < counts[i]; ++pos) {
            ADTool::setIntoModifyBuffer(bufMod[pos + displsMod[i]], buf[pos + displs[i]]);
          }
        }
      }
    }

    static inline void handleSendBuffer(const Type* buf, const int* counts, const int* displs, int ranks, ModifiedType* &bufMod, int* &displsMod, IndexType* &indices, int* &indexCounts, int* &indexDispls, int &totalSize) {
      if(ADTool::IS_RequiresModifiedBuffer) {
        delete [] bufMod;
        delete [] displsMod;
      }
    }

    static inline void prepareRecvBuffer(Type* buf, const int* counts, const int* displs, int ranks, ModifiedType* &bufMod, int* &displsMod, IndexType* &indices, int* &indexCounts, int* &indexDispls, int &totalSizeBuf) {
      if(ADTool::IS_RequiresModifiedBuffer) {
        displsMod = new int[ranks];

        int totalSize = 0;
        for(int i = 0; i < ranks; ++i) {
          displsMod[i] = totalSize;
          totalSize += counts[i];
        }

        bufMod = new ModifiedType[totalSize];
      } else {
        bufMod = buf;
        displsMod = const_cast<int*>(displs);
      }

      //TODO: add index deletion here
    }

    static inline void handleRecvBuffer(Type* buf, const int* counts, const int* displs, int ranks, ModifiedType* &bufMod, int* &displsMod, IndexType* &indices, int* &indexCounts, int* &indexDispls, int &totalSizeBuf) {
      for(int i = 0; i < ranks; ++i) {
        for(int pos = 0; pos < counts[i]; ++pos) {
          ADTool::getFromModifyBuffer(bufMod[pos + displsMod[i]], buf[pos + displs[i]]);
        }
      }

      if(ADTool::isActive()) {
        indexCounts = new int[ranks];
        indexDispls = new int[ranks];

        int totalSize = 0;
        for(int i = 0; i < ranks; ++i) {
          indexCounts[i] = counts[i];
          indexDispls[i] = totalSize;
          totalSize += counts[i];
        }

        indices = new int[totalSize];
        totalSizeBuf = totalSize;
        for(int i = 0; i < ranks; ++i) {
          for(int pos = 0; pos < counts[i]; ++pos) {
            indices[pos + indexDispls[i]] = ADTool::getIndex(buf[pos + displs[i]]);
          }
        }
      }

      if(ADTool::IS_RequiresModifiedBuffer) {
        delete [] bufMod;
        delete [] displsMod;
      }
    }

    static inline void getValues(const Type* buf, int count, double* &primals) {
      primals = new PassiveType[count];
      for(int pos = 0; pos < count; ++pos) {
        primals[pos] = ADTool::getValue(buf[pos]);
      }
    }

    static inline void getAdjoints(const int* indices, int count, AdjointType* adjoints) {
      for(int pos = 0; pos < count; ++pos) {
        adjoints[pos] = ADTool::getAdjoint(indices[pos]);
      }
    }

    static inline void updateAdjoints(const int* indices, int count, const AdjointType* adjoints) {
      for(int pos = 0; pos < count; ++pos) {
        ADTool::updateAdjoint(indices[pos], adjoints[pos]);
      }
    }
  };
}
