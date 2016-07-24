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
    /*const*/ MPI_Op modifiedPrimalFunction;

    /*const*/ PreAdjointOperation preAdjointOperation;
    /*const*/ PostAdjointOperation postAdjointOperation;

    bool hasAdjoint;



    TAMPI_Op() :
      requiresPrimal(false),
      requiresPrimalSend(false),
      primalFunction(MPI_SUM),
      modifiedPrimalFunction(MPI_SUM),
      preAdjointOperation(noPreAdjointOperation),
      postAdjointOperation(noPostAdjointOperation),
      hasAdjoint(false) {}

    void init(const bool requiresPrimal, const bool requiresPrimalSend, MPI_Op primalFunction, MPI_Op modifiedPrimalFunction, const PreAdjointOperation preAdjointOperation, const PostAdjointOperation postAdjointOperation) {
      this->requiresPrimal = requiresPrimal;
      this->requiresPrimalSend = requiresPrimalSend;
      this->primalFunction = primalFunction;
      this->modifiedPrimalFunction = modifiedPrimalFunction;
      this->preAdjointOperation = preAdjointOperation;
      this->postAdjointOperation = postAdjointOperation;
      this->hasAdjoint = true;
    }

    void init(MPI_Op primalFunction) {
      this->requiresPrimal = false;
      this->requiresPrimalSend = false;
      this->primalFunction = primalFunction;
      this->modifiedPrimalFunction = MPI_SUM;
      this->preAdjointOperation = noPreAdjointOperation;
      this->postAdjointOperation = noPostAdjointOperation;
      this->hasAdjoint = false;
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

  inline int computeDisplacementsTotalSize(const int* counts, int ranks) {
    int totalSize = 0;
    for(int i = 0; i < ranks; ++i) {
      totalSize += counts[i];
    }

    return totalSize;
  }

  inline int* createLinearDisplacements(const int* counts, int ranks) {
    int* displs = new int[ranks];

    displs[0] = 0;
    for(int i = 1; i < ranks; ++i) {
      displs[i] = counts[i - 1] +  displs[i - 1];
    }

    return displs;
  }

  inline void createLinearIndexDisplacements(int* &linearCounts, int* &linearDispls, const int* counts, int ranks, int indicesPerElement) {
    linearCounts = new int[ranks];
    linearDispls = new int[ranks];

    linearCounts[0] = counts[0] * indicesPerElement;
    linearDispls[0] = 0;
    for(int i = 1; i < ranks; ++i) {
      linearCounts[i] = counts[i] * indicesPerElement;
      linearDispls[i] = linearCounts[i - 1] +  linearDispls[i - 1];
    }
  }

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
      static inline void clearIndex(const Type& value) { MEDI_UNUSED(value); }
      static inline Type getValue(const Type& value) { return value; }
      static inline void setIntoModifyBuffer(ModifiedType& modValue, const Type& value) { MEDI_UNUSED(modValue); MEDI_UNUSED(value); }
      static inline void getFromModifyBuffer(const ModifiedType& modValue, Type& value) { MEDI_UNUSED(modValue); MEDI_UNUSED(value); }
      static inline int registerValue(Type& value) { MEDI_UNUSED(value); return 0;}
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

      const static int IndicesPerElement = 0;
      const static bool IS_RequiresModifiedBuffer = false;

      static inline void copyIntoModifiedBuffer(const Type* buf, ModifiedType* bufMod, int elements) {
        MEDI_UNUSED(buf);
        MEDI_UNUSED(bufMod);
        MEDI_UNUSED(elements);
      }

      static inline void copyFromModifiedBuffer(Type* buf, const ModifiedType* bufMod, int elements) {
        MEDI_UNUSED(buf);
        MEDI_UNUSED(bufMod);
        MEDI_UNUSED(elements);
      }

      static inline void getIndices(const Type* buf, IndexType* indices, int elements) {
        MEDI_UNUSED(buf);
        MEDI_UNUSED(indices);
        MEDI_UNUSED(elements);
      }

      static inline void registerValue(Type* buf, IndexType* indices, int elements) {
        MEDI_UNUSED(buf);
        MEDI_UNUSED(indices);
        MEDI_UNUSED(elements);
      }

      static inline void clearIndices(Type* buf, int elements) {
        MEDI_UNUSED(buf);
        MEDI_UNUSED(elements);
      }

      static inline void getValues(const Type* buf, int count, PassiveType* &primals) {
        MEDI_UNUSED(buf);
        MEDI_UNUSED(count);
        MEDI_UNUSED(primals);
      }

      static inline void getAdjoints(const int* indices, int count, AdjointType* adjoints) {
        MEDI_UNUSED(indices);
        MEDI_UNUSED(count);
        MEDI_UNUSED(adjoints);
      }

      static inline void updateAdjoints(const int* indices, int count, const AdjointType* adjoints) {
        MEDI_UNUSED(indices);
        MEDI_UNUSED(count);
        MEDI_UNUSED(adjoints);
      }


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

    const static int IndicesPerElement = 1;

    const static bool IS_RequiresModifiedBuffer = ADTool::IS_RequiresModifiedBuffer;

    static inline void copyIntoModifiedBuffer(const Type* buf, ModifiedType* bufMod, int elements) {
      if(ADTool::IS_RequiresModifiedBuffer) {
        for(int i = 0; i < elements; ++i) {
          ADTool::setIntoModifyBuffer(bufMod[i], buf[i]);
        }
      }
    }

    static inline void copyFromModifiedBuffer(Type* buf, const ModifiedType* bufMod, int elements) {
      if(ADTool::IS_RequiresModifiedBuffer) {
        for(int i = 0; i < elements; ++i) {
          ADTool::getFromModifyBuffer(bufMod[i], buf[i]);
        }
      }
    }

    static inline void getIndices(const Type* buf, IndexType* indices, int elements) {
      for(int i = 0; i < elements; ++i) {
        indices[i] = ADTool::getIndex(buf[i]);
      }
    }

    static inline void registerValue(Type* buf, IndexType* indices, int elements) {
      for(int i = 0; i < elements; ++i) {
        indices[i] = ADTool::registerValue(buf[i]);;
      }
    }

    static inline void clearIndices(Type* buf, int elements) {
      for(int i = 0; i < elements; ++i) {
        ADTool::clearIndex(buf[i]);
      }
    }

    static inline void getValues(const Type* buf, int count, PassiveType* &primals) {
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
