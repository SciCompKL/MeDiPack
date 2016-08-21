#pragma once


#include <mpi.h>

#include "macros.h"
#include "mpiTypeInterface.hpp"
#include "typeDefinitions.h"

namespace medi {

  #define AMPI_COMM_WORLD MPI_COMM_WORLD
  #define AMPI_STATUS_IGNORE MPI_STATUS_IGNORE
  #define AMPI_STATUSES_IGNORE MPI_STATUSES_IGNORE
  #define AMPI_IN_PLACE MPI_IN_PLACE

  typedef int Range[3];

  typedef MPI_Aint AMPI_Aint;
  typedef MPI_Comm_errhandler_function AMPI_Comm_errhandler_function;
  typedef MPI_Errhandler AMPI_Errhandler;
  typedef MPI_Errhandler AMPI_Errhandler;
  typedef MPI_File_errhandler_function AMPI_File_errhandler_function;
  typedef MPI_File AMPI_File;
  typedef MPI_Info AMPI_Info;
  typedef MPI_Message AMPI_Message;
  typedef MPI_Win_errhandler_function AMPI_Win_errhandler_function;
  typedef MPI_Win AMPI_Win;
  typedef MPI_Comm AMPI_Comm;
  typedef MPI_Status AMPI_Status;
  typedef MPI_Comm_copy_attr_function AMPI_Comm_copy_attr_function;
  typedef MPI_Comm_delete_attr_function AMPI_Comm_delete_attr_function;
  typedef MPI_Group AMPI_Group;
  typedef MPI_Type_copy_attr_function AMPI_Type_copy_attr_function;
  typedef MPI_Type_delete_attr_function AMPI_Type_delete_attr_function;
  typedef MPI_Win_copy_attr_function AMPI_Win_copy_attr_function;
  typedef MPI_Win_delete_attr_function AMPI_Win_delete_attr_function;

  typedef MpiTypeInterface* AMPI_Datatype;

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
}
