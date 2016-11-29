#pragma once


#include <mpi.h>

#include "macros.h"
#include "mpiTypeInterface.hpp"
#include "typeDefinitions.h"

namespace medi {

  #define AMPI_COMM_WORLD MPI_COMM_WORLD
  #define AMPI_STATUS_IGNORE MPI_STATUS_IGNORE
  #define AMPI_STATUSES_IGNORE MPI_STATUSES_IGNORE

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

  template<typename Datatype>
  inline void createLinearIndexDisplacements(int* &linearCounts, int* &linearDispls, const int* counts, int ranks, Datatype* type) {
    linearCounts = new int[ranks];
    linearDispls = new int[ranks];

    linearCounts[0] = type->computeActiveElements(counts[0]);
    linearDispls[0] = 0;
    for(int i = 1; i < ranks; ++i) {
      linearCounts[i] = type->computeActiveElements(counts[i]);
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
}
