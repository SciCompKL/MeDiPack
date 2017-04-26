#pragma once


#include <mpi.h>

#include "macros.h"
#include "mpiTypeInterface.hpp"
#include "typeDefinitions.h"

namespace medi {

  typedef int Range[3];

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
