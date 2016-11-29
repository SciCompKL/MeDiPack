#include "../../../include/medi/ampi/operatorFunctions.hpp"

namespace medi {

  void emptyFunction(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {}

  AMPI_Op AMPI_OP_NULL;

  void initializeOperators() {
    AMPI_OP_NULL.init(MPI_OP_NULL);
  }
}
