#pragma once

#include "../macros.h"
#include "../mpiOp.hpp"
#include "../../../generated/medi/ampiDefinitions.h"

namespace medi {
  inline int AMPI_Op_create(AMPI_User_function* user_fn, int commute, AMPI_Op* op) {
    return op->init(user_fn, commute);
  }

  // TODO: implement advanced version

  inline int AMPI_Op_free(AMPI_Op* op) {

    return op->free();
  }
}
