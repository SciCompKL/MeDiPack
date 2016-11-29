#pragma once

#include <mpi.h>

#include "macros.h"
#include "typeDefinitions.h"

namespace medi {

  static void noPreAdjointOperation(void* adjoints, void* primals, int count) { MEDI_UNUSED(adjoints); MEDI_UNUSED(primals); MEDI_UNUSED(count); }
  static void noPostAdjointOperation(void* adjoints, void* primals, void* rootPrimals, int count) { MEDI_UNUSED(adjoints); MEDI_UNUSED(primals); MEDI_UNUSED(rootPrimals); MEDI_UNUSED(count); }

  struct AMPI_Op {
    /*const*/ bool requiresPrimal;
    /*const*/ bool requiresPrimalSend;

    /*const*/ MPI_Op primalFunction;
    /*const*/ MPI_Op modifiedPrimalFunction;

    /*const*/ PreAdjointOperation preAdjointOperation;
    /*const*/ PostAdjointOperation postAdjointOperation;

    bool hasAdjoint;

    AMPI_Op() :
      requiresPrimal(false),
      requiresPrimalSend(false),
      primalFunction(MPI_SUM),
      modifiedPrimalFunction(MPI_SUM),
      preAdjointOperation(noPreAdjointOperation),
      postAdjointOperation(noPostAdjointOperation),
      hasAdjoint(false) {}

    int init(const bool requiresPrimal, const bool requiresPrimalSend, MPI_User_function* primalFunction, int primalFunctionCommute, MPI_User_function* modifiedPrimalFunction, int modifiedPrimalFunctionCommute, const PreAdjointOperation preAdjointOperation, const PostAdjointOperation postAdjointOperation) {
      int result1 = MPI_Op_create(primalFunction, primalFunctionCommute, &this->primalFunction);
      MPI_Op_create(modifiedPrimalFunction, modifiedPrimalFunctionCommute, &this->modifiedPrimalFunction);

      this->requiresPrimal = requiresPrimal;
      this->requiresPrimalSend = requiresPrimalSend;
      this->preAdjointOperation = preAdjointOperation;
      this->postAdjointOperation = postAdjointOperation;
      this->hasAdjoint = true;

      return result1;
    }

    int init(MPI_User_function* user_fn, int commute) {
      MPI_Op op;

      int result = MPI_Op_create(user_fn, commute, &op);
      init(op);

      return result;
    }

    void init(MPI_Op op) {
      this->primalFunction = op;
      this->requiresPrimal = false;
      this->requiresPrimalSend = false;
      this->modifiedPrimalFunction = MPI_SUM;
      this->preAdjointOperation = noPreAdjointOperation;
      this->postAdjointOperation = noPostAdjointOperation;
      this->hasAdjoint = false;
    }

    int free() {
      if(this->hasAdjoint) {
        MPI_Op_free(&this->modifiedPrimalFunction);
      }

      return MPI_Op_free(&this->primalFunction);
    }
  };
}
