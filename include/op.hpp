#pragma once

#include <mpi.h>

#include "macros.h"
#include "typeDefinitions.h"

namespace medi {

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
}
