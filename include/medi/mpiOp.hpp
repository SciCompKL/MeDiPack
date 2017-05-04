/*
 * MeDiPack, a Message Differentiation Package
 *
 * Copyright (C) 2017 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (codi@scicomp.uni-kl.de)
 *
 * Lead developers: Max Sagebaum (SciComp, TU Kaiserslautern)
 *
 * This file is part of MeDiPack (http://www.scicomp.uni-kl.de/software/codi).
 *
 * MeDiPack is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * MeDiPack is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU
 * General Public License along with MeDiPack.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Max Sagebaum (SciComp, TU Kaiserslautern)
 */

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
