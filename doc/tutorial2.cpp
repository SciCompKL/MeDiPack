/*
 * MeDiPack, a Message Differentiation Package
 *
 * Copyright (C) 2015-2023 Chair for Scientific Computing (SciComp), University of Kaiserslautern-Landau
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (codi@scicomp.uni-kl.de)
 *
 * Lead developers: Max Sagebaum (SciComp, University of Kaiserslautern-Landau)
 *
 * This file is part of MeDiPack (http://www.scicomp.uni-kl.de/software/codi).
 *
 * MeDiPack is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * MeDiPack is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License for more details.
 * You should have received a copy of the GNU
 * Lesser General Public License along with MeDiPack.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Max Sagebaum, Tim Albring (SciComp, University of Kaiserslautern-Landau)
 */

#include <medi/medi.hpp>

#include <codi.hpp>
#include <codi/externals/codiMpiTypes.hpp>

#include <iostream>

using namespace medi;

using MpiTypes = CoDiMpiTypes<codi::RealReverse>;
using MpiTool = MpiTypes::Tool;
MpiTypes* mpiTypes;

struct Residuals {
  codi::RealReverse l1;
  codi::RealReverse lMax;
};

void customOpp(Residuals* invec, Residuals* inoutvec, int* len, MPI_Datatype* datatype) {
  for(int i = 0; i < *len; ++i) {
    inoutvec[i].l1 += invec[i].l1;
    inoutvec[i].lMax = max(inoutvec[i].lMax, invec[i].lMax);
  }
}

void customOperator() {
  int rank;
  AMPI_Comm_rank(AMPI_COMM_WORLD, &rank);

  AMPI_Datatype residualMpiType;
  AMPI_Type_create_contiguous(2, mpiTypes->MPI_TYPE, &residualMpiType);
  AMPI_Type_commit(&residualMpiType);

  AMPI_Op op;
  AMPI_Op_create((MPI_User_function*)customOpp, 1, &op);

  codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
  tape.setActive();

  Residuals res;
  res.l1 = rank;
  res.lMax = -rank;

  tape.registerInput(res.l1);
  tape.registerInput(res.lMax);

  Residuals resRed;
  AMPI_Reduce(&res, &resRed, 1, residualMpiType, op, 0, MPI_COMM_WORLD);

  if(0 == rank) {
    tape.registerOutput(resRed.l1);
    tape.registerOutput(resRed.lMax);

    resRed.l1.setGradient(100.0);
    resRed.lMax.setGradient(200.0);
  }

  tape.evaluate();

  std::cout << "CustomOP: Adjoint of   l1 on rank " << rank << ": " << res.l1.getGradient() << std::endl;
  std::cout << "CustomOP: Adjoint of lMax on rank " << rank << ": " << res.lMax.getGradient() << std::endl;
}

void modifiedCustomOpp(Residuals* invec, Residuals* inoutvec, int* len, MPI_Datatype* datatype) {

  // Special treatment for CoDiPack for online dependency analysis. Take a look at the Tool implementation for CoDiPack.
  for(int i = 0; i < *len; ++i) {
    MpiTool::modifyDependency(invec[i].l1, inoutvec[i].l1);
    MpiTool::modifyDependency(invec[i].lMax, inoutvec[i].lMax);
  }

  for(int i = 0; i < *len; ++i) {
    inoutvec[i].l1.value() += invec[i].l1.value();
    inoutvec[i].lMax.value() = std::max(inoutvec[i].lMax.value(), invec[i].lMax.value());
  }
}

void postAdjResidual(double* adjoints, double* primals, double* rootPrimals, int count) {
  // no += needs to be evaluated. This is done by MeDiPack and the AD tool.
  for(int i = 0; i < count; ++i) {
    if(0 == i % 2) {
      // no handling for l1 required
    } else {
      if(rootPrimals[i] != primals[i]) {
        adjoints[i] = 0.0; // not the maximum set the adjoint to zero
      }
    }
  }
}

void optimizedCustomOperator() {

  int rank;
  AMPI_Comm_rank(AMPI_COMM_WORLD, &rank);

  AMPI_Datatype residualMpiType;
  AMPI_Type_create_contiguous(2, mpiTypes->MPI_TYPE, &residualMpiType);
  AMPI_Type_commit(&residualMpiType);

  AMPI_Op op2;
  AMPI_Op_create(true, true,
                 (MPI_User_function*)customOpp, 1,
                 (MPI_User_function*)modifiedCustomOpp, 1,
                 noPreAdjointOperation,
                 (PostAdjointOperation)postAdjResidual,
                 &op2);

  codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
  tape.setActive();

  Residuals res;
  res.l1 = rank;
  res.lMax = -rank;

  tape.registerInput(res.l1);
  tape.registerInput(res.lMax);

  Residuals resRed;
  AMPI_Reduce(&res, &resRed, 1, residualMpiType, op2, 0, MPI_COMM_WORLD);

  if(0 == rank) {
    tape.registerOutput(resRed.l1);
    tape.registerOutput(resRed.lMax);

    resRed.l1.setGradient(100.0);
    resRed.lMax.setGradient(200.0);
  }

  tape.evaluate();

  std::cout << "OptimizedCustomOperator: Adjoint of   l1 on rank " << rank << ": " << res.l1.getGradient() << std::endl;
  std::cout << "OptimizedCustomOperator: Adjoint of lMax on rank " << rank << ": " << res.lMax.getGradient() << std::endl;
}

int main(int nargs, char** args) {
  AMPI_Init(&nargs, &args);

  int size;

  AMPI_Comm_size(AMPI_COMM_WORLD, &size);
  if(size != 2) {
    std::cout << "Please start the tutorial with two processes." << std::endl;
  } else {

    mpiTypes = new MpiTypes();
    codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();

    customOperator();

    tape.reset();

    optimizedCustomOperator();

    delete mpiTypes;
  }

  AMPI_Finalize();
}

#include <medi/medi.cpp>
