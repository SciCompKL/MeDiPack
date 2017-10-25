#include <codi.hpp>
#include <medi/medi.hpp>
#include <medi/codiMediPackTypes.hpp>

#include <iostream>

using /**
 * @brief Global namespace for MeDiPack - Message Differentiation Package
 */
namespace medi ;

#define TOOL CoDiPackTool<codi::RealReverse>

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
  AMPI_Type_create_contiguous(2, TOOL::MPI_TYPE, &residualMpiType);
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
    TOOL::modifyDependency(invec[i].l1, inoutvec[i].l1);
    TOOL::modifyDependency(invec[i].lMax, inoutvec[i].lMax);
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
  AMPI_Type_create_contiguous(2, TOOL::MPI_TYPE, &residualMpiType);
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

  TOOL::init();
  codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();

  customOperator();

  tape.reset();

  optimizedCustomOperator();

  TOOL::finalize();

  AMPI_Finalize();
}

#include <medi/medi.cpp>
