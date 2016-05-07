#include <mpi.h>
#include <codi.hpp>

#include "test.hpp"

#include "medipack.h"
#include "codiMediPackTypes.hpp"

#include <iostream>


typedef codi::RealReverse Number;

typedef medi::DefaultDataType<CoDiPackTool> CoDiDataType;


const int n = 10;
int main(int nargs, char** args) {
  Number *nums = NULL;

  nums = new Number[n];

  MPI_Init(&nargs, &args);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  for(int i = 0; i < n; ++i) {
    nums[i] = (double)(i) + world_rank * 10;
  }

  Number::TapeType& tape = Number::getGlobalTape();
  tape.setActive();

  for(int i = 0; i < n; ++i) {
    tape.registerInput(nums[i]);
  }

  int blockLength[2] = {1,1};
  MPI_Aint displacements[2] = {0, sizeof(double)};
  MPI_Datatype types[2] = {MPI_DOUBLE, MPI_INT};
  MPI_Datatype codiMpiType;
  MPI_Type_create_struct(2, blockLength, displacements, types, &codiMpiType);
  MPI_Type_commit(&codiMpiType);
  CoDiPackTool::init(codiMpiType);
  medi::PassiveTool<Number>::init(codiMpiType);

  MPI_Op codiAddOpMod;
  MPI_Op_create((MPI_User_function*)codiMpiAdd<Number>, true, &codiAddOpMod);
  medi::TAMPI_Op codiAddTOpMod(false, codiAddOpMod, NULL);
  MPI_Op codiAddOp;
  MPI_Op_create((MPI_User_function*)codiValueAdd<Number>, true, &codiAddOp);
  medi::TAMPI_Op codiAddTOp(false, codiAddOp,&codiAddTOpMod);

  MPI_Op codiMulOpMod;
  MPI_Op_create((MPI_User_function*)codiMpiMul<Number>, true, &codiMulOpMod);
  medi::TAMPI_Op codiMulTOpMod(true, codiMulOpMod, NULL);
  MPI_Op codiMulOp;
  MPI_Op_create((MPI_User_function*)codiValueMul<Number>, true, &codiMulOp);
  medi::TAMPI_Op codiMulTOp(true, codiMulOp,&codiMulTOpMod);

  MPI_Op codiMaxOpMod;
  MPI_Op_create((MPI_User_function*)codiMpiMax<Number>, true, &codiMaxOpMod);
  medi::TAMPI_Op codiMaxTOpMod(true, codiMaxOpMod, NULL);
  MPI_Op codiMaxOp;
  MPI_Op_create((MPI_User_function*)codiValueMax<Number>, true, &codiMaxOp);
  medi::TAMPI_Op codiMaxTOp(true, codiMaxOp,&codiMaxTOpMod);

  MPI_Op codiMinOpMod;
  MPI_Op_create((MPI_User_function*)codiMpiMin<Number>, true, &codiMinOpMod);
  medi::TAMPI_Op codiMinTOpMod(true, codiMinOpMod, NULL);
  MPI_Op codiMinOp;
  MPI_Op_create((MPI_User_function*)codiValueMin<Number>, true, &codiMinOp);
  medi::TAMPI_Op codiMinTOp(true, codiMinOp,&codiMinTOpMod);


  // Reduce all of the local sums into the global sum
  Number* global_sum = new Number[n];
  medi::TAMPI_Reduce<CoDiDataType>(nums, global_sum, n, codiMaxTOp, 0, MPI_COMM_WORLD);

  // Print the result
  if (world_rank == 0) {
    for(int i = 0; i < n; ++i) {
      std::cout << i << ": " << global_sum[i].getValue() << " " << global_sum[i].getGradientData() << std::endl;
    }
  }

  MPI_Finalize();
}
