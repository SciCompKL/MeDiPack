#include <toolDefines.h>

#include <adolc/interfaces.h>
#include <adolc/taping.h>
#include <adolc/drivers/drivers.h>

#include <iostream>
#include <vector>

MPI_NUMBER* mpiNumberType;

int main(int nargs, char** args) {

  medi::AMPI_Init(&nargs, &args);

  int world_rank;
  medi::AMPI_Comm_rank(AMPI_COMM_WORLD, &world_rank);
  int world_size;
  medi::AMPI_Comm_size(AMPI_COMM_WORLD, &world_size);

  TOOL::init();

  mpiNumberType = new MPI_NUMBER();

  int evalPoints = getEvalPointsCount();
  int inputs = getInputCount();
  int outputs = getOutputCount();
  NUMBER* x = new NUMBER[inputs];
  NUMBER* y = new NUMBER[outputs];

  double* grad = new double[inputs];
  double* seed = new double[outputs];

  for(int curPoint = 0; curPoint < evalPoints; ++curPoint) {

    trace_on(1, 1);
    std::cout << "Point " << curPoint << " : {";

    for(int i = 0; i < inputs; ++i) {
      if(i != 0) {
        std::cout << ", ";
      }
      double val = getEvalPoint(curPoint, world_rank, i);
      std::cout << val;

      x[i] <<= val;
    }
    std::cout << "}\n";

    for(int i = 0; i < outputs; ++i) {
      y[i] = 0.0;
    }

    func(x, y);

    for(int i = 0; i < outputs; ++i) {
      double temp;
      y[i] >>= temp;
    }

    trace_off();

    std::cout << "Seed " << curPoint << " : {";
    for(int i = 0; i < outputs; ++i) {
      if(i != 0) {
        std::cout << ", ";
      }
      double val = getEvalSeed(curPoint, world_rank, i);
      std::cout << val;

      seed[i] = val;
    }
    std::cout << "}\n";

    fos_reverse((short)1, (int)outputs, (int)inputs, seed, grad);

    for(int curIn = 0; curIn < inputs; ++curIn) {
      std::cout << curIn << " " << grad[curIn] << std::endl;
    }


  }

  delete [] grad;
  delete [] seed;
  delete [] y;
  delete [] x;

  delete mpiNumberType;

  TOOL::finalize();
  medi::AMPI_Finalize();
}

#include <medi/medi.cpp>
#include <medi/adolcMeDiPackTypes.cpp>
