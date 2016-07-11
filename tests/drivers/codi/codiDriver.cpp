#include <toolDefines.h>

#include <iostream>
#include <vector>

int main(int nargs, char** args) {

  TAMPI_Init(&nargs, &args);

  int world_rank;
  TAMPI_Comm_rank(TAMPI_COMM_WORLD, &world_rank);
  int world_size;
  TAMPI_Comm_size(TAMPI_COMM_WORLD, &world_size);


  TOOL::init();

  int evalPoints = getEvalPointsCount();
  int inputs = getInputCount();
  int outputs = getOutputCount();
  NUMBER* x = new NUMBER[inputs];
  NUMBER* y = new NUMBER[outputs];

  NUMBER::TapeType& tape = NUMBER::getGlobalTape();
  tape.resize(2, 3);
  tape.setActive();

  for(int curPoint = 0; curPoint < evalPoints; ++curPoint) {
    std::cout << "Point " << curPoint << " : {";

    for(int i = 0; i < inputs; ++i) {
      if(i != 0) {
        std::cout << ", ";
      }
      double val = getEvalPoint(curPoint, world_rank, i);
      std::cout << val;

      x[i] = (NUMBER)(val);
    }
    std::cout << "}\n";

    for(int i = 0; i < outputs; ++i) {
      y[i] = 0.0;
    }

    for(int i = 0; i < inputs; ++i) {
      tape.registerInput(x[i]);
    }

    func(x, y);

    for(int i = 0; i < outputs; ++i) {
      tape.registerOutput(y[i]);
    }

    std::cout << "Seed " << curPoint << " : {";
    for(int i = 0; i < outputs; ++i) {
      if(i != 0) {
        std::cout << ", ";
      }
      double val = getEvalSeed(curPoint, world_rank, i);
      std::cout << val;

      y[i].setGradient(val);
    }
    std::cout << "}\n";

    tape.evaluate();

    for(int curIn = 0; curIn < inputs; ++curIn) {
      std::cout << curIn << " " << x[curIn].getGradient() << std::endl;
    }

    tape.reset();
  }

  MPI_Finalize();
}

#include <tampi/async.cpp>
