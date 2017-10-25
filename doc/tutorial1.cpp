#include <codi.hpp>
#include <medi/medi.hpp>
#include <medi/codiMediPackTypes.hpp>

#include <iostream>

using /**
 * @brief Global namespace for MeDiPack - Message Differentiation Package
 */
namespace medi ;

#define TOOL CoDiPackTool<codi::RealReverse>

int main(int nargs, char** args) {
  AMPI_Init(&nargs, &args);

  TOOL::init();

  int rank;

  AMPI_Comm_rank(AMPI_COMM_WORLD, &rank);

  codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
  tape.setActive();

  codi::RealReverse a = 3.0;
  if( 0 == rank ) {
    tape.registerInput(a);

    AMPI_Send(&a, 1, TOOL::MPI_TYPE, 1, 42, AMPI_COMM_WORLD);
  } else {
    AMPI_Recv(&a, 1, TOOL::MPI_TYPE, 0, 42, AMPI_COMM_WORLD, AMPI_STATUS_IGNORE);

    tape.registerOutput(a);

    a.setGradient(100.0);
  }

  tape.setPassive();

  tape.evaluate();

  if(0 == rank) {
    std::cout << "Adjoint of 'a' on rank 0 is: " << a.getGradient() << std::endl;
  }

  TOOL::finalize();

  AMPI_Finalize();
}

#include <medi/medi.cpp>
