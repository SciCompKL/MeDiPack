#include <toolDefines.h>

IN(1)
OUT(1)
POINTS(1) = {{{2.0}, {12.0}}};
SEEDS(1) = {{{1.0}, {11.0}}};

void func(NUMBER* x, NUMBER* y) {
  int world_rank;
  medi::AMPI_Comm_rank(AMPI_COMM_WORLD, &world_rank);
  int world_size;
  medi::AMPI_Comm_size(AMPI_COMM_WORLD, &world_size);

  y[0] = 5.0 * x[0];
  NUMBER temp = x[0] * y[0];

  NUMBER yLocal = temp;
  if(world_rank == 0) {
    medi::AMPI_Send(&temp, 1, mpiNumberType, 1, 42, AMPI_COMM_WORLD);
  } else {
    medi::AMPI_Recv(y, 1, mpiNumberType, 0, 42, AMPI_COMM_WORLD, AMPI_STATUS_IGNORE);
  }

  y[0] += yLocal;
}
