#include <toolDefines.h>

IN(10)
OUT(10)
POINTS(1) = {{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}, {11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0}}};
SEEDS(1) = {{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}, {11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0}}};

void func(NUMBER* x, NUMBER* y) {
  int world_rank;
  AMPI_Comm_rank(AMPI_COMM_WORLD, &world_rank);
  int world_size;
  AMPI_Comm_size(AMPI_COMM_WORLD, &world_size);

  if(0 == world_rank) {
    medi::AMPI_Sendrecv(x, 10, mpiNumberType, 1, 42, y, 10, mpiNumberType, 1, 43, AMPI_COMM_WORLD, AMPI_STATUS_IGNORE);
  } else {
    medi::AMPI_Sendrecv(x, 10, mpiNumberType, 0, 43, y, 10, mpiNumberType, 0, 42, AMPI_COMM_WORLD, AMPI_STATUS_IGNORE);
  }
}
