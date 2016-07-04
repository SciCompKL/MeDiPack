#include <toolDefines.h>

#include <unistd.h>

IN(10)
OUT(10)
POINTS(1) = {{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}, {11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0}}};
SEEDS(1) = {{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}, {11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0}}};

void func(NUMBER* x, NUMBER* y) {
  int world_rank;
  TAMPI_Comm_rank(TAMPI_COMM_WORLD, &world_rank);
  int world_size;
  TAMPI_Comm_size(TAMPI_COMM_WORLD, &world_size);

  medi::TAMPI_Request request;
  if(world_rank == 0) {
    usleep(250000); // sleep for a quarter second
    medi::TAMPI_Irsend<MPI_NUMBER>(x, 10, 1, 42, TAMPI_COMM_WORLD, &request);
  } else {
    medi::TAMPI_Irecv<MPI_NUMBER>(y, 10, 0, 42, TAMPI_COMM_WORLD, &request);
  }

  medi::TAMPI_Wait(&request, TAMPI_STATUS_IGNORE);
}
