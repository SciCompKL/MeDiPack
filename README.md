# MeDiPack

[MeDiPack](http://www.scicomp.uni-kl.de/software/medi/) (Message Differentiation Package) is a tool that handles the MPI communication of Algorithmic Differentiation (AD) tools like
[CoDiPack](http://www.scicomp.uni-kl.de/software/codi/).

The features of the initial release are:
  - Full forward of the MPI 3.1 specification
  - AD handling for nearly all MPI functions
    - This includes:
      - Point to point communication
      - Collective communication
      - Blocking as well as non-blocking variants
    - This excludes (will be handled later):
      - One-Sided communication
      - IO functions

The [Scientific Computing Group](http://www.scicomp.uni-kl.de) at the TU Kaiserslautern develops MeDiPack and
will enhance and extend MeDiPack in the future. There is a newsletter available at [codi-info@uni-kl.de](https://lists.uni-kl.de/uni-kl/subscribe/codi-info)
(The newsletter for MeDiPack and CoDiPack is the same.)

## List of supported and unsupported functions

### Supported

Most of the MPI standard is covered by the MeDiPack library. The functions that are not supported are mostly very
specialized routines. The following general list describes *features* of MPI that are handled, but makes no claim to be
complete:
 - Asynchronous communication
 - Custom data types
 - In place buffers
 - Operators
   - Here the interface needed to be extended for AD handling. The default creation of operators will still work but the
     reduction operations for AD types are handled by performing a gather. Afterwards a local reduce is performed.
     See the tutorial for further information.

Statistics about the handled functions:
- MPI 1.* 117/129 (90 %)
- MPI 2.* 153/183 (83 %)
- MPI 3.* 70/109 (64 %)
- Total  340/421 (80 %)

### Unsupported

Our aim is to support all MPI functions in MeDiPack, therefore most of the functions in this listing will be removed
in the next releases. If you require a function that is in the list below please feel free to contact us.

In general the following class of functions are not supported:
 - One sided communication
 - *_init methods
 - *w methods
 - Fortran conversion functions
 - \*neighbor\* functions
 - Handling intercommunicators

 The MPI IO functions are just forwarded to there MPI versions. A special handling for the AD types is not implemented.

The missing functions by MPI version:
 - MPI 1.0
   - Bsend_init, Recv_init, Rsend_init, Send_init, Sendrecv_replace, Ssend_init, Start, Startall, Pack, Pack_size, Unpack, Reduce_scatter
 - MPI 2.0
   - Pack_external, Pack_external_size, Type_create_darray, Unpack_external, Alltoallw, Accumulate, Get, Put, Win_complete, Win_create, Win_fence, Win_free, Win_get_group, Win_lock, Win_post, Win_start, Win_test, Win_wait, Type_create_f90_complex, Type_create_f90_integer, Type_create_f90_real, Type_match_size, Op_c2f, Op_f2c, Request_c2f, Request_f2c, Type_c2f, Type_f2c
 - MPI 2.2
   - Reduce_scatter_block
 - MPI 3.0
   - Ialltoallw, Ireduce_scatter, Ireduce_scatter_block, Ineighbor_allgather, Ineighbor_allgatherv, Ineighbor_alltoall, Ineighbor_alltoallv, Ineighbor_alltoallw, Neighbor_allgather, Neighbor_allgatherv, Neighbor_alltoall, Neighbor_alltoallv, Neighbor_alltoallw, Compare_and_swap, Fetch_and_op, Get_accumulate, Raccumulate, Rget, Rget_accumulate, Rput, Win_allocate, Win_allocate_shared, Win_attach, Win_create_dynamic, Win_detach, Win_flush, Win_flush_all, Win_flush_local, Win_flush_local_all, Win_get_info, Win_lock_all, Win_set_info, Win_shared_query, Win_sync, Win_unlock_all, Message_c2f, Message_f2c, T_cvar_get_info, T_pvar_get_info

## Usage

In order to use MeDiPack in your application the following steps have to be taken:
 - Include <medi/medi.hpp> in a global header
 - Use the MeDiPack namespace (using namespace medi;)
 - Rename all uses of MPI_ to AMPI_
 - Include <medi/medi.cpp> file in a translation unit of you program
 - Initialize the specific implementation of your AD tool

## Hello World Example

The example uses [CoDiPack](http://www.scicomp.uni-kl.de/software/codi/) as an AD tool.

~~~
#include <codi.hpp>
#include <medi/medi.hpp>
#include <medi/codiMediPackTypes.hpp>

#include <iostream>

using namespace medi;

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
~~~

Please visit the [tutorial page](http://www.scicomp.uni-kl.de/medi/db/d3c/tutorialPage.html) for further information.
