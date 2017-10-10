# MeDiPack

[MeDiPack](http://www.scicomp.uni-kl.de/software/medi/) (Message Differentiation Package) is a tool that handles the MPI communication of Algorithmic Differentiation (AD) tools like
[CoDiPack](http://www.scicomp.uni-kl.de/software/codi/).

This is currently only a preview release of MeDiPack in order to make it available to partners and testers.
We hope that we can release the version 1.0 of MeDiPack during this summer.
Some of the features that we have in the 1.0 version of MeDiPack:
  - Full forward of the MPI 3.1 specification
  - AD handling for nearly all MPI functions
    - This includes:
      - Point to point communication
      - Collective communication
      - Blocking as well as non-blocking variants
    - This excludes (will be handled later):
      - One-Sided communication
      - IO functions

In the full release there will be a detailed specification of all functions, that are completely handled and which ones are just forwarded.

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
     reduction operations for AD types are send handled by performing a gather. Afterwards a local reduce is performed.
     See the tutorial for further information.

Statistics about the handled functions:
- MPI 1.* 114/126 (90 %)
- MPI 2.* 153/183 (83 %)
- MPI 3.* 70/109 (64 %)
- Total  337/418 (80 %)

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
