Changelog      {#Changelog}
============

### v 1.3.1 - 2024-12-27
 - Possibility to attach debug information to MeDiPack handles.

### v 1.3.0 - 2024-07-16
 - Recv, Mrecv, Irecv and Mirecv have now an optional parameter for the definition of the adjoint send.
 - Use optimized operator for add reduce.

### v 1.2.2 - 2023-05-11
 - CMake support.

### v 1.2.1 - 2022-11-30
 - License change to GNU LGPL v3
 - Minor/fixes:
   - Additional tests for async requests.
   - Missing activity checks.
   - Improved MPI_OP_NULL handling.
   - Comparisons for AMPI_Op.
   - Optional sendbuf constness.
   - Deprecation updates.

### v 1.2 - 2020-04-28
 - Support for persistent communication requests
 - Improved accessibility of internal handles and reverse requests
 - Changes to the AD tool interface (not backwards compatible)
 - Fixes for MS-MPI and Open MPI
 - Disabled ADOL-C tests (until ADOL-C is patched)
 - Handling of 82% of the MPI standard

### v 1.1.2 - 2019-01-07
 - Documentation change for new CoDiPack folder layout

### v 1.1.1 - 2018-11-07
 - Added missing include of new

### v 1.1 - 2018-10-30
 - Added primal and tangent evaluation functions
 - CoDiPack interface implementation is moved to CoDiPack

### v 1.0 - 2018-03-15
 - Handling of 80% of the MPI standard
