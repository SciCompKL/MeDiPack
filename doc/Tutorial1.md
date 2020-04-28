Tutorial 1: Basic use {#Tutorial1}
============

MeDiPack is a Message Differentiation Package.
The goal is to provide a complete wrapper implementation for MPI such that an Algorithmic Differentiation (AD) tool can be used
in all communication routines.
For an introduction to an AD tool see e.g. [CoDiPack](http://www.scicomp.uni-kl.de/software/codi/).

MeDiPack is designed such that the integration into an existing application should pose no major problems.
It is assumed here, that the application is already able to use an AD tool for the generation of derivative results.
Thus only the additional steps for the MeDiPack integration are discussed here.
The general steps for the interation of MeDiPack are:
 - Include <medi/medi.hpp> in a global header
 - Use the MeDiPack namespace
 - Rename all uses of MPI_ to AMPI_
 - Include <medi/medi.cpp> file in a translation unit of you program
 - Initialize the specific implementation of your AD tool
 
The download of MeDiPack is available from the github web page of [MeDiPack](http://www.scicomp.uni-kl.de/software/medi/)
or it can be directly cloned from github:
~~~
git clone https://github.com/SciCompKL/MeDiPack.git
cd MeDiPack
export MEDIPACK_DIR=$PWD
~~~

In order to be able to include the header and source file the compiler command line needs to be extended by:
~~~
-I$MEDIPACK_DIR/include -I$MEDIPACK_DIR/src
~~~

Now the MeDiPack header can be included in a global header file:
~~~
#include <medi/medi.hpp>

using namespace medi;
~~~

The rename of MPI_ to AMPI_ can be performed with the regular expression "\bMPI_" -> "AMPI_". This should find all uses of
MPI in you files and replace them with the MeDiPack replacement. This change is necessary since MeDiPack needs to modify
the default data structures of MPI. The AMPI methods are using this data structure and call the appropriate version of
the MPI method. If the method does not need any handling for the AD tool it is directly forwarded to the MPI call. If
a special treatment for AD is required the additional operations are performed before and/or after the MPI method is called.

The next step is the initialization of the AD tool. It needs to be done after MPI is initialized. Usually the AD tool
provides an implementation of the medi::ADToolInterface from MeDiPack. For CoDiPack the interfaces are provided in the externals folder.
~~~
#include <medi/medi.hpp>

#include <codi.hpp>
#include <codi/externals/codiMpiTypes.hpp>

using namespace medi;

using MpiTypes = CoDiMpiTypes<codi::RealReverse>;
MpiTypes* mpiTypes;

int main(int nargs, char** args) {
  ...
  AMPI_Init(&nargs, &args);

  mpiTypes = new MpiTypes();
  
  ...
  
  delete mpiTypes;

  AMPI_Finalize();
}

#include <medi/medi.cpp>
~~~

The constructor call `MpiTypes()` will create all the specifics for the AD tool. The MPI datatype is then initialized
and can be used to send AD variables over the network. For a simple send/recv pair the code is:

~~~
  int rank;

  AMPI_Comm_rank(AMPI_COMM_WORLD, &rank);

  codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
  tape.setActive();

  codi::RealReverse a = 3.0;
  if( 0 == rank ) {
    tape.registerInput(a);

    AMPI_Send(&a, 1, mpiTypes->MPI_TYPE, 1, 42, AMPI_COMM_WORLD);
  } else {
    AMPI_Recv(&a, 1, mpiTypes->MPI_TYPE, 0, 42, AMPI_COMM_WORLD, AMPI_STATUS_IGNORE);

    tape.registerOutput(a);

    a.setGradient(100.0);
  }

  tape.setPassive();

  tape.evaluate();

  if(0 == rank) {
    std::cout << "Adjoint of 'a' on rank 0 is: " << a.getGradient() << std::endl;
  }
~~~

For all MPI functions and variables the AMPI replacements are used. Otherwise the only other change is the use of the
MPI data type provided by the AD tool. `mpiTypes->MPI_TYPE` provides all information for MeDiPack such that the special handling
for AD can be executed.

### Additional remarks

#### Compilation

During the compilation of a MPI application that uses MeDiPack you might get errors of the kind:
~~~
error: cannot convert ‘int*’ to ‘const Type* {aka const double*}’ for argument ‘1’ to ‘int medi::AMPI_Send(const typename DATATYPE::Type*, int, DATATYPE*, int, int, MPI_Comm) [with DATATYPE = medi::MpiTypePassive<double>; typename DATATYPE::Type = double; MPI_Comm = ompi_communicator_t*]’
     AMPI_Send(&rank, 1, AMPI_DOUBLE, 1, 42, AMPI_COMM_WORLD);
~~~
MeDiPack changes the definitions of the MPI functions such that they are type safe. The type given by `AMPI_INT` will
force the corresponding buffer to be of the type `int*`. If that is not the case errors like the one above are generated.

This type checking is only possible if the direct type definitions of the AD tools or the default MPI types are used.
If the above line is changed to:
~~~
  AMPI_Datatype type = AMPI_DOUBLE;
  AMPI_Send(&rank, 1, type, 1, 42, AMPI_COMM_WORLD);
~~~
the type checking can no longer be performed. This is also true if custom user types are used in MeDiPack.

#### Compilation options

There are some additional options that can be used to configure MeDiPack on a global level:
 - Some MPI implementations do not declare send buffers as constant.
   With the preprocessor option `-DMEDI_NO_CONST_SEND`, MeDiPack can be configured to adhere to this specification.
 - The MPI target (e.g. 1.0, 2.0, etc.) is determined by MeDiPack from the preprocessor macros `MPI_VERSION` and
   `MPI_SUBVERSION`. If these macros are not available the default is 3.1. The default and detection can be overwritten
   by setting the macro `MEDI_MPI_TARGET` to *MMmm* where *MM* is the major version and *mm* the minor version. For MPI 3.1
   this would be 301.
   
### Full code

The complete code for this tutorial is:
~~~
#include <medi/medi.hpp>

#include <codi.hpp>
#include <codi/externals/codiMpiTypes.hpp>

#include <iostream>

using namespace medi;

using MpiTypes = CoDiMpiTypes<codi::RealReverse>;
MpiTypes* mpiTypes;

int main(int nargs, char** args) {
  AMPI_Init(&nargs, &args);

  mpiTypes = new MpiTypes();

  int rank;
  int size;

  AMPI_Comm_rank(AMPI_COMM_WORLD, &rank);
  AMPI_Comm_size(AMPI_COMM_WORLD, &size);

  if(size != 2) {
    std::cout << "Please start the tutorial with two processes." << std::endl;
  } else {

    codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
    tape.setActive();

    codi::RealReverse a = 3.0;
    if( 0 == rank ) {
      tape.registerInput(a);

      AMPI_Send(&a, 1, mpiTypes->MPI_TYPE, 1, 42, AMPI_COMM_WORLD);
    } else {
      AMPI_Recv(&a, 1, mpiTypes->MPI_TYPE, 0, 42, AMPI_COMM_WORLD, AMPI_STATUS_IGNORE);

      tape.registerOutput(a);

      a.setGradient(100.0);
    }

    tape.setPassive();

    tape.evaluate();

    if(0 == rank) {
      std::cout << "Adjoint of 'a' on rank 0 is: " << a.getGradient() << std::endl;
    }
  }

  delete mpiTypes;

  AMPI_Finalize();
}

#include <medi/medi.cpp>
~~~
