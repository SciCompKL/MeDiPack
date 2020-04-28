Tutorial 2: Reduce functions and custom operators {#Tutorial2}
============

Reduce operations for the common reduce operators `MIN`, `MAX`, `SUM` and `PROD` are handled without any additional changes to
the source code. That is
~~~
  codi::RealReverse a = ...;
  codi::RealReverse c;
  AMPI_Reduce(&a, &c, 1, mpiTypes->MPI_TYPE, AMPI_SUM, 0, AMPI_COMM_WORLD);
~~~

will work without any problems and the MPI data type provided by the AD behaves like a build in MPI type.

The same is true for the `MINLOC` and `MAXLOC` operators. An example is:
~~~
  struct IntType {
    codi::RealReverse p;
    int i;
  };

  IntType a;
  a.p = ...;
  a.i = rank;
  IntType c;
  AMPI_Reduce(&a, &c, 1, mpiTypes->MPI_INT_TYPE, AMPI_MINLOC, 0, MPI_COMM_WORLD);
~~~

The process gets more involved if custom operators are used in the application. The default handling of MeDiPack in such
cases is to change the reduce operation into a gather operation and to perform the reduction on the local processors.
This enables the AD tool to see the operations and perform the correct adjoint implementation. An example for such an
operation is:
~~~
  struct Residuals {
    codi::RealReverse l1;
    codi::RealReverse lMax;
  };

  void customOpp(Residuals* invec, Residuals* inoutvec, int* len, MPI_Datatype* datatype) {
    for(int i = 0; i < *len; ++i) {
      inoutvec[i].l1 += invec[i].l1;
      inoutvec[i].lMax = max(inoutvec[i].lMax, invec[i].lMax);
    }
  }

  ...

    AMPI_Datatype residualMpiType;
    AMPI_Type_create_contiguous(2, mpiTypes->MPI_TYPE, &residualMpiType);
    AMPI_Type_commit(&residualMpiType);

    AMPI_Op op;
    AMPI_Op_create((MPI_User_function*)customOpp, 1, &op);

    Residuals res;
    res.l1 = localL1;
    res.lMax = localLMax;

    Residuals resRed;
    AMPI_Reduce(&res, &resRed, 1, residualMpiType, op, 0, MPI_COMM_WORLD);
~~~

It is important that the implementation of the custom operator is done with the AD type, otherwise a wrong data structure
will be used in the custom operator. Since the function is the implemenation of the MPI user defined function only at this
place the MPI_Datatype is used and not the AMPI_Datatype.

The default handling from MeDiPack of such custom operators will always work but it is inefficient for large applications.
There is a new overload of the MPI_Op_create function that enables the user to provide additional information such that
a custom operator can do the reduce operations in the MPI nodes.

The mathematical model for a reduce operation is
\f[
   y = h(x_0, \ldots, x_{n-1})
\f]
where \f$x_i\f$ represents the value of one MPI rank, \f$n\f$ the number of ranks and \f$y\f$ the final result. \f$h\f$ is
the mathematical representation of the the reduce operation. The reverse AD formulation for this operation is
\f[
   \bar x_i \aeq \frac{\d h}{\d x_i}(x_0, \ldots, x_{n-1}) \bar y, \quad \forall i = 0, \ldots, n-1 \eqdot
\f]
The most important change, this equation represents, is the communication structure.
The reduce operation is changed into a broadcast operation and therefore no custom operator can be executed inside of
the broadcast operation. It is also problematic to calculate the derivative \f$\frac{\d h}{\d x_i}\f$ on each processor
since the values are not available. The above reverse AD formulation is reinterpreted as
\f[
   \bar x_i \aeq post(x_i, y, pre(y, \bar y)) \quad \forall i = 0, \ldots, n-1
\f]
where *pre* and *post* represent a pre and post broadcast operation. The equation is evaluated by MeDiPack in the form:
~~~
  double y_b = ...;
  double y_b_mod;
  if(rank == root) {
    y_b_mod = pre(y, y_b);
  }
  MPI_Bcast(&y_b_mod, 1, MPI_DOUBLE, root, comm);

  double x_b = post(x, y, y_b_mod);

  tape.updateAdjoint(x_b);
~~~

The availability of *y* on the sending side of the communication is handled by MeDiPack if that is requested by the user.
The definitions for the methods can be found in \link medi::AMPI_Op_create(const bool, const bool, MPI_User_function*, int, MPI_User_function*, int, const PreAdjointOperation, const PostAdjointOperation, AMPI_Op*) AMPI_Op_create \endlink.
The definition is
~~~
AMPI_Op_create(const bool requiresPrimal, const bool requiresPrimalSend,
               MPI_User_function* primalFunction, int primalFunctionCommute,
               MPI_User_function* modifiedPrimalFunction, int modifiedPrimalFunctionCommute,
               const PreAdjointOperation preAdjointOperation,
               const PostAdjointOperation postAdjointOperation,
               AMPI_Op* op) {
~~~
 - The argument `requiresPrimal` will tell MeDiPack to store the primal values (*x* and *y*) on there respective communication
sides.
 - If `requiresPrimalSend` is set to `true`, *y* is made available on the receiving side.
 - `primalFunction` defines the user defined function as implemented above. It should use the AD type such that
   the AD tool can track the operations. This function maybe used as a fallback method.
 - `modifiedPrimalFunction` defines the user function that is evaluated in the MPI nodes. This function needs to be
   implemented such that it uses the modified type of the AD tool. This type is the one, that is defined by the AD tool
   for the use in the buffers that are send through the network. For details see the implementation for the specific AD
   tool.
 - `preAdjointOperation` the function that is represented by *pre* in the above equations.
 - `postAdjointOperation` the function that is represented by *post* in the above equations.

 The definition for the last two arguments is:
 ~~~
   typedef void (*PreAdjointOperation)(void* adjoints, void* primals, int count);
   typedef void (*PostAdjointOperation)(void* adjoints, void* primals, void* rootPrimals, int count);
 ~~~

If the primal value is not requested or not requested to be send, the appropriate arguments will be null pointers.
Otherwise the methods follow the concept of the MPI user defined functions.

The above custom operator will now be implemented in an optimized way. The handling for the summation of the L1 residual
is trivial and the adjoints for this member do not need to be handled. The LMax residual is more difficult to handle.
The adjoint is only valid on the processor which provided the maximum value for the computation. On the other
processors the adjoint value needs to be set to zero. For 2 ranks this primal computation would be
~~~
void max(double a, double b, double& c) {
  if(a > b) {
    c = a;
  } else {
    c = b;
  }
}
~~~
the adjoint for this method can then be implemented by
~~~
void max_b(double a, double& a_b, double b, double& b_b, double c, double& c_b) {
  if(c == a) {
    a_b = c_b;
  }
  if(c == b) {
    b_b = c_b;
  }
}
~~~
which is the technique described above. The implementation can now be done with MeDiPack. For this it is
important to know that the CoDiPack types are not modified in the MPI buffers. So the implementation can directly use
the CoDiPack type.
~~~
using MpiTool = MpiTypes::Tool;

void modifiedCustomOpp(Residuals* invec, Residuals* inoutvec, int* len, MPI_Datatype* datatype) {

  // Special treatment for CoDiPack for online dependency analysis. Take a look at the Tool implementation for CoDiPack.
  for(int i = 0; i < *len; ++i) {
    MpiTool::modifyDependency(invec[i].l1, inoutvec[i].l1);
    MpiTool::modifyDependency(invec[i].lMax, inoutvec[i].lMax);
  }

  for(int i = 0; i < *len; ++i) {
    inoutvec[i].l1.value() += invec[i].l1.value();
    inoutvec[i].lMax.value() = std::max(inoutvec[i].lMax.value(), invec[i].lMax.value());
  }
}
~~~
The implementation just adds the `.value()` property access such that the operations are hidden from
CoDiPack. The first loop is just a special handling for CoDiPack such that an online activity analysis can be performed.
The next step implements the post operation since there is no pre operation required for this operator. The tricky part
in the implementation is, that there are two adjoint values for each value in the primal computation. Each first adjoint
value represents the L1 adjoint and each second represents the LMax adjoint.
~~~
void postAdjResidual(double* adjoints, double* primals, double* rootPrimals, int count) {
  // no += needs to be evaluated. This is done by MeDiPack and the AD tool.
  for(int i = 0; i < count; ++i) {
    if(0 == i % 2) {
      // no handling for L1 required
    } else {
      if(rootPrimals[i] != primals[i]) {
        adjoints[i] = 0.0; // not the maximum set the adjoint to zero
      }
    }
  }
}
~~~
The creation of the MPI operator can now be done with
~~~
  AMPI_Op op2;
  AMPI_Op_create(true, true,
                 (MPI_User_function*)customOpp, 1,
                 (MPI_User_function*)modifiedCustomOpp, 1,
                 noPreAdjointOperation,
                 (PostAdjointOperation)postAdjResidual,
                 &op2);
~~~


###Full code

~~~
#include <medi/medi.hpp>

#include <codi.hpp>
#include <codi/externals/codiMpiTypes.hpp>

#include <iostream>

using namespace medi;

using MpiTypes = CoDiMpiTypes<codi::RealReverse>;
using MpiTool = MpiTypes::Tool;
MpiTypes* mpiTypes;

struct Residuals {
  codi::RealReverse l1;
  codi::RealReverse lMax;
};

void customOpp(Residuals* invec, Residuals* inoutvec, int* len, MPI_Datatype* datatype) {
  for(int i = 0; i < *len; ++i) {
    inoutvec[i].l1 += invec[i].l1;
    inoutvec[i].lMax = max(inoutvec[i].lMax, invec[i].lMax);
  }
}

void customOperator() {
  int rank;
  AMPI_Comm_rank(AMPI_COMM_WORLD, &rank);

  AMPI_Datatype residualMpiType;
  AMPI_Type_create_contiguous(2, mpiTypes->MPI_TYPE, &residualMpiType);
  AMPI_Type_commit(&residualMpiType);

  AMPI_Op op;
  AMPI_Op_create((MPI_User_function*)customOpp, 1, &op);

  codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
  tape.setActive();

  Residuals res;
  res.l1 = rank;
  res.lMax = -rank;

  tape.registerInput(res.l1);
  tape.registerInput(res.lMax);

  Residuals resRed;
  AMPI_Reduce(&res, &resRed, 1, residualMpiType, op, 0, MPI_COMM_WORLD);

  if(0 == rank) {
    tape.registerOutput(resRed.l1);
    tape.registerOutput(resRed.lMax);

    resRed.l1.setGradient(100.0);
    resRed.lMax.setGradient(200.0);
  }

  tape.evaluate();

  std::cout << "CustomOP: Adjoint of   l1 on rank " << rank << ": " << res.l1.getGradient() << std::endl;
  std::cout << "CustomOP: Adjoint of lMax on rank " << rank << ": " << res.lMax.getGradient() << std::endl;
}

void modifiedCustomOpp(Residuals* invec, Residuals* inoutvec, int* len, MPI_Datatype* datatype) {

  // Special treatment for CoDiPack for online dependency analysis. Take a look at the Tool implementation for CoDiPack.
  for(int i = 0; i < *len; ++i) {
    MpiTool::modifyDependency(invec[i].l1, inoutvec[i].l1);
    MpiTool::modifyDependency(invec[i].lMax, inoutvec[i].lMax);
  }

  for(int i = 0; i < *len; ++i) {
    inoutvec[i].l1.value() += invec[i].l1.value();
    inoutvec[i].lMax.value() = std::max(inoutvec[i].lMax.value(), invec[i].lMax.value());
  }
}

void postAdjResidual(double* adjoints, double* primals, double* rootPrimals, int count) {
  // no += needs to be evaluated. This is done by MeDiPack and the AD tool.
  for(int i = 0; i < count; ++i) {
    if(0 == i % 2) {
      // no handling for l1 required
    } else {
      if(rootPrimals[i] != primals[i]) {
        adjoints[i] = 0.0; // not the maximum set the adjoint to zero
      }
    }
  }
}

void optimizedCustomOperator() {

  int rank;
  AMPI_Comm_rank(AMPI_COMM_WORLD, &rank);

  AMPI_Datatype residualMpiType;
  AMPI_Type_create_contiguous(2, mpiTypes->MPI_TYPE, &residualMpiType);
  AMPI_Type_commit(&residualMpiType);

  AMPI_Op op2;
  AMPI_Op_create(true, true,
                 (MPI_User_function*)customOpp, 1,
                 (MPI_User_function*)modifiedCustomOpp, 1,
                 noPreAdjointOperation,
                 (PostAdjointOperation)postAdjResidual,
                 &op2);

  codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
  tape.setActive();

  Residuals res;
  res.l1 = rank;
  res.lMax = -rank;

  tape.registerInput(res.l1);
  tape.registerInput(res.lMax);

  Residuals resRed;
  AMPI_Reduce(&res, &resRed, 1, residualMpiType, op2, 0, MPI_COMM_WORLD);

  if(0 == rank) {
    tape.registerOutput(resRed.l1);
    tape.registerOutput(resRed.lMax);

    resRed.l1.setGradient(100.0);
    resRed.lMax.setGradient(200.0);
  }

  tape.evaluate();

  std::cout << "OptimizedCustomOperator: Adjoint of   l1 on rank " << rank << ": " << res.l1.getGradient() << std::endl;
  std::cout << "OptimizedCustomOperator: Adjoint of lMax on rank " << rank << ": " << res.lMax.getGradient() << std::endl;
}

int main(int nargs, char** args) {
  AMPI_Init(&nargs, &args);

  int size;

  AMPI_Comm_size(AMPI_COMM_WORLD, &size);
  if(size != 2) {
    std::cout << "Please start the tutorial with two processes." << std::endl;
  } else {

    mpiTypes = new MpiTypes();
    codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();

    customOperator();

    tape.reset();

    optimizedCustomOperator();

    delete mpiTypes;
  }

  AMPI_Finalize();
}

#include <medi/medi.cpp>
~~~
