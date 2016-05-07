#pragma once

#include "medipack.h"

template <typename T>
void codiMpiAdd(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  for(int i = 0; i < *len; ++i) {
    inoutvec[i].value() += invec[i].value();
  }
}

template <typename T>
void codiMpiMul(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  for(int i = 0; i < *len; ++i) {
    inoutvec[i].value() *= invec[i].value();
  }
}

template <typename T>
void codiMpiMax(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  using std::max;
  for(int i = 0; i < *len; ++i) {
    inoutvec[i].value() = max(inoutvec[i].value(), invec[i].value());
  }
}

template <typename T>
void codiMpiMin(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  using std::min;
  for(int i = 0; i < *len; ++i) {
    inoutvec[i].value() = min(inoutvec[i].value(), invec[i].value());
  }
}

template <typename T>
void codiValueAdd(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  for(int i = 0; i < *len; ++i) {
    inoutvec[i] += invec[i];
  }
}

template <typename T>
void codiValueMul(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  for(int i = 0; i < *len; ++i) {
    inoutvec[i] *= invec[i];
  }
}

template <typename T>
void codiValueMax(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  using std::max;
  for(int i = 0; i < *len; ++i) {
    inoutvec[i] = max(inoutvec[i], invec[i]);
  }
}

template <typename T>
void codiValueMin(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  using std::min;
  for(int i = 0; i < *len; ++i) {
    inoutvec[i] = min(inoutvec[i], invec[i]);
  }
}


struct CoDiPackTool {
  typedef codi::RealReverse Type;
  typedef codi::RealReverse::GradientValue AdjointType;
  typedef codi::RealReverse ModifiedType;
  typedef double PassiveType;

  const static bool IS_ActiveType = true;
  const static bool IS_RequiresModifiedBuffer = false;

  static MPI_Datatype MPIType;
  typedef medi::PassiveDataType<Type> ModifiedNested;

  static void init(MPI_Datatype mpiType) {
    MPIType = mpiType;
  }

  static bool isHandleRequired() {
    return Type::getGlobalTape().isActive();
  }

  static inline void startAssembly(medi::Handle* h) {

  }

  static inline void addToolAction(medi::Handle* h) {
    if(NULL != h) {
      Type::getGlobalTape().pushExternalFunctionHandle(callFunc, h, deleteFunc);
    }
  }

  static inline void stopAssembly(medi::Handle* h) {

  }

  static void callFunc(void* h) {
    medi::Handle* handle = static_cast<medi::Handle*>(h);
    handle->func(handle);
  }

  static void deleteFunc(void* h) {
    medi::Handle* handle = static_cast<medi::Handle*>(h);
    delete handle;
  }

  static inline int getIndex(const Type& value) {
    return value.getGradientData();
  }

  static inline PassiveType getValue(const Type& value) {
    return value.getValue();
  }

  static inline AdjointType getAdjoint(const int index) {
    return Type::getGlobalTape().getGradient(index);
  }

  static inline void updateAdjoint(const int index, const AdjointType& adjoint) {
    return Type::getGlobalTape().gradient(index) += adjoint;
  }

  static inline void setIntoModifyBuffer(ModifiedType& modValue, const Type& value) {
    (void)value;
    (void)modValue;
  }

  static inline void getFromModifyBuffer(const ModifiedType& modValue, Type& value) {
    (void)modValue;
    if(0 != value.getGradientData()) {
      value.getGradientData() = 0;
      Type::getGlobalTape().registerInput(value);
    }
  }

  static bool isActive() {
    return Type::getGlobalTape().isActive();
  }
};

MPI_Datatype CoDiPackTool::MPIType;
