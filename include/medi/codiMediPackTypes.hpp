#pragma once

#include "medipack.h"

#include "adToolInterface.h"

template <typename T>
void codiModifiedAdd(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  for(int i = 0; i < *len; ++i) {
    inoutvec[i].value() += invec[i].value();
  }
}

template <typename T>
void codiModifiedMul(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  for(int i = 0; i < *len; ++i) {
    inoutvec[i].value() *= invec[i].value();
  }
}

template <typename T>
void codiModifiedMax(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  using std::max;
  for(int i = 0; i < *len; ++i) {
    inoutvec[i].value() = max(inoutvec[i].value(), invec[i].value());
  }
}

template <typename T>
void codiModifiedMin(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  using std::min;
  for(int i = 0; i < *len; ++i) {
    inoutvec[i].value() = min(inoutvec[i].value(), invec[i].value());
  }
}

template <typename T>
void codiUnmodifiedAdd(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  for(int i = 0; i < *len; ++i) {
    inoutvec[i] += invec[i];
  }
}

template <typename T>
void codiUnmodifiedMul(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  for(int i = 0; i < *len; ++i) {
    inoutvec[i] *= invec[i];
  }
}

template <typename T>
void codiUnmodifiedMax(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  using std::max;
  for(int i = 0; i < *len; ++i) {
    inoutvec[i] = max(inoutvec[i], invec[i]);
  }
}

template <typename T>
void codiUnmodifiedMin(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
  MEDI_UNUSED(datatype);

  using std::min;
  for(int i = 0; i < *len; ++i) {
    inoutvec[i] = min(inoutvec[i], invec[i]);
  }
}

template <typename AT, typename PT>
void codiPreAdjMul(AT* adjoints, PT* primals, int count) {
  for(int i = 0; i < count; ++i) {
    adjoints[i] *= primals[i];
  }
}

template <typename AT, typename PT>
void codiPostAdjMul(AT* adjoints, PT* primals, PT* rootPrimals, int count) {
  CODI_UNUSED(rootPrimals);

  for(int i = 0; i < count; ++i) {
    if(0.0 != primals[i]) {
      adjoints[i] /= primals[i];
    }
  }
}

template <typename AT, typename PT>
void codiPostAdjMinMax(AT* adjoints, PT* primals, PT* rootPrimals, int count) {
  for(int i = 0; i < count; ++i) {
    if(rootPrimals[i] != primals[i]) {
      adjoints[i] = 0.0; // the primal of this process was not the minimum or maximum so do not perfrom the adjoint update
    }
  }
}

template<typename CoDiType>
struct CoDiPackTool final : public medi::ADToolInterface {
  typedef CoDiType Type;
  typedef typename CoDiType::GradientValue AdjointType;
  typedef CoDiType ModifiedType;
  typedef typename CoDiType::PassiveReal PassiveType;
  typedef typename CoDiType::GradientData IndexType;

  const static bool IS_ActiveType = true;
  const static bool IS_RequiresModifiedBuffer = false;

  static MPI_Datatype MpiType;
  static MPI_Datatype ModifiedMpiType;
  static MPI_Datatype AdjointMpiType;
  static medi::AMPI_Op OP_ADD;
  static medi::AMPI_Op OP_MUL;
  static medi::AMPI_Op OP_MIN;
  static medi::AMPI_Op OP_MAX;

  static void initTypes() {
    // create the mpi type for CoDiPack
    // this type is used in this type and the passive formulation
    int blockLength[2] = {1,1};
    MPI_Aint displacements[2] = {0, sizeof(double)};
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_INT};
    MPI_Type_create_struct(2, blockLength, displacements, types, &MpiType);
    MPI_Type_commit(&MpiType);

    ModifiedMpiType = MpiType;
    AdjointMpiType = MPI_DOUBLE;
  }

  static void initOperator(medi::AMPI_Op& op, bool requiresPrimal, bool requiresPrimalSend, MPI_User_function* modifiedFunc, MPI_User_function* primalFunc, const medi::PreAdjointOperation preAdjointOperation, const medi::PostAdjointOperation postAdjointOperation) {
    MPI_Op modifiedTypeOperator;
    MPI_Op_create(modifiedFunc, true, &modifiedTypeOperator);
    MPI_Op valueTypeOperator;
    MPI_Op_create(primalFunc, true, &valueTypeOperator);
    op.init(requiresPrimal, requiresPrimalSend, valueTypeOperator, modifiedTypeOperator, preAdjointOperation, postAdjointOperation);
  }

  static void initOperator(medi::AMPI_Op& op, MPI_User_function* primalFunc) {
    MPI_Op valueTypeOperator;
    MPI_Op_create(primalFunc, true, &valueTypeOperator);
    op.init(valueTypeOperator);
  }

  static void initOperators() {
    initOperator(OP_ADD, false, false, (MPI_User_function*)codiModifiedAdd<Type>, (MPI_User_function*)codiUnmodifiedAdd<Type>, medi::noPreAdjointOperation, medi::noPostAdjointOperation);
    initOperator(OP_MUL, (MPI_User_function*)codiUnmodifiedMul<Type>);
    initOperator(OP_MIN, true, true, (MPI_User_function*)codiModifiedMin<Type>, (MPI_User_function*)codiUnmodifiedMin<Type>, medi::noPreAdjointOperation, (medi::PostAdjointOperation)codiPostAdjMinMax<double, double>);
    initOperator(OP_MAX, true, true, (MPI_User_function*)codiModifiedMax<Type>, (MPI_User_function*)codiUnmodifiedMax<Type>, medi::noPreAdjointOperation, (medi::PostAdjointOperation)codiPostAdjMinMax<double, double>);
  }

  static void init() {
    initTypes();
    initOperators();
  }

  static void finalizeOperators() {
    MPI_Op_free(&OP_ADD.primalFunction);
    if(OP_ADD.hasAdjoint) {
      MPI_Op_free(&OP_ADD.modifiedPrimalFunction);
    }

    MPI_Op_free(&OP_MUL.primalFunction);
    if(OP_MUL.hasAdjoint) {
      MPI_Op_free(&OP_MUL.modifiedPrimalFunction);
    }

    MPI_Op_free(&OP_MIN.primalFunction);
    if(OP_MIN.hasAdjoint) {
      MPI_Op_free(&OP_MIN.modifiedPrimalFunction);
    }

    MPI_Op_free(&OP_MAX.primalFunction);
    if(OP_MAX.hasAdjoint) {
      MPI_Op_free(&OP_MAX.modifiedPrimalFunction);
    }
  }

  static void finalizeTypes() {
    MPI_Type_free(&MpiType);
  }

  static void finalize() {
    finalizeOperators();
    finalizeTypes();
  }

  inline bool isActiveType() const {
    return true;
  }

  inline  bool isHandleRequired() const {
    return Type::getGlobalTape().isActive();
  }

  inline bool isOldPrimalsRequired() const {
    return false;
  }

  inline void startAssembly(medi::HandleBase* h) {
    MEDI_UNUSED(h);

  }

  inline void addToolAction(medi::HandleBase* h) {
    if(NULL != h) {
      Type::getGlobalTape().pushExternalFunctionHandle(callFunc, h, deleteFunc);
    }
  }

  inline void stopAssembly(medi::HandleBase* h) {
    MEDI_UNUSED(h);
  }

  static void callFunc(void* h) {
    medi::HandleBase* handle = static_cast<medi::HandleBase*>(h);
    handle->func(handle);
  }

  static void deleteFunc(void* h) {
    medi::HandleBase* handle = static_cast<medi::HandleBase*>(h);
    delete handle;
  }

  static inline int getIndex(const Type& value) {
    return value.getGradientData();
  }

  static inline void clearIndex(Type& value) {
    value.~Type();
    value.getGradientData() = 0;
  }

  static inline PassiveType getValue(const Type& value) {
    return value.getValue();
  }

  static inline void setValue(const IndexType& index, const PassiveType& primal) {
    MEDI_UNUSED(index);
    MEDI_UNUSED(primal);
  }

  static inline AdjointType getAdjoint(const int index) {
    return Type::getGlobalTape().getGradient(index);
  }

  static inline void updateAdjoint(const int index, const AdjointType& adjoint) {
    int indexCopy = index;
    Type::getGlobalTape().gradient(indexCopy) += adjoint;
  }

  static inline void setIntoModifyBuffer(ModifiedType& modValue, const Type& value) {
    MEDI_UNUSED(modValue);
    MEDI_UNUSED(value);
  }

  static inline void getFromModifyBuffer(const ModifiedType& modValue, Type& value) {
    MEDI_UNUSED(modValue);
    if(0 != value.getGradientData()) {
      value.getGradientData() = 0;
      Type::getGlobalTape().registerInput(value);
    }
  }

  static inline int registerValue(Type& value) {
    value.getGradientData() = 0;
    Type::getGlobalTape().registerInput(value);

    return value.getGradientData();
  }

  static bool isActive() {
    return Type::getGlobalTape().isActive();
  }
};

template<typename CoDiType> MPI_Datatype CoDiPackTool<CoDiType>::MpiType;
template<typename CoDiType> MPI_Datatype CoDiPackTool<CoDiType>::ModifiedMpiType;
template<typename CoDiType> MPI_Datatype CoDiPackTool<CoDiType>::AdjointMpiType;
template<typename CoDiType> medi::AMPI_Op CoDiPackTool<CoDiType>::OP_ADD;
template<typename CoDiType> medi::AMPI_Op CoDiPackTool<CoDiType>::OP_MUL;
template<typename CoDiType> medi::AMPI_Op CoDiPackTool<CoDiType>::OP_MIN;
template<typename CoDiType> medi::AMPI_Op CoDiPackTool<CoDiType>::OP_MAX;
