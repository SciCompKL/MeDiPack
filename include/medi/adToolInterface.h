/*
 * MeDiPack, a Message Differentiation Package
 *
 * Copyright (C) 2017 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (codi@scicomp.uni-kl.de)
 *
 * Lead developers: Max Sagebaum (SciComp, TU Kaiserslautern)
 *
 * This file is part of MeDiPack (http://www.scicomp.uni-kl.de/software/codi).
 *
 * MeDiPack is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * MeDiPack is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU
 * General Public License along with MeDiPack.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Max Sagebaum (SciComp, TU Kaiserslautern)
 */

#pragma once

#include "ampi/op.hpp"
#include "typeDefinitions.h"

/**
 * @brief Global namespace for MeDiPack - Message Differentiation Package
 */
namespace medi {

  /**
   * @brief The interface for the AD tool that is accessed by MeDiPack.
   */
  class ADToolInterface {

      MPI_Datatype adjointMpiType;

    public:

      /**
       * @brief The actual type that the AD implementation uses.
       */
      typedef void Type;

      /**
       * @brief The type that is send through the modified buffers.
       */
      typedef void ModifiedType;

      /**
       * @brief The data type that is used for the adjoint variables.
       */
      typedef void AdjointType;

      /**
       * @brief The data type used for the floating point data.
       */
      typedef void PassiveType;

      /**
       * @brief The data type from the AD tool for the identification of AD variables.
       */
      typedef void IndexType;

      /**
       * @brief Create an interface for the AD type.
       * @param[in] adjointMpiType  The mpi data type for the adjoint type.
       */
      ADToolInterface(MPI_Datatype adjointMpiType) :
        adjointMpiType(adjointMpiType) {}

      virtual ~ADToolInterface() {}

      /**
       * @brief The mpi data type for the adjoint type.
       * @return The adjoint mpi data type.
       */
      MPI_Datatype getAdjointMpiType() const {
        return adjointMpiType;
      }

      /**
       * @brief If this AD interface represents an AD type.
       * @return true if it is an AD type.
       */
      virtual bool isActiveType() const = 0;

      /**
       * @brief The handle needs to be created if an adjoint action is required by the AD tool.
       *
       * @return True if an adjoint action is required.
       */
      virtual bool isHandleRequired() const  = 0;

      /**
       * @brief Indicates if the AD tool needs to modify the buffer in order to send the correct data.
       * @return true if a new buffer needs to be created.
       */
      virtual bool isModifiedBufferRequired() const = 0;

      /**
       * @brief Indicates if MeDiPack needs store the overwritten primal values for the AD tool
       * @return True if overwritten primal values need to be restored.
       */
      virtual bool isOldPrimalsRequired() const = 0;

      /**
       * @brief Indicates to the AD tool that an adjoint action is in the progress of beeing recorded.
       * @param[in,out] h  The handle that is used by MeDiPack for the data storing.
       */
      virtual void startAssembly(HandleBase* h) const = 0;

      /**
       * @brief Indicates to the AD tool that an adjoint action is beeing finished.
       * @param[in,out] h  The handle that is used by MeDiPack for the data storing.
       */
      virtual void stopAssembly(HandleBase* h) const = 0;

      /**
       * @brief Register the handle so that the AD tool can evaluate it in the reverse sweep.
       *
       * The AD tool needs to store the handle and do the following call:
       *
       *  h->func(h);
       *
       * This call will evaluate the necessary adjoint actions by MeDiPack.
       *
       * The handle pointer is now in the possion of the AD tool and the AD tool needs to delete the handle with the call
       *
       * delete h;
       *
       * The virtual function call will take care of the proper deletion of the MeDiPack structure.
       *
       * @param[in,out] h  The handle that is used by MeDiPack for the data storing.
       */
      virtual void addToolAction(HandleBase* h) const = 0;

      /**
       * @brief Convert the mpi intrinsic operators like MPI_SUM to the specific one for the AD tool.
       *
       * @param[in] op  The intrinsic mpi operator.
       * @return The opertor that can handle the AD type.
       */
      virtual AMPI_Op convertOperator(AMPI_Op op) const = 0;

      /**
       * @brief Get the adjoints for the indices from the AD tool.
       *
       * @param[in]   indices  The indices from the AD tool for the variables in the buffer.
       * @param[out] adjoints  The vector for the adjoint variables.
       * @param[in]  elements  The number of elements in the vectors.
       */
      virtual void getAdjoints(const void* indices, void* adjoints, int elements) const = 0;

      /**
       * @brief Add the adjoint varaibles to the ones in the AD tool. That is the AD tool should perform the
       * operation:
       *
       * internalAdjoints[indices[i]] += adjoints[i];
       *
       * @param[in]   indices  The indices from the AD tool for the variables in the buffer.
       * @param[out] adjoints  The vector with the adjoint variables.
       * @param[in]  elements  The number of elements in the vectors.
       */
      virtual void updateAdjoints(const void* indices, const void* adjoints, int elements) const = 0;

      /**
       * @brief Restore the old primal values from the floating point values in the buffer.
       *
       * @param[in]   indices  The indices from the AD tool for the variables in the buffer.
       * @param[out] adjoints  The vector with the old primal variables.
       * @param[in]  elements  The number of elements in the vectors.
       */
      virtual void setReverseValues(const void* indices, const void* primals, int elements) const = 0;

      /**
       * @brief Perform a reduction in the first element of the buffer.
       * @param[in,out]  buf  The buffer with adjoint values its size is elements * ranks
       * @param[in] elements  The number of elements in the vectors.
       * @param[in]    ranks  The number of ranks in the communication.
       */
      virtual void combineAdjoints(void* buf, const int elements, const int ranks) const = 0;

      /**
       * @brief Create an array for the adjoint variables.
       *
       * @param[out] buf  The pointer for the buffer.
       * @param[in] size  The size of the buffer.
       */
      virtual void createAdjointTypeBuffer(void* &buf, size_t size) const = 0;

      /**
       * @brief Create an array for the passive variables.
       *
       * @param[out] buf  The pointer for the buffer.
       * @param[in] size  The size of the buffer.
       */
      virtual void createPassiveTypeBuffer(void* &buf, size_t size) const = 0;

      /**
       * @brief Create an array for the index variables.
       *
       * @param[out] buf  The pointer for the buffer.
       * @param[in] size  The size of the buffer.
       */
      virtual void createIndexTypeBuffer(void* &buf, size_t size) const = 0;

      /**
       * @brief Delete the array of the adjoint variables.
       *
       * @param[in,out] buf  The pointer for the buffer.
       */
      virtual void deleteAdjointTypeBuffer(void* &buf) const = 0;

      /**
       * @brief Delete the array of the passive variables.
       *
       * @param[in,out] buf  The pointer for the buffer.
       */
      virtual void deletePassiveTypeBuffer(void* &buf) const = 0;

      /**
       * @brief Delete the array of the index variables.
       *
       * @param[in,out] buf  The pointer for the buffer.
       */
      virtual void deleteIndexTypeBuffer(void* &buf) const = 0;
  };

  /**
   * @brief The static methods for the AD tool interface.
   *
   * All these static methods need to be implemented by the AD tool
   */
  struct StaticADToolInterface : public ADToolInterface {

      /**
       * @brief The actual type that the AD implementation uses.
       */
      typedef double Type;

      /**
       * @brief The type that is send through the modified buffers.
       */
      typedef double ModifiedType;

      /**
       * @brief The data type used for the floating point data.
       */
      typedef double PassiveType;

      /**
       * @brief The data type that is used for the adjoint variables.
       */
      typedef double AdjointType;

      /**
       * @brief The data type from the AD tool for the identification of AD variables.
       */
      typedef int IndexType;

      /**
       * @brief Copies the nescessary data from the user buffer into the buffer crated by MeDiPack.
       *
       * @param[out] modValue  The value in the modified buffer
       * @param[in]     value  The value in the user buffer.
       */
      static void setIntoModifyBuffer(ModifiedType& modValue, const Type& value);

      /**
       * @brief Copies the nescessary data from the received MeDiPack buffer into the user buffer.
       *
       * @param[in]  modValue  The value in the modified buffer
       * @param[out]    value  The value in the user buffer.
       */
      static void getFromModifyBuffer(const ModifiedType& modValue, Type& value);

      /**
       * @brief Get the AD identifier for this value.
       * @param[in] value  The AD value.
       * @return The identifier for the AD value.
       */
      static IndexType getIndex(const Type& value);

      /**
       * @brief Register an AD value on the receiving side of the communication.
       *
       * @param[in,out]    value  The AD value in the user buffer on the receiving side.
       * @param[out]   oldPrimal  The old primal value that was overwritten by this value.
       * @return THe identifier for the registered AD value.
       */
      static IndexType registerValue(Type& value, PassiveType& oldPrimal);

      /**
       * @brief Delete the index in a buffer such that the buffer can be overwritten.
       * @param[in,out] value  The AD value in the buffer.
       */
      static void clearIndex(Type& value);

      /**
       * @brief Get the primal floating point value of the AD value.
       * @param[in] value  The AD value.
       * @return The primal floating point value that is represented by the AD value.
       */
      static PassiveType getValue(const Type& value);
  };


  /**
   * A type save implementation of the AD tool interface.
   *
   * All functions with void types are forwarded to the type safe implementation.
   *
   * @tparam         Impl  The class that implements the actual interface in a type save manner.
   * @tparam AdjointTypeB  The data type for the adjoint variables that the implementation uses.
   * @tparam PassiveTypeB  The data type for the passive variables that the implementation uses.
   * @tparam   IndexTypeB  The data type for the index variables that the implementation uses.
   */
  template <typename Impl, typename AdjointTypeB, typename PassiveTypeB, typename IndexTypeB>
  class ADToolBase : public ADToolInterface {
    public:

      /**
       * @brief Construct the type safe wrapper.
       * @param[in] adjointMpiType  The mpi data type for the adjoint type.
       */
      ADToolBase(MPI_Datatype adjointMpiType) :
        ADToolInterface(adjointMpiType) {}

      void getAdjoints(const void* indices, void* adjoints, int elements) const {
        cast().getAdjoints(castBuffer<IndexTypeB>(indices), castBuffer<AdjointTypeB>(adjoints), elements);
      }

      void updateAdjoints(const void* indices, const void* adjoints, int elements) const {
        cast().updateAdjoints(castBuffer<IndexTypeB>(indices), castBuffer<AdjointTypeB>(adjoints), elements);
      }

      void setReverseValues(const void* indices, const void* primals, int elements) const {
        cast().setReverseValues(castBuffer<IndexTypeB>(indices), castBuffer<PassiveTypeB>(primals), elements);
      }

      void combineAdjoints(void* buf, const int elements, const int ranks) const {
        cast().combineAdjoints(castBuffer<AdjointTypeB>(buf), elements, ranks);
      }

      void createAdjointTypeBuffer(void* &buf, size_t size) const {
        cast().createAdjointTypeBuffer(castBuffer<AdjointTypeB>(buf), size);
      }

      void createPassiveTypeBuffer(void* &buf, size_t size) const {
        cast().createPassiveTypeBuffer(castBuffer<PassiveTypeB>(buf), size);
      }

      void createIndexTypeBuffer(void* &buf, size_t size) const {
        cast().createIndexTypeBuffer(castBuffer<IndexTypeB>(buf), size);
      }

      void deleteAdjointTypeBuffer(void* &buf) const {
        cast().deleteAdjointTypeBuffer(castBuffer<AdjointTypeB>(buf));
      }

      void deletePassiveTypeBuffer(void* &buf) const {
        cast().deletePassiveTypeBuffer(castBuffer<PassiveTypeB>(buf));
      }

      void deleteIndexTypeBuffer(void* &buf) const {
        cast().deleteIndexTypeBuffer(castBuffer<IndexTypeB>(buf));
      }

    private:

      inline Impl& cast() {
        return *reinterpret_cast<Impl*>(this);
      }

      inline const Impl& cast() const {
        return *reinterpret_cast<const Impl*>(this);
      }

      template <typename T>
      inline T*& castBuffer(void*& buf) const {
        return reinterpret_cast<T*&>(buf);
      }

      template <typename T>
      inline const T*& castBuffer(const void* &buf) const {
        return reinterpret_cast<const T*&>(buf);
      }
  };
}
