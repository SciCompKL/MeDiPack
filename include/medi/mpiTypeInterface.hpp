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

#include <mpi.h>

#include "adToolInterface.h"
#include "mpiOp.hpp"

namespace medi {
  class MpiTypeInterface {
    private:

      MPI_Datatype mpiType;
      MPI_Datatype modifiedMpiType;
    public:

      typedef void Type;
      typedef void ModifiedType;
      typedef void AdjointType;
      typedef void PassiveType;
      typedef void IndexType;

      MpiTypeInterface(MPI_Datatype mpiType, MPI_Datatype modifiedMpiType) :
        mpiType(mpiType),
        modifiedMpiType(modifiedMpiType) {}

      virtual ~MpiTypeInterface() {}

      MPI_Datatype getMpiType() const {
        return mpiType;
      }

      MPI_Datatype getModifiedMpiType() const {
        return modifiedMpiType;
      }


      virtual bool isModifiedBufferRequired() const = 0;

      virtual int computeActiveElements(const int count) const = 0;

      virtual ADToolInterface& getADTool() = 0;

      virtual void copyIntoModifiedBuffer(const void* buf, size_t bufOffset, void* bufMod, size_t bufModOffset, int elements) const = 0;

      virtual void copyFromModifiedBuffer(void* buf, size_t bufOffset, const void* bufMod, size_t bufModOffset, int elements) const = 0;

      virtual void getIndices(const void* buf, size_t bufOffset, void* indices, size_t bufModOffset, int elements) const = 0;

      virtual void registerValue(void* buf, size_t bufOffset, void* indices, void* oldPrimals, size_t bufModOffset, int elements) const = 0;

      virtual void clearIndices(void* buf, size_t bufOffset, int elements) const = 0;

      virtual void getValues(const void* buf, size_t bufOffset, void* primals, size_t bufModOffset, int elements) const = 0;

      virtual void performReduce(void* buf, void* target, int count, AMPI_Op op, int ranks) const = 0;

      virtual void copy(void* from, size_t fromOffset, void* to, size_t toOffset, int count) const  = 0;

      virtual void createTypeBuffer(void* &buf, size_t size) const = 0;

      virtual void createModifiedTypeBuffer(void* &buf, size_t size) const = 0;

      virtual void deleteTypeBuffer(void* &buf) const = 0;

      virtual void deleteModifiedTypeBuffer(void* &buf) const = 0;
  };

  /**
   * Implements all functions from MpiType that require a type change.
   *
   * Functions that are not implemented: isModifiedBufferRequired
   */
  template <typename Impl, typename TypeB, typename ModifiedTypeB, typename ADToolB>
  class MpiTypeBase : public MpiTypeInterface {
    public:

      typedef typename ADToolB::PassiveType PassiveTypeB;
      typedef typename ADToolB::IndexType IndexTypeB;

      MpiTypeBase(MPI_Datatype mpiType, MPI_Datatype modifiedMpiType) :
        MpiTypeInterface(mpiType, modifiedMpiType) {}

      int computeActiveElements(const int count) const {
        return cast().computeActiveElements(count);
      }

      void copyIntoModifiedBuffer(const void* buf, size_t bufOffset, void* bufMod, size_t bufModOffset, int elements) const {
        cast().copyIntoModifiedBuffer(castBuffer<TypeB>(buf), bufOffset, castBuffer<ModifiedTypeB>(bufMod), bufModOffset, elements);
      }

      void copyFromModifiedBuffer(void* buf, size_t bufOffset, const void* bufMod, size_t bufModOffset, int elements) const {
        cast().copyFromModifiedBuffer(castBuffer<TypeB>(buf), bufOffset, castBuffer<ModifiedTypeB>(bufMod), bufModOffset, elements);
      }

      void getIndices(const void* buf, size_t bufOffset, void* indices, size_t bufModOffset, int elements) const {
        cast().getIndices(castBuffer<TypeB>(buf), bufOffset, castBuffer<IndexTypeB>(indices), bufModOffset, elements);
      }

      void registerValue(void* buf, size_t bufOffset, void* indices, void* oldPrimals, size_t bufModOffset, int elements) const {
        cast().registerValue(castBuffer<TypeB>(buf), bufOffset, castBuffer<IndexTypeB>(indices), castBuffer<PassiveTypeB>(oldPrimals), bufModOffset, elements);
      }

      void clearIndices(void* buf, size_t bufOffset, int elements) const {
        cast().clearIndices(castBuffer<TypeB>(buf), bufOffset, elements);
      }

      void getValues(const void* buf, size_t bufOffset, void* primals, size_t bufModOffset, int elements) const {
        cast().getValues(castBuffer<TypeB>(buf), bufOffset, castBuffer<PassiveTypeB>(primals), bufModOffset, elements);
      }

      void performReduce(void* buf, void* target, int count, AMPI_Op op, int ranks) const {
        cast().performReduce(castBuffer<TypeB>(buf), castBuffer<TypeB>(target), count, op, ranks);
      }

      void copy(void* from, size_t fromOffset, void* to, size_t toOffset, int count) const {
        cast().copy(castBuffer<TypeB>(from), fromOffset, castBuffer<TypeB>(to), toOffset, count);
      }

      void createTypeBuffer(void* &buf, size_t size) const {
        cast().createTypeBuffer(castBuffer<TypeB>(buf), size);
      }

      void createModifiedTypeBuffer(void* &buf, size_t size) const {
        cast().createModifiedTypeBuffer(castBuffer<ModifiedTypeB>(buf), size);
      }

      void deleteTypeBuffer(void* &buf) const {
        cast().deleteTypeBuffer(castBuffer<TypeB>(buf));
      }

      void deleteModifiedTypeBuffer(void* &buf) const {
        cast().deleteModifiedTypeBuffer(castBuffer<ModifiedTypeB>(buf));
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
