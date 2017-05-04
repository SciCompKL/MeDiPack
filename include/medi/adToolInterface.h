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

#include "typeDefinitions.h"

namespace medi {

  /**
   * @brief The interface for the AD tool that is accessed by MeDiPack.
   */
  class ADToolInterface {

      MPI_Datatype adjointMpiType;

    public:

      typedef void AdjointType;
      typedef void PassiveType;
      typedef void IndexType;

      ADToolInterface(MPI_Datatype adjointMpiType) :
        adjointMpiType(adjointMpiType) {}

      virtual ~ADToolInterface() {}

      MPI_Datatype getAdjointMpiType() const {
        return adjointMpiType;
      }

      virtual bool isActiveType() const = 0;
      virtual bool isHandleRequired() const  = 0;
      virtual bool isOldPrimalsRequired() const = 0;
      virtual void startAssembly(HandleBase* h) = 0;
      virtual void stopAssembly(HandleBase* h) = 0;
      virtual void addToolAction(HandleBase* h) = 0;

      virtual void getAdjoints(const void* indices, void* adjoints, int elements) const = 0;

      virtual void updateAdjoints(const void* indices, const void* adjoints, int elements) const = 0;

      virtual void setReverseValues(const void* indices, const void* primals, int elements) const = 0;

      virtual void combineAdjoints(void* buf, const int elements, const int ranks) const = 0;

      virtual void createAdjointTypeBuffer(void* &buf, size_t size) const = 0;

      virtual void createPassiveTypeBuffer(void* &buf, size_t size) const = 0;

      virtual void createIndexTypeBuffer(void* &buf, size_t size) const = 0;

      virtual void deleteAdjointTypeBuffer(void* &buf) const = 0;

      virtual void deletePassiveTypeBuffer(void* &buf) const = 0;

      virtual void deleteIndexTypeBuffer(void* &buf) const = 0;
  };

  template <typename Impl, typename AdjointTypeB, typename PassiveTypeB, typename IndexTypeB>
  class ADToolBase : public ADToolInterface {
    public:

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
