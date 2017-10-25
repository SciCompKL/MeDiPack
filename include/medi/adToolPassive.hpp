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

#include "adToolInterface.h"
#include "macros.h"
#include "typeDefinitions.h"

/**
 * @brief Global namespace for MeDiPack - Message Differentiation Package
 */
namespace medi {

  /**
   * @brief Implementation for the AD tool interface of a type that is no AD type.
   *
   * All methods in this implementation contain no logic.
   */
  class ADToolPassive final : public ADToolBase<ADToolPassive, void, void, void> {
    public:

      typedef void PassiveType;
      typedef void AdjointType;
      typedef void IndexType;

      ADToolPassive(MPI_Datatype adjointType) :
        ADToolBase<ADToolPassive, void, void, void>(adjointType)
      {}

      inline bool isActiveType() const {return false;}
      inline bool isHandleRequired() const {return false;}
      inline bool isModifiedBufferRequired() const {return false;}
      inline bool isOldPrimalsRequired() const {return false;}
      inline void startAssembly(HandleBase* h) const {MEDI_UNUSED(h);}
      inline void stopAssembly(HandleBase* h) const {MEDI_UNUSED(h);}
      inline void addToolAction(HandleBase* h) const {MEDI_UNUSED(h);}

      inline AMPI_Op convertOperator(AMPI_Op op) const {
        return op;
      }
      inline void getAdjoints(const IndexType* indices, AdjointType* adjoints, int elements) const {
        MEDI_UNUSED(indices);
        MEDI_UNUSED(adjoints);
        MEDI_UNUSED(elements);
      }

      inline void updateAdjoints(const IndexType* indices, const AdjointType* adjoints, int elements) const {
        MEDI_UNUSED(indices);
        MEDI_UNUSED(adjoints);
        MEDI_UNUSED(elements);
      }

      inline void setReverseValues(const IndexType* indices, const PassiveType* primals, int elements) const {
        MEDI_UNUSED(indices);
        MEDI_UNUSED(primals);
        MEDI_UNUSED(elements);
      }

      inline void combineAdjoints(AdjointType* buf, const int elements, const int ranks) const {
        MEDI_UNUSED(buf);
        MEDI_UNUSED(elements);
        MEDI_UNUSED(ranks);
      }

      inline void createAdjointTypeBuffer(AdjointType* &buf, size_t size) const {
        MEDI_UNUSED(size);

        buf = nullptr;
      }

      inline void createPassiveTypeBuffer(PassiveType* &buf, size_t size) const {
        MEDI_UNUSED(size);

        buf = nullptr;
      }

      inline void createIndexTypeBuffer(IndexType* &buf, size_t size) const {
        MEDI_UNUSED(size);

        buf = nullptr;
      }

      inline void deleteAdjointTypeBuffer(AdjointType* &buf) const {
        buf = nullptr;
      }

      inline void deletePassiveTypeBuffer(PassiveType* &buf) const {
        buf = nullptr;
      }

      inline void deleteIndexTypeBuffer(IndexType* &buf) const {
        buf = nullptr;
      }
  };
}
