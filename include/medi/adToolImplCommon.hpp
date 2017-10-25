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
#include "adToolInterface.h"

/**
 * @brief Global namespace for MeDiPack - Message Differentiation Package
 */
namespace medi {

  template <typename Impl, bool restorePrimal, bool modifiedBuffer, typename Type, typename AdjointType, typename PassiveType, typename IndexType>
  class ADToolImplCommon : public ADToolBase<Impl, AdjointType, PassiveType, IndexType> {
    public:

      ADToolImplCommon(MPI_Datatype adjointMpiType) :
        ADToolBase<Impl, AdjointType, PassiveType, IndexType>(adjointMpiType) {}

      inline bool isActiveType() const {
        return true;
      }

      inline bool isModifiedBufferRequired() const {
        return modifiedBuffer;
      }

      inline bool isOldPrimalsRequired() const {
        return restorePrimal;
      }

      inline void createAdjointTypeBuffer(AdjointType* &buf, size_t size) const {
        buf = new AdjointType[size];
      }

      inline void createPassiveTypeBuffer(PassiveType* &buf, size_t size) const {
        buf = new PassiveType[size];
      }

      inline void createIndexTypeBuffer(IndexType* &buf, size_t size) const {
        buf = new IndexType[size];
      }

      inline void deleteAdjointTypeBuffer(AdjointType* &buf) const {
        if(NULL != buf) {
          delete [] buf;
          buf = NULL;
        }
      }

      inline void deletePassiveTypeBuffer(PassiveType* &buf) const {
        if(NULL != buf) {
          delete [] buf;
          buf = NULL;
        }
      }

      inline void deleteIndexTypeBuffer(IndexType* &buf) const {
        if(NULL != buf) {
          delete [] buf;
          buf = NULL;
        }
      }


  };
}
