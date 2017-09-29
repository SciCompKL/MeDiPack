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

#include "../macros.h"
#include "../mpiTypeInterface.hpp"

namespace medi {

  class MpiStructType final : public MpiTypeInterface {

    private:
      bool mdificationRequired;
      int valuesPerElement;

      const ADToolInterface* adInterface;

      size_t typeExtend;
      size_t typeOffset;
      size_t modifiedExtend;

      int nTypes;
      int* blockLengths;
      int* blockOffsets;
      int* modifiedBlockOffsets;
      MpiTypeInterface** types;


    public:
      typedef void Type;
      typedef void ModifiedType;
      typedef void AdjointType;
      typedef void PassiveType;
      typedef void IndexType;


      MpiStructType(int count, const int* array_of_blocklengths, const MPI_Aint* array_of_displacements, MpiTypeInterface* const * array_of_types) :
        MpiTypeInterface(MPI_INT, MPI_INT) {

        MPI_Datatype* mpiTypes = new MPI_Datatype[count];

        MPI_Datatype newMpiType;
        MPI_Datatype newModMpiType;

        nTypes = count;
        blockLengths = new int[count];
        blockOffsets = new int[count];
        types = new MpiTypeInterface*[count];

        // check if a modified buffer is required and populate the mpiTypes as well as the arrayes
        mdificationRequired = false;
        valuesPerElement = 0;
        for(int i = 0; i < count; ++i) {
          mdificationRequired |= array_of_types[i]->isModifiedBufferRequired();
          if(array_of_types[i]->getADTool().isActiveType()) {
            adInterface = &array_of_types[i]->getADTool();
          }

          valuesPerElement += array_of_types[i]->computeActiveElements(array_of_blocklengths[i]);

          blockLengths[i] = array_of_blocklengths[i];
          blockOffsets[i] = array_of_displacements[i];
          mpiTypes[i] = array_of_types[i]->getMpiType();
          types[i] = array_of_types[i];
        }

        MPI_Type_create_struct(count, array_of_blocklengths, array_of_displacements, mpiTypes, &newMpiType);

        if(mdificationRequired) {
          MPI_Aint* modifiedDisplacements = new MPI_Aint[count];
          MPI_Datatype* modifiedMpiTypes = new MPI_Datatype[count];

          modifiedBlockOffsets = new int[count];

          MPI_Aint totalDisplacement = 0;
          for(int i = 0; i < count; ++i) {
            MPI_Aint curLowerBound;
            MPI_Aint curExtend;

            modifiedMpiTypes[i] = array_of_types[i]->getModifiedMpiType();
            MPI_Type_get_extent(modifiedMpiTypes[i], &curLowerBound, &curExtend);

            mediAssert(0 == curLowerBound); // The modified types are always packed without any holes.
            modifiedDisplacements[i] = totalDisplacement;
            modifiedBlockOffsets[i] = totalDisplacement;
            totalDisplacement += curExtend * array_of_blocklengths[i];
          }

          MPI_Type_create_struct(count, array_of_blocklengths, modifiedDisplacements, modifiedMpiTypes, &newModMpiType);

          delete [] modifiedMpiTypes;
          delete [] modifiedDisplacements;
        } else {

          modifiedBlockOffsets = nullptr;
          newModMpiType = newMpiType;
        }

        MPI_Aint lb = 0;
        MPI_Aint ext = 0;
        MPI_Type_get_extent(newMpiType, &lb, &ext);
        typeOffset = lb;
        typeExtend = ext;

        MPI_Type_get_extent(newModMpiType, &lb, &ext);
        modifiedExtend = ext;

        setMpiTypes(newMpiType, newModMpiType);

        delete [] mpiTypes;
      }

      ~MpiStructType() {
        if(nullptr != modifiedBlockOffsets) {
          delete [] modifiedBlockOffsets;
        }
        delete [] blockOffsets;
        delete [] blockLengths;
        delete [] types;
      }

      int computeBufOffset(size_t element) const {
        return (int)(element * typeExtend);
      }

      int computeModOffset(size_t element) const {
        return (int)(element * modifiedExtend);
      }

      const void* computeBufferPointer(const void* buf, size_t offset) const {
        return (const void*)((const char*) buf + offset);
      }

      void* computeBufferPointer(void* buf, size_t offset) const {
        return (void*)((char*) buf + offset);
      }

      bool isModifiedBufferRequired() const {
        return mdificationRequired;
      }

      int computeActiveElements(const int count) const {
        return count * valuesPerElement;
      }


      const ADToolInterface& getADTool() const {
        return *adInterface;
      }

      void copyIntoModifiedBuffer(const void* buf, size_t bufOffset, void* bufMod, size_t bufModOffset, int elements) const {
        for(int i = 0; i < elements; ++i) {
          int totalBufOffset = computeBufOffset(i + bufOffset);
          int totalModOffset = computeModOffset(i + bufModOffset);

          for(int curType = 0; curType < nTypes; ++curType) {
            types[curType]->copyIntoModifiedBuffer(computeBufferPointer(buf, totalBufOffset + blockOffsets[curType]), 0, computeBufferPointer(bufMod, totalModOffset + modifiedBlockOffsets[curType]), 0, blockLengths[curType]);
          }
        }
      }

      void copyFromModifiedBuffer(void* buf, size_t bufOffset, const void* bufMod, size_t bufModOffset, int elements) const {
        for(int i = 0; i < elements; ++i) {
          int totalBufOffset = computeBufOffset(i + bufOffset);
          int totalModOffset = computeModOffset(i + bufModOffset);


          for(int curType = 0; curType < nTypes; ++curType) {
            types[curType]->copyFromModifiedBuffer(computeBufferPointer(buf, totalBufOffset + blockOffsets[curType]), 0, computeBufferPointer(bufMod, totalModOffset + modifiedBlockOffsets[curType]), 0, blockLengths[curType]);
          }
        }
      }

      void getIndices(const void* buf, size_t bufOffset, void* indices, size_t bufModOffset, int elements) const {
        int totalIndexOffset = computeActiveElements(bufModOffset);  // indices are lineralized and counted up in the loop

        for(int i = 0; i < elements; ++i) {
          int totalBufOffset = computeBufOffset(i + bufOffset);

          for(int curType = 0; curType < nTypes; ++curType) {
            types[curType]->getIndices(computeBufferPointer(buf, totalBufOffset + blockOffsets[curType]), 0, indices, totalIndexOffset, blockLengths[curType]);
            totalIndexOffset += types[curType]->computeActiveElements(blockLengths[curType]);
          }
        }
      }

      void registerValue(void* buf, size_t bufOffset, void* indices, void* oldPrimals, size_t bufModOffset, int elements) const {
        int totalIndexOffset = computeActiveElements(bufModOffset);  // indices are lineralized and counted up in the loop

        for(int i = 0; i < elements; ++i) {
          int totalBufOffset = computeBufOffset(i + bufOffset);

          for(int curType = 0; curType < nTypes; ++curType) {
            types[curType]->registerValue(computeBufferPointer(buf, totalBufOffset + blockOffsets[curType]), 0, indices, oldPrimals, totalIndexOffset, blockLengths[curType]);
            totalIndexOffset +=  types[curType]->computeActiveElements(blockLengths[curType]);
          }
        }
      }

      void clearIndices(void* buf, size_t bufOffset, int elements) const {
        for(int i = 0; i < elements; ++i) {
          int totalBufOffset = computeBufOffset(i + bufOffset);

          for(int curType = 0; curType < nTypes; ++curType) {
            types[curType]->clearIndices(computeBufferPointer(buf, totalBufOffset + blockOffsets[curType]), 0, blockLengths[curType]);
          }
        }
      }

      void getValues(const void* buf, size_t bufOffset, void* primals, size_t bufModOffset, int elements) const {
        int totalPrimalsOffset = computeActiveElements(bufModOffset);  // indices are lineralized and counted up in the loop
        for(int i = 0; i < elements; ++i) {
          int totalBufOffset = computeBufOffset(i + bufOffset);

          for(int curType = 0; curType < nTypes; ++curType) {
            types[curType]->getValues(computeBufferPointer(buf, totalBufOffset + blockOffsets[curType]), 0, primals, totalPrimalsOffset, blockLengths[curType]);
            totalPrimalsOffset += types[curType]->computeActiveElements(blockLengths[curType]);
          }
        }
      }

      void performReduce(void* buf, void* target, int count, AMPI_Op op, int ranks) const {
        for(int j = 1; j < ranks; ++j) {
          int totalBufOffset = computeBufOffset(count * j);

          MPI_Reduce_local(computeBufferPointer(buf, totalBufOffset), buf, count, this->getMpiType(), op.primalFunction);
        }

        if(0 != ranks) {
          copy(buf, 0, target, 0, count);
        }
      }

      void copy(void* from, size_t fromOffset, void* to, size_t toOffset, int count) const {
        for(int i = 0; i < count; ++i) {
          int totalFromOffset = computeBufOffset(i + fromOffset);
          int totalToOffset = computeBufOffset(i + toOffset);

          for(int curType = 0; curType < nTypes; ++curType) {
            types[curType]->copy(computeBufferPointer(from, totalFromOffset + blockOffsets[curType]), 0, computeBufferPointer(to, totalToOffset + modifiedBlockOffsets[curType]), 0, blockLengths[curType]);
          }
        }
      }

      void createTypeBuffer(void* &buf, size_t size) const {
        char* b = (char*)calloc(size, typeExtend);
        b -= typeOffset;
        buf = (void*)b;
      }

      void createModifiedTypeBuffer(void* &buf, size_t size) const {
        buf = calloc(size, modifiedExtend);
      }


      void deleteTypeBuffer(void* &buf) const {
        char* b = (char*)buf;
        b += typeOffset;
        free(b);
        buf = nullptr;
      }

      void deleteModifiedTypeBuffer(void* &buf) const {
        free(buf);
        buf = nullptr;
      }
  };

  inline int AMPI_Type_create_struct(int count, const int* array_of_blocklengths, const MPI_Aint* array_of_displacements, MpiTypeInterface* const* array_of_types, MpiTypeInterface** newtype) {

    *newtype = new MpiStructType(count, array_of_blocklengths, array_of_displacements, array_of_types);

    return 0;
  }

  inline int AMPI_Type_commit(MpiTypeInterface** d) {
    MpiTypeInterface* datatype = *d;

    if(datatype->isModifiedBufferRequired()) {
      MPI_Datatype modType = datatype->getModifiedMpiType();
      MPI_Type_commit(&modType);
    }

    MPI_Datatype type = datatype->getMpiType();
    return MPI_Type_commit(&type);
  }

  inline int AMPI_Type_free(MpiTypeInterface** d) {
    MpiTypeInterface* datatype = *d;
    if(datatype->isModifiedBufferRequired()) {
      MPI_Datatype modType = datatype->getModifiedMpiType();
      MPI_Type_free(&modType);
    }

    MPI_Datatype type = datatype->getMpiType();
    int ret = MPI_Type_free(&type);

    delete datatype;

    return ret;
  }
}
