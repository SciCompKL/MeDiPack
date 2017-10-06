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
#include "../exceptions.hpp"

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

      MpiStructType(const MpiStructType* other) :
        MpiTypeInterface(MPI_INT, MPI_INT)
      {
        adInterface = other->adInterface;
        typeExtend = other->typeExtend;
        typeOffset = other->typeOffset;
        modifiedExtend = other->modifiedExtend;

        nTypes = other->nTypes;
        blockLengths = new int [nTypes];
        blockOffsets = new int [nTypes];
        types = new MpiTypeInterface* [nTypes];

        if(nullptr != other->modifiedBlockOffsets) {
          modifiedBlockOffsets = new int [nTypes];
        }

        for(int i = 0; i < nTypes; ++i) {
          blockLengths[i] = other->blockLengths[i];
          blockOffsets[i] = other->blockOffsets[i];
          types[i] = other->types[i]->clone();

          if(nullptr != modifiedBlockOffsets) {
            modifiedBlockOffsets[i] = other->modifiedBlockOffsets[i];
          }
        }

        MPI_Datatype type;
        MPI_Datatype modType;

        MPI_Type_dup(this->getMpiType(), &type);
        if(this->getMpiType() != this->getModifiedMpiType()) {
          MPI_Type_dup(this->getModifiedMpiType(), &modType);
        } else {
          modType = type;
        }

        setMpiTypes(type, modType);
      }


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
          types[i] = array_of_types[i]->clone();
        }

        MPI_Type_create_struct(count, array_of_blocklengths, array_of_displacements, mpiTypes, &newMpiType);

        if(mdificationRequired) {
          MPI_Aint* modifiedDisplacements = new MPI_Aint[count + 1]; // We might need to add padding so add an extra element
          MPI_Datatype* modifiedMpiTypes = new MPI_Datatype[count + 1];  // We might need to add padding so add an extra element
          int* modifiedArrayLength = new int[count + 1];   // We might need to add padding so add an extra element

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
            modifiedArrayLength[i] = array_of_blocklengths[i];
            totalDisplacement += curExtend * array_of_blocklengths[i];
          }

          // TODO: This code assumes a 64-bit machine and if the last member of a struct is a byte, that these are
          //       padding bytes
          size_t paddingBytes = totalDisplacement % sizeof(double);
          int numberOfTypes = count; // This is obvious default we might increase this in the ifs below
          if(paddingBytes != 0) {
            // yuck we have to add padding

            // check if the last member was a byte, if yes recompute the total displacement
            if(modifiedMpiTypes[count - 1] == MPI_BYTE) {
              totalDisplacement -= array_of_blocklengths[count - 1]; // extend is one
              paddingBytes = totalDisplacement % sizeof(double);

              if(paddingBytes != 0) {
                // we still have padding bytes so change the array size
                modifiedArrayLength[count - 1] = paddingBytes;
                totalDisplacement += paddingBytes;
              } else {
                // no padding bytes left remove the padding bytes from the creation
                modifiedArrayLength[count - 1] = 0;
              }

              // since we assume now that the last type are padding bytes, we can remove it from the iterations
              blockLengths[count - 1] = 0;
            } else {
              // The last type is now padding type so we have to add a new type to the lists
              modifiedMpiTypes[count] = MPI_BYTE;
              modifiedDisplacements[count] = totalDisplacement;
              modifiedArrayLength[count] = paddingBytes;
              totalDisplacement += paddingBytes;

              numberOfTypes += 1;
            }
          }

          MPI_Type_create_struct(numberOfTypes, modifiedArrayLength, modifiedDisplacements, modifiedMpiTypes, &newModMpiType);

          delete [] modifiedMpiTypes;
          delete [] modifiedDisplacements;
          delete [] modifiedArrayLength;
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

        MPI_Datatype temp;
        if(this->getModifiedMpiType() != this->getMpiType()) {
          temp = this->getModifiedMpiType();
          MPI_Type_free(&temp);
        }
        temp = this->getMpiType();
        MPI_Type_free(&temp);

        for(int i = 0; i < nTypes; ++i) {
          delete types[i];
        }

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

      MpiStructType* clone() const {
        return new MpiStructType(this);
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

    delete datatype;

    *d = nullptr;
    return 0;
  }

  inline int AMPI_Type_dup(MpiTypeInterface* oldtype, MpiTypeInterface** newtype) {

    *newtype  = oldtype->clone();

    return 0;
  }
}
