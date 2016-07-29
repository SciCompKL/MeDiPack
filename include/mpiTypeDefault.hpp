#pragma once

#include "macros.h"
#include "mpiTypeInterface.hpp"

namespace medi {

  template<typename ADTool>
  class MpiTypeDefault
      : public MpiTypeBase<
          MpiTypeDefault<ADTool>,
          typename ADTool::Type,
          typename ADTool::ModifiedType,
          typename ADTool::PassiveType,
          typename ADTool::AdjointType,
          typename ADTool::IndexType>
  {

    public:

      typedef typename ADTool::Type Type;
      typedef typename ADTool::ModifiedType ModifiedType;
      typedef typename ADTool::PassiveType PassiveType;
      typedef typename ADTool::AdjointType AdjointType;
      typedef typename ADTool::IndexType IndexType;

      typedef ADTool Tool;

      Tool adTool;

      MpiTypeDefault() :
        MpiTypeBase<MpiTypeDefault<ADTool>, Type, ModifiedType, PassiveType, AdjointType, IndexType>(Tool::MPIType),
        adTool() {}

      Tool& getADTool() {
        return adTool;
      }

      const int getValuesPerElement(const Type* buf) const {
        MEDI_UNUSED(buf);

        return 1;
      }

      bool isModifiedBufferRequired() const {
        return Tool::IS_RequiresModifiedBuffer;
      }

      void copyIntoModifiedBuffer(const Type* buf, ModifiedType* bufMod, int elements) const {
        if(ADTool::IS_RequiresModifiedBuffer) {
          for(int i = 0; i < elements; ++i) {
            ADTool::setIntoModifyBuffer(bufMod[i], buf[i]);
          }
        }
      }

      inline void copyFromModifiedBuffer(Type* buf, const ModifiedType* bufMod, int elements) const {
        if(ADTool::IS_RequiresModifiedBuffer) {
          for(int i = 0; i < elements; ++i) {
            ADTool::getFromModifyBuffer(bufMod[i], buf[i]);
          }
        }
      }

      inline void getIndices(const Type* buf, IndexType* indices, int elements) const {
        for(int i = 0; i < elements; ++i) {
          indices[i] = ADTool::getIndex(buf[i]);
        }
      }

      inline void registerValue(Type* buf, IndexType* indices, int elements) const {
        for(int i = 0; i < elements; ++i) {
          indices[i] = ADTool::registerValue(buf[i]);;
        }
      }

      inline void clearIndices(Type* buf, int elements) const {
        for(int i = 0; i < elements; ++i) {
          ADTool::clearIndex(buf[i]);
        }
      }

      inline void getValues(const Type* buf, PassiveType* primals, int elements) const {
        for(int pos = 0; pos < elements; ++pos) {
          primals[pos] = ADTool::getValue(buf[pos]);
        }
      }

      inline void getAdjoints(const int* indices, int count, AdjointType* adjoints) const {
        for(int pos = 0; pos < count; ++pos) {
          adjoints[pos] = ADTool::getAdjoint(indices[pos]);
        }
      }

      inline void updateAdjoints(const int* indices, int count, const AdjointType* adjoints) const {
        for(int pos = 0; pos < count; ++pos) {
          ADTool::updateAdjoint(indices[pos], adjoints[pos]);
        }
      }
  };
}
