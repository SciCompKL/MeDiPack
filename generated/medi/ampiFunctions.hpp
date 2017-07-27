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

#include "../../include/medi/medipack.h"
#include "../../include/medi/reverseFunctions.hpp"
#include "../../include/medi/ampi/async.hpp"

namespace medi {
  template<typename DATATYPE>
  struct AMPI_Bsend_AdjointHandle : public HandleBase {
    int bufTotalSize;
    typename DATATYPE::IndexType* bufIndices;
    typename DATATYPE::PassiveType* bufPrimals;
    /* required for async */ typename DATATYPE::PassiveType* bufAdjoints;
    int bufCount;
    int count;
    DATATYPE* datatype;
    int dest;
    int tag;
    AMPI_Comm comm;

    ~AMPI_Bsend_AdjointHandle () {
      if(nullptr != bufIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(bufIndices);
        bufIndices = nullptr;
      }
      if(nullptr != bufPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(bufPrimals);
        bufPrimals = nullptr;
      }
    }
  };


  template<typename DATATYPE>
  void AMPI_Bsend_b(HandleBase* handle) {
    AMPI_Bsend_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Bsend_AdjointHandle<DATATYPE>*>(handle);

    h->bufAdjoints = nullptr;
    h->datatype->getADTool().createAdjointTypeBuffer(h->bufAdjoints, h->bufTotalSize );

    AMPI_Bsend_adj<DATATYPE>(h->bufAdjoints, h->bufCount, h->count, h->datatype, h->dest, h->tag, h->comm);

    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->datatype->getADTool().updateAdjoints(h->bufIndices, h->bufAdjoints, h->bufTotalSize);
    h->datatype->getADTool().deleteAdjointTypeBuffer(h->bufAdjoints);
  }

  template<typename DATATYPE>
  int AMPI_Bsend(const typename DATATYPE::Type* buf, int count, DATATYPE* datatype, int dest, int tag, AMPI_Comm comm) {
    int rStatus;
    if(!datatype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Bsend(buf, count, datatype->getMpiType(), dest, tag, comm);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Bsend_AdjointHandle<DATATYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(datatype->getADTool().isHandleRequired()) {
        h = new AMPI_Bsend_AdjointHandle<DATATYPE>();
      }
      datatype->getADTool().startAssembly(h);
      typename DATATYPE::ModifiedType* bufMod = nullptr;
      int bufElements = 0;

      // compute the total size of the buffer
      bufElements = count;

      if(datatype->isModifiedBufferRequired() ) {
        datatype->createModifiedTypeBuffer(bufMod, bufElements);
      } else {
        bufMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(buf));
      }

      if(datatype->isModifiedBufferRequired()) {
        datatype->copyIntoModifiedBuffer(buf, 0, bufMod, 0, count);
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        h->bufCount = datatype->computeActiveElements(count);
        h->bufTotalSize = datatype->computeActiveElements(bufElements);
        datatype->getADTool().createIndexTypeBuffer(h->bufIndices, h->bufTotalSize);




        datatype->getIndices(buf, 0, h->bufIndices, 0, count);


        // pack all the variables in the handle
        h->func = AMPI_Bsend_b<DATATYPE>;
        h->count = count;
        h->datatype = datatype;
        h->dest = dest;
        h->tag = tag;
        h->comm = comm;
      }

      rStatus = MPI_Bsend(bufMod, count, datatype->getModifiedMpiType(), dest, tag, comm);
      datatype->getADTool().addToolAction(h);


      if(nullptr != h) {
        // handle the recv buffers
      }

      datatype->getADTool().stopAssembly(h);

      if(datatype->isModifiedBufferRequired() ) {
        datatype->deleteModifiedTypeBuffer(bufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename DATATYPE>
  struct AMPI_Ibsend_AdjointHandle : public HandleBase {
    int bufTotalSize;
    typename DATATYPE::IndexType* bufIndices;
    typename DATATYPE::PassiveType* bufPrimals;
    /* required for async */ typename DATATYPE::PassiveType* bufAdjoints;
    int bufCount;
    int count;
    DATATYPE* datatype;
    int dest;
    int tag;
    AMPI_Comm comm;
    AMPI_Request requestReverse;

    ~AMPI_Ibsend_AdjointHandle () {
      if(nullptr != bufIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(bufIndices);
        bufIndices = nullptr;
      }
      if(nullptr != bufPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(bufPrimals);
        bufPrimals = nullptr;
      }
    }
  };

  template<typename DATATYPE>
  struct AMPI_Ibsend_AsyncHandle : public HandleBase {
    const  typename DATATYPE::Type* buf;
    typename DATATYPE::ModifiedType* bufMod;
    int count;
    DATATYPE* datatype;
    int dest;
    int tag;
    AMPI_Comm comm;
    AMPI_Request* request;
    AMPI_Ibsend_AdjointHandle<DATATYPE>* h;
  };

  template<typename DATATYPE>
  void AMPI_Ibsend_b(HandleBase* handle) {
    AMPI_Ibsend_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Ibsend_AdjointHandle<DATATYPE>*>(handle);

    h->bufAdjoints = nullptr;
    h->datatype->getADTool().createAdjointTypeBuffer(h->bufAdjoints, h->bufTotalSize );

    AMPI_Ibsend_adj<DATATYPE>(h->bufAdjoints, h->bufCount, h->count, h->datatype, h->dest, h->tag, h->comm,
                              &h->requestReverse);

  }

  template<typename DATATYPE>
  void AMPI_Ibsend_b_finish(HandleBase* handle) {

    AMPI_Ibsend_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Ibsend_AdjointHandle<DATATYPE>*>(handle);
    MPI_Wait(&h->requestReverse.request, MPI_STATUS_IGNORE);

    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->datatype->getADTool().updateAdjoints(h->bufIndices, h->bufAdjoints, h->bufTotalSize);
    h->datatype->getADTool().deleteAdjointTypeBuffer(h->bufAdjoints);
  }

  template<typename DATATYPE>
  int AMPI_Ibsend_finish(HandleBase* handle);
  template<typename DATATYPE>
  int AMPI_Ibsend(const typename DATATYPE::Type* buf, int count, DATATYPE* datatype, int dest, int tag, AMPI_Comm comm,
                  AMPI_Request* request) {
    int rStatus;
    if(!datatype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Ibsend(buf, count, datatype->getMpiType(), dest, tag, comm, &request->request);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Ibsend_AdjointHandle<DATATYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(datatype->getADTool().isHandleRequired()) {
        h = new AMPI_Ibsend_AdjointHandle<DATATYPE>();
      }
      datatype->getADTool().startAssembly(h);
      typename DATATYPE::ModifiedType* bufMod = nullptr;
      int bufElements = 0;

      // compute the total size of the buffer
      bufElements = count;

      if(datatype->isModifiedBufferRequired() ) {
        datatype->createModifiedTypeBuffer(bufMod, bufElements);
      } else {
        bufMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(buf));
      }

      if(datatype->isModifiedBufferRequired()) {
        datatype->copyIntoModifiedBuffer(buf, 0, bufMod, 0, count);
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        h->bufCount = datatype->computeActiveElements(count);
        h->bufTotalSize = datatype->computeActiveElements(bufElements);
        datatype->getADTool().createIndexTypeBuffer(h->bufIndices, h->bufTotalSize);




        datatype->getIndices(buf, 0, h->bufIndices, 0, count);


        // pack all the variables in the handle
        h->func = AMPI_Ibsend_b<DATATYPE>;
        h->count = count;
        h->datatype = datatype;
        h->dest = dest;
        h->tag = tag;
        h->comm = comm;
      }

      rStatus = MPI_Ibsend(bufMod, count, datatype->getModifiedMpiType(), dest, tag, comm, &request->request);

      AMPI_Ibsend_AsyncHandle<DATATYPE>* asyncHandle = new AMPI_Ibsend_AsyncHandle<DATATYPE>();
      asyncHandle->buf = buf;
      asyncHandle->bufMod = bufMod;
      asyncHandle->count = count;
      asyncHandle->datatype = datatype;
      asyncHandle->dest = dest;
      asyncHandle->tag = tag;
      asyncHandle->comm = comm;
      asyncHandle->h = h;
      request->handle = asyncHandle;
      request->func = (ContinueFunction)AMPI_Ibsend_finish<DATATYPE>;

      // create adjoint wait
      if(nullptr != h) {
        WaitHandle* waitH = new WaitHandle((ContinueFunction)AMPI_Ibsend_b_finish<DATATYPE>, h);
        datatype->getADTool().addToolAction(waitH);
      }
    }

    return rStatus;
  }

  template<typename DATATYPE>
  int AMPI_Ibsend_finish(HandleBase* handle) {
    int rStatus = 0;

    AMPI_Ibsend_AsyncHandle<DATATYPE>* asyncHandle = static_cast<AMPI_Ibsend_AsyncHandle<DATATYPE>*>(handle);
    const  typename DATATYPE::Type* buf = asyncHandle->buf;
    typename DATATYPE::ModifiedType* bufMod = asyncHandle->bufMod;
    int count = asyncHandle->count;
    DATATYPE* datatype = asyncHandle->datatype;
    int dest = asyncHandle->dest;
    int tag = asyncHandle->tag;
    AMPI_Comm comm = asyncHandle->comm;
    AMPI_Request* request = asyncHandle->request;
    AMPI_Ibsend_AdjointHandle<DATATYPE>* h = asyncHandle->h;
    MEDI_UNUSED(buf); // Unused generated to ignore warnings
    MEDI_UNUSED(bufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(count); // Unused generated to ignore warnings
    MEDI_UNUSED(datatype); // Unused generated to ignore warnings
    MEDI_UNUSED(dest); // Unused generated to ignore warnings
    MEDI_UNUSED(tag); // Unused generated to ignore warnings
    MEDI_UNUSED(comm); // Unused generated to ignore warnings
    MEDI_UNUSED(request); // Unused generated to ignore warnings
    MEDI_UNUSED(h); // Unused generated to ignore warnings

    delete asyncHandle;

    if(datatype->getADTool().isActiveType()) {

      datatype->getADTool().addToolAction(h);


      if(nullptr != h) {
        // handle the recv buffers
      }

      datatype->getADTool().stopAssembly(h);

      if(datatype->isModifiedBufferRequired() ) {
        datatype->deleteModifiedTypeBuffer(bufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename DATATYPE>
  struct AMPI_Irecv_AdjointHandle : public HandleBase {
    int bufTotalSize;
    typename DATATYPE::IndexType* bufIndices;
    typename DATATYPE::PassiveType* bufPrimals;
    typename DATATYPE::PassiveType* bufOldPrimals;
    /* required for async */ typename DATATYPE::PassiveType* bufAdjoints;
    int bufCount;
    int count;
    DATATYPE* datatype;
    int source;
    int tag;
    AMPI_Comm comm;
    AMPI_Request requestReverse;

    ~AMPI_Irecv_AdjointHandle () {
      if(nullptr != bufIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(bufIndices);
        bufIndices = nullptr;
      }
      if(nullptr != bufPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(bufPrimals);
        bufPrimals = nullptr;
      }
      if(nullptr != bufOldPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(bufOldPrimals);
        bufOldPrimals = nullptr;
      }
    }
  };

  template<typename DATATYPE>
  struct AMPI_Irecv_AsyncHandle : public HandleBase {
    typename DATATYPE::Type* buf;
    typename DATATYPE::ModifiedType* bufMod;
    int count;
    DATATYPE* datatype;
    int source;
    int tag;
    AMPI_Comm comm;
    AMPI_Request* request;
    AMPI_Irecv_AdjointHandle<DATATYPE>* h;
  };

  template<typename DATATYPE>
  void AMPI_Irecv_b(HandleBase* handle) {
    AMPI_Irecv_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Irecv_AdjointHandle<DATATYPE>*>(handle);

    h->bufAdjoints = nullptr;
    h->datatype->getADTool().createAdjointTypeBuffer(h->bufAdjoints, h->bufTotalSize );
    // Adjoint buffers are always linear in space so we can accesses them in one sweep
    h->datatype->getADTool().getAdjoints(h->bufIndices, h->bufAdjoints, h->bufTotalSize);

    if(h->datatype->getADTool().isOldPrimalsRequired()) {
      h->datatype->getADTool().setReverseValues(h->bufIndices, h->bufOldPrimals, h->bufTotalSize);
    }

    AMPI_Irecv_adj<DATATYPE>(h->bufAdjoints, h->bufCount, h->count, h->datatype, h->source, h->tag, h->comm,
                             &h->requestReverse);

  }

  template<typename DATATYPE>
  void AMPI_Irecv_b_finish(HandleBase* handle) {

    AMPI_Irecv_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Irecv_AdjointHandle<DATATYPE>*>(handle);
    MPI_Wait(&h->requestReverse.request, MPI_STATUS_IGNORE);

    h->datatype->getADTool().deleteAdjointTypeBuffer(h->bufAdjoints);
  }

  template<typename DATATYPE>
  int AMPI_Irecv_finish(HandleBase* handle);
  template<typename DATATYPE>
  int AMPI_Irecv(typename DATATYPE::Type* buf, int count, DATATYPE* datatype, int source, int tag, AMPI_Comm comm,
                 AMPI_Request* request) {
    int rStatus;
    if(!datatype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Irecv(buf, count, datatype->getMpiType(), source, tag, comm, &request->request);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Irecv_AdjointHandle<DATATYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(datatype->getADTool().isHandleRequired()) {
        h = new AMPI_Irecv_AdjointHandle<DATATYPE>();
      }
      datatype->getADTool().startAssembly(h);
      typename DATATYPE::ModifiedType* bufMod = nullptr;
      int bufElements = 0;

      // compute the total size of the buffer
      bufElements = count;

      if(datatype->isModifiedBufferRequired() ) {
        datatype->createModifiedTypeBuffer(bufMod, bufElements);
      } else {
        bufMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(buf));
      }


      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        h->bufCount = datatype->computeActiveElements(count);
        h->bufTotalSize = datatype->computeActiveElements(bufElements);
        datatype->getADTool().createIndexTypeBuffer(h->bufIndices, h->bufTotalSize);


        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(datatype->getADTool().isOldPrimalsRequired()) {
          datatype->getADTool().createPassiveTypeBuffer(h->bufOldPrimals, h->bufTotalSize);
          datatype->getValues(buf, 0, h->bufOldPrimals, 0, count);
        }



        if(!datatype->isModifiedBufferRequired()) {
          datatype->clearIndices(buf, 0, count);
        }

        // pack all the variables in the handle
        h->func = AMPI_Irecv_b<DATATYPE>;
        h->count = count;
        h->datatype = datatype;
        h->source = source;
        h->tag = tag;
        h->comm = comm;
      }

      rStatus = MPI_Irecv(bufMod, count, datatype->getModifiedMpiType(), source, tag, comm, &request->request);

      AMPI_Irecv_AsyncHandle<DATATYPE>* asyncHandle = new AMPI_Irecv_AsyncHandle<DATATYPE>();
      asyncHandle->buf = buf;
      asyncHandle->bufMod = bufMod;
      asyncHandle->count = count;
      asyncHandle->datatype = datatype;
      asyncHandle->source = source;
      asyncHandle->tag = tag;
      asyncHandle->comm = comm;
      asyncHandle->h = h;
      request->handle = asyncHandle;
      request->func = (ContinueFunction)AMPI_Irecv_finish<DATATYPE>;

      // create adjoint wait
      if(nullptr != h) {
        WaitHandle* waitH = new WaitHandle((ContinueFunction)AMPI_Irecv_b_finish<DATATYPE>, h);
        datatype->getADTool().addToolAction(waitH);
      }
    }

    return rStatus;
  }

  template<typename DATATYPE>
  int AMPI_Irecv_finish(HandleBase* handle) {
    int rStatus = 0;

    AMPI_Irecv_AsyncHandle<DATATYPE>* asyncHandle = static_cast<AMPI_Irecv_AsyncHandle<DATATYPE>*>(handle);
    typename DATATYPE::Type* buf = asyncHandle->buf;
    typename DATATYPE::ModifiedType* bufMod = asyncHandle->bufMod;
    int count = asyncHandle->count;
    DATATYPE* datatype = asyncHandle->datatype;
    int source = asyncHandle->source;
    int tag = asyncHandle->tag;
    AMPI_Comm comm = asyncHandle->comm;
    AMPI_Request* request = asyncHandle->request;
    AMPI_Irecv_AdjointHandle<DATATYPE>* h = asyncHandle->h;
    MEDI_UNUSED(buf); // Unused generated to ignore warnings
    MEDI_UNUSED(bufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(count); // Unused generated to ignore warnings
    MEDI_UNUSED(datatype); // Unused generated to ignore warnings
    MEDI_UNUSED(source); // Unused generated to ignore warnings
    MEDI_UNUSED(tag); // Unused generated to ignore warnings
    MEDI_UNUSED(comm); // Unused generated to ignore warnings
    MEDI_UNUSED(request); // Unused generated to ignore warnings
    MEDI_UNUSED(h); // Unused generated to ignore warnings

    delete asyncHandle;

    if(datatype->getADTool().isActiveType()) {

      datatype->getADTool().addToolAction(h);

      if(datatype->isModifiedBufferRequired()) {
        datatype->copyFromModifiedBuffer(buf, 0, bufMod, 0, count);
      }

      if(nullptr != h) {
        // handle the recv buffers
        datatype->registerValue(buf, 0, h->bufIndices, h->bufOldPrimals, 0, count);
      }

      datatype->getADTool().stopAssembly(h);

      if(datatype->isModifiedBufferRequired() ) {
        datatype->deleteModifiedTypeBuffer(bufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename DATATYPE>
  struct AMPI_Irsend_AdjointHandle : public HandleBase {
    int bufTotalSize;
    typename DATATYPE::IndexType* bufIndices;
    typename DATATYPE::PassiveType* bufPrimals;
    /* required for async */ typename DATATYPE::PassiveType* bufAdjoints;
    int bufCount;
    int count;
    DATATYPE* datatype;
    int dest;
    int tag;
    AMPI_Comm comm;
    AMPI_Request requestReverse;

    ~AMPI_Irsend_AdjointHandle () {
      if(nullptr != bufIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(bufIndices);
        bufIndices = nullptr;
      }
      if(nullptr != bufPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(bufPrimals);
        bufPrimals = nullptr;
      }
    }
  };

  template<typename DATATYPE>
  struct AMPI_Irsend_AsyncHandle : public HandleBase {
    const  typename DATATYPE::Type* buf;
    typename DATATYPE::ModifiedType* bufMod;
    int count;
    DATATYPE* datatype;
    int dest;
    int tag;
    AMPI_Comm comm;
    AMPI_Request* request;
    AMPI_Irsend_AdjointHandle<DATATYPE>* h;
  };

  template<typename DATATYPE>
  void AMPI_Irsend_b(HandleBase* handle) {
    AMPI_Irsend_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Irsend_AdjointHandle<DATATYPE>*>(handle);

    h->bufAdjoints = nullptr;
    h->datatype->getADTool().createAdjointTypeBuffer(h->bufAdjoints, h->bufTotalSize );

    AMPI_Irsend_adj<DATATYPE>(h->bufAdjoints, h->bufCount, h->count, h->datatype, h->dest, h->tag, h->comm,
                              &h->requestReverse);

  }

  template<typename DATATYPE>
  void AMPI_Irsend_b_finish(HandleBase* handle) {

    AMPI_Irsend_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Irsend_AdjointHandle<DATATYPE>*>(handle);
    MPI_Wait(&h->requestReverse.request, MPI_STATUS_IGNORE);

    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->datatype->getADTool().updateAdjoints(h->bufIndices, h->bufAdjoints, h->bufTotalSize);
    h->datatype->getADTool().deleteAdjointTypeBuffer(h->bufAdjoints);
  }

  template<typename DATATYPE>
  int AMPI_Irsend_finish(HandleBase* handle);
  template<typename DATATYPE>
  int AMPI_Irsend(const typename DATATYPE::Type* buf, int count, DATATYPE* datatype, int dest, int tag, AMPI_Comm comm,
                  AMPI_Request* request) {
    int rStatus;
    if(!datatype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Irsend(buf, count, datatype->getMpiType(), dest, tag, comm, &request->request);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Irsend_AdjointHandle<DATATYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(datatype->getADTool().isHandleRequired()) {
        h = new AMPI_Irsend_AdjointHandle<DATATYPE>();
      }
      datatype->getADTool().startAssembly(h);
      typename DATATYPE::ModifiedType* bufMod = nullptr;
      int bufElements = 0;

      // compute the total size of the buffer
      bufElements = count;

      if(datatype->isModifiedBufferRequired() ) {
        datatype->createModifiedTypeBuffer(bufMod, bufElements);
      } else {
        bufMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(buf));
      }

      if(datatype->isModifiedBufferRequired()) {
        datatype->copyIntoModifiedBuffer(buf, 0, bufMod, 0, count);
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        h->bufCount = datatype->computeActiveElements(count);
        h->bufTotalSize = datatype->computeActiveElements(bufElements);
        datatype->getADTool().createIndexTypeBuffer(h->bufIndices, h->bufTotalSize);




        datatype->getIndices(buf, 0, h->bufIndices, 0, count);


        // pack all the variables in the handle
        h->func = AMPI_Irsend_b<DATATYPE>;
        h->count = count;
        h->datatype = datatype;
        h->dest = dest;
        h->tag = tag;
        h->comm = comm;
      }

      rStatus = MPI_Irsend(bufMod, count, datatype->getModifiedMpiType(), dest, tag, comm, &request->request);

      AMPI_Irsend_AsyncHandle<DATATYPE>* asyncHandle = new AMPI_Irsend_AsyncHandle<DATATYPE>();
      asyncHandle->buf = buf;
      asyncHandle->bufMod = bufMod;
      asyncHandle->count = count;
      asyncHandle->datatype = datatype;
      asyncHandle->dest = dest;
      asyncHandle->tag = tag;
      asyncHandle->comm = comm;
      asyncHandle->h = h;
      request->handle = asyncHandle;
      request->func = (ContinueFunction)AMPI_Irsend_finish<DATATYPE>;

      // create adjoint wait
      if(nullptr != h) {
        WaitHandle* waitH = new WaitHandle((ContinueFunction)AMPI_Irsend_b_finish<DATATYPE>, h);
        datatype->getADTool().addToolAction(waitH);
      }
    }

    return rStatus;
  }

  template<typename DATATYPE>
  int AMPI_Irsend_finish(HandleBase* handle) {
    int rStatus = 0;

    AMPI_Irsend_AsyncHandle<DATATYPE>* asyncHandle = static_cast<AMPI_Irsend_AsyncHandle<DATATYPE>*>(handle);
    const  typename DATATYPE::Type* buf = asyncHandle->buf;
    typename DATATYPE::ModifiedType* bufMod = asyncHandle->bufMod;
    int count = asyncHandle->count;
    DATATYPE* datatype = asyncHandle->datatype;
    int dest = asyncHandle->dest;
    int tag = asyncHandle->tag;
    AMPI_Comm comm = asyncHandle->comm;
    AMPI_Request* request = asyncHandle->request;
    AMPI_Irsend_AdjointHandle<DATATYPE>* h = asyncHandle->h;
    MEDI_UNUSED(buf); // Unused generated to ignore warnings
    MEDI_UNUSED(bufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(count); // Unused generated to ignore warnings
    MEDI_UNUSED(datatype); // Unused generated to ignore warnings
    MEDI_UNUSED(dest); // Unused generated to ignore warnings
    MEDI_UNUSED(tag); // Unused generated to ignore warnings
    MEDI_UNUSED(comm); // Unused generated to ignore warnings
    MEDI_UNUSED(request); // Unused generated to ignore warnings
    MEDI_UNUSED(h); // Unused generated to ignore warnings

    delete asyncHandle;

    if(datatype->getADTool().isActiveType()) {

      datatype->getADTool().addToolAction(h);


      if(nullptr != h) {
        // handle the recv buffers
      }

      datatype->getADTool().stopAssembly(h);

      if(datatype->isModifiedBufferRequired() ) {
        datatype->deleteModifiedTypeBuffer(bufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename DATATYPE>
  struct AMPI_Isend_AdjointHandle : public HandleBase {
    int bufTotalSize;
    typename DATATYPE::IndexType* bufIndices;
    typename DATATYPE::PassiveType* bufPrimals;
    /* required for async */ typename DATATYPE::PassiveType* bufAdjoints;
    int bufCount;
    int count;
    DATATYPE* datatype;
    int dest;
    int tag;
    AMPI_Comm comm;
    AMPI_Request requestReverse;

    ~AMPI_Isend_AdjointHandle () {
      if(nullptr != bufIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(bufIndices);
        bufIndices = nullptr;
      }
      if(nullptr != bufPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(bufPrimals);
        bufPrimals = nullptr;
      }
    }
  };

  template<typename DATATYPE>
  struct AMPI_Isend_AsyncHandle : public HandleBase {
    const  typename DATATYPE::Type* buf;
    typename DATATYPE::ModifiedType* bufMod;
    int count;
    DATATYPE* datatype;
    int dest;
    int tag;
    AMPI_Comm comm;
    AMPI_Request* request;
    AMPI_Isend_AdjointHandle<DATATYPE>* h;
  };

  template<typename DATATYPE>
  void AMPI_Isend_b(HandleBase* handle) {
    AMPI_Isend_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Isend_AdjointHandle<DATATYPE>*>(handle);

    h->bufAdjoints = nullptr;
    h->datatype->getADTool().createAdjointTypeBuffer(h->bufAdjoints, h->bufTotalSize );

    AMPI_Isend_adj<DATATYPE>(h->bufAdjoints, h->bufCount, h->count, h->datatype, h->dest, h->tag, h->comm,
                             &h->requestReverse);

  }

  template<typename DATATYPE>
  void AMPI_Isend_b_finish(HandleBase* handle) {

    AMPI_Isend_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Isend_AdjointHandle<DATATYPE>*>(handle);
    MPI_Wait(&h->requestReverse.request, MPI_STATUS_IGNORE);

    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->datatype->getADTool().updateAdjoints(h->bufIndices, h->bufAdjoints, h->bufTotalSize);
    h->datatype->getADTool().deleteAdjointTypeBuffer(h->bufAdjoints);
  }

  template<typename DATATYPE>
  int AMPI_Isend_finish(HandleBase* handle);
  template<typename DATATYPE>
  int AMPI_Isend(const typename DATATYPE::Type* buf, int count, DATATYPE* datatype, int dest, int tag, AMPI_Comm comm,
                 AMPI_Request* request) {
    int rStatus;
    if(!datatype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Isend(buf, count, datatype->getMpiType(), dest, tag, comm, &request->request);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Isend_AdjointHandle<DATATYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(datatype->getADTool().isHandleRequired()) {
        h = new AMPI_Isend_AdjointHandle<DATATYPE>();
      }
      datatype->getADTool().startAssembly(h);
      typename DATATYPE::ModifiedType* bufMod = nullptr;
      int bufElements = 0;

      // compute the total size of the buffer
      bufElements = count;

      if(datatype->isModifiedBufferRequired() ) {
        datatype->createModifiedTypeBuffer(bufMod, bufElements);
      } else {
        bufMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(buf));
      }

      if(datatype->isModifiedBufferRequired()) {
        datatype->copyIntoModifiedBuffer(buf, 0, bufMod, 0, count);
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        h->bufCount = datatype->computeActiveElements(count);
        h->bufTotalSize = datatype->computeActiveElements(bufElements);
        datatype->getADTool().createIndexTypeBuffer(h->bufIndices, h->bufTotalSize);




        datatype->getIndices(buf, 0, h->bufIndices, 0, count);


        // pack all the variables in the handle
        h->func = AMPI_Isend_b<DATATYPE>;
        h->count = count;
        h->datatype = datatype;
        h->dest = dest;
        h->tag = tag;
        h->comm = comm;
      }

      rStatus = MPI_Isend(bufMod, count, datatype->getModifiedMpiType(), dest, tag, comm, &request->request);

      AMPI_Isend_AsyncHandle<DATATYPE>* asyncHandle = new AMPI_Isend_AsyncHandle<DATATYPE>();
      asyncHandle->buf = buf;
      asyncHandle->bufMod = bufMod;
      asyncHandle->count = count;
      asyncHandle->datatype = datatype;
      asyncHandle->dest = dest;
      asyncHandle->tag = tag;
      asyncHandle->comm = comm;
      asyncHandle->h = h;
      request->handle = asyncHandle;
      request->func = (ContinueFunction)AMPI_Isend_finish<DATATYPE>;

      // create adjoint wait
      if(nullptr != h) {
        WaitHandle* waitH = new WaitHandle((ContinueFunction)AMPI_Isend_b_finish<DATATYPE>, h);
        datatype->getADTool().addToolAction(waitH);
      }
    }

    return rStatus;
  }

  template<typename DATATYPE>
  int AMPI_Isend_finish(HandleBase* handle) {
    int rStatus = 0;

    AMPI_Isend_AsyncHandle<DATATYPE>* asyncHandle = static_cast<AMPI_Isend_AsyncHandle<DATATYPE>*>(handle);
    const  typename DATATYPE::Type* buf = asyncHandle->buf;
    typename DATATYPE::ModifiedType* bufMod = asyncHandle->bufMod;
    int count = asyncHandle->count;
    DATATYPE* datatype = asyncHandle->datatype;
    int dest = asyncHandle->dest;
    int tag = asyncHandle->tag;
    AMPI_Comm comm = asyncHandle->comm;
    AMPI_Request* request = asyncHandle->request;
    AMPI_Isend_AdjointHandle<DATATYPE>* h = asyncHandle->h;
    MEDI_UNUSED(buf); // Unused generated to ignore warnings
    MEDI_UNUSED(bufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(count); // Unused generated to ignore warnings
    MEDI_UNUSED(datatype); // Unused generated to ignore warnings
    MEDI_UNUSED(dest); // Unused generated to ignore warnings
    MEDI_UNUSED(tag); // Unused generated to ignore warnings
    MEDI_UNUSED(comm); // Unused generated to ignore warnings
    MEDI_UNUSED(request); // Unused generated to ignore warnings
    MEDI_UNUSED(h); // Unused generated to ignore warnings

    delete asyncHandle;

    if(datatype->getADTool().isActiveType()) {

      datatype->getADTool().addToolAction(h);


      if(nullptr != h) {
        // handle the recv buffers
      }

      datatype->getADTool().stopAssembly(h);

      if(datatype->isModifiedBufferRequired() ) {
        datatype->deleteModifiedTypeBuffer(bufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename DATATYPE>
  struct AMPI_Issend_AdjointHandle : public HandleBase {
    int bufTotalSize;
    typename DATATYPE::IndexType* bufIndices;
    typename DATATYPE::PassiveType* bufPrimals;
    /* required for async */ typename DATATYPE::PassiveType* bufAdjoints;
    int bufCount;
    int count;
    DATATYPE* datatype;
    int dest;
    int tag;
    AMPI_Comm comm;
    AMPI_Request requestReverse;

    ~AMPI_Issend_AdjointHandle () {
      if(nullptr != bufIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(bufIndices);
        bufIndices = nullptr;
      }
      if(nullptr != bufPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(bufPrimals);
        bufPrimals = nullptr;
      }
    }
  };

  template<typename DATATYPE>
  struct AMPI_Issend_AsyncHandle : public HandleBase {
    const  typename DATATYPE::Type* buf;
    typename DATATYPE::ModifiedType* bufMod;
    int count;
    DATATYPE* datatype;
    int dest;
    int tag;
    AMPI_Comm comm;
    AMPI_Request* request;
    AMPI_Issend_AdjointHandle<DATATYPE>* h;
  };

  template<typename DATATYPE>
  void AMPI_Issend_b(HandleBase* handle) {
    AMPI_Issend_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Issend_AdjointHandle<DATATYPE>*>(handle);

    h->bufAdjoints = nullptr;
    h->datatype->getADTool().createAdjointTypeBuffer(h->bufAdjoints, h->bufTotalSize );

    AMPI_Issend_adj<DATATYPE>(h->bufAdjoints, h->bufCount, h->count, h->datatype, h->dest, h->tag, h->comm,
                              &h->requestReverse);

  }

  template<typename DATATYPE>
  void AMPI_Issend_b_finish(HandleBase* handle) {

    AMPI_Issend_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Issend_AdjointHandle<DATATYPE>*>(handle);
    MPI_Wait(&h->requestReverse.request, MPI_STATUS_IGNORE);

    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->datatype->getADTool().updateAdjoints(h->bufIndices, h->bufAdjoints, h->bufTotalSize);
    h->datatype->getADTool().deleteAdjointTypeBuffer(h->bufAdjoints);
  }

  template<typename DATATYPE>
  int AMPI_Issend_finish(HandleBase* handle);
  template<typename DATATYPE>
  int AMPI_Issend(const typename DATATYPE::Type* buf, int count, DATATYPE* datatype, int dest, int tag, AMPI_Comm comm,
                  AMPI_Request* request) {
    int rStatus;
    if(!datatype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Issend(buf, count, datatype->getMpiType(), dest, tag, comm, &request->request);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Issend_AdjointHandle<DATATYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(datatype->getADTool().isHandleRequired()) {
        h = new AMPI_Issend_AdjointHandle<DATATYPE>();
      }
      datatype->getADTool().startAssembly(h);
      typename DATATYPE::ModifiedType* bufMod = nullptr;
      int bufElements = 0;

      // compute the total size of the buffer
      bufElements = count;

      if(datatype->isModifiedBufferRequired() ) {
        datatype->createModifiedTypeBuffer(bufMod, bufElements);
      } else {
        bufMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(buf));
      }

      if(datatype->isModifiedBufferRequired()) {
        datatype->copyIntoModifiedBuffer(buf, 0, bufMod, 0, count);
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        h->bufCount = datatype->computeActiveElements(count);
        h->bufTotalSize = datatype->computeActiveElements(bufElements);
        datatype->getADTool().createIndexTypeBuffer(h->bufIndices, h->bufTotalSize);




        datatype->getIndices(buf, 0, h->bufIndices, 0, count);


        // pack all the variables in the handle
        h->func = AMPI_Issend_b<DATATYPE>;
        h->count = count;
        h->datatype = datatype;
        h->dest = dest;
        h->tag = tag;
        h->comm = comm;
      }

      rStatus = MPI_Issend(bufMod, count, datatype->getModifiedMpiType(), dest, tag, comm, &request->request);

      AMPI_Issend_AsyncHandle<DATATYPE>* asyncHandle = new AMPI_Issend_AsyncHandle<DATATYPE>();
      asyncHandle->buf = buf;
      asyncHandle->bufMod = bufMod;
      asyncHandle->count = count;
      asyncHandle->datatype = datatype;
      asyncHandle->dest = dest;
      asyncHandle->tag = tag;
      asyncHandle->comm = comm;
      asyncHandle->h = h;
      request->handle = asyncHandle;
      request->func = (ContinueFunction)AMPI_Issend_finish<DATATYPE>;

      // create adjoint wait
      if(nullptr != h) {
        WaitHandle* waitH = new WaitHandle((ContinueFunction)AMPI_Issend_b_finish<DATATYPE>, h);
        datatype->getADTool().addToolAction(waitH);
      }
    }

    return rStatus;
  }

  template<typename DATATYPE>
  int AMPI_Issend_finish(HandleBase* handle) {
    int rStatus = 0;

    AMPI_Issend_AsyncHandle<DATATYPE>* asyncHandle = static_cast<AMPI_Issend_AsyncHandle<DATATYPE>*>(handle);
    const  typename DATATYPE::Type* buf = asyncHandle->buf;
    typename DATATYPE::ModifiedType* bufMod = asyncHandle->bufMod;
    int count = asyncHandle->count;
    DATATYPE* datatype = asyncHandle->datatype;
    int dest = asyncHandle->dest;
    int tag = asyncHandle->tag;
    AMPI_Comm comm = asyncHandle->comm;
    AMPI_Request* request = asyncHandle->request;
    AMPI_Issend_AdjointHandle<DATATYPE>* h = asyncHandle->h;
    MEDI_UNUSED(buf); // Unused generated to ignore warnings
    MEDI_UNUSED(bufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(count); // Unused generated to ignore warnings
    MEDI_UNUSED(datatype); // Unused generated to ignore warnings
    MEDI_UNUSED(dest); // Unused generated to ignore warnings
    MEDI_UNUSED(tag); // Unused generated to ignore warnings
    MEDI_UNUSED(comm); // Unused generated to ignore warnings
    MEDI_UNUSED(request); // Unused generated to ignore warnings
    MEDI_UNUSED(h); // Unused generated to ignore warnings

    delete asyncHandle;

    if(datatype->getADTool().isActiveType()) {

      datatype->getADTool().addToolAction(h);


      if(nullptr != h) {
        // handle the recv buffers
      }

      datatype->getADTool().stopAssembly(h);

      if(datatype->isModifiedBufferRequired() ) {
        datatype->deleteModifiedTypeBuffer(bufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename DATATYPE>
  struct AMPI_Recv_AdjointHandle : public HandleBase {
    int bufTotalSize;
    typename DATATYPE::IndexType* bufIndices;
    typename DATATYPE::PassiveType* bufPrimals;
    typename DATATYPE::PassiveType* bufOldPrimals;
    /* required for async */ typename DATATYPE::PassiveType* bufAdjoints;
    int bufCount;
    int count;
    DATATYPE* datatype;
    int source;
    int tag;
    AMPI_Comm comm;

    ~AMPI_Recv_AdjointHandle () {
      if(nullptr != bufIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(bufIndices);
        bufIndices = nullptr;
      }
      if(nullptr != bufPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(bufPrimals);
        bufPrimals = nullptr;
      }
      if(nullptr != bufOldPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(bufOldPrimals);
        bufOldPrimals = nullptr;
      }
    }
  };


  template<typename DATATYPE>
  void AMPI_Recv_b(HandleBase* handle) {
    AMPI_Recv_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Recv_AdjointHandle<DATATYPE>*>(handle);

    MPI_Status status;
    h->bufAdjoints = nullptr;
    h->datatype->getADTool().createAdjointTypeBuffer(h->bufAdjoints, h->bufTotalSize );
    // Adjoint buffers are always linear in space so we can accesses them in one sweep
    h->datatype->getADTool().getAdjoints(h->bufIndices, h->bufAdjoints, h->bufTotalSize);

    if(h->datatype->getADTool().isOldPrimalsRequired()) {
      h->datatype->getADTool().setReverseValues(h->bufIndices, h->bufOldPrimals, h->bufTotalSize);
    }

    AMPI_Recv_adj<DATATYPE>(h->bufAdjoints, h->bufCount, h->count, h->datatype, h->source, h->tag, h->comm, &status);

    h->datatype->getADTool().deleteAdjointTypeBuffer(h->bufAdjoints);
  }

  template<typename DATATYPE>
  int AMPI_Recv(typename DATATYPE::Type* buf, int count, DATATYPE* datatype, int source, int tag, AMPI_Comm comm,
                AMPI_Status* status) {
    int rStatus;
    if(!datatype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Recv(buf, count, datatype->getMpiType(), source, tag, comm, status);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Recv_AdjointHandle<DATATYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(datatype->getADTool().isHandleRequired()) {
        h = new AMPI_Recv_AdjointHandle<DATATYPE>();
      }
      datatype->getADTool().startAssembly(h);
      typename DATATYPE::ModifiedType* bufMod = nullptr;
      int bufElements = 0;

      // compute the total size of the buffer
      bufElements = count;

      if(datatype->isModifiedBufferRequired() ) {
        datatype->createModifiedTypeBuffer(bufMod, bufElements);
      } else {
        bufMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(buf));
      }


      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        h->bufCount = datatype->computeActiveElements(count);
        h->bufTotalSize = datatype->computeActiveElements(bufElements);
        datatype->getADTool().createIndexTypeBuffer(h->bufIndices, h->bufTotalSize);


        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(datatype->getADTool().isOldPrimalsRequired()) {
          datatype->getADTool().createPassiveTypeBuffer(h->bufOldPrimals, h->bufTotalSize);
          datatype->getValues(buf, 0, h->bufOldPrimals, 0, count);
        }



        if(!datatype->isModifiedBufferRequired()) {
          datatype->clearIndices(buf, 0, count);
        }

        // pack all the variables in the handle
        h->func = AMPI_Recv_b<DATATYPE>;
        h->count = count;
        h->datatype = datatype;
        h->source = source;
        h->tag = tag;
        h->comm = comm;
      }

      rStatus = MPI_Recv(bufMod, count, datatype->getModifiedMpiType(), source, tag, comm, status);
      datatype->getADTool().addToolAction(h);

      if(datatype->isModifiedBufferRequired()) {
        datatype->copyFromModifiedBuffer(buf, 0, bufMod, 0, count);
      }

      if(nullptr != h) {
        // handle the recv buffers
        datatype->registerValue(buf, 0, h->bufIndices, h->bufOldPrimals, 0, count);
      }

      datatype->getADTool().stopAssembly(h);

      if(datatype->isModifiedBufferRequired() ) {
        datatype->deleteModifiedTypeBuffer(bufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename DATATYPE>
  struct AMPI_Rsend_AdjointHandle : public HandleBase {
    int bufTotalSize;
    typename DATATYPE::IndexType* bufIndices;
    typename DATATYPE::PassiveType* bufPrimals;
    /* required for async */ typename DATATYPE::PassiveType* bufAdjoints;
    int bufCount;
    int count;
    DATATYPE* datatype;
    int dest;
    int tag;
    AMPI_Comm comm;

    ~AMPI_Rsend_AdjointHandle () {
      if(nullptr != bufIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(bufIndices);
        bufIndices = nullptr;
      }
      if(nullptr != bufPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(bufPrimals);
        bufPrimals = nullptr;
      }
    }
  };


  template<typename DATATYPE>
  void AMPI_Rsend_b(HandleBase* handle) {
    AMPI_Rsend_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Rsend_AdjointHandle<DATATYPE>*>(handle);

    h->bufAdjoints = nullptr;
    h->datatype->getADTool().createAdjointTypeBuffer(h->bufAdjoints, h->bufTotalSize );

    AMPI_Rsend_adj<DATATYPE>(h->bufAdjoints, h->bufCount, h->count, h->datatype, h->dest, h->tag, h->comm);

    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->datatype->getADTool().updateAdjoints(h->bufIndices, h->bufAdjoints, h->bufTotalSize);
    h->datatype->getADTool().deleteAdjointTypeBuffer(h->bufAdjoints);
  }

  template<typename DATATYPE>
  int AMPI_Rsend(const typename DATATYPE::Type* buf, int count, DATATYPE* datatype, int dest, int tag, AMPI_Comm comm) {
    int rStatus;
    if(!datatype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Rsend(buf, count, datatype->getMpiType(), dest, tag, comm);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Rsend_AdjointHandle<DATATYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(datatype->getADTool().isHandleRequired()) {
        h = new AMPI_Rsend_AdjointHandle<DATATYPE>();
      }
      datatype->getADTool().startAssembly(h);
      typename DATATYPE::ModifiedType* bufMod = nullptr;
      int bufElements = 0;

      // compute the total size of the buffer
      bufElements = count;

      if(datatype->isModifiedBufferRequired() ) {
        datatype->createModifiedTypeBuffer(bufMod, bufElements);
      } else {
        bufMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(buf));
      }

      if(datatype->isModifiedBufferRequired()) {
        datatype->copyIntoModifiedBuffer(buf, 0, bufMod, 0, count);
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        h->bufCount = datatype->computeActiveElements(count);
        h->bufTotalSize = datatype->computeActiveElements(bufElements);
        datatype->getADTool().createIndexTypeBuffer(h->bufIndices, h->bufTotalSize);




        datatype->getIndices(buf, 0, h->bufIndices, 0, count);


        // pack all the variables in the handle
        h->func = AMPI_Rsend_b<DATATYPE>;
        h->count = count;
        h->datatype = datatype;
        h->dest = dest;
        h->tag = tag;
        h->comm = comm;
      }

      rStatus = MPI_Rsend(bufMod, count, datatype->getModifiedMpiType(), dest, tag, comm);
      datatype->getADTool().addToolAction(h);


      if(nullptr != h) {
        // handle the recv buffers
      }

      datatype->getADTool().stopAssembly(h);

      if(datatype->isModifiedBufferRequired() ) {
        datatype->deleteModifiedTypeBuffer(bufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename DATATYPE>
  struct AMPI_Send_AdjointHandle : public HandleBase {
    int bufTotalSize;
    typename DATATYPE::IndexType* bufIndices;
    typename DATATYPE::PassiveType* bufPrimals;
    /* required for async */ typename DATATYPE::PassiveType* bufAdjoints;
    int bufCount;
    int count;
    DATATYPE* datatype;
    int dest;
    int tag;
    AMPI_Comm comm;

    ~AMPI_Send_AdjointHandle () {
      if(nullptr != bufIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(bufIndices);
        bufIndices = nullptr;
      }
      if(nullptr != bufPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(bufPrimals);
        bufPrimals = nullptr;
      }
    }
  };


  template<typename DATATYPE>
  void AMPI_Send_b(HandleBase* handle) {
    AMPI_Send_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Send_AdjointHandle<DATATYPE>*>(handle);

    h->bufAdjoints = nullptr;
    h->datatype->getADTool().createAdjointTypeBuffer(h->bufAdjoints, h->bufTotalSize );

    AMPI_Send_adj<DATATYPE>(h->bufAdjoints, h->bufCount, h->count, h->datatype, h->dest, h->tag, h->comm);

    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->datatype->getADTool().updateAdjoints(h->bufIndices, h->bufAdjoints, h->bufTotalSize);
    h->datatype->getADTool().deleteAdjointTypeBuffer(h->bufAdjoints);
  }

  template<typename DATATYPE>
  int AMPI_Send(const typename DATATYPE::Type* buf, int count, DATATYPE* datatype, int dest, int tag, AMPI_Comm comm) {
    int rStatus;
    if(!datatype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Send(buf, count, datatype->getMpiType(), dest, tag, comm);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Send_AdjointHandle<DATATYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(datatype->getADTool().isHandleRequired()) {
        h = new AMPI_Send_AdjointHandle<DATATYPE>();
      }
      datatype->getADTool().startAssembly(h);
      typename DATATYPE::ModifiedType* bufMod = nullptr;
      int bufElements = 0;

      // compute the total size of the buffer
      bufElements = count;

      if(datatype->isModifiedBufferRequired() ) {
        datatype->createModifiedTypeBuffer(bufMod, bufElements);
      } else {
        bufMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(buf));
      }

      if(datatype->isModifiedBufferRequired()) {
        datatype->copyIntoModifiedBuffer(buf, 0, bufMod, 0, count);
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        h->bufCount = datatype->computeActiveElements(count);
        h->bufTotalSize = datatype->computeActiveElements(bufElements);
        datatype->getADTool().createIndexTypeBuffer(h->bufIndices, h->bufTotalSize);




        datatype->getIndices(buf, 0, h->bufIndices, 0, count);


        // pack all the variables in the handle
        h->func = AMPI_Send_b<DATATYPE>;
        h->count = count;
        h->datatype = datatype;
        h->dest = dest;
        h->tag = tag;
        h->comm = comm;
      }

      rStatus = MPI_Send(bufMod, count, datatype->getModifiedMpiType(), dest, tag, comm);
      datatype->getADTool().addToolAction(h);


      if(nullptr != h) {
        // handle the recv buffers
      }

      datatype->getADTool().stopAssembly(h);

      if(datatype->isModifiedBufferRequired() ) {
        datatype->deleteModifiedTypeBuffer(bufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Sendrecv_AdjointHandle : public HandleBase {
    int sendbufTotalSize;
    typename SENDTYPE::IndexType* sendbufIndices;
    typename SENDTYPE::PassiveType* sendbufPrimals;
    /* required for async */ typename SENDTYPE::PassiveType* sendbufAdjoints;
    int sendbufCount;
    int sendcount;
    SENDTYPE* sendtype;
    int dest;
    int sendtag;
    int recvbufTotalSize;
    typename RECVTYPE::IndexType* recvbufIndices;
    typename RECVTYPE::PassiveType* recvbufPrimals;
    typename RECVTYPE::PassiveType* recvbufOldPrimals;
    /* required for async */ typename RECVTYPE::PassiveType* recvbufAdjoints;
    int recvbufCount;
    int recvcount;
    RECVTYPE* recvtype;
    int source;
    int recvtag;
    AMPI_Comm comm;

    ~AMPI_Sendrecv_AdjointHandle () {
      if(nullptr != sendbufIndices) {
        sendtype->getADTool().deleteIndexTypeBuffer(sendbufIndices);
        sendbufIndices = nullptr;
      }
      if(nullptr != sendbufPrimals) {
        sendtype->getADTool().deletePassiveTypeBuffer(sendbufPrimals);
        sendbufPrimals = nullptr;
      }
      if(nullptr != recvbufIndices) {
        recvtype->getADTool().deleteIndexTypeBuffer(recvbufIndices);
        recvbufIndices = nullptr;
      }
      if(nullptr != recvbufPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufPrimals);
        recvbufPrimals = nullptr;
      }
      if(nullptr != recvbufOldPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufOldPrimals);
        recvbufOldPrimals = nullptr;
      }
    }
  };


  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Sendrecv_b(HandleBase* handle) {
    AMPI_Sendrecv_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Sendrecv_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);

    MPI_Status status;
    h->recvbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->recvbufAdjoints, h->recvbufTotalSize );
    // Adjoint buffers are always linear in space so we can accesses them in one sweep
    h->recvtype->getADTool().getAdjoints(h->recvbufIndices, h->recvbufAdjoints, h->recvbufTotalSize);

    if(h->recvtype->getADTool().isOldPrimalsRequired()) {
      h->recvtype->getADTool().setReverseValues(h->recvbufIndices, h->recvbufOldPrimals, h->recvbufTotalSize);
    }
    h->sendbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->sendbufAdjoints, h->sendbufTotalSize );

    AMPI_Sendrecv_adj<SENDTYPE, RECVTYPE>(h->sendbufAdjoints, h->sendbufCount, h->sendcount, h->sendtype, h->dest,
                                          h->sendtag, h->recvbufAdjoints, h->recvbufCount, h->recvcount, h->recvtype, h->source, h->recvtag, h->comm, &status);

    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->recvtype->getADTool().updateAdjoints(h->sendbufIndices, h->sendbufAdjoints, h->sendbufTotalSize);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->sendbufAdjoints);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->recvbufAdjoints);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Sendrecv(const typename SENDTYPE::Type* sendbuf, int sendcount, SENDTYPE* sendtype, int dest, int sendtag,
                    typename RECVTYPE::Type* recvbuf, int recvcount, RECVTYPE* recvtype, int source, int recvtag, AMPI_Comm comm,
                    AMPI_Status* status) {
    int rStatus;
    if(!recvtype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Sendrecv(sendbuf, sendcount, sendtype->getMpiType(), dest, sendtag, recvbuf, recvcount,
                             recvtype->getMpiType(), source, recvtag, comm, status);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Sendrecv_AdjointHandle<SENDTYPE, RECVTYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(recvtype->getADTool().isHandleRequired()) {
        h = new AMPI_Sendrecv_AdjointHandle<SENDTYPE, RECVTYPE>();
      }
      recvtype->getADTool().startAssembly(h);
      typename SENDTYPE::ModifiedType* sendbufMod = nullptr;
      int sendbufElements = 0;

      // compute the total size of the buffer
      sendbufElements = sendcount;

      if(sendtype->isModifiedBufferRequired() ) {
        sendtype->createModifiedTypeBuffer(sendbufMod, sendbufElements);
      } else {
        sendbufMod = reinterpret_cast<typename SENDTYPE::ModifiedType*>(const_cast<typename SENDTYPE::Type*>(sendbuf));
      }
      typename RECVTYPE::ModifiedType* recvbufMod = nullptr;
      int recvbufElements = 0;

      // compute the total size of the buffer
      recvbufElements = recvcount;

      if(recvtype->isModifiedBufferRequired() ) {
        recvtype->createModifiedTypeBuffer(recvbufMod, recvbufElements);
      } else {
        recvbufMod = reinterpret_cast<typename RECVTYPE::ModifiedType*>(const_cast<typename RECVTYPE::Type*>(recvbuf));
      }

      if(sendtype->isModifiedBufferRequired()) {
        sendtype->copyIntoModifiedBuffer(sendbuf, 0, sendbufMod, 0, sendcount);
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        h->sendbufCount = sendtype->computeActiveElements(sendcount);
        h->sendbufTotalSize = sendtype->computeActiveElements(sendbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->sendbufIndices, h->sendbufTotalSize);
        h->recvbufCount = recvtype->computeActiveElements(recvcount);
        h->recvbufTotalSize = recvtype->computeActiveElements(recvbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->recvbufIndices, h->recvbufTotalSize);


        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(recvtype->getADTool().isOldPrimalsRequired()) {
          recvtype->getADTool().createPassiveTypeBuffer(h->recvbufOldPrimals, h->recvbufTotalSize);
          recvtype->getValues(recvbuf, 0, h->recvbufOldPrimals, 0, recvcount);
        }


        sendtype->getIndices(sendbuf, 0, h->sendbufIndices, 0, sendcount);

        if(!recvtype->isModifiedBufferRequired()) {
          recvtype->clearIndices(recvbuf, 0, recvcount);
        }

        // pack all the variables in the handle
        h->func = AMPI_Sendrecv_b<SENDTYPE, RECVTYPE>;
        h->sendcount = sendcount;
        h->sendtype = sendtype;
        h->dest = dest;
        h->sendtag = sendtag;
        h->recvcount = recvcount;
        h->recvtype = recvtype;
        h->source = source;
        h->recvtag = recvtag;
        h->comm = comm;
      }

      rStatus = MPI_Sendrecv(sendbufMod, sendcount, sendtype->getModifiedMpiType(), dest, sendtag, recvbufMod, recvcount,
                             recvtype->getModifiedMpiType(), source, recvtag, comm, status);
      recvtype->getADTool().addToolAction(h);

      if(recvtype->isModifiedBufferRequired()) {
        recvtype->copyFromModifiedBuffer(recvbuf, 0, recvbufMod, 0, recvcount);
      }

      if(nullptr != h) {
        // handle the recv buffers
        recvtype->registerValue(recvbuf, 0, h->recvbufIndices, h->recvbufOldPrimals, 0, recvcount);
      }

      recvtype->getADTool().stopAssembly(h);

      if(sendtype->isModifiedBufferRequired() ) {
        sendtype->deleteModifiedTypeBuffer(sendbufMod);
      }
      if(recvtype->isModifiedBufferRequired() ) {
        recvtype->deleteModifiedTypeBuffer(recvbufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename DATATYPE>
  struct AMPI_Ssend_AdjointHandle : public HandleBase {
    int bufTotalSize;
    typename DATATYPE::IndexType* bufIndices;
    typename DATATYPE::PassiveType* bufPrimals;
    /* required for async */ typename DATATYPE::PassiveType* bufAdjoints;
    int bufCount;
    int count;
    DATATYPE* datatype;
    int dest;
    int tag;
    AMPI_Comm comm;

    ~AMPI_Ssend_AdjointHandle () {
      if(nullptr != bufIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(bufIndices);
        bufIndices = nullptr;
      }
      if(nullptr != bufPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(bufPrimals);
        bufPrimals = nullptr;
      }
    }
  };


  template<typename DATATYPE>
  void AMPI_Ssend_b(HandleBase* handle) {
    AMPI_Ssend_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Ssend_AdjointHandle<DATATYPE>*>(handle);

    h->bufAdjoints = nullptr;
    h->datatype->getADTool().createAdjointTypeBuffer(h->bufAdjoints, h->bufTotalSize );

    AMPI_Ssend_adj<DATATYPE>(h->bufAdjoints, h->bufCount, h->count, h->datatype, h->dest, h->tag, h->comm);

    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->datatype->getADTool().updateAdjoints(h->bufIndices, h->bufAdjoints, h->bufTotalSize);
    h->datatype->getADTool().deleteAdjointTypeBuffer(h->bufAdjoints);
  }

  template<typename DATATYPE>
  int AMPI_Ssend(const typename DATATYPE::Type* buf, int count, DATATYPE* datatype, int dest, int tag, AMPI_Comm comm) {
    int rStatus;
    if(!datatype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Ssend(buf, count, datatype->getMpiType(), dest, tag, comm);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Ssend_AdjointHandle<DATATYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(datatype->getADTool().isHandleRequired()) {
        h = new AMPI_Ssend_AdjointHandle<DATATYPE>();
      }
      datatype->getADTool().startAssembly(h);
      typename DATATYPE::ModifiedType* bufMod = nullptr;
      int bufElements = 0;

      // compute the total size of the buffer
      bufElements = count;

      if(datatype->isModifiedBufferRequired() ) {
        datatype->createModifiedTypeBuffer(bufMod, bufElements);
      } else {
        bufMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(buf));
      }

      if(datatype->isModifiedBufferRequired()) {
        datatype->copyIntoModifiedBuffer(buf, 0, bufMod, 0, count);
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        h->bufCount = datatype->computeActiveElements(count);
        h->bufTotalSize = datatype->computeActiveElements(bufElements);
        datatype->getADTool().createIndexTypeBuffer(h->bufIndices, h->bufTotalSize);




        datatype->getIndices(buf, 0, h->bufIndices, 0, count);


        // pack all the variables in the handle
        h->func = AMPI_Ssend_b<DATATYPE>;
        h->count = count;
        h->datatype = datatype;
        h->dest = dest;
        h->tag = tag;
        h->comm = comm;
      }

      rStatus = MPI_Ssend(bufMod, count, datatype->getModifiedMpiType(), dest, tag, comm);
      datatype->getADTool().addToolAction(h);


      if(nullptr != h) {
        // handle the recv buffers
      }

      datatype->getADTool().stopAssembly(h);

      if(datatype->isModifiedBufferRequired() ) {
        datatype->deleteModifiedTypeBuffer(bufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Allgather_AdjointHandle : public HandleBase {
    int sendbufTotalSize;
    typename SENDTYPE::IndexType* sendbufIndices;
    typename SENDTYPE::PassiveType* sendbufPrimals;
    /* required for async */ typename SENDTYPE::PassiveType* sendbufAdjoints;
    int sendbufCount;
    int sendcount;
    SENDTYPE* sendtype;
    int recvbufTotalSize;
    typename RECVTYPE::IndexType* recvbufIndices;
    typename RECVTYPE::PassiveType* recvbufPrimals;
    typename RECVTYPE::PassiveType* recvbufOldPrimals;
    /* required for async */ typename RECVTYPE::PassiveType* recvbufAdjoints;
    int recvbufCount;
    int recvcount;
    RECVTYPE* recvtype;
    AMPI_Comm comm;

    ~AMPI_Allgather_AdjointHandle () {
      if(nullptr != sendbufIndices) {
        sendtype->getADTool().deleteIndexTypeBuffer(sendbufIndices);
        sendbufIndices = nullptr;
      }
      if(nullptr != sendbufPrimals) {
        sendtype->getADTool().deletePassiveTypeBuffer(sendbufPrimals);
        sendbufPrimals = nullptr;
      }
      if(nullptr != recvbufIndices) {
        recvtype->getADTool().deleteIndexTypeBuffer(recvbufIndices);
        recvbufIndices = nullptr;
      }
      if(nullptr != recvbufPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufPrimals);
        recvbufPrimals = nullptr;
      }
      if(nullptr != recvbufOldPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufOldPrimals);
        recvbufOldPrimals = nullptr;
      }
    }
  };


  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Allgather_b(HandleBase* handle) {
    AMPI_Allgather_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Allgather_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);

    h->recvbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->recvbufAdjoints, h->recvbufTotalSize );
    // Adjoint buffers are always linear in space so we can accesses them in one sweep
    h->recvtype->getADTool().getAdjoints(h->recvbufIndices, h->recvbufAdjoints, h->recvbufTotalSize);

    if(h->recvtype->getADTool().isOldPrimalsRequired()) {
      h->recvtype->getADTool().setReverseValues(h->recvbufIndices, h->recvbufOldPrimals, h->recvbufTotalSize);
    }
    h->sendbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->sendbufAdjoints, h->sendbufTotalSize * getCommSize(h->comm));

    AMPI_Allgather_adj<SENDTYPE, RECVTYPE>(h->sendbufAdjoints, h->sendbufCount, h->sendcount, h->sendtype,
                                           h->recvbufAdjoints, h->recvbufCount, h->recvcount, h->recvtype, h->comm);

    h->recvtype->getADTool().combineAdjoints(h->sendbufAdjoints, h->sendbufTotalSize, getCommSize(h->comm));
    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->recvtype->getADTool().updateAdjoints(h->sendbufIndices, h->sendbufAdjoints, h->sendbufTotalSize);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->sendbufAdjoints);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->recvbufAdjoints);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Allgather(const typename SENDTYPE::Type* sendbuf, int sendcount, SENDTYPE* sendtype,
                     typename RECVTYPE::Type* recvbuf, int recvcount, RECVTYPE* recvtype, AMPI_Comm comm) {
    int rStatus;
    if(!recvtype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Allgather(sendbuf, sendcount, sendtype->getMpiType(), recvbuf, recvcount, recvtype->getMpiType(), comm);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Allgather_AdjointHandle<SENDTYPE, RECVTYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(recvtype->getADTool().isHandleRequired()) {
        h = new AMPI_Allgather_AdjointHandle<SENDTYPE, RECVTYPE>();
      }
      recvtype->getADTool().startAssembly(h);
      typename SENDTYPE::ModifiedType* sendbufMod = nullptr;
      int sendbufElements = 0;

      // compute the total size of the buffer
      if(AMPI_IN_PLACE != sendbuf) {
        sendbufElements = sendcount;
      } else {
        sendbufElements = recvcount;
      }

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->createModifiedTypeBuffer(sendbufMod, sendbufElements);
      } else {
        sendbufMod = reinterpret_cast<typename SENDTYPE::ModifiedType*>(const_cast<typename SENDTYPE::Type*>(sendbuf));
      }
      typename RECVTYPE::ModifiedType* recvbufMod = nullptr;
      int recvbufElements = 0;

      // compute the total size of the buffer
      recvbufElements = recvcount * getCommSize(comm);

      if(recvtype->isModifiedBufferRequired() ) {
        recvtype->createModifiedTypeBuffer(recvbufMod, recvbufElements);
      } else {
        recvbufMod = reinterpret_cast<typename RECVTYPE::ModifiedType*>(const_cast<typename RECVTYPE::Type*>(recvbuf));
      }

      if(sendtype->isModifiedBufferRequired()) {
        if(AMPI_IN_PLACE != sendbuf) {
          sendtype->copyIntoModifiedBuffer(sendbuf, 0, sendbufMod, 0, sendcount);
        } else {
          recvtype->copyIntoModifiedBuffer(recvbuf, recvcount * getCommRank(comm), recvbufMod, recvcount * getCommRank(comm),
                                           recvcount);
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(AMPI_IN_PLACE != sendbuf) {
          h->sendbufCount = sendtype->computeActiveElements(sendcount);
        } else {
          h->sendbufCount = recvtype->computeActiveElements(recvcount);
        }
        h->sendbufTotalSize = sendtype->computeActiveElements(sendbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->sendbufIndices, h->sendbufTotalSize);
        h->recvbufCount = recvtype->computeActiveElements(recvcount);
        h->recvbufTotalSize = recvtype->computeActiveElements(recvbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->recvbufIndices, h->recvbufTotalSize);


        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(recvtype->getADTool().isOldPrimalsRequired()) {
          recvtype->getADTool().createPassiveTypeBuffer(h->recvbufOldPrimals, h->recvbufTotalSize);
          recvtype->getValues(recvbuf, 0, h->recvbufOldPrimals, 0, recvcount * getCommSize(comm));
        }


        if(AMPI_IN_PLACE != sendbuf) {
          sendtype->getIndices(sendbuf, 0, h->sendbufIndices, 0, sendcount);
        } else {
          recvtype->getIndices(recvbuf, recvcount * getCommRank(comm), h->sendbufIndices, 0, recvcount);
        }

        if(!recvtype->isModifiedBufferRequired()) {
          recvtype->clearIndices(recvbuf, 0, recvcount * getCommSize(comm));
        }

        // pack all the variables in the handle
        h->func = AMPI_Allgather_b<SENDTYPE, RECVTYPE>;
        h->sendcount = sendcount;
        h->sendtype = sendtype;
        h->recvcount = recvcount;
        h->recvtype = recvtype;
        h->comm = comm;
      }

      rStatus = MPI_Allgather(sendbufMod, sendcount, sendtype->getModifiedMpiType(), recvbufMod, recvcount,
                              recvtype->getModifiedMpiType(), comm);
      recvtype->getADTool().addToolAction(h);

      if(recvtype->isModifiedBufferRequired()) {
        recvtype->copyFromModifiedBuffer(recvbuf, 0, recvbufMod, 0, recvcount * getCommSize(comm));
      }

      if(nullptr != h) {
        // handle the recv buffers
        recvtype->registerValue(recvbuf, 0, h->recvbufIndices, h->recvbufOldPrimals, 0, recvcount * getCommSize(comm));
      }

      recvtype->getADTool().stopAssembly(h);

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->deleteModifiedTypeBuffer(sendbufMod);
      }
      if(recvtype->isModifiedBufferRequired() ) {
        recvtype->deleteModifiedTypeBuffer(recvbufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Allgatherv_AdjointHandle : public HandleBase {
    int sendbufTotalSize;
    typename SENDTYPE::IndexType* sendbufIndices;
    typename SENDTYPE::PassiveType* sendbufPrimals;
    /* required for async */ typename SENDTYPE::PassiveType* sendbufAdjoints;
    int sendbufCount;
    int sendcount;
    SENDTYPE* sendtype;
    int recvbufTotalSize;
    typename RECVTYPE::IndexType* recvbufIndices;
    typename RECVTYPE::PassiveType* recvbufPrimals;
    typename RECVTYPE::PassiveType* recvbufOldPrimals;
    /* required for async */ typename RECVTYPE::PassiveType* recvbufAdjoints;
    int* recvbufCount;
    int* recvbufDispls;
    const  int* recvcounts;
    const  int* displs;
    RECVTYPE* recvtype;
    AMPI_Comm comm;

    ~AMPI_Allgatherv_AdjointHandle () {
      if(nullptr != sendbufIndices) {
        sendtype->getADTool().deleteIndexTypeBuffer(sendbufIndices);
        sendbufIndices = nullptr;
      }
      if(nullptr != sendbufPrimals) {
        sendtype->getADTool().deletePassiveTypeBuffer(sendbufPrimals);
        sendbufPrimals = nullptr;
      }
      if(nullptr != recvbufIndices) {
        recvtype->getADTool().deleteIndexTypeBuffer(recvbufIndices);
        recvbufIndices = nullptr;
      }
      if(nullptr != recvbufPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufPrimals);
        recvbufPrimals = nullptr;
      }
      if(nullptr != recvbufOldPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufOldPrimals);
        recvbufOldPrimals = nullptr;
      }
      if(nullptr != recvbufCount) {
        delete [] recvbufCount;
        recvbufCount = nullptr;
      }
      if(nullptr != recvbufDispls) {
        delete [] recvbufDispls;
        recvbufDispls = nullptr;
      }
    }
  };


  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Allgatherv_b(HandleBase* handle) {
    AMPI_Allgatherv_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Allgatherv_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);

    h->recvbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->recvbufAdjoints, h->recvbufTotalSize );
    // Adjoint buffers are always linear in space so we can accesses them in one sweep
    h->recvtype->getADTool().getAdjoints(h->recvbufIndices, h->recvbufAdjoints, h->recvbufTotalSize);

    if(h->recvtype->getADTool().isOldPrimalsRequired()) {
      h->recvtype->getADTool().setReverseValues(h->recvbufIndices, h->recvbufOldPrimals, h->recvbufTotalSize);
    }
    h->sendbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->sendbufAdjoints, h->sendbufTotalSize * getCommSize(h->comm));

    AMPI_Allgatherv_adj<SENDTYPE, RECVTYPE>(h->sendbufAdjoints, h->sendbufCount, h->sendcount, h->sendtype,
                                            h->recvbufAdjoints, h->recvbufCount, h->recvbufDispls, h->recvcounts, h->displs, h->recvtype, h->comm);

    h->recvtype->getADTool().combineAdjoints(h->sendbufAdjoints, h->sendbufTotalSize, getCommSize(h->comm));
    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->recvtype->getADTool().updateAdjoints(h->sendbufIndices, h->sendbufAdjoints, h->sendbufTotalSize);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->sendbufAdjoints);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->recvbufAdjoints);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Allgatherv(const typename SENDTYPE::Type* sendbuf, int sendcount, SENDTYPE* sendtype,
                      typename RECVTYPE::Type* recvbuf, const int* recvcounts, const int* displs, RECVTYPE* recvtype, AMPI_Comm comm) {
    int rStatus;
    if(!recvtype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Allgatherv(sendbuf, sendcount, sendtype->getMpiType(), recvbuf, recvcounts, displs,
                               recvtype->getMpiType(), comm);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Allgatherv_AdjointHandle<SENDTYPE, RECVTYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(recvtype->getADTool().isHandleRequired()) {
        h = new AMPI_Allgatherv_AdjointHandle<SENDTYPE, RECVTYPE>();
      }
      recvtype->getADTool().startAssembly(h);
      const int* displsMod = displs;
      int displsTotalSize = 0;
      if(nullptr != displs) {
        displsTotalSize = computeDisplacementsTotalSize(recvcounts, getCommSize(comm));
        if(recvtype->isModifiedBufferRequired()) {
          displsMod = createLinearDisplacements(recvcounts, getCommSize(comm));
        }
      }
      typename SENDTYPE::ModifiedType* sendbufMod = nullptr;
      int sendbufElements = 0;

      // compute the total size of the buffer
      if(AMPI_IN_PLACE != sendbuf) {
        sendbufElements = sendcount;
      } else {
        sendbufElements = recvcounts[getCommRank(comm)];
      }

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->createModifiedTypeBuffer(sendbufMod, sendbufElements);
      } else {
        sendbufMod = reinterpret_cast<typename SENDTYPE::ModifiedType*>(const_cast<typename SENDTYPE::Type*>(sendbuf));
      }
      typename RECVTYPE::ModifiedType* recvbufMod = nullptr;
      int recvbufElements = 0;

      // compute the total size of the buffer
      recvbufElements = displsTotalSize;

      if(recvtype->isModifiedBufferRequired() ) {
        recvtype->createModifiedTypeBuffer(recvbufMod, recvbufElements);
      } else {
        recvbufMod = reinterpret_cast<typename RECVTYPE::ModifiedType*>(const_cast<typename RECVTYPE::Type*>(recvbuf));
      }

      if(sendtype->isModifiedBufferRequired()) {
        if(AMPI_IN_PLACE != sendbuf) {
          sendtype->copyIntoModifiedBuffer(sendbuf, 0, sendbufMod, 0, sendcount);
        } else {
          {
            const int rank = getCommRank(comm);
            recvtype->copyIntoModifiedBuffer(recvbuf, displs[rank], recvbufMod, displsMod[rank], recvcounts[rank]);
          }
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(AMPI_IN_PLACE != sendbuf) {
          h->sendbufCount = sendtype->computeActiveElements(sendcount);
        } else {
          h->sendbufCount = recvtype->computeActiveElements(recvcounts[getCommRank(comm)]);
        }
        h->sendbufTotalSize = sendtype->computeActiveElements(sendbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->sendbufIndices, h->sendbufTotalSize);
        createLinearIndexDisplacements(h->recvbufCount, h->recvbufDispls, recvcounts, getCommSize(comm), recvtype);
        h->recvbufTotalSize = recvtype->computeActiveElements(recvbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->recvbufIndices, h->recvbufTotalSize);


        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(recvtype->getADTool().isOldPrimalsRequired()) {
          recvtype->getADTool().createPassiveTypeBuffer(h->recvbufOldPrimals, h->recvbufTotalSize);
          for(int i = 0; i < getCommSize(comm); ++i) {
            recvtype->getValues(recvbuf, displs[i], h->recvbufOldPrimals, displsMod[i], recvcounts[i]);
          }
        }


        if(AMPI_IN_PLACE != sendbuf) {
          sendtype->getIndices(sendbuf, 0, h->sendbufIndices, 0, sendcount);
        } else {
          {
            const int rank = getCommRank(comm);
            recvtype->getIndices(recvbuf, displs[rank], h->sendbufIndices, 0, recvcounts[rank]);
          }
        }

        if(!recvtype->isModifiedBufferRequired()) {
          for(int i = 0; i < getCommSize(comm); ++i) {
            recvtype->clearIndices(recvbuf, displs[i], recvcounts[i]);
          }
        }

        // pack all the variables in the handle
        h->func = AMPI_Allgatherv_b<SENDTYPE, RECVTYPE>;
        h->sendcount = sendcount;
        h->sendtype = sendtype;
        h->recvcounts = recvcounts;
        h->displs = displs;
        h->recvtype = recvtype;
        h->comm = comm;
      }

      rStatus = MPI_Allgatherv(sendbufMod, sendcount, sendtype->getModifiedMpiType(), recvbufMod, recvcounts, displsMod,
                               recvtype->getModifiedMpiType(), comm);
      recvtype->getADTool().addToolAction(h);

      if(recvtype->isModifiedBufferRequired()) {
        for(int i = 0; i < getCommSize(comm); ++i) {
          recvtype->copyFromModifiedBuffer(recvbuf, displs[i], recvbufMod, displsMod[i], recvcounts[i]);
        }
      }

      if(nullptr != h) {
        // handle the recv buffers
        for(int i = 0; i < getCommSize(comm); ++i) {
          recvtype->registerValue(recvbuf, displs[i], h->recvbufIndices, h->recvbufOldPrimals, displsMod[i], recvcounts[i]);
        }
      }

      recvtype->getADTool().stopAssembly(h);
      if(recvtype->isModifiedBufferRequired()) {
        delete [] displsMod;
      }

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->deleteModifiedTypeBuffer(sendbufMod);
      }
      if(recvtype->isModifiedBufferRequired() ) {
        recvtype->deleteModifiedTypeBuffer(recvbufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename DATATYPE>
  struct AMPI_Allreduce_global_AdjointHandle : public HandleBase {
    int sendbufTotalSize;
    typename DATATYPE::IndexType* sendbufIndices;
    typename DATATYPE::PassiveType* sendbufPrimals;
    /* required for async */ typename DATATYPE::PassiveType* sendbufAdjoints;
    int sendbufCount;
    int recvbufTotalSize;
    typename DATATYPE::IndexType* recvbufIndices;
    typename DATATYPE::PassiveType* recvbufPrimals;
    typename DATATYPE::PassiveType* recvbufOldPrimals;
    /* required for async */ typename DATATYPE::PassiveType* recvbufAdjoints;
    int recvbufCount;
    int count;
    DATATYPE* datatype;
    AMPI_Op op;
    AMPI_Comm comm;

    ~AMPI_Allreduce_global_AdjointHandle () {
      if(nullptr != sendbufIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(sendbufIndices);
        sendbufIndices = nullptr;
      }
      if(nullptr != sendbufPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(sendbufPrimals);
        sendbufPrimals = nullptr;
      }
      if(nullptr != recvbufIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(recvbufIndices);
        recvbufIndices = nullptr;
      }
      if(nullptr != recvbufPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(recvbufPrimals);
        recvbufPrimals = nullptr;
      }
      if(nullptr != recvbufOldPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(recvbufOldPrimals);
        recvbufOldPrimals = nullptr;
      }
    }
  };


  template<typename DATATYPE>
  void AMPI_Allreduce_global_b(HandleBase* handle) {
    AMPI_Allreduce_global_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Allreduce_global_AdjointHandle<DATATYPE>*>(handle);

    h->recvbufAdjoints = nullptr;
    h->datatype->getADTool().createAdjointTypeBuffer(h->recvbufAdjoints, h->recvbufTotalSize );
    // Adjoint buffers are always linear in space so we can accesses them in one sweep
    h->datatype->getADTool().getAdjoints(h->recvbufIndices, h->recvbufAdjoints, h->recvbufTotalSize);

    h->op.preAdjointOperation(h->recvbufAdjoints, h->recvbufPrimals, h->recvbufCount);
    if(h->datatype->getADTool().isOldPrimalsRequired()) {
      h->datatype->getADTool().setReverseValues(h->recvbufIndices, h->recvbufOldPrimals, h->recvbufTotalSize);
    }
    h->sendbufAdjoints = nullptr;
    h->datatype->getADTool().createAdjointTypeBuffer(h->sendbufAdjoints, h->sendbufTotalSize * getCommSize(h->comm));

    AMPI_Allreduce_global_adj<DATATYPE>(h->sendbufAdjoints, h->sendbufCount, h->recvbufAdjoints, h->recvbufCount, h->count,
                                        h->datatype, h->op, h->comm);

    h->datatype->getADTool().combineAdjoints(h->sendbufAdjoints, h->sendbufTotalSize, getCommSize(h->comm));
    // the primals of the recive buffer are always given to the function. The operator should ignore them if not needed.
    // The wrapper functions make sure that for operators that need the primals an all* action is perfomed (e.g. Allreduce instead of Reduce)
    h->op.postAdjointOperation(h->sendbufAdjoints, h->sendbufPrimals, h->recvbufPrimals, h->sendbufTotalSize);
    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->datatype->getADTool().updateAdjoints(h->sendbufIndices, h->sendbufAdjoints, h->sendbufTotalSize);
    h->datatype->getADTool().deleteAdjointTypeBuffer(h->sendbufAdjoints);
    h->datatype->getADTool().deleteAdjointTypeBuffer(h->recvbufAdjoints);
  }

  template<typename DATATYPE>
  int AMPI_Allreduce_global(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count,
                            DATATYPE* datatype, AMPI_Op op, AMPI_Comm comm) {
    int rStatus;
    if(!datatype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Allreduce(sendbuf, recvbuf, count, datatype->getMpiType(), op.primalFunction, comm);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Allreduce_global_AdjointHandle<DATATYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(datatype->getADTool().isHandleRequired()) {
        h = new AMPI_Allreduce_global_AdjointHandle<DATATYPE>();
      }
      datatype->getADTool().startAssembly(h);
      typename DATATYPE::ModifiedType* sendbufMod = nullptr;
      int sendbufElements = 0;

      // compute the total size of the buffer
      if(AMPI_IN_PLACE != sendbuf) {
        sendbufElements = count;
      } else {
        sendbufElements = count;
      }

      if(datatype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        datatype->createModifiedTypeBuffer(sendbufMod, sendbufElements);
      } else {
        sendbufMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(sendbuf));
      }
      typename DATATYPE::ModifiedType* recvbufMod = nullptr;
      int recvbufElements = 0;

      // compute the total size of the buffer
      recvbufElements = count;

      if(datatype->isModifiedBufferRequired() ) {
        datatype->createModifiedTypeBuffer(recvbufMod, recvbufElements);
      } else {
        recvbufMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(recvbuf));
      }

      if(datatype->isModifiedBufferRequired()) {
        if(AMPI_IN_PLACE != sendbuf) {
          datatype->copyIntoModifiedBuffer(sendbuf, 0, sendbufMod, 0, count);
        } else {
          datatype->copyIntoModifiedBuffer(recvbuf, 0, recvbufMod, 0, count);
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(AMPI_IN_PLACE != sendbuf) {
          h->sendbufCount = datatype->computeActiveElements(count);
        } else {
          h->sendbufCount = datatype->computeActiveElements(count);
        }
        h->sendbufTotalSize = datatype->computeActiveElements(sendbufElements);
        datatype->getADTool().createIndexTypeBuffer(h->sendbufIndices, h->sendbufTotalSize);
        h->recvbufCount = datatype->computeActiveElements(count);
        h->recvbufTotalSize = datatype->computeActiveElements(recvbufElements);
        datatype->getADTool().createIndexTypeBuffer(h->recvbufIndices, h->recvbufTotalSize);

        // extract the primal values for the operator if required
        if(op.requiresPrimal) {
          datatype->getADTool().createPassiveTypeBuffer(h->sendbufPrimals, h->sendbufTotalSize);
          if(AMPI_IN_PLACE != sendbuf) {
            datatype->getValues(sendbuf, 0, h->sendbufPrimals, 0, count);
          } else {
            datatype->getValues(recvbuf, 0, h->sendbufPrimals, 0, count);
          }
        }

        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(datatype->getADTool().isOldPrimalsRequired()) {
          datatype->getADTool().createPassiveTypeBuffer(h->recvbufOldPrimals, h->recvbufTotalSize);
          datatype->getValues(recvbuf, 0, h->recvbufOldPrimals, 0, count);
        }


        if(AMPI_IN_PLACE != sendbuf) {
          datatype->getIndices(sendbuf, 0, h->sendbufIndices, 0, count);
        } else {
          datatype->getIndices(recvbuf, 0, h->sendbufIndices, 0, count);
        }

        if(!datatype->isModifiedBufferRequired()) {
          datatype->clearIndices(recvbuf, 0, count);
        }

        // pack all the variables in the handle
        h->func = AMPI_Allreduce_global_b<DATATYPE>;
        h->count = count;
        h->datatype = datatype;
        h->op = op;
        h->comm = comm;
      }

      rStatus = MPI_Allreduce(sendbufMod, recvbufMod, count, datatype->getModifiedMpiType(), op.modifiedPrimalFunction, comm);
      datatype->getADTool().addToolAction(h);

      if(datatype->isModifiedBufferRequired()) {
        datatype->copyFromModifiedBuffer(recvbuf, 0, recvbufMod, 0, count);
      }

      if(nullptr != h) {
        // handle the recv buffers
        datatype->registerValue(recvbuf, 0, h->recvbufIndices, h->recvbufOldPrimals, 0, count);
      }
      // extract the primal values for the operator if required
      if(nullptr != h && op.requiresPrimal) {
        datatype->getADTool().createPassiveTypeBuffer(h->recvbufPrimals, h->recvbufTotalSize);
        datatype->getValues(recvbuf, 0, h->recvbufPrimals, 0, count);
      }

      datatype->getADTool().stopAssembly(h);

      if(datatype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        datatype->deleteModifiedTypeBuffer(sendbufMod);
      }
      if(datatype->isModifiedBufferRequired() ) {
        datatype->deleteModifiedTypeBuffer(recvbufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Alltoall_AdjointHandle : public HandleBase {
    int sendbufTotalSize;
    typename SENDTYPE::IndexType* sendbufIndices;
    typename SENDTYPE::PassiveType* sendbufPrimals;
    /* required for async */ typename SENDTYPE::PassiveType* sendbufAdjoints;
    int sendbufCount;
    int sendcount;
    SENDTYPE* sendtype;
    int recvbufTotalSize;
    typename RECVTYPE::IndexType* recvbufIndices;
    typename RECVTYPE::PassiveType* recvbufPrimals;
    typename RECVTYPE::PassiveType* recvbufOldPrimals;
    /* required for async */ typename RECVTYPE::PassiveType* recvbufAdjoints;
    int recvbufCount;
    int recvcount;
    RECVTYPE* recvtype;
    AMPI_Comm comm;

    ~AMPI_Alltoall_AdjointHandle () {
      if(nullptr != sendbufIndices) {
        sendtype->getADTool().deleteIndexTypeBuffer(sendbufIndices);
        sendbufIndices = nullptr;
      }
      if(nullptr != sendbufPrimals) {
        sendtype->getADTool().deletePassiveTypeBuffer(sendbufPrimals);
        sendbufPrimals = nullptr;
      }
      if(nullptr != recvbufIndices) {
        recvtype->getADTool().deleteIndexTypeBuffer(recvbufIndices);
        recvbufIndices = nullptr;
      }
      if(nullptr != recvbufPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufPrimals);
        recvbufPrimals = nullptr;
      }
      if(nullptr != recvbufOldPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufOldPrimals);
        recvbufOldPrimals = nullptr;
      }
    }
  };


  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Alltoall_b(HandleBase* handle) {
    AMPI_Alltoall_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Alltoall_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);

    h->recvbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->recvbufAdjoints, h->recvbufTotalSize );
    // Adjoint buffers are always linear in space so we can accesses them in one sweep
    h->recvtype->getADTool().getAdjoints(h->recvbufIndices, h->recvbufAdjoints, h->recvbufTotalSize);

    if(h->recvtype->getADTool().isOldPrimalsRequired()) {
      h->recvtype->getADTool().setReverseValues(h->recvbufIndices, h->recvbufOldPrimals, h->recvbufTotalSize);
    }
    h->sendbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->sendbufAdjoints, h->sendbufTotalSize );

    AMPI_Alltoall_adj<SENDTYPE, RECVTYPE>(h->sendbufAdjoints, h->sendbufCount, h->sendcount, h->sendtype,
                                          h->recvbufAdjoints, h->recvbufCount, h->recvcount, h->recvtype, h->comm);

    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->recvtype->getADTool().updateAdjoints(h->sendbufIndices, h->sendbufAdjoints, h->sendbufTotalSize);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->sendbufAdjoints);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->recvbufAdjoints);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Alltoall(const typename SENDTYPE::Type* sendbuf, int sendcount, SENDTYPE* sendtype,
                    typename RECVTYPE::Type* recvbuf, int recvcount, RECVTYPE* recvtype, AMPI_Comm comm) {
    int rStatus;
    if(!recvtype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Alltoall(sendbuf, sendcount, sendtype->getMpiType(), recvbuf, recvcount, recvtype->getMpiType(), comm);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Alltoall_AdjointHandle<SENDTYPE, RECVTYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(recvtype->getADTool().isHandleRequired()) {
        h = new AMPI_Alltoall_AdjointHandle<SENDTYPE, RECVTYPE>();
      }
      recvtype->getADTool().startAssembly(h);
      typename SENDTYPE::ModifiedType* sendbufMod = nullptr;
      int sendbufElements = 0;

      // compute the total size of the buffer
      if(AMPI_IN_PLACE != sendbuf) {
        sendbufElements = sendcount * getCommSize(comm);
      } else {
        sendbufElements = recvcount * getCommSize(comm);
      }

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->createModifiedTypeBuffer(sendbufMod, sendbufElements);
      } else {
        sendbufMod = reinterpret_cast<typename SENDTYPE::ModifiedType*>(const_cast<typename SENDTYPE::Type*>(sendbuf));
      }
      typename RECVTYPE::ModifiedType* recvbufMod = nullptr;
      int recvbufElements = 0;

      // compute the total size of the buffer
      recvbufElements = recvcount * getCommSize(comm);

      if(recvtype->isModifiedBufferRequired() ) {
        recvtype->createModifiedTypeBuffer(recvbufMod, recvbufElements);
      } else {
        recvbufMod = reinterpret_cast<typename RECVTYPE::ModifiedType*>(const_cast<typename RECVTYPE::Type*>(recvbuf));
      }

      if(sendtype->isModifiedBufferRequired()) {
        if(AMPI_IN_PLACE != sendbuf) {
          sendtype->copyIntoModifiedBuffer(sendbuf, 0, sendbufMod, 0, sendcount * getCommSize(comm));
        } else {
          recvtype->copyIntoModifiedBuffer(recvbuf, 0, recvbufMod, 0, recvcount * getCommSize(comm));
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(AMPI_IN_PLACE != sendbuf) {
          h->sendbufCount = sendtype->computeActiveElements(sendcount);
        } else {
          h->sendbufCount = recvtype->computeActiveElements(recvcount);
        }
        h->sendbufTotalSize = sendtype->computeActiveElements(sendbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->sendbufIndices, h->sendbufTotalSize);
        h->recvbufCount = recvtype->computeActiveElements(recvcount);
        h->recvbufTotalSize = recvtype->computeActiveElements(recvbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->recvbufIndices, h->recvbufTotalSize);


        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(recvtype->getADTool().isOldPrimalsRequired()) {
          recvtype->getADTool().createPassiveTypeBuffer(h->recvbufOldPrimals, h->recvbufTotalSize);
          recvtype->getValues(recvbuf, 0, h->recvbufOldPrimals, 0, recvcount * getCommSize(comm));
        }


        if(AMPI_IN_PLACE != sendbuf) {
          sendtype->getIndices(sendbuf, 0, h->sendbufIndices, 0, sendcount * getCommSize(comm));
        } else {
          recvtype->getIndices(recvbuf, 0, h->sendbufIndices, 0, recvcount * getCommSize(comm));
        }

        if(!recvtype->isModifiedBufferRequired()) {
          recvtype->clearIndices(recvbuf, 0, recvcount * getCommSize(comm));
        }

        // pack all the variables in the handle
        h->func = AMPI_Alltoall_b<SENDTYPE, RECVTYPE>;
        h->sendcount = sendcount;
        h->sendtype = sendtype;
        h->recvcount = recvcount;
        h->recvtype = recvtype;
        h->comm = comm;
      }

      rStatus = MPI_Alltoall(sendbufMod, sendcount, sendtype->getModifiedMpiType(), recvbufMod, recvcount,
                             recvtype->getModifiedMpiType(), comm);
      recvtype->getADTool().addToolAction(h);

      if(recvtype->isModifiedBufferRequired()) {
        recvtype->copyFromModifiedBuffer(recvbuf, 0, recvbufMod, 0, recvcount * getCommSize(comm));
      }

      if(nullptr != h) {
        // handle the recv buffers
        recvtype->registerValue(recvbuf, 0, h->recvbufIndices, h->recvbufOldPrimals, 0, recvcount * getCommSize(comm));
      }

      recvtype->getADTool().stopAssembly(h);

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->deleteModifiedTypeBuffer(sendbufMod);
      }
      if(recvtype->isModifiedBufferRequired() ) {
        recvtype->deleteModifiedTypeBuffer(recvbufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Alltoallv_AdjointHandle : public HandleBase {
    int sendbufTotalSize;
    typename SENDTYPE::IndexType* sendbufIndices;
    typename SENDTYPE::PassiveType* sendbufPrimals;
    /* required for async */ typename SENDTYPE::PassiveType* sendbufAdjoints;
    int* sendbufCount;
    int* sendbufDispls;
    const  int* sendcounts;
    const  int* sdispls;
    SENDTYPE* sendtype;
    int recvbufTotalSize;
    typename RECVTYPE::IndexType* recvbufIndices;
    typename RECVTYPE::PassiveType* recvbufPrimals;
    typename RECVTYPE::PassiveType* recvbufOldPrimals;
    /* required for async */ typename RECVTYPE::PassiveType* recvbufAdjoints;
    int* recvbufCount;
    int* recvbufDispls;
    const  int* recvcounts;
    const  int* rdispls;
    RECVTYPE* recvtype;
    AMPI_Comm comm;

    ~AMPI_Alltoallv_AdjointHandle () {
      if(nullptr != sendbufIndices) {
        sendtype->getADTool().deleteIndexTypeBuffer(sendbufIndices);
        sendbufIndices = nullptr;
      }
      if(nullptr != sendbufPrimals) {
        sendtype->getADTool().deletePassiveTypeBuffer(sendbufPrimals);
        sendbufPrimals = nullptr;
      }
      if(nullptr != sendbufCount) {
        delete [] sendbufCount;
        sendbufCount = nullptr;
      }
      if(nullptr != sendbufDispls) {
        delete [] sendbufDispls;
        sendbufDispls = nullptr;
      }
      if(nullptr != recvbufIndices) {
        recvtype->getADTool().deleteIndexTypeBuffer(recvbufIndices);
        recvbufIndices = nullptr;
      }
      if(nullptr != recvbufPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufPrimals);
        recvbufPrimals = nullptr;
      }
      if(nullptr != recvbufOldPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufOldPrimals);
        recvbufOldPrimals = nullptr;
      }
      if(nullptr != recvbufCount) {
        delete [] recvbufCount;
        recvbufCount = nullptr;
      }
      if(nullptr != recvbufDispls) {
        delete [] recvbufDispls;
        recvbufDispls = nullptr;
      }
    }
  };


  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Alltoallv_b(HandleBase* handle) {
    AMPI_Alltoallv_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Alltoallv_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);

    h->recvbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->recvbufAdjoints, h->recvbufTotalSize );
    // Adjoint buffers are always linear in space so we can accesses them in one sweep
    h->recvtype->getADTool().getAdjoints(h->recvbufIndices, h->recvbufAdjoints, h->recvbufTotalSize);

    if(h->recvtype->getADTool().isOldPrimalsRequired()) {
      h->recvtype->getADTool().setReverseValues(h->recvbufIndices, h->recvbufOldPrimals, h->recvbufTotalSize);
    }
    h->sendbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->sendbufAdjoints, h->sendbufTotalSize );

    AMPI_Alltoallv_adj<SENDTYPE, RECVTYPE>(h->sendbufAdjoints, h->sendbufCount, h->sendbufDispls, h->sendcounts, h->sdispls,
                                           h->sendtype, h->recvbufAdjoints, h->recvbufCount, h->recvbufDispls, h->recvcounts, h->rdispls, h->recvtype, h->comm);

    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->recvtype->getADTool().updateAdjoints(h->sendbufIndices, h->sendbufAdjoints, h->sendbufTotalSize);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->sendbufAdjoints);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->recvbufAdjoints);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Alltoallv(const typename SENDTYPE::Type* sendbuf, const int* sendcounts, const int* sdispls,
                     SENDTYPE* sendtype, typename RECVTYPE::Type* recvbuf, const int* recvcounts, const int* rdispls, RECVTYPE* recvtype,
                     AMPI_Comm comm) {
    int rStatus;
    if(!recvtype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype->getMpiType(), recvbuf, recvcounts, rdispls,
                              recvtype->getMpiType(), comm);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Alltoallv_AdjointHandle<SENDTYPE, RECVTYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(recvtype->getADTool().isHandleRequired()) {
        h = new AMPI_Alltoallv_AdjointHandle<SENDTYPE, RECVTYPE>();
      }
      recvtype->getADTool().startAssembly(h);
      const int* sdisplsMod = sdispls;
      int sdisplsTotalSize = 0;
      if(nullptr != sdispls) {
        sdisplsTotalSize = computeDisplacementsTotalSize(sendcounts, getCommSize(comm));
        if(recvtype->isModifiedBufferRequired()) {
          sdisplsMod = createLinearDisplacements(sendcounts, getCommSize(comm));
        }
      }
      const int* rdisplsMod = rdispls;
      int rdisplsTotalSize = 0;
      if(nullptr != rdispls) {
        rdisplsTotalSize = computeDisplacementsTotalSize(recvcounts, getCommSize(comm));
        if(recvtype->isModifiedBufferRequired()) {
          rdisplsMod = createLinearDisplacements(recvcounts, getCommSize(comm));
        }
      }
      typename SENDTYPE::ModifiedType* sendbufMod = nullptr;
      int sendbufElements = 0;

      // compute the total size of the buffer
      if(AMPI_IN_PLACE != sendbuf) {
        sendbufElements = sdisplsTotalSize;
      } else {
        sendbufElements = rdisplsTotalSize;
      }

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->createModifiedTypeBuffer(sendbufMod, sendbufElements);
      } else {
        sendbufMod = reinterpret_cast<typename SENDTYPE::ModifiedType*>(const_cast<typename SENDTYPE::Type*>(sendbuf));
      }
      typename RECVTYPE::ModifiedType* recvbufMod = nullptr;
      int recvbufElements = 0;

      // compute the total size of the buffer
      recvbufElements = rdisplsTotalSize;

      if(recvtype->isModifiedBufferRequired() ) {
        recvtype->createModifiedTypeBuffer(recvbufMod, recvbufElements);
      } else {
        recvbufMod = reinterpret_cast<typename RECVTYPE::ModifiedType*>(const_cast<typename RECVTYPE::Type*>(recvbuf));
      }

      if(sendtype->isModifiedBufferRequired()) {
        if(AMPI_IN_PLACE != sendbuf) {
          for(int i = 0; i < getCommSize(comm); ++i) {
            sendtype->copyIntoModifiedBuffer(sendbuf, sdispls[i], sendbufMod, sdisplsMod[i], sendcounts[i]);
          }
        } else {
          for(int i = 0; i < getCommSize(comm); ++i) {
            recvtype->copyIntoModifiedBuffer(recvbuf, rdispls[i], recvbufMod, rdisplsMod[i], recvcounts[i]);
          }
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(AMPI_IN_PLACE != sendbuf) {
          createLinearIndexDisplacements(h->sendbufCount, h->sendbufDispls, sendcounts, getCommSize(comm), sendtype);
        } else {
          createLinearIndexDisplacements(h->sendbufCount, h->sendbufDispls, recvcounts, getCommSize(comm), recvtype);
        }
        h->sendbufTotalSize = sendtype->computeActiveElements(sendbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->sendbufIndices, h->sendbufTotalSize);
        createLinearIndexDisplacements(h->recvbufCount, h->recvbufDispls, recvcounts, getCommSize(comm), recvtype);
        h->recvbufTotalSize = recvtype->computeActiveElements(recvbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->recvbufIndices, h->recvbufTotalSize);


        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(recvtype->getADTool().isOldPrimalsRequired()) {
          recvtype->getADTool().createPassiveTypeBuffer(h->recvbufOldPrimals, h->recvbufTotalSize);
          for(int i = 0; i < getCommSize(comm); ++i) {
            recvtype->getValues(recvbuf, rdispls[i], h->recvbufOldPrimals, rdisplsMod[i], recvcounts[i]);
          }
        }


        if(AMPI_IN_PLACE != sendbuf) {
          for(int i = 0; i < getCommSize(comm); ++i) {
            sendtype->getIndices(sendbuf, sdispls[i], h->sendbufIndices, sdisplsMod[i], sendcounts[i]);
          }
        } else {
          for(int i = 0; i < getCommSize(comm); ++i) {
            recvtype->getIndices(recvbuf, rdispls[i], h->sendbufIndices, rdisplsMod[i], recvcounts[i]);
          }
        }

        if(!recvtype->isModifiedBufferRequired()) {
          for(int i = 0; i < getCommSize(comm); ++i) {
            recvtype->clearIndices(recvbuf, rdispls[i], recvcounts[i]);
          }
        }

        // pack all the variables in the handle
        h->func = AMPI_Alltoallv_b<SENDTYPE, RECVTYPE>;
        h->sendcounts = sendcounts;
        h->sdispls = sdispls;
        h->sendtype = sendtype;
        h->recvcounts = recvcounts;
        h->rdispls = rdispls;
        h->recvtype = recvtype;
        h->comm = comm;
      }

      rStatus = MPI_Alltoallv(sendbufMod, sendcounts, sdisplsMod, sendtype->getModifiedMpiType(), recvbufMod, recvcounts,
                              rdisplsMod, recvtype->getModifiedMpiType(), comm);
      recvtype->getADTool().addToolAction(h);

      if(recvtype->isModifiedBufferRequired()) {
        for(int i = 0; i < getCommSize(comm); ++i) {
          recvtype->copyFromModifiedBuffer(recvbuf, rdispls[i], recvbufMod, rdisplsMod[i], recvcounts[i]);
        }
      }

      if(nullptr != h) {
        // handle the recv buffers
        for(int i = 0; i < getCommSize(comm); ++i) {
          recvtype->registerValue(recvbuf, rdispls[i], h->recvbufIndices, h->recvbufOldPrimals, rdisplsMod[i], recvcounts[i]);
        }
      }

      recvtype->getADTool().stopAssembly(h);
      if(recvtype->isModifiedBufferRequired()) {
        delete [] sdisplsMod;
      }
      if(recvtype->isModifiedBufferRequired()) {
        delete [] rdisplsMod;
      }

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->deleteModifiedTypeBuffer(sendbufMod);
      }
      if(recvtype->isModifiedBufferRequired() ) {
        recvtype->deleteModifiedTypeBuffer(recvbufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename DATATYPE>
  struct AMPI_Bcast_wrap_AdjointHandle : public HandleBase {
    int bufferSendTotalSize;
    typename DATATYPE::IndexType* bufferSendIndices;
    typename DATATYPE::PassiveType* bufferSendPrimals;
    /* required for async */ typename DATATYPE::PassiveType* bufferSendAdjoints;
    int bufferSendCount;
    int bufferRecvTotalSize;
    typename DATATYPE::IndexType* bufferRecvIndices;
    typename DATATYPE::PassiveType* bufferRecvPrimals;
    typename DATATYPE::PassiveType* bufferRecvOldPrimals;
    /* required for async */ typename DATATYPE::PassiveType* bufferRecvAdjoints;
    int bufferRecvCount;
    int count;
    DATATYPE* datatype;
    int root;
    AMPI_Comm comm;

    ~AMPI_Bcast_wrap_AdjointHandle () {
      if(nullptr != bufferSendIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(bufferSendIndices);
        bufferSendIndices = nullptr;
      }
      if(nullptr != bufferSendPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(bufferSendPrimals);
        bufferSendPrimals = nullptr;
      }
      if(nullptr != bufferRecvIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(bufferRecvIndices);
        bufferRecvIndices = nullptr;
      }
      if(nullptr != bufferRecvPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(bufferRecvPrimals);
        bufferRecvPrimals = nullptr;
      }
      if(nullptr != bufferRecvOldPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(bufferRecvOldPrimals);
        bufferRecvOldPrimals = nullptr;
      }
    }
  };


  template<typename DATATYPE>
  void AMPI_Bcast_wrap_b(HandleBase* handle) {
    AMPI_Bcast_wrap_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Bcast_wrap_AdjointHandle<DATATYPE>*>(handle);

    h->bufferRecvAdjoints = nullptr;
    h->datatype->getADTool().createAdjointTypeBuffer(h->bufferRecvAdjoints, h->bufferRecvTotalSize );
    // Adjoint buffers are always linear in space so we can accesses them in one sweep
    h->datatype->getADTool().getAdjoints(h->bufferRecvIndices, h->bufferRecvAdjoints, h->bufferRecvTotalSize);

    if(h->datatype->getADTool().isOldPrimalsRequired()) {
      h->datatype->getADTool().setReverseValues(h->bufferRecvIndices, h->bufferRecvOldPrimals, h->bufferRecvTotalSize);
    }
    h->bufferSendAdjoints = nullptr;
    if(h->root == getCommRank(h->comm)) {
      h->datatype->getADTool().createAdjointTypeBuffer(h->bufferSendAdjoints, h->bufferSendTotalSize * getCommSize(h->comm));
    }

    AMPI_Bcast_wrap_adj<DATATYPE>(h->bufferSendAdjoints, h->bufferSendCount, h->bufferRecvAdjoints, h->bufferRecvCount,
                                  h->count, h->datatype, h->root, h->comm);

    if(h->root == getCommRank(h->comm)) {
      h->datatype->getADTool().combineAdjoints(h->bufferSendAdjoints, h->bufferSendTotalSize, getCommSize(h->comm));
      // Adjoint buffers are always linear in space so they can be accessed in one sweep
      h->datatype->getADTool().updateAdjoints(h->bufferSendIndices, h->bufferSendAdjoints, h->bufferSendTotalSize);
      h->datatype->getADTool().deleteAdjointTypeBuffer(h->bufferSendAdjoints);
    }
    h->datatype->getADTool().deleteAdjointTypeBuffer(h->bufferRecvAdjoints);
  }

  template<typename DATATYPE>
  int AMPI_Bcast_wrap(typename DATATYPE::Type* bufferSend, typename DATATYPE::Type* bufferRecv, int count,
                      DATATYPE* datatype, int root, AMPI_Comm comm) {
    int rStatus;
    if(!datatype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Bcast_wrap(bufferSend, bufferRecv, count, datatype->getMpiType(), root, comm);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Bcast_wrap_AdjointHandle<DATATYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(datatype->getADTool().isHandleRequired()) {
        h = new AMPI_Bcast_wrap_AdjointHandle<DATATYPE>();
      }
      datatype->getADTool().startAssembly(h);
      typename DATATYPE::ModifiedType* bufferSendMod = nullptr;
      int bufferSendElements = 0;

      if(root == getCommRank(comm)) {
        // compute the total size of the buffer
        if(AMPI_IN_PLACE != bufferSend) {
          bufferSendElements = count;
        } else {
          bufferSendElements = count;
        }

        if(datatype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == bufferSend)) {
          datatype->createModifiedTypeBuffer(bufferSendMod, bufferSendElements);
        } else {
          bufferSendMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(bufferSend));
        }
      }
      typename DATATYPE::ModifiedType* bufferRecvMod = nullptr;
      int bufferRecvElements = 0;

      // compute the total size of the buffer
      bufferRecvElements = count;

      if(datatype->isModifiedBufferRequired() ) {
        datatype->createModifiedTypeBuffer(bufferRecvMod, bufferRecvElements);
      } else {
        bufferRecvMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(bufferRecv));
      }

      if(root == getCommRank(comm)) {
        if(datatype->isModifiedBufferRequired()) {
          if(AMPI_IN_PLACE != bufferSend) {
            datatype->copyIntoModifiedBuffer(bufferSend, 0, bufferSendMod, 0, count);
          } else {
            datatype->copyIntoModifiedBuffer(bufferRecv, 0, bufferRecvMod, 0, count);
          }
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(root == getCommRank(comm)) {
          if(AMPI_IN_PLACE != bufferSend) {
            h->bufferSendCount = datatype->computeActiveElements(count);
          } else {
            h->bufferSendCount = datatype->computeActiveElements(count);
          }
          h->bufferSendTotalSize = datatype->computeActiveElements(bufferSendElements);
          datatype->getADTool().createIndexTypeBuffer(h->bufferSendIndices, h->bufferSendTotalSize);
        }
        h->bufferRecvCount = datatype->computeActiveElements(count);
        h->bufferRecvTotalSize = datatype->computeActiveElements(bufferRecvElements);
        datatype->getADTool().createIndexTypeBuffer(h->bufferRecvIndices, h->bufferRecvTotalSize);


        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(datatype->getADTool().isOldPrimalsRequired()) {
          datatype->getADTool().createPassiveTypeBuffer(h->bufferRecvOldPrimals, h->bufferRecvTotalSize);
          datatype->getValues(bufferRecv, 0, h->bufferRecvOldPrimals, 0, count);
        }


        if(root == getCommRank(comm)) {
          if(AMPI_IN_PLACE != bufferSend) {
            datatype->getIndices(bufferSend, 0, h->bufferSendIndices, 0, count);
          } else {
            datatype->getIndices(bufferRecv, 0, h->bufferSendIndices, 0, count);
          }
        }

        if(!datatype->isModifiedBufferRequired()) {
          datatype->clearIndices(bufferRecv, 0, count);
        }

        // pack all the variables in the handle
        h->func = AMPI_Bcast_wrap_b<DATATYPE>;
        h->count = count;
        h->datatype = datatype;
        h->root = root;
        h->comm = comm;
      }

      rStatus = MPI_Bcast_wrap(bufferSendMod, bufferRecvMod, count, datatype->getModifiedMpiType(), root, comm);
      datatype->getADTool().addToolAction(h);

      if(datatype->isModifiedBufferRequired()) {
        datatype->copyFromModifiedBuffer(bufferRecv, 0, bufferRecvMod, 0, count);
      }

      if(nullptr != h) {
        // handle the recv buffers
        datatype->registerValue(bufferRecv, 0, h->bufferRecvIndices, h->bufferRecvOldPrimals, 0, count);
      }

      datatype->getADTool().stopAssembly(h);

      if(root == getCommRank(comm)) {
        if(datatype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == bufferSend)) {
          datatype->deleteModifiedTypeBuffer(bufferSendMod);
        }
      }
      if(datatype->isModifiedBufferRequired() ) {
        datatype->deleteModifiedTypeBuffer(bufferRecvMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Gather_AdjointHandle : public HandleBase {
    int sendbufTotalSize;
    typename SENDTYPE::IndexType* sendbufIndices;
    typename SENDTYPE::PassiveType* sendbufPrimals;
    /* required for async */ typename SENDTYPE::PassiveType* sendbufAdjoints;
    int sendbufCount;
    int sendcount;
    SENDTYPE* sendtype;
    int recvbufTotalSize;
    typename RECVTYPE::IndexType* recvbufIndices;
    typename RECVTYPE::PassiveType* recvbufPrimals;
    typename RECVTYPE::PassiveType* recvbufOldPrimals;
    /* required for async */ typename RECVTYPE::PassiveType* recvbufAdjoints;
    int recvbufCount;
    int recvcount;
    RECVTYPE* recvtype;
    int root;
    AMPI_Comm comm;

    ~AMPI_Gather_AdjointHandle () {
      if(nullptr != sendbufIndices) {
        sendtype->getADTool().deleteIndexTypeBuffer(sendbufIndices);
        sendbufIndices = nullptr;
      }
      if(nullptr != sendbufPrimals) {
        sendtype->getADTool().deletePassiveTypeBuffer(sendbufPrimals);
        sendbufPrimals = nullptr;
      }
      if(nullptr != recvbufIndices) {
        recvtype->getADTool().deleteIndexTypeBuffer(recvbufIndices);
        recvbufIndices = nullptr;
      }
      if(nullptr != recvbufPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufPrimals);
        recvbufPrimals = nullptr;
      }
      if(nullptr != recvbufOldPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufOldPrimals);
        recvbufOldPrimals = nullptr;
      }
    }
  };


  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Gather_b(HandleBase* handle) {
    AMPI_Gather_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Gather_AdjointHandle<SENDTYPE, RECVTYPE>*>(handle);

    h->recvbufAdjoints = nullptr;
    if(h->root == getCommRank(h->comm)) {
      h->recvtype->getADTool().createAdjointTypeBuffer(h->recvbufAdjoints, h->recvbufTotalSize );
      // Adjoint buffers are always linear in space so we can accesses them in one sweep
      h->recvtype->getADTool().getAdjoints(h->recvbufIndices, h->recvbufAdjoints, h->recvbufTotalSize);

    }
    if(h->recvtype->getADTool().isOldPrimalsRequired()) {
      if(h->root == getCommRank(h->comm)) {
        h->recvtype->getADTool().setReverseValues(h->recvbufIndices, h->recvbufOldPrimals, h->recvbufTotalSize);
      }
    }
    h->sendbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->sendbufAdjoints, h->sendbufTotalSize );

    AMPI_Gather_adj<SENDTYPE, RECVTYPE>(h->sendbufAdjoints, h->sendbufCount, h->sendcount, h->sendtype, h->recvbufAdjoints,
                                        h->recvbufCount, h->recvcount, h->recvtype, h->root, h->comm);

    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->recvtype->getADTool().updateAdjoints(h->sendbufIndices, h->sendbufAdjoints, h->sendbufTotalSize);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->sendbufAdjoints);
    if(h->root == getCommRank(h->comm)) {
      h->recvtype->getADTool().deleteAdjointTypeBuffer(h->recvbufAdjoints);
    }
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Gather(const typename SENDTYPE::Type* sendbuf, int sendcount, SENDTYPE* sendtype,
                  typename RECVTYPE::Type* recvbuf, int recvcount, RECVTYPE* recvtype, int root, AMPI_Comm comm) {
    int rStatus;
    if(!recvtype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Gather(sendbuf, sendcount, sendtype->getMpiType(), recvbuf, recvcount, recvtype->getMpiType(), root,
                           comm);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Gather_AdjointHandle<SENDTYPE, RECVTYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(recvtype->getADTool().isHandleRequired()) {
        h = new AMPI_Gather_AdjointHandle<SENDTYPE, RECVTYPE>();
      }
      recvtype->getADTool().startAssembly(h);
      typename SENDTYPE::ModifiedType* sendbufMod = nullptr;
      int sendbufElements = 0;

      // compute the total size of the buffer
      if(AMPI_IN_PLACE != sendbuf) {
        sendbufElements = sendcount;
      } else {
        sendbufElements = recvcount;
      }

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->createModifiedTypeBuffer(sendbufMod, sendbufElements);
      } else {
        sendbufMod = reinterpret_cast<typename SENDTYPE::ModifiedType*>(const_cast<typename SENDTYPE::Type*>(sendbuf));
      }
      typename RECVTYPE::ModifiedType* recvbufMod = nullptr;
      int recvbufElements = 0;

      if(root == getCommRank(comm)) {
        // compute the total size of the buffer
        recvbufElements = recvcount * getCommSize(comm);

        if(recvtype->isModifiedBufferRequired() ) {
          recvtype->createModifiedTypeBuffer(recvbufMod, recvbufElements);
        } else {
          recvbufMod = reinterpret_cast<typename RECVTYPE::ModifiedType*>(const_cast<typename RECVTYPE::Type*>(recvbuf));
        }
      }

      if(sendtype->isModifiedBufferRequired()) {
        if(AMPI_IN_PLACE != sendbuf) {
          sendtype->copyIntoModifiedBuffer(sendbuf, 0, sendbufMod, 0, sendcount);
        } else {
          recvtype->copyIntoModifiedBuffer(recvbuf, recvcount * getCommRank(comm), recvbufMod, recvcount * getCommRank(comm),
                                           recvcount);
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(AMPI_IN_PLACE != sendbuf) {
          h->sendbufCount = sendtype->computeActiveElements(sendcount);
        } else {
          h->sendbufCount = recvtype->computeActiveElements(recvcount);
        }
        h->sendbufTotalSize = sendtype->computeActiveElements(sendbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->sendbufIndices, h->sendbufTotalSize);
        if(root == getCommRank(comm)) {
          h->recvbufCount = recvtype->computeActiveElements(recvcount);
          h->recvbufTotalSize = recvtype->computeActiveElements(recvbufElements);
          recvtype->getADTool().createIndexTypeBuffer(h->recvbufIndices, h->recvbufTotalSize);
        }


        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(recvtype->getADTool().isOldPrimalsRequired()) {
          if(root == getCommRank(comm)) {
            recvtype->getADTool().createPassiveTypeBuffer(h->recvbufOldPrimals, h->recvbufTotalSize);
            if(root == getCommRank(comm)) {
              recvtype->getValues(recvbuf, 0, h->recvbufOldPrimals, 0, recvcount * getCommSize(comm));
            }
          }
        }


        if(AMPI_IN_PLACE != sendbuf) {
          sendtype->getIndices(sendbuf, 0, h->sendbufIndices, 0, sendcount);
        } else {
          recvtype->getIndices(recvbuf, recvcount * getCommRank(comm), h->sendbufIndices, 0, recvcount);
        }

        if(root == getCommRank(comm)) {
          if(!recvtype->isModifiedBufferRequired()) {
            recvtype->clearIndices(recvbuf, 0, recvcount * getCommSize(comm));
          }
        }

        // pack all the variables in the handle
        h->func = AMPI_Gather_b<SENDTYPE, RECVTYPE>;
        h->sendcount = sendcount;
        h->sendtype = sendtype;
        h->recvcount = recvcount;
        h->recvtype = recvtype;
        h->root = root;
        h->comm = comm;
      }

      rStatus = MPI_Gather(sendbufMod, sendcount, sendtype->getModifiedMpiType(), recvbufMod, recvcount,
                           recvtype->getModifiedMpiType(), root, comm);
      recvtype->getADTool().addToolAction(h);

      if(root == getCommRank(comm)) {
        if(recvtype->isModifiedBufferRequired()) {
          recvtype->copyFromModifiedBuffer(recvbuf, 0, recvbufMod, 0, recvcount * getCommSize(comm));
        }
      }

      if(nullptr != h) {
        // handle the recv buffers
        if(root == getCommRank(comm)) {
          recvtype->registerValue(recvbuf, 0, h->recvbufIndices, h->recvbufOldPrimals, 0, recvcount * getCommSize(comm));
        }
      }

      recvtype->getADTool().stopAssembly(h);

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->deleteModifiedTypeBuffer(sendbufMod);
      }
      if(root == getCommRank(comm)) {
        if(recvtype->isModifiedBufferRequired() ) {
          recvtype->deleteModifiedTypeBuffer(recvbufMod);
        }
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Gatherv_AdjointHandle : public HandleBase {
    int sendbufTotalSize;
    typename SENDTYPE::IndexType* sendbufIndices;
    typename SENDTYPE::PassiveType* sendbufPrimals;
    /* required for async */ typename SENDTYPE::PassiveType* sendbufAdjoints;
    int sendbufCount;
    int sendcount;
    SENDTYPE* sendtype;
    int recvbufTotalSize;
    typename RECVTYPE::IndexType* recvbufIndices;
    typename RECVTYPE::PassiveType* recvbufPrimals;
    typename RECVTYPE::PassiveType* recvbufOldPrimals;
    /* required for async */ typename RECVTYPE::PassiveType* recvbufAdjoints;
    int* recvbufCount;
    int* recvbufDispls;
    const  int* recvcounts;
    const  int* displs;
    RECVTYPE* recvtype;
    int root;
    AMPI_Comm comm;

    ~AMPI_Gatherv_AdjointHandle () {
      if(nullptr != sendbufIndices) {
        sendtype->getADTool().deleteIndexTypeBuffer(sendbufIndices);
        sendbufIndices = nullptr;
      }
      if(nullptr != sendbufPrimals) {
        sendtype->getADTool().deletePassiveTypeBuffer(sendbufPrimals);
        sendbufPrimals = nullptr;
      }
      if(nullptr != recvbufIndices) {
        recvtype->getADTool().deleteIndexTypeBuffer(recvbufIndices);
        recvbufIndices = nullptr;
      }
      if(nullptr != recvbufPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufPrimals);
        recvbufPrimals = nullptr;
      }
      if(nullptr != recvbufOldPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufOldPrimals);
        recvbufOldPrimals = nullptr;
      }
      if(nullptr != recvbufCount) {
        delete [] recvbufCount;
        recvbufCount = nullptr;
      }
      if(nullptr != recvbufDispls) {
        delete [] recvbufDispls;
        recvbufDispls = nullptr;
      }
    }
  };


  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Gatherv_b(HandleBase* handle) {
    AMPI_Gatherv_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Gatherv_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);

    h->recvbufAdjoints = nullptr;
    if(h->root == getCommRank(h->comm)) {
      h->recvtype->getADTool().createAdjointTypeBuffer(h->recvbufAdjoints, h->recvbufTotalSize );
      // Adjoint buffers are always linear in space so we can accesses them in one sweep
      h->recvtype->getADTool().getAdjoints(h->recvbufIndices, h->recvbufAdjoints, h->recvbufTotalSize);

    }
    if(h->recvtype->getADTool().isOldPrimalsRequired()) {
      if(h->root == getCommRank(h->comm)) {
        h->recvtype->getADTool().setReverseValues(h->recvbufIndices, h->recvbufOldPrimals, h->recvbufTotalSize);
      }
    }
    h->sendbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->sendbufAdjoints, h->sendbufTotalSize );

    AMPI_Gatherv_adj<SENDTYPE, RECVTYPE>(h->sendbufAdjoints, h->sendbufCount, h->sendcount, h->sendtype, h->recvbufAdjoints,
                                         h->recvbufCount, h->recvbufDispls, h->recvcounts, h->displs, h->recvtype, h->root, h->comm);

    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->recvtype->getADTool().updateAdjoints(h->sendbufIndices, h->sendbufAdjoints, h->sendbufTotalSize);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->sendbufAdjoints);
    if(h->root == getCommRank(h->comm)) {
      h->recvtype->getADTool().deleteAdjointTypeBuffer(h->recvbufAdjoints);
    }
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Gatherv(const typename SENDTYPE::Type* sendbuf, int sendcount, SENDTYPE* sendtype,
                   typename RECVTYPE::Type* recvbuf, const int* recvcounts, const int* displs, RECVTYPE* recvtype, int root,
                   AMPI_Comm comm) {
    int rStatus;
    if(!recvtype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Gatherv(sendbuf, sendcount, sendtype->getMpiType(), recvbuf, recvcounts, displs, recvtype->getMpiType(),
                            root, comm);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Gatherv_AdjointHandle<SENDTYPE, RECVTYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(recvtype->getADTool().isHandleRequired()) {
        h = new AMPI_Gatherv_AdjointHandle<SENDTYPE, RECVTYPE>();
      }
      recvtype->getADTool().startAssembly(h);
      const int* displsMod = displs;
      int displsTotalSize = 0;
      if(nullptr != displs) {
        displsTotalSize = computeDisplacementsTotalSize(recvcounts, getCommSize(comm));
        if(recvtype->isModifiedBufferRequired()) {
          displsMod = createLinearDisplacements(recvcounts, getCommSize(comm));
        }
      }
      typename SENDTYPE::ModifiedType* sendbufMod = nullptr;
      int sendbufElements = 0;

      // compute the total size of the buffer
      if(AMPI_IN_PLACE != sendbuf) {
        sendbufElements = sendcount;
      } else {
        sendbufElements = recvcounts[getCommRank(comm)];
      }

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->createModifiedTypeBuffer(sendbufMod, sendbufElements);
      } else {
        sendbufMod = reinterpret_cast<typename SENDTYPE::ModifiedType*>(const_cast<typename SENDTYPE::Type*>(sendbuf));
      }
      typename RECVTYPE::ModifiedType* recvbufMod = nullptr;
      int recvbufElements = 0;

      if(root == getCommRank(comm)) {
        // compute the total size of the buffer
        recvbufElements = displsTotalSize;

        if(recvtype->isModifiedBufferRequired() ) {
          recvtype->createModifiedTypeBuffer(recvbufMod, recvbufElements);
        } else {
          recvbufMod = reinterpret_cast<typename RECVTYPE::ModifiedType*>(const_cast<typename RECVTYPE::Type*>(recvbuf));
        }
      }

      if(sendtype->isModifiedBufferRequired()) {
        if(AMPI_IN_PLACE != sendbuf) {
          sendtype->copyIntoModifiedBuffer(sendbuf, 0, sendbufMod, 0, sendcount);
        } else {
          {
            const int rank = getCommRank(comm);
            recvtype->copyIntoModifiedBuffer(recvbuf, displs[rank], recvbufMod, displsMod[rank], recvcounts[rank]);
          }
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(AMPI_IN_PLACE != sendbuf) {
          h->sendbufCount = sendtype->computeActiveElements(sendcount);
        } else {
          h->sendbufCount = recvtype->computeActiveElements(recvcounts[getCommRank(comm)]);
        }
        h->sendbufTotalSize = sendtype->computeActiveElements(sendbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->sendbufIndices, h->sendbufTotalSize);
        if(root == getCommRank(comm)) {
          createLinearIndexDisplacements(h->recvbufCount, h->recvbufDispls, recvcounts, getCommSize(comm), recvtype);
          h->recvbufTotalSize = recvtype->computeActiveElements(recvbufElements);
          recvtype->getADTool().createIndexTypeBuffer(h->recvbufIndices, h->recvbufTotalSize);
        }


        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(recvtype->getADTool().isOldPrimalsRequired()) {
          if(root == getCommRank(comm)) {
            recvtype->getADTool().createPassiveTypeBuffer(h->recvbufOldPrimals, h->recvbufTotalSize);
            if(root == getCommRank(comm)) {
              for(int i = 0; i < getCommSize(comm); ++i) {
                recvtype->getValues(recvbuf, displs[i], h->recvbufOldPrimals, displsMod[i], recvcounts[i]);
              }
            }
          }
        }


        if(AMPI_IN_PLACE != sendbuf) {
          sendtype->getIndices(sendbuf, 0, h->sendbufIndices, 0, sendcount);
        } else {
          {
            const int rank = getCommRank(comm);
            recvtype->getIndices(recvbuf, displs[rank], h->sendbufIndices, 0, recvcounts[rank]);
          }
        }

        if(root == getCommRank(comm)) {
          if(!recvtype->isModifiedBufferRequired()) {
            for(int i = 0; i < getCommSize(comm); ++i) {
              recvtype->clearIndices(recvbuf, displs[i], recvcounts[i]);
            }
          }
        }

        // pack all the variables in the handle
        h->func = AMPI_Gatherv_b<SENDTYPE, RECVTYPE>;
        h->sendcount = sendcount;
        h->sendtype = sendtype;
        h->recvcounts = recvcounts;
        h->displs = displs;
        h->recvtype = recvtype;
        h->root = root;
        h->comm = comm;
      }

      rStatus = MPI_Gatherv(sendbufMod, sendcount, sendtype->getModifiedMpiType(), recvbufMod, recvcounts, displsMod,
                            recvtype->getModifiedMpiType(), root, comm);
      recvtype->getADTool().addToolAction(h);

      if(root == getCommRank(comm)) {
        if(recvtype->isModifiedBufferRequired()) {
          for(int i = 0; i < getCommSize(comm); ++i) {
            recvtype->copyFromModifiedBuffer(recvbuf, displs[i], recvbufMod, displsMod[i], recvcounts[i]);
          }
        }
      }

      if(nullptr != h) {
        // handle the recv buffers
        if(root == getCommRank(comm)) {
          for(int i = 0; i < getCommSize(comm); ++i) {
            recvtype->registerValue(recvbuf, displs[i], h->recvbufIndices, h->recvbufOldPrimals, displsMod[i], recvcounts[i]);
          }
        }
      }

      recvtype->getADTool().stopAssembly(h);
      if(recvtype->isModifiedBufferRequired()) {
        delete [] displsMod;
      }

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->deleteModifiedTypeBuffer(sendbufMod);
      }
      if(root == getCommRank(comm)) {
        if(recvtype->isModifiedBufferRequired() ) {
          recvtype->deleteModifiedTypeBuffer(recvbufMod);
        }
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Iallgather_AdjointHandle : public HandleBase {
    int sendbufTotalSize;
    typename SENDTYPE::IndexType* sendbufIndices;
    typename SENDTYPE::PassiveType* sendbufPrimals;
    /* required for async */ typename SENDTYPE::PassiveType* sendbufAdjoints;
    int sendbufCount;
    int sendcount;
    SENDTYPE* sendtype;
    int recvbufTotalSize;
    typename RECVTYPE::IndexType* recvbufIndices;
    typename RECVTYPE::PassiveType* recvbufPrimals;
    typename RECVTYPE::PassiveType* recvbufOldPrimals;
    /* required for async */ typename RECVTYPE::PassiveType* recvbufAdjoints;
    int recvbufCount;
    int recvcount;
    RECVTYPE* recvtype;
    AMPI_Comm comm;
    AMPI_Request requestReverse;

    ~AMPI_Iallgather_AdjointHandle () {
      if(nullptr != sendbufIndices) {
        sendtype->getADTool().deleteIndexTypeBuffer(sendbufIndices);
        sendbufIndices = nullptr;
      }
      if(nullptr != sendbufPrimals) {
        sendtype->getADTool().deletePassiveTypeBuffer(sendbufPrimals);
        sendbufPrimals = nullptr;
      }
      if(nullptr != recvbufIndices) {
        recvtype->getADTool().deleteIndexTypeBuffer(recvbufIndices);
        recvbufIndices = nullptr;
      }
      if(nullptr != recvbufPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufPrimals);
        recvbufPrimals = nullptr;
      }
      if(nullptr != recvbufOldPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufOldPrimals);
        recvbufOldPrimals = nullptr;
      }
    }
  };

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Iallgather_AsyncHandle : public HandleBase {
    const  typename SENDTYPE::Type* sendbuf;
    typename SENDTYPE::ModifiedType* sendbufMod;
    int sendcount;
    SENDTYPE* sendtype;
    typename RECVTYPE::Type* recvbuf;
    typename RECVTYPE::ModifiedType* recvbufMod;
    int recvcount;
    RECVTYPE* recvtype;
    AMPI_Comm comm;
    AMPI_Request* request;
    AMPI_Iallgather_AdjointHandle<SENDTYPE, RECVTYPE>* h;
  };

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Iallgather_b(HandleBase* handle) {
    AMPI_Iallgather_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Iallgather_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);

    h->recvbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->recvbufAdjoints, h->recvbufTotalSize );
    // Adjoint buffers are always linear in space so we can accesses them in one sweep
    h->recvtype->getADTool().getAdjoints(h->recvbufIndices, h->recvbufAdjoints, h->recvbufTotalSize);

    if(h->recvtype->getADTool().isOldPrimalsRequired()) {
      h->recvtype->getADTool().setReverseValues(h->recvbufIndices, h->recvbufOldPrimals, h->recvbufTotalSize);
    }
    h->sendbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->sendbufAdjoints, h->sendbufTotalSize * getCommSize(h->comm));

    AMPI_Iallgather_adj<SENDTYPE, RECVTYPE>(h->sendbufAdjoints, h->sendbufCount, h->sendcount, h->sendtype,
                                            h->recvbufAdjoints, h->recvbufCount, h->recvcount, h->recvtype, h->comm, &h->requestReverse);

  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Iallgather_b_finish(HandleBase* handle) {

    AMPI_Iallgather_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Iallgather_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);
    MPI_Wait(&h->requestReverse.request, MPI_STATUS_IGNORE);

    h->recvtype->getADTool().combineAdjoints(h->sendbufAdjoints, h->sendbufTotalSize, getCommSize(h->comm));
    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->recvtype->getADTool().updateAdjoints(h->sendbufIndices, h->sendbufAdjoints, h->sendbufTotalSize);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->sendbufAdjoints);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->recvbufAdjoints);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Iallgather_finish(HandleBase* handle);
  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Iallgather(const typename SENDTYPE::Type* sendbuf, int sendcount, SENDTYPE* sendtype,
                      typename RECVTYPE::Type* recvbuf, int recvcount, RECVTYPE* recvtype, AMPI_Comm comm, AMPI_Request* request) {
    int rStatus;
    if(!recvtype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Iallgather(sendbuf, sendcount, sendtype->getMpiType(), recvbuf, recvcount, recvtype->getMpiType(), comm,
                               &request->request);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Iallgather_AdjointHandle<SENDTYPE, RECVTYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(recvtype->getADTool().isHandleRequired()) {
        h = new AMPI_Iallgather_AdjointHandle<SENDTYPE, RECVTYPE>();
      }
      recvtype->getADTool().startAssembly(h);
      typename SENDTYPE::ModifiedType* sendbufMod = nullptr;
      int sendbufElements = 0;

      // compute the total size of the buffer
      if(AMPI_IN_PLACE != sendbuf) {
        sendbufElements = sendcount;
      } else {
        sendbufElements = recvcount;
      }

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->createModifiedTypeBuffer(sendbufMod, sendbufElements);
      } else {
        sendbufMod = reinterpret_cast<typename SENDTYPE::ModifiedType*>(const_cast<typename SENDTYPE::Type*>(sendbuf));
      }
      typename RECVTYPE::ModifiedType* recvbufMod = nullptr;
      int recvbufElements = 0;

      // compute the total size of the buffer
      recvbufElements = recvcount * getCommSize(comm);

      if(recvtype->isModifiedBufferRequired() ) {
        recvtype->createModifiedTypeBuffer(recvbufMod, recvbufElements);
      } else {
        recvbufMod = reinterpret_cast<typename RECVTYPE::ModifiedType*>(const_cast<typename RECVTYPE::Type*>(recvbuf));
      }

      if(sendtype->isModifiedBufferRequired()) {
        if(AMPI_IN_PLACE != sendbuf) {
          sendtype->copyIntoModifiedBuffer(sendbuf, 0, sendbufMod, 0, sendcount);
        } else {
          recvtype->copyIntoModifiedBuffer(recvbuf, recvcount * getCommRank(comm), recvbufMod, recvcount * getCommRank(comm),
                                           recvcount);
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(AMPI_IN_PLACE != sendbuf) {
          h->sendbufCount = sendtype->computeActiveElements(sendcount);
        } else {
          h->sendbufCount = recvtype->computeActiveElements(recvcount);
        }
        h->sendbufTotalSize = sendtype->computeActiveElements(sendbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->sendbufIndices, h->sendbufTotalSize);
        h->recvbufCount = recvtype->computeActiveElements(recvcount);
        h->recvbufTotalSize = recvtype->computeActiveElements(recvbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->recvbufIndices, h->recvbufTotalSize);


        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(recvtype->getADTool().isOldPrimalsRequired()) {
          recvtype->getADTool().createPassiveTypeBuffer(h->recvbufOldPrimals, h->recvbufTotalSize);
          recvtype->getValues(recvbuf, 0, h->recvbufOldPrimals, 0, recvcount * getCommSize(comm));
        }


        if(AMPI_IN_PLACE != sendbuf) {
          sendtype->getIndices(sendbuf, 0, h->sendbufIndices, 0, sendcount);
        } else {
          recvtype->getIndices(recvbuf, recvcount * getCommRank(comm), h->sendbufIndices, 0, recvcount);
        }

        if(!recvtype->isModifiedBufferRequired()) {
          recvtype->clearIndices(recvbuf, 0, recvcount * getCommSize(comm));
        }

        // pack all the variables in the handle
        h->func = AMPI_Iallgather_b<SENDTYPE, RECVTYPE>;
        h->sendcount = sendcount;
        h->sendtype = sendtype;
        h->recvcount = recvcount;
        h->recvtype = recvtype;
        h->comm = comm;
      }

      rStatus = MPI_Iallgather(sendbufMod, sendcount, sendtype->getModifiedMpiType(), recvbufMod, recvcount,
                               recvtype->getModifiedMpiType(), comm, &request->request);

      AMPI_Iallgather_AsyncHandle<SENDTYPE, RECVTYPE>* asyncHandle = new AMPI_Iallgather_AsyncHandle<SENDTYPE, RECVTYPE>();
      asyncHandle->sendbuf = sendbuf;
      asyncHandle->sendbufMod = sendbufMod;
      asyncHandle->sendcount = sendcount;
      asyncHandle->sendtype = sendtype;
      asyncHandle->recvbuf = recvbuf;
      asyncHandle->recvbufMod = recvbufMod;
      asyncHandle->recvcount = recvcount;
      asyncHandle->recvtype = recvtype;
      asyncHandle->comm = comm;
      asyncHandle->h = h;
      request->handle = asyncHandle;
      request->func = (ContinueFunction)AMPI_Iallgather_finish<SENDTYPE, RECVTYPE>;

      // create adjoint wait
      if(nullptr != h) {
        WaitHandle* waitH = new WaitHandle((ContinueFunction)AMPI_Iallgather_b_finish<SENDTYPE, RECVTYPE>, h);
        recvtype->getADTool().addToolAction(waitH);
      }
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Iallgather_finish(HandleBase* handle) {
    int rStatus = 0;

    AMPI_Iallgather_AsyncHandle<SENDTYPE, RECVTYPE>* asyncHandle =
      static_cast<AMPI_Iallgather_AsyncHandle<SENDTYPE, RECVTYPE>*>(handle);
    const  typename SENDTYPE::Type* sendbuf = asyncHandle->sendbuf;
    typename SENDTYPE::ModifiedType* sendbufMod = asyncHandle->sendbufMod;
    int sendcount = asyncHandle->sendcount;
    SENDTYPE* sendtype = asyncHandle->sendtype;
    typename RECVTYPE::Type* recvbuf = asyncHandle->recvbuf;
    typename RECVTYPE::ModifiedType* recvbufMod = asyncHandle->recvbufMod;
    int recvcount = asyncHandle->recvcount;
    RECVTYPE* recvtype = asyncHandle->recvtype;
    AMPI_Comm comm = asyncHandle->comm;
    AMPI_Request* request = asyncHandle->request;
    AMPI_Iallgather_AdjointHandle<SENDTYPE, RECVTYPE>* h = asyncHandle->h;
    MEDI_UNUSED(sendbuf); // Unused generated to ignore warnings
    MEDI_UNUSED(sendbufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(sendcount); // Unused generated to ignore warnings
    MEDI_UNUSED(sendtype); // Unused generated to ignore warnings
    MEDI_UNUSED(recvbuf); // Unused generated to ignore warnings
    MEDI_UNUSED(recvbufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(recvcount); // Unused generated to ignore warnings
    MEDI_UNUSED(recvtype); // Unused generated to ignore warnings
    MEDI_UNUSED(comm); // Unused generated to ignore warnings
    MEDI_UNUSED(request); // Unused generated to ignore warnings
    MEDI_UNUSED(h); // Unused generated to ignore warnings

    delete asyncHandle;

    if(recvtype->getADTool().isActiveType()) {

      recvtype->getADTool().addToolAction(h);

      if(recvtype->isModifiedBufferRequired()) {
        recvtype->copyFromModifiedBuffer(recvbuf, 0, recvbufMod, 0, recvcount * getCommSize(comm));
      }

      if(nullptr != h) {
        // handle the recv buffers
        recvtype->registerValue(recvbuf, 0, h->recvbufIndices, h->recvbufOldPrimals, 0, recvcount * getCommSize(comm));
      }

      recvtype->getADTool().stopAssembly(h);

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->deleteModifiedTypeBuffer(sendbufMod);
      }
      if(recvtype->isModifiedBufferRequired() ) {
        recvtype->deleteModifiedTypeBuffer(recvbufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Iallgatherv_AdjointHandle : public HandleBase {
    int sendbufTotalSize;
    typename SENDTYPE::IndexType* sendbufIndices;
    typename SENDTYPE::PassiveType* sendbufPrimals;
    /* required for async */ typename SENDTYPE::PassiveType* sendbufAdjoints;
    int sendbufCount;
    int sendcount;
    SENDTYPE* sendtype;
    int recvbufTotalSize;
    typename RECVTYPE::IndexType* recvbufIndices;
    typename RECVTYPE::PassiveType* recvbufPrimals;
    typename RECVTYPE::PassiveType* recvbufOldPrimals;
    /* required for async */ typename RECVTYPE::PassiveType* recvbufAdjoints;
    int* recvbufCount;
    int* recvbufDispls;
    const  int* recvcounts;
    const  int* displs;
    RECVTYPE* recvtype;
    AMPI_Comm comm;
    AMPI_Request requestReverse;

    ~AMPI_Iallgatherv_AdjointHandle () {
      if(nullptr != sendbufIndices) {
        sendtype->getADTool().deleteIndexTypeBuffer(sendbufIndices);
        sendbufIndices = nullptr;
      }
      if(nullptr != sendbufPrimals) {
        sendtype->getADTool().deletePassiveTypeBuffer(sendbufPrimals);
        sendbufPrimals = nullptr;
      }
      if(nullptr != recvbufIndices) {
        recvtype->getADTool().deleteIndexTypeBuffer(recvbufIndices);
        recvbufIndices = nullptr;
      }
      if(nullptr != recvbufPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufPrimals);
        recvbufPrimals = nullptr;
      }
      if(nullptr != recvbufOldPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufOldPrimals);
        recvbufOldPrimals = nullptr;
      }
      if(nullptr != recvbufCount) {
        delete [] recvbufCount;
        recvbufCount = nullptr;
      }
      if(nullptr != recvbufDispls) {
        delete [] recvbufDispls;
        recvbufDispls = nullptr;
      }
    }
  };

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Iallgatherv_AsyncHandle : public HandleBase {
    const  typename SENDTYPE::Type* sendbuf;
    typename SENDTYPE::ModifiedType* sendbufMod;
    int sendcount;
    SENDTYPE* sendtype;
    typename RECVTYPE::Type* recvbuf;
    typename RECVTYPE::ModifiedType* recvbufMod;
    const int* displsMod;
    const  int* recvcounts;
    const  int* displs;
    RECVTYPE* recvtype;
    AMPI_Comm comm;
    AMPI_Request* request;
    AMPI_Iallgatherv_AdjointHandle<SENDTYPE, RECVTYPE>* h;
  };

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Iallgatherv_b(HandleBase* handle) {
    AMPI_Iallgatherv_AdjointHandle<SENDTYPE, RECVTYPE>* h =
      static_cast<AMPI_Iallgatherv_AdjointHandle<SENDTYPE, RECVTYPE>*>(handle);

    h->recvbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->recvbufAdjoints, h->recvbufTotalSize );
    // Adjoint buffers are always linear in space so we can accesses them in one sweep
    h->recvtype->getADTool().getAdjoints(h->recvbufIndices, h->recvbufAdjoints, h->recvbufTotalSize);

    if(h->recvtype->getADTool().isOldPrimalsRequired()) {
      h->recvtype->getADTool().setReverseValues(h->recvbufIndices, h->recvbufOldPrimals, h->recvbufTotalSize);
    }
    h->sendbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->sendbufAdjoints, h->sendbufTotalSize * getCommSize(h->comm));

    AMPI_Iallgatherv_adj<SENDTYPE, RECVTYPE>(h->sendbufAdjoints, h->sendbufCount, h->sendcount, h->sendtype,
        h->recvbufAdjoints, h->recvbufCount, h->recvbufDispls, h->recvcounts, h->displs, h->recvtype, h->comm,
        &h->requestReverse);

  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Iallgatherv_b_finish(HandleBase* handle) {

    AMPI_Iallgatherv_AdjointHandle<SENDTYPE, RECVTYPE>* h =
      static_cast<AMPI_Iallgatherv_AdjointHandle<SENDTYPE, RECVTYPE>*>(handle);
    MPI_Wait(&h->requestReverse.request, MPI_STATUS_IGNORE);

    h->recvtype->getADTool().combineAdjoints(h->sendbufAdjoints, h->sendbufTotalSize, getCommSize(h->comm));
    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->recvtype->getADTool().updateAdjoints(h->sendbufIndices, h->sendbufAdjoints, h->sendbufTotalSize);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->sendbufAdjoints);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->recvbufAdjoints);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Iallgatherv_finish(HandleBase* handle);
  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Iallgatherv(const typename SENDTYPE::Type* sendbuf, int sendcount, SENDTYPE* sendtype,
                       typename RECVTYPE::Type* recvbuf, const int* recvcounts, const int* displs, RECVTYPE* recvtype, AMPI_Comm comm,
                       AMPI_Request* request) {
    int rStatus;
    if(!recvtype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Iallgatherv(sendbuf, sendcount, sendtype->getMpiType(), recvbuf, recvcounts, displs,
                                recvtype->getMpiType(), comm, &request->request);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Iallgatherv_AdjointHandle<SENDTYPE, RECVTYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(recvtype->getADTool().isHandleRequired()) {
        h = new AMPI_Iallgatherv_AdjointHandle<SENDTYPE, RECVTYPE>();
      }
      recvtype->getADTool().startAssembly(h);
      const int* displsMod = displs;
      int displsTotalSize = 0;
      if(nullptr != displs) {
        displsTotalSize = computeDisplacementsTotalSize(recvcounts, getCommSize(comm));
        if(recvtype->isModifiedBufferRequired()) {
          displsMod = createLinearDisplacements(recvcounts, getCommSize(comm));
        }
      }
      typename SENDTYPE::ModifiedType* sendbufMod = nullptr;
      int sendbufElements = 0;

      // compute the total size of the buffer
      if(AMPI_IN_PLACE != sendbuf) {
        sendbufElements = sendcount;
      } else {
        sendbufElements = recvcounts[getCommRank(comm)];
      }

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->createModifiedTypeBuffer(sendbufMod, sendbufElements);
      } else {
        sendbufMod = reinterpret_cast<typename SENDTYPE::ModifiedType*>(const_cast<typename SENDTYPE::Type*>(sendbuf));
      }
      typename RECVTYPE::ModifiedType* recvbufMod = nullptr;
      int recvbufElements = 0;

      // compute the total size of the buffer
      recvbufElements = displsTotalSize;

      if(recvtype->isModifiedBufferRequired() ) {
        recvtype->createModifiedTypeBuffer(recvbufMod, recvbufElements);
      } else {
        recvbufMod = reinterpret_cast<typename RECVTYPE::ModifiedType*>(const_cast<typename RECVTYPE::Type*>(recvbuf));
      }

      if(sendtype->isModifiedBufferRequired()) {
        if(AMPI_IN_PLACE != sendbuf) {
          sendtype->copyIntoModifiedBuffer(sendbuf, 0, sendbufMod, 0, sendcount);
        } else {
          {
            const int rank = getCommRank(comm);
            recvtype->copyIntoModifiedBuffer(recvbuf, displs[rank], recvbufMod, displsMod[rank], recvcounts[rank]);
          }
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(AMPI_IN_PLACE != sendbuf) {
          h->sendbufCount = sendtype->computeActiveElements(sendcount);
        } else {
          h->sendbufCount = recvtype->computeActiveElements(recvcounts[getCommRank(comm)]);
        }
        h->sendbufTotalSize = sendtype->computeActiveElements(sendbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->sendbufIndices, h->sendbufTotalSize);
        createLinearIndexDisplacements(h->recvbufCount, h->recvbufDispls, recvcounts, getCommSize(comm), recvtype);
        h->recvbufTotalSize = recvtype->computeActiveElements(recvbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->recvbufIndices, h->recvbufTotalSize);


        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(recvtype->getADTool().isOldPrimalsRequired()) {
          recvtype->getADTool().createPassiveTypeBuffer(h->recvbufOldPrimals, h->recvbufTotalSize);
          for(int i = 0; i < getCommSize(comm); ++i) {
            recvtype->getValues(recvbuf, displs[i], h->recvbufOldPrimals, displsMod[i], recvcounts[i]);
          }
        }


        if(AMPI_IN_PLACE != sendbuf) {
          sendtype->getIndices(sendbuf, 0, h->sendbufIndices, 0, sendcount);
        } else {
          {
            const int rank = getCommRank(comm);
            recvtype->getIndices(recvbuf, displs[rank], h->sendbufIndices, 0, recvcounts[rank]);
          }
        }

        if(!recvtype->isModifiedBufferRequired()) {
          for(int i = 0; i < getCommSize(comm); ++i) {
            recvtype->clearIndices(recvbuf, displs[i], recvcounts[i]);
          }
        }

        // pack all the variables in the handle
        h->func = AMPI_Iallgatherv_b<SENDTYPE, RECVTYPE>;
        h->sendcount = sendcount;
        h->sendtype = sendtype;
        h->recvcounts = recvcounts;
        h->displs = displs;
        h->recvtype = recvtype;
        h->comm = comm;
      }

      rStatus = MPI_Iallgatherv(sendbufMod, sendcount, sendtype->getModifiedMpiType(), recvbufMod, recvcounts, displsMod,
                                recvtype->getModifiedMpiType(), comm, &request->request);

      AMPI_Iallgatherv_AsyncHandle<SENDTYPE, RECVTYPE>* asyncHandle = new AMPI_Iallgatherv_AsyncHandle<SENDTYPE, RECVTYPE>();
      asyncHandle->sendbuf = sendbuf;
      asyncHandle->sendbufMod = sendbufMod;
      asyncHandle->sendcount = sendcount;
      asyncHandle->sendtype = sendtype;
      asyncHandle->recvbuf = recvbuf;
      asyncHandle->recvbufMod = recvbufMod;
      asyncHandle->displsMod = displsMod;
      asyncHandle->recvcounts = recvcounts;
      asyncHandle->displs = displs;
      asyncHandle->recvtype = recvtype;
      asyncHandle->comm = comm;
      asyncHandle->h = h;
      request->handle = asyncHandle;
      request->func = (ContinueFunction)AMPI_Iallgatherv_finish<SENDTYPE, RECVTYPE>;

      // create adjoint wait
      if(nullptr != h) {
        WaitHandle* waitH = new WaitHandle((ContinueFunction)AMPI_Iallgatherv_b_finish<SENDTYPE, RECVTYPE>, h);
        recvtype->getADTool().addToolAction(waitH);
      }
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Iallgatherv_finish(HandleBase* handle) {
    int rStatus = 0;

    AMPI_Iallgatherv_AsyncHandle<SENDTYPE, RECVTYPE>* asyncHandle =
      static_cast<AMPI_Iallgatherv_AsyncHandle<SENDTYPE, RECVTYPE>*>(handle);
    const  typename SENDTYPE::Type* sendbuf = asyncHandle->sendbuf;
    typename SENDTYPE::ModifiedType* sendbufMod = asyncHandle->sendbufMod;
    int sendcount = asyncHandle->sendcount;
    SENDTYPE* sendtype = asyncHandle->sendtype;
    typename RECVTYPE::Type* recvbuf = asyncHandle->recvbuf;
    typename RECVTYPE::ModifiedType* recvbufMod = asyncHandle->recvbufMod;
    const int* displsMod = asyncHandle->displsMod;
    const  int* recvcounts = asyncHandle->recvcounts;
    const  int* displs = asyncHandle->displs;
    RECVTYPE* recvtype = asyncHandle->recvtype;
    AMPI_Comm comm = asyncHandle->comm;
    AMPI_Request* request = asyncHandle->request;
    AMPI_Iallgatherv_AdjointHandle<SENDTYPE, RECVTYPE>* h = asyncHandle->h;
    MEDI_UNUSED(sendbuf); // Unused generated to ignore warnings
    MEDI_UNUSED(sendbufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(sendcount); // Unused generated to ignore warnings
    MEDI_UNUSED(sendtype); // Unused generated to ignore warnings
    MEDI_UNUSED(recvbuf); // Unused generated to ignore warnings
    MEDI_UNUSED(recvbufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(displsMod); // Unused generated to ignore warnings
    MEDI_UNUSED(recvcounts); // Unused generated to ignore warnings
    MEDI_UNUSED(displs); // Unused generated to ignore warnings
    MEDI_UNUSED(recvtype); // Unused generated to ignore warnings
    MEDI_UNUSED(comm); // Unused generated to ignore warnings
    MEDI_UNUSED(request); // Unused generated to ignore warnings
    MEDI_UNUSED(h); // Unused generated to ignore warnings

    delete asyncHandle;

    if(recvtype->getADTool().isActiveType()) {

      recvtype->getADTool().addToolAction(h);

      if(recvtype->isModifiedBufferRequired()) {
        for(int i = 0; i < getCommSize(comm); ++i) {
          recvtype->copyFromModifiedBuffer(recvbuf, displs[i], recvbufMod, displsMod[i], recvcounts[i]);
        }
      }

      if(nullptr != h) {
        // handle the recv buffers
        for(int i = 0; i < getCommSize(comm); ++i) {
          recvtype->registerValue(recvbuf, displs[i], h->recvbufIndices, h->recvbufOldPrimals, displsMod[i], recvcounts[i]);
        }
      }

      recvtype->getADTool().stopAssembly(h);
      if(recvtype->isModifiedBufferRequired()) {
        delete [] displsMod;
      }

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->deleteModifiedTypeBuffer(sendbufMod);
      }
      if(recvtype->isModifiedBufferRequired() ) {
        recvtype->deleteModifiedTypeBuffer(recvbufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename DATATYPE>
  struct AMPI_Iallreduce_global_AdjointHandle : public HandleBase {
    int sendbufTotalSize;
    typename DATATYPE::IndexType* sendbufIndices;
    typename DATATYPE::PassiveType* sendbufPrimals;
    /* required for async */ typename DATATYPE::PassiveType* sendbufAdjoints;
    int sendbufCount;
    int recvbufTotalSize;
    typename DATATYPE::IndexType* recvbufIndices;
    typename DATATYPE::PassiveType* recvbufPrimals;
    typename DATATYPE::PassiveType* recvbufOldPrimals;
    /* required for async */ typename DATATYPE::PassiveType* recvbufAdjoints;
    int recvbufCount;
    int count;
    DATATYPE* datatype;
    AMPI_Op op;
    AMPI_Comm comm;
    AMPI_Request requestReverse;

    ~AMPI_Iallreduce_global_AdjointHandle () {
      if(nullptr != sendbufIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(sendbufIndices);
        sendbufIndices = nullptr;
      }
      if(nullptr != sendbufPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(sendbufPrimals);
        sendbufPrimals = nullptr;
      }
      if(nullptr != recvbufIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(recvbufIndices);
        recvbufIndices = nullptr;
      }
      if(nullptr != recvbufPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(recvbufPrimals);
        recvbufPrimals = nullptr;
      }
      if(nullptr != recvbufOldPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(recvbufOldPrimals);
        recvbufOldPrimals = nullptr;
      }
    }
  };

  template<typename DATATYPE>
  struct AMPI_Iallreduce_global_AsyncHandle : public HandleBase {
    const  typename DATATYPE::Type* sendbuf;
    typename DATATYPE::ModifiedType* sendbufMod;
    typename DATATYPE::Type* recvbuf;
    typename DATATYPE::ModifiedType* recvbufMod;
    int count;
    DATATYPE* datatype;
    AMPI_Op op;
    AMPI_Comm comm;
    AMPI_Request* request;
    AMPI_Iallreduce_global_AdjointHandle<DATATYPE>* h;
  };

  template<typename DATATYPE>
  void AMPI_Iallreduce_global_b(HandleBase* handle) {
    AMPI_Iallreduce_global_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Iallreduce_global_AdjointHandle<DATATYPE>*>
        (handle);

    h->recvbufAdjoints = nullptr;
    h->datatype->getADTool().createAdjointTypeBuffer(h->recvbufAdjoints, h->recvbufTotalSize );
    // Adjoint buffers are always linear in space so we can accesses them in one sweep
    h->datatype->getADTool().getAdjoints(h->recvbufIndices, h->recvbufAdjoints, h->recvbufTotalSize);

    h->op.preAdjointOperation(h->recvbufAdjoints, h->recvbufPrimals, h->recvbufCount);
    if(h->datatype->getADTool().isOldPrimalsRequired()) {
      h->datatype->getADTool().setReverseValues(h->recvbufIndices, h->recvbufOldPrimals, h->recvbufTotalSize);
    }
    h->sendbufAdjoints = nullptr;
    h->datatype->getADTool().createAdjointTypeBuffer(h->sendbufAdjoints, h->sendbufTotalSize * getCommSize(h->comm));

    AMPI_Iallreduce_global_adj<DATATYPE>(h->sendbufAdjoints, h->sendbufCount, h->recvbufAdjoints, h->recvbufCount, h->count,
                                         h->datatype, h->op, h->comm, &h->requestReverse);

  }

  template<typename DATATYPE>
  void AMPI_Iallreduce_global_b_finish(HandleBase* handle) {

    AMPI_Iallreduce_global_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Iallreduce_global_AdjointHandle<DATATYPE>*>
        (handle);
    MPI_Wait(&h->requestReverse.request, MPI_STATUS_IGNORE);

    h->datatype->getADTool().combineAdjoints(h->sendbufAdjoints, h->sendbufTotalSize, getCommSize(h->comm));
    // the primals of the recive buffer are always given to the function. The operator should ignore them if not needed.
    // The wrapper functions make sure that for operators that need the primals an all* action is perfomed (e.g. Allreduce instead of Reduce)
    h->op.postAdjointOperation(h->sendbufAdjoints, h->sendbufPrimals, h->recvbufPrimals, h->sendbufTotalSize);
    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->datatype->getADTool().updateAdjoints(h->sendbufIndices, h->sendbufAdjoints, h->sendbufTotalSize);
    h->datatype->getADTool().deleteAdjointTypeBuffer(h->sendbufAdjoints);
    h->datatype->getADTool().deleteAdjointTypeBuffer(h->recvbufAdjoints);
  }

  template<typename DATATYPE>
  int AMPI_Iallreduce_global_finish(HandleBase* handle);
  template<typename DATATYPE>
  int AMPI_Iallreduce_global(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count,
                             DATATYPE* datatype, AMPI_Op op, AMPI_Comm comm, AMPI_Request* request) {
    int rStatus;
    if(!datatype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Iallreduce(sendbuf, recvbuf, count, datatype->getMpiType(), op.primalFunction, comm, &request->request);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Iallreduce_global_AdjointHandle<DATATYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(datatype->getADTool().isHandleRequired()) {
        h = new AMPI_Iallreduce_global_AdjointHandle<DATATYPE>();
      }
      datatype->getADTool().startAssembly(h);
      typename DATATYPE::ModifiedType* sendbufMod = nullptr;
      int sendbufElements = 0;

      // compute the total size of the buffer
      if(AMPI_IN_PLACE != sendbuf) {
        sendbufElements = count;
      } else {
        sendbufElements = count;
      }

      if(datatype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        datatype->createModifiedTypeBuffer(sendbufMod, sendbufElements);
      } else {
        sendbufMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(sendbuf));
      }
      typename DATATYPE::ModifiedType* recvbufMod = nullptr;
      int recvbufElements = 0;

      // compute the total size of the buffer
      recvbufElements = count;

      if(datatype->isModifiedBufferRequired() ) {
        datatype->createModifiedTypeBuffer(recvbufMod, recvbufElements);
      } else {
        recvbufMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(recvbuf));
      }

      if(datatype->isModifiedBufferRequired()) {
        if(AMPI_IN_PLACE != sendbuf) {
          datatype->copyIntoModifiedBuffer(sendbuf, 0, sendbufMod, 0, count);
        } else {
          datatype->copyIntoModifiedBuffer(recvbuf, 0, recvbufMod, 0, count);
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(AMPI_IN_PLACE != sendbuf) {
          h->sendbufCount = datatype->computeActiveElements(count);
        } else {
          h->sendbufCount = datatype->computeActiveElements(count);
        }
        h->sendbufTotalSize = datatype->computeActiveElements(sendbufElements);
        datatype->getADTool().createIndexTypeBuffer(h->sendbufIndices, h->sendbufTotalSize);
        h->recvbufCount = datatype->computeActiveElements(count);
        h->recvbufTotalSize = datatype->computeActiveElements(recvbufElements);
        datatype->getADTool().createIndexTypeBuffer(h->recvbufIndices, h->recvbufTotalSize);

        // extract the primal values for the operator if required
        if(op.requiresPrimal) {
          datatype->getADTool().createPassiveTypeBuffer(h->sendbufPrimals, h->sendbufTotalSize);
          if(AMPI_IN_PLACE != sendbuf) {
            datatype->getValues(sendbuf, 0, h->sendbufPrimals, 0, count);
          } else {
            datatype->getValues(recvbuf, 0, h->sendbufPrimals, 0, count);
          }
        }

        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(datatype->getADTool().isOldPrimalsRequired()) {
          datatype->getADTool().createPassiveTypeBuffer(h->recvbufOldPrimals, h->recvbufTotalSize);
          datatype->getValues(recvbuf, 0, h->recvbufOldPrimals, 0, count);
        }


        if(AMPI_IN_PLACE != sendbuf) {
          datatype->getIndices(sendbuf, 0, h->sendbufIndices, 0, count);
        } else {
          datatype->getIndices(recvbuf, 0, h->sendbufIndices, 0, count);
        }

        if(!datatype->isModifiedBufferRequired()) {
          datatype->clearIndices(recvbuf, 0, count);
        }

        // pack all the variables in the handle
        h->func = AMPI_Iallreduce_global_b<DATATYPE>;
        h->count = count;
        h->datatype = datatype;
        h->op = op;
        h->comm = comm;
      }

      rStatus = MPI_Iallreduce(sendbufMod, recvbufMod, count, datatype->getModifiedMpiType(), op.modifiedPrimalFunction, comm,
                               &request->request);

      AMPI_Iallreduce_global_AsyncHandle<DATATYPE>* asyncHandle = new AMPI_Iallreduce_global_AsyncHandle<DATATYPE>();
      asyncHandle->sendbuf = sendbuf;
      asyncHandle->sendbufMod = sendbufMod;
      asyncHandle->recvbuf = recvbuf;
      asyncHandle->recvbufMod = recvbufMod;
      asyncHandle->count = count;
      asyncHandle->datatype = datatype;
      asyncHandle->op = op;
      asyncHandle->comm = comm;
      asyncHandle->h = h;
      request->handle = asyncHandle;
      request->func = (ContinueFunction)AMPI_Iallreduce_global_finish<DATATYPE>;

      // create adjoint wait
      if(nullptr != h) {
        WaitHandle* waitH = new WaitHandle((ContinueFunction)AMPI_Iallreduce_global_b_finish<DATATYPE>, h);
        datatype->getADTool().addToolAction(waitH);
      }
    }

    return rStatus;
  }

  template<typename DATATYPE>
  int AMPI_Iallreduce_global_finish(HandleBase* handle) {
    int rStatus = 0;

    AMPI_Iallreduce_global_AsyncHandle<DATATYPE>* asyncHandle = static_cast<AMPI_Iallreduce_global_AsyncHandle<DATATYPE>*>
        (handle);
    const  typename DATATYPE::Type* sendbuf = asyncHandle->sendbuf;
    typename DATATYPE::ModifiedType* sendbufMod = asyncHandle->sendbufMod;
    typename DATATYPE::Type* recvbuf = asyncHandle->recvbuf;
    typename DATATYPE::ModifiedType* recvbufMod = asyncHandle->recvbufMod;
    int count = asyncHandle->count;
    DATATYPE* datatype = asyncHandle->datatype;
    AMPI_Op op = asyncHandle->op;
    AMPI_Comm comm = asyncHandle->comm;
    AMPI_Request* request = asyncHandle->request;
    AMPI_Iallreduce_global_AdjointHandle<DATATYPE>* h = asyncHandle->h;
    MEDI_UNUSED(sendbuf); // Unused generated to ignore warnings
    MEDI_UNUSED(sendbufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(recvbuf); // Unused generated to ignore warnings
    MEDI_UNUSED(recvbufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(count); // Unused generated to ignore warnings
    MEDI_UNUSED(datatype); // Unused generated to ignore warnings
    MEDI_UNUSED(op); // Unused generated to ignore warnings
    MEDI_UNUSED(comm); // Unused generated to ignore warnings
    MEDI_UNUSED(request); // Unused generated to ignore warnings
    MEDI_UNUSED(h); // Unused generated to ignore warnings

    delete asyncHandle;

    if(datatype->getADTool().isActiveType()) {

      datatype->getADTool().addToolAction(h);

      if(datatype->isModifiedBufferRequired()) {
        datatype->copyFromModifiedBuffer(recvbuf, 0, recvbufMod, 0, count);
      }

      if(nullptr != h) {
        // handle the recv buffers
        datatype->registerValue(recvbuf, 0, h->recvbufIndices, h->recvbufOldPrimals, 0, count);
      }
      // extract the primal values for the operator if required
      if(nullptr != h && op.requiresPrimal) {
        datatype->getADTool().createPassiveTypeBuffer(h->recvbufPrimals, h->recvbufTotalSize);
        datatype->getValues(recvbuf, 0, h->recvbufPrimals, 0, count);
      }

      datatype->getADTool().stopAssembly(h);

      if(datatype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        datatype->deleteModifiedTypeBuffer(sendbufMod);
      }
      if(datatype->isModifiedBufferRequired() ) {
        datatype->deleteModifiedTypeBuffer(recvbufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Ialltoall_AdjointHandle : public HandleBase {
    int sendbufTotalSize;
    typename SENDTYPE::IndexType* sendbufIndices;
    typename SENDTYPE::PassiveType* sendbufPrimals;
    /* required for async */ typename SENDTYPE::PassiveType* sendbufAdjoints;
    int sendbufCount;
    int sendcount;
    SENDTYPE* sendtype;
    int recvbufTotalSize;
    typename RECVTYPE::IndexType* recvbufIndices;
    typename RECVTYPE::PassiveType* recvbufPrimals;
    typename RECVTYPE::PassiveType* recvbufOldPrimals;
    /* required for async */ typename RECVTYPE::PassiveType* recvbufAdjoints;
    int recvbufCount;
    int recvcount;
    RECVTYPE* recvtype;
    AMPI_Comm comm;
    AMPI_Request requestReverse;

    ~AMPI_Ialltoall_AdjointHandle () {
      if(nullptr != sendbufIndices) {
        sendtype->getADTool().deleteIndexTypeBuffer(sendbufIndices);
        sendbufIndices = nullptr;
      }
      if(nullptr != sendbufPrimals) {
        sendtype->getADTool().deletePassiveTypeBuffer(sendbufPrimals);
        sendbufPrimals = nullptr;
      }
      if(nullptr != recvbufIndices) {
        recvtype->getADTool().deleteIndexTypeBuffer(recvbufIndices);
        recvbufIndices = nullptr;
      }
      if(nullptr != recvbufPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufPrimals);
        recvbufPrimals = nullptr;
      }
      if(nullptr != recvbufOldPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufOldPrimals);
        recvbufOldPrimals = nullptr;
      }
    }
  };

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Ialltoall_AsyncHandle : public HandleBase {
    const  typename SENDTYPE::Type* sendbuf;
    typename SENDTYPE::ModifiedType* sendbufMod;
    int sendcount;
    SENDTYPE* sendtype;
    typename RECVTYPE::Type* recvbuf;
    typename RECVTYPE::ModifiedType* recvbufMod;
    int recvcount;
    RECVTYPE* recvtype;
    AMPI_Comm comm;
    AMPI_Request* request;
    AMPI_Ialltoall_AdjointHandle<SENDTYPE, RECVTYPE>* h;
  };

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Ialltoall_b(HandleBase* handle) {
    AMPI_Ialltoall_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Ialltoall_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);

    h->recvbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->recvbufAdjoints, h->recvbufTotalSize );
    // Adjoint buffers are always linear in space so we can accesses them in one sweep
    h->recvtype->getADTool().getAdjoints(h->recvbufIndices, h->recvbufAdjoints, h->recvbufTotalSize);

    if(h->recvtype->getADTool().isOldPrimalsRequired()) {
      h->recvtype->getADTool().setReverseValues(h->recvbufIndices, h->recvbufOldPrimals, h->recvbufTotalSize);
    }
    h->sendbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->sendbufAdjoints, h->sendbufTotalSize );

    AMPI_Ialltoall_adj<SENDTYPE, RECVTYPE>(h->sendbufAdjoints, h->sendbufCount, h->sendcount, h->sendtype,
                                           h->recvbufAdjoints, h->recvbufCount, h->recvcount, h->recvtype, h->comm, &h->requestReverse);

  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Ialltoall_b_finish(HandleBase* handle) {

    AMPI_Ialltoall_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Ialltoall_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);
    MPI_Wait(&h->requestReverse.request, MPI_STATUS_IGNORE);

    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->recvtype->getADTool().updateAdjoints(h->sendbufIndices, h->sendbufAdjoints, h->sendbufTotalSize);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->sendbufAdjoints);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->recvbufAdjoints);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Ialltoall_finish(HandleBase* handle);
  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Ialltoall(const typename SENDTYPE::Type* sendbuf, int sendcount, SENDTYPE* sendtype,
                     typename RECVTYPE::Type* recvbuf, int recvcount, RECVTYPE* recvtype, AMPI_Comm comm, AMPI_Request* request) {
    int rStatus;
    if(!recvtype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Ialltoall(sendbuf, sendcount, sendtype->getMpiType(), recvbuf, recvcount, recvtype->getMpiType(), comm,
                              &request->request);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Ialltoall_AdjointHandle<SENDTYPE, RECVTYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(recvtype->getADTool().isHandleRequired()) {
        h = new AMPI_Ialltoall_AdjointHandle<SENDTYPE, RECVTYPE>();
      }
      recvtype->getADTool().startAssembly(h);
      typename SENDTYPE::ModifiedType* sendbufMod = nullptr;
      int sendbufElements = 0;

      // compute the total size of the buffer
      if(AMPI_IN_PLACE != sendbuf) {
        sendbufElements = sendcount * getCommSize(comm);
      } else {
        sendbufElements = recvcount * getCommSize(comm);
      }

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->createModifiedTypeBuffer(sendbufMod, sendbufElements);
      } else {
        sendbufMod = reinterpret_cast<typename SENDTYPE::ModifiedType*>(const_cast<typename SENDTYPE::Type*>(sendbuf));
      }
      typename RECVTYPE::ModifiedType* recvbufMod = nullptr;
      int recvbufElements = 0;

      // compute the total size of the buffer
      recvbufElements = recvcount * getCommSize(comm);

      if(recvtype->isModifiedBufferRequired() ) {
        recvtype->createModifiedTypeBuffer(recvbufMod, recvbufElements);
      } else {
        recvbufMod = reinterpret_cast<typename RECVTYPE::ModifiedType*>(const_cast<typename RECVTYPE::Type*>(recvbuf));
      }

      if(sendtype->isModifiedBufferRequired()) {
        if(AMPI_IN_PLACE != sendbuf) {
          sendtype->copyIntoModifiedBuffer(sendbuf, 0, sendbufMod, 0, sendcount * getCommSize(comm));
        } else {
          recvtype->copyIntoModifiedBuffer(recvbuf, 0, recvbufMod, 0, recvcount * getCommSize(comm));
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(AMPI_IN_PLACE != sendbuf) {
          h->sendbufCount = sendtype->computeActiveElements(sendcount);
        } else {
          h->sendbufCount = recvtype->computeActiveElements(recvcount);
        }
        h->sendbufTotalSize = sendtype->computeActiveElements(sendbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->sendbufIndices, h->sendbufTotalSize);
        h->recvbufCount = recvtype->computeActiveElements(recvcount);
        h->recvbufTotalSize = recvtype->computeActiveElements(recvbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->recvbufIndices, h->recvbufTotalSize);


        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(recvtype->getADTool().isOldPrimalsRequired()) {
          recvtype->getADTool().createPassiveTypeBuffer(h->recvbufOldPrimals, h->recvbufTotalSize);
          recvtype->getValues(recvbuf, 0, h->recvbufOldPrimals, 0, recvcount * getCommSize(comm));
        }


        if(AMPI_IN_PLACE != sendbuf) {
          sendtype->getIndices(sendbuf, 0, h->sendbufIndices, 0, sendcount * getCommSize(comm));
        } else {
          recvtype->getIndices(recvbuf, 0, h->sendbufIndices, 0, recvcount * getCommSize(comm));
        }

        if(!recvtype->isModifiedBufferRequired()) {
          recvtype->clearIndices(recvbuf, 0, recvcount * getCommSize(comm));
        }

        // pack all the variables in the handle
        h->func = AMPI_Ialltoall_b<SENDTYPE, RECVTYPE>;
        h->sendcount = sendcount;
        h->sendtype = sendtype;
        h->recvcount = recvcount;
        h->recvtype = recvtype;
        h->comm = comm;
      }

      rStatus = MPI_Ialltoall(sendbufMod, sendcount, sendtype->getModifiedMpiType(), recvbufMod, recvcount,
                              recvtype->getModifiedMpiType(), comm, &request->request);

      AMPI_Ialltoall_AsyncHandle<SENDTYPE, RECVTYPE>* asyncHandle = new AMPI_Ialltoall_AsyncHandle<SENDTYPE, RECVTYPE>();
      asyncHandle->sendbuf = sendbuf;
      asyncHandle->sendbufMod = sendbufMod;
      asyncHandle->sendcount = sendcount;
      asyncHandle->sendtype = sendtype;
      asyncHandle->recvbuf = recvbuf;
      asyncHandle->recvbufMod = recvbufMod;
      asyncHandle->recvcount = recvcount;
      asyncHandle->recvtype = recvtype;
      asyncHandle->comm = comm;
      asyncHandle->h = h;
      request->handle = asyncHandle;
      request->func = (ContinueFunction)AMPI_Ialltoall_finish<SENDTYPE, RECVTYPE>;

      // create adjoint wait
      if(nullptr != h) {
        WaitHandle* waitH = new WaitHandle((ContinueFunction)AMPI_Ialltoall_b_finish<SENDTYPE, RECVTYPE>, h);
        recvtype->getADTool().addToolAction(waitH);
      }
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Ialltoall_finish(HandleBase* handle) {
    int rStatus = 0;

    AMPI_Ialltoall_AsyncHandle<SENDTYPE, RECVTYPE>* asyncHandle =
      static_cast<AMPI_Ialltoall_AsyncHandle<SENDTYPE, RECVTYPE>*>(handle);
    const  typename SENDTYPE::Type* sendbuf = asyncHandle->sendbuf;
    typename SENDTYPE::ModifiedType* sendbufMod = asyncHandle->sendbufMod;
    int sendcount = asyncHandle->sendcount;
    SENDTYPE* sendtype = asyncHandle->sendtype;
    typename RECVTYPE::Type* recvbuf = asyncHandle->recvbuf;
    typename RECVTYPE::ModifiedType* recvbufMod = asyncHandle->recvbufMod;
    int recvcount = asyncHandle->recvcount;
    RECVTYPE* recvtype = asyncHandle->recvtype;
    AMPI_Comm comm = asyncHandle->comm;
    AMPI_Request* request = asyncHandle->request;
    AMPI_Ialltoall_AdjointHandle<SENDTYPE, RECVTYPE>* h = asyncHandle->h;
    MEDI_UNUSED(sendbuf); // Unused generated to ignore warnings
    MEDI_UNUSED(sendbufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(sendcount); // Unused generated to ignore warnings
    MEDI_UNUSED(sendtype); // Unused generated to ignore warnings
    MEDI_UNUSED(recvbuf); // Unused generated to ignore warnings
    MEDI_UNUSED(recvbufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(recvcount); // Unused generated to ignore warnings
    MEDI_UNUSED(recvtype); // Unused generated to ignore warnings
    MEDI_UNUSED(comm); // Unused generated to ignore warnings
    MEDI_UNUSED(request); // Unused generated to ignore warnings
    MEDI_UNUSED(h); // Unused generated to ignore warnings

    delete asyncHandle;

    if(recvtype->getADTool().isActiveType()) {

      recvtype->getADTool().addToolAction(h);

      if(recvtype->isModifiedBufferRequired()) {
        recvtype->copyFromModifiedBuffer(recvbuf, 0, recvbufMod, 0, recvcount * getCommSize(comm));
      }

      if(nullptr != h) {
        // handle the recv buffers
        recvtype->registerValue(recvbuf, 0, h->recvbufIndices, h->recvbufOldPrimals, 0, recvcount * getCommSize(comm));
      }

      recvtype->getADTool().stopAssembly(h);

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->deleteModifiedTypeBuffer(sendbufMod);
      }
      if(recvtype->isModifiedBufferRequired() ) {
        recvtype->deleteModifiedTypeBuffer(recvbufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Ialltoallv_AdjointHandle : public HandleBase {
    int sendbufTotalSize;
    typename SENDTYPE::IndexType* sendbufIndices;
    typename SENDTYPE::PassiveType* sendbufPrimals;
    /* required for async */ typename SENDTYPE::PassiveType* sendbufAdjoints;
    int* sendbufCount;
    int* sendbufDispls;
    const  int* sendcounts;
    const  int* sdispls;
    SENDTYPE* sendtype;
    int recvbufTotalSize;
    typename RECVTYPE::IndexType* recvbufIndices;
    typename RECVTYPE::PassiveType* recvbufPrimals;
    typename RECVTYPE::PassiveType* recvbufOldPrimals;
    /* required for async */ typename RECVTYPE::PassiveType* recvbufAdjoints;
    int* recvbufCount;
    int* recvbufDispls;
    const  int* recvcounts;
    const  int* rdispls;
    RECVTYPE* recvtype;
    AMPI_Comm comm;
    AMPI_Request requestReverse;

    ~AMPI_Ialltoallv_AdjointHandle () {
      if(nullptr != sendbufIndices) {
        sendtype->getADTool().deleteIndexTypeBuffer(sendbufIndices);
        sendbufIndices = nullptr;
      }
      if(nullptr != sendbufPrimals) {
        sendtype->getADTool().deletePassiveTypeBuffer(sendbufPrimals);
        sendbufPrimals = nullptr;
      }
      if(nullptr != sendbufCount) {
        delete [] sendbufCount;
        sendbufCount = nullptr;
      }
      if(nullptr != sendbufDispls) {
        delete [] sendbufDispls;
        sendbufDispls = nullptr;
      }
      if(nullptr != recvbufIndices) {
        recvtype->getADTool().deleteIndexTypeBuffer(recvbufIndices);
        recvbufIndices = nullptr;
      }
      if(nullptr != recvbufPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufPrimals);
        recvbufPrimals = nullptr;
      }
      if(nullptr != recvbufOldPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufOldPrimals);
        recvbufOldPrimals = nullptr;
      }
      if(nullptr != recvbufCount) {
        delete [] recvbufCount;
        recvbufCount = nullptr;
      }
      if(nullptr != recvbufDispls) {
        delete [] recvbufDispls;
        recvbufDispls = nullptr;
      }
    }
  };

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Ialltoallv_AsyncHandle : public HandleBase {
    const  typename SENDTYPE::Type* sendbuf;
    typename SENDTYPE::ModifiedType* sendbufMod;
    const int* sdisplsMod;
    const  int* sendcounts;
    const  int* sdispls;
    SENDTYPE* sendtype;
    typename RECVTYPE::Type* recvbuf;
    typename RECVTYPE::ModifiedType* recvbufMod;
    const int* rdisplsMod;
    const  int* recvcounts;
    const  int* rdispls;
    RECVTYPE* recvtype;
    AMPI_Comm comm;
    AMPI_Request* request;
    AMPI_Ialltoallv_AdjointHandle<SENDTYPE, RECVTYPE>* h;
  };

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Ialltoallv_b(HandleBase* handle) {
    AMPI_Ialltoallv_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Ialltoallv_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);

    h->recvbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->recvbufAdjoints, h->recvbufTotalSize );
    // Adjoint buffers are always linear in space so we can accesses them in one sweep
    h->recvtype->getADTool().getAdjoints(h->recvbufIndices, h->recvbufAdjoints, h->recvbufTotalSize);

    if(h->recvtype->getADTool().isOldPrimalsRequired()) {
      h->recvtype->getADTool().setReverseValues(h->recvbufIndices, h->recvbufOldPrimals, h->recvbufTotalSize);
    }
    h->sendbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->sendbufAdjoints, h->sendbufTotalSize );

    AMPI_Ialltoallv_adj<SENDTYPE, RECVTYPE>(h->sendbufAdjoints, h->sendbufCount, h->sendbufDispls, h->sendcounts,
                                            h->sdispls, h->sendtype, h->recvbufAdjoints, h->recvbufCount, h->recvbufDispls, h->recvcounts, h->rdispls, h->recvtype,
                                            h->comm, &h->requestReverse);

  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Ialltoallv_b_finish(HandleBase* handle) {

    AMPI_Ialltoallv_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Ialltoallv_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);
    MPI_Wait(&h->requestReverse.request, MPI_STATUS_IGNORE);

    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->recvtype->getADTool().updateAdjoints(h->sendbufIndices, h->sendbufAdjoints, h->sendbufTotalSize);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->sendbufAdjoints);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->recvbufAdjoints);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Ialltoallv_finish(HandleBase* handle);
  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Ialltoallv(const typename SENDTYPE::Type* sendbuf, const int* sendcounts, const int* sdispls,
                      SENDTYPE* sendtype, typename RECVTYPE::Type* recvbuf, const int* recvcounts, const int* rdispls, RECVTYPE* recvtype,
                      AMPI_Comm comm, AMPI_Request* request) {
    int rStatus;
    if(!recvtype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Ialltoallv(sendbuf, sendcounts, sdispls, sendtype->getMpiType(), recvbuf, recvcounts, rdispls,
                               recvtype->getMpiType(), comm, &request->request);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Ialltoallv_AdjointHandle<SENDTYPE, RECVTYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(recvtype->getADTool().isHandleRequired()) {
        h = new AMPI_Ialltoallv_AdjointHandle<SENDTYPE, RECVTYPE>();
      }
      recvtype->getADTool().startAssembly(h);
      const int* sdisplsMod = sdispls;
      int sdisplsTotalSize = 0;
      if(nullptr != sdispls) {
        sdisplsTotalSize = computeDisplacementsTotalSize(sendcounts, getCommSize(comm));
        if(recvtype->isModifiedBufferRequired()) {
          sdisplsMod = createLinearDisplacements(sendcounts, getCommSize(comm));
        }
      }
      const int* rdisplsMod = rdispls;
      int rdisplsTotalSize = 0;
      if(nullptr != rdispls) {
        rdisplsTotalSize = computeDisplacementsTotalSize(recvcounts, getCommSize(comm));
        if(recvtype->isModifiedBufferRequired()) {
          rdisplsMod = createLinearDisplacements(recvcounts, getCommSize(comm));
        }
      }
      typename SENDTYPE::ModifiedType* sendbufMod = nullptr;
      int sendbufElements = 0;

      // compute the total size of the buffer
      if(AMPI_IN_PLACE != sendbuf) {
        sendbufElements = sdisplsTotalSize;
      } else {
        sendbufElements = rdisplsTotalSize;
      }

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->createModifiedTypeBuffer(sendbufMod, sendbufElements);
      } else {
        sendbufMod = reinterpret_cast<typename SENDTYPE::ModifiedType*>(const_cast<typename SENDTYPE::Type*>(sendbuf));
      }
      typename RECVTYPE::ModifiedType* recvbufMod = nullptr;
      int recvbufElements = 0;

      // compute the total size of the buffer
      recvbufElements = rdisplsTotalSize;

      if(recvtype->isModifiedBufferRequired() ) {
        recvtype->createModifiedTypeBuffer(recvbufMod, recvbufElements);
      } else {
        recvbufMod = reinterpret_cast<typename RECVTYPE::ModifiedType*>(const_cast<typename RECVTYPE::Type*>(recvbuf));
      }

      if(sendtype->isModifiedBufferRequired()) {
        if(AMPI_IN_PLACE != sendbuf) {
          for(int i = 0; i < getCommSize(comm); ++i) {
            sendtype->copyIntoModifiedBuffer(sendbuf, sdispls[i], sendbufMod, sdisplsMod[i], sendcounts[i]);
          }
        } else {
          for(int i = 0; i < getCommSize(comm); ++i) {
            recvtype->copyIntoModifiedBuffer(recvbuf, rdispls[i], recvbufMod, rdisplsMod[i], recvcounts[i]);
          }
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(AMPI_IN_PLACE != sendbuf) {
          createLinearIndexDisplacements(h->sendbufCount, h->sendbufDispls, sendcounts, getCommSize(comm), sendtype);
        } else {
          createLinearIndexDisplacements(h->sendbufCount, h->sendbufDispls, recvcounts, getCommSize(comm), recvtype);
        }
        h->sendbufTotalSize = sendtype->computeActiveElements(sendbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->sendbufIndices, h->sendbufTotalSize);
        createLinearIndexDisplacements(h->recvbufCount, h->recvbufDispls, recvcounts, getCommSize(comm), recvtype);
        h->recvbufTotalSize = recvtype->computeActiveElements(recvbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->recvbufIndices, h->recvbufTotalSize);


        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(recvtype->getADTool().isOldPrimalsRequired()) {
          recvtype->getADTool().createPassiveTypeBuffer(h->recvbufOldPrimals, h->recvbufTotalSize);
          for(int i = 0; i < getCommSize(comm); ++i) {
            recvtype->getValues(recvbuf, rdispls[i], h->recvbufOldPrimals, rdisplsMod[i], recvcounts[i]);
          }
        }


        if(AMPI_IN_PLACE != sendbuf) {
          for(int i = 0; i < getCommSize(comm); ++i) {
            sendtype->getIndices(sendbuf, sdispls[i], h->sendbufIndices, sdisplsMod[i], sendcounts[i]);
          }
        } else {
          for(int i = 0; i < getCommSize(comm); ++i) {
            recvtype->getIndices(recvbuf, rdispls[i], h->sendbufIndices, rdisplsMod[i], recvcounts[i]);
          }
        }

        if(!recvtype->isModifiedBufferRequired()) {
          for(int i = 0; i < getCommSize(comm); ++i) {
            recvtype->clearIndices(recvbuf, rdispls[i], recvcounts[i]);
          }
        }

        // pack all the variables in the handle
        h->func = AMPI_Ialltoallv_b<SENDTYPE, RECVTYPE>;
        h->sendcounts = sendcounts;
        h->sdispls = sdispls;
        h->sendtype = sendtype;
        h->recvcounts = recvcounts;
        h->rdispls = rdispls;
        h->recvtype = recvtype;
        h->comm = comm;
      }

      rStatus = MPI_Ialltoallv(sendbufMod, sendcounts, sdisplsMod, sendtype->getModifiedMpiType(), recvbufMod, recvcounts,
                               rdisplsMod, recvtype->getModifiedMpiType(), comm, &request->request);

      AMPI_Ialltoallv_AsyncHandle<SENDTYPE, RECVTYPE>* asyncHandle = new AMPI_Ialltoallv_AsyncHandle<SENDTYPE, RECVTYPE>();
      asyncHandle->sendbuf = sendbuf;
      asyncHandle->sendbufMod = sendbufMod;
      asyncHandle->sdisplsMod = sdisplsMod;
      asyncHandle->sendcounts = sendcounts;
      asyncHandle->sdispls = sdispls;
      asyncHandle->sendtype = sendtype;
      asyncHandle->recvbuf = recvbuf;
      asyncHandle->recvbufMod = recvbufMod;
      asyncHandle->rdisplsMod = rdisplsMod;
      asyncHandle->recvcounts = recvcounts;
      asyncHandle->rdispls = rdispls;
      asyncHandle->recvtype = recvtype;
      asyncHandle->comm = comm;
      asyncHandle->h = h;
      request->handle = asyncHandle;
      request->func = (ContinueFunction)AMPI_Ialltoallv_finish<SENDTYPE, RECVTYPE>;

      // create adjoint wait
      if(nullptr != h) {
        WaitHandle* waitH = new WaitHandle((ContinueFunction)AMPI_Ialltoallv_b_finish<SENDTYPE, RECVTYPE>, h);
        recvtype->getADTool().addToolAction(waitH);
      }
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Ialltoallv_finish(HandleBase* handle) {
    int rStatus = 0;

    AMPI_Ialltoallv_AsyncHandle<SENDTYPE, RECVTYPE>* asyncHandle =
      static_cast<AMPI_Ialltoallv_AsyncHandle<SENDTYPE, RECVTYPE>*>(handle);
    const  typename SENDTYPE::Type* sendbuf = asyncHandle->sendbuf;
    typename SENDTYPE::ModifiedType* sendbufMod = asyncHandle->sendbufMod;
    const int* sdisplsMod = asyncHandle->sdisplsMod;
    const  int* sendcounts = asyncHandle->sendcounts;
    const  int* sdispls = asyncHandle->sdispls;
    SENDTYPE* sendtype = asyncHandle->sendtype;
    typename RECVTYPE::Type* recvbuf = asyncHandle->recvbuf;
    typename RECVTYPE::ModifiedType* recvbufMod = asyncHandle->recvbufMod;
    const int* rdisplsMod = asyncHandle->rdisplsMod;
    const  int* recvcounts = asyncHandle->recvcounts;
    const  int* rdispls = asyncHandle->rdispls;
    RECVTYPE* recvtype = asyncHandle->recvtype;
    AMPI_Comm comm = asyncHandle->comm;
    AMPI_Request* request = asyncHandle->request;
    AMPI_Ialltoallv_AdjointHandle<SENDTYPE, RECVTYPE>* h = asyncHandle->h;
    MEDI_UNUSED(sendbuf); // Unused generated to ignore warnings
    MEDI_UNUSED(sendbufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(sdisplsMod); // Unused generated to ignore warnings
    MEDI_UNUSED(sendcounts); // Unused generated to ignore warnings
    MEDI_UNUSED(sdispls); // Unused generated to ignore warnings
    MEDI_UNUSED(sendtype); // Unused generated to ignore warnings
    MEDI_UNUSED(recvbuf); // Unused generated to ignore warnings
    MEDI_UNUSED(recvbufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(rdisplsMod); // Unused generated to ignore warnings
    MEDI_UNUSED(recvcounts); // Unused generated to ignore warnings
    MEDI_UNUSED(rdispls); // Unused generated to ignore warnings
    MEDI_UNUSED(recvtype); // Unused generated to ignore warnings
    MEDI_UNUSED(comm); // Unused generated to ignore warnings
    MEDI_UNUSED(request); // Unused generated to ignore warnings
    MEDI_UNUSED(h); // Unused generated to ignore warnings

    delete asyncHandle;

    if(recvtype->getADTool().isActiveType()) {

      recvtype->getADTool().addToolAction(h);

      if(recvtype->isModifiedBufferRequired()) {
        for(int i = 0; i < getCommSize(comm); ++i) {
          recvtype->copyFromModifiedBuffer(recvbuf, rdispls[i], recvbufMod, rdisplsMod[i], recvcounts[i]);
        }
      }

      if(nullptr != h) {
        // handle the recv buffers
        for(int i = 0; i < getCommSize(comm); ++i) {
          recvtype->registerValue(recvbuf, rdispls[i], h->recvbufIndices, h->recvbufOldPrimals, rdisplsMod[i], recvcounts[i]);
        }
      }

      recvtype->getADTool().stopAssembly(h);
      if(recvtype->isModifiedBufferRequired()) {
        delete [] sdisplsMod;
      }
      if(recvtype->isModifiedBufferRequired()) {
        delete [] rdisplsMod;
      }

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->deleteModifiedTypeBuffer(sendbufMod);
      }
      if(recvtype->isModifiedBufferRequired() ) {
        recvtype->deleteModifiedTypeBuffer(recvbufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename DATATYPE>
  struct AMPI_Ibcast_wrap_AdjointHandle : public HandleBase {
    int bufferSendTotalSize;
    typename DATATYPE::IndexType* bufferSendIndices;
    typename DATATYPE::PassiveType* bufferSendPrimals;
    /* required for async */ typename DATATYPE::PassiveType* bufferSendAdjoints;
    int bufferSendCount;
    int bufferRecvTotalSize;
    typename DATATYPE::IndexType* bufferRecvIndices;
    typename DATATYPE::PassiveType* bufferRecvPrimals;
    typename DATATYPE::PassiveType* bufferRecvOldPrimals;
    /* required for async */ typename DATATYPE::PassiveType* bufferRecvAdjoints;
    int bufferRecvCount;
    int count;
    DATATYPE* datatype;
    int root;
    AMPI_Comm comm;
    AMPI_Request requestReverse;

    ~AMPI_Ibcast_wrap_AdjointHandle () {
      if(nullptr != bufferSendIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(bufferSendIndices);
        bufferSendIndices = nullptr;
      }
      if(nullptr != bufferSendPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(bufferSendPrimals);
        bufferSendPrimals = nullptr;
      }
      if(nullptr != bufferRecvIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(bufferRecvIndices);
        bufferRecvIndices = nullptr;
      }
      if(nullptr != bufferRecvPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(bufferRecvPrimals);
        bufferRecvPrimals = nullptr;
      }
      if(nullptr != bufferRecvOldPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(bufferRecvOldPrimals);
        bufferRecvOldPrimals = nullptr;
      }
    }
  };

  template<typename DATATYPE>
  struct AMPI_Ibcast_wrap_AsyncHandle : public HandleBase {
    typename DATATYPE::Type* bufferSend;
    typename DATATYPE::ModifiedType* bufferSendMod;
    typename DATATYPE::Type* bufferRecv;
    typename DATATYPE::ModifiedType* bufferRecvMod;
    int count;
    DATATYPE* datatype;
    int root;
    AMPI_Comm comm;
    AMPI_Request* request;
    AMPI_Ibcast_wrap_AdjointHandle<DATATYPE>* h;
  };

  template<typename DATATYPE>
  void AMPI_Ibcast_wrap_b(HandleBase* handle) {
    AMPI_Ibcast_wrap_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Ibcast_wrap_AdjointHandle<DATATYPE>*>(handle);

    h->bufferRecvAdjoints = nullptr;
    h->datatype->getADTool().createAdjointTypeBuffer(h->bufferRecvAdjoints, h->bufferRecvTotalSize );
    // Adjoint buffers are always linear in space so we can accesses them in one sweep
    h->datatype->getADTool().getAdjoints(h->bufferRecvIndices, h->bufferRecvAdjoints, h->bufferRecvTotalSize);

    if(h->datatype->getADTool().isOldPrimalsRequired()) {
      h->datatype->getADTool().setReverseValues(h->bufferRecvIndices, h->bufferRecvOldPrimals, h->bufferRecvTotalSize);
    }
    h->bufferSendAdjoints = nullptr;
    if(h->root == getCommRank(h->comm)) {
      h->datatype->getADTool().createAdjointTypeBuffer(h->bufferSendAdjoints, h->bufferSendTotalSize * getCommSize(h->comm));
    }

    AMPI_Ibcast_wrap_adj<DATATYPE>(h->bufferSendAdjoints, h->bufferSendCount, h->bufferRecvAdjoints, h->bufferRecvCount,
                                   h->count, h->datatype, h->root, h->comm, &h->requestReverse);

  }

  template<typename DATATYPE>
  void AMPI_Ibcast_wrap_b_finish(HandleBase* handle) {

    AMPI_Ibcast_wrap_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Ibcast_wrap_AdjointHandle<DATATYPE>*>(handle);
    MPI_Wait(&h->requestReverse.request, MPI_STATUS_IGNORE);

    if(h->root == getCommRank(h->comm)) {
      h->datatype->getADTool().combineAdjoints(h->bufferSendAdjoints, h->bufferSendTotalSize, getCommSize(h->comm));
      // Adjoint buffers are always linear in space so they can be accessed in one sweep
      h->datatype->getADTool().updateAdjoints(h->bufferSendIndices, h->bufferSendAdjoints, h->bufferSendTotalSize);
      h->datatype->getADTool().deleteAdjointTypeBuffer(h->bufferSendAdjoints);
    }
    h->datatype->getADTool().deleteAdjointTypeBuffer(h->bufferRecvAdjoints);
  }

  template<typename DATATYPE>
  int AMPI_Ibcast_wrap_finish(HandleBase* handle);
  template<typename DATATYPE>
  int AMPI_Ibcast_wrap(typename DATATYPE::Type* bufferSend, typename DATATYPE::Type* bufferRecv, int count,
                       DATATYPE* datatype, int root, AMPI_Comm comm, AMPI_Request* request) {
    int rStatus;
    if(!datatype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Ibcast_wrap(bufferSend, bufferRecv, count, datatype->getMpiType(), root, comm, &request->request);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Ibcast_wrap_AdjointHandle<DATATYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(datatype->getADTool().isHandleRequired()) {
        h = new AMPI_Ibcast_wrap_AdjointHandle<DATATYPE>();
      }
      datatype->getADTool().startAssembly(h);
      typename DATATYPE::ModifiedType* bufferSendMod = nullptr;
      int bufferSendElements = 0;

      if(root == getCommRank(comm)) {
        // compute the total size of the buffer
        if(AMPI_IN_PLACE != bufferSend) {
          bufferSendElements = count;
        } else {
          bufferSendElements = count;
        }

        if(datatype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == bufferSend)) {
          datatype->createModifiedTypeBuffer(bufferSendMod, bufferSendElements);
        } else {
          bufferSendMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(bufferSend));
        }
      }
      typename DATATYPE::ModifiedType* bufferRecvMod = nullptr;
      int bufferRecvElements = 0;

      // compute the total size of the buffer
      bufferRecvElements = count;

      if(datatype->isModifiedBufferRequired() ) {
        datatype->createModifiedTypeBuffer(bufferRecvMod, bufferRecvElements);
      } else {
        bufferRecvMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(bufferRecv));
      }

      if(root == getCommRank(comm)) {
        if(datatype->isModifiedBufferRequired()) {
          if(AMPI_IN_PLACE != bufferSend) {
            datatype->copyIntoModifiedBuffer(bufferSend, 0, bufferSendMod, 0, count);
          } else {
            datatype->copyIntoModifiedBuffer(bufferRecv, 0, bufferRecvMod, 0, count);
          }
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(root == getCommRank(comm)) {
          if(AMPI_IN_PLACE != bufferSend) {
            h->bufferSendCount = datatype->computeActiveElements(count);
          } else {
            h->bufferSendCount = datatype->computeActiveElements(count);
          }
          h->bufferSendTotalSize = datatype->computeActiveElements(bufferSendElements);
          datatype->getADTool().createIndexTypeBuffer(h->bufferSendIndices, h->bufferSendTotalSize);
        }
        h->bufferRecvCount = datatype->computeActiveElements(count);
        h->bufferRecvTotalSize = datatype->computeActiveElements(bufferRecvElements);
        datatype->getADTool().createIndexTypeBuffer(h->bufferRecvIndices, h->bufferRecvTotalSize);


        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(datatype->getADTool().isOldPrimalsRequired()) {
          datatype->getADTool().createPassiveTypeBuffer(h->bufferRecvOldPrimals, h->bufferRecvTotalSize);
          datatype->getValues(bufferRecv, 0, h->bufferRecvOldPrimals, 0, count);
        }


        if(root == getCommRank(comm)) {
          if(AMPI_IN_PLACE != bufferSend) {
            datatype->getIndices(bufferSend, 0, h->bufferSendIndices, 0, count);
          } else {
            datatype->getIndices(bufferRecv, 0, h->bufferSendIndices, 0, count);
          }
        }

        if(!datatype->isModifiedBufferRequired()) {
          datatype->clearIndices(bufferRecv, 0, count);
        }

        // pack all the variables in the handle
        h->func = AMPI_Ibcast_wrap_b<DATATYPE>;
        h->count = count;
        h->datatype = datatype;
        h->root = root;
        h->comm = comm;
      }

      rStatus = MPI_Ibcast_wrap(bufferSendMod, bufferRecvMod, count, datatype->getModifiedMpiType(), root, comm,
                                &request->request);

      AMPI_Ibcast_wrap_AsyncHandle<DATATYPE>* asyncHandle = new AMPI_Ibcast_wrap_AsyncHandle<DATATYPE>();
      asyncHandle->bufferSend = bufferSend;
      asyncHandle->bufferSendMod = bufferSendMod;
      asyncHandle->bufferRecv = bufferRecv;
      asyncHandle->bufferRecvMod = bufferRecvMod;
      asyncHandle->count = count;
      asyncHandle->datatype = datatype;
      asyncHandle->root = root;
      asyncHandle->comm = comm;
      asyncHandle->h = h;
      request->handle = asyncHandle;
      request->func = (ContinueFunction)AMPI_Ibcast_wrap_finish<DATATYPE>;

      // create adjoint wait
      if(nullptr != h) {
        WaitHandle* waitH = new WaitHandle((ContinueFunction)AMPI_Ibcast_wrap_b_finish<DATATYPE>, h);
        datatype->getADTool().addToolAction(waitH);
      }
    }

    return rStatus;
  }

  template<typename DATATYPE>
  int AMPI_Ibcast_wrap_finish(HandleBase* handle) {
    int rStatus = 0;

    AMPI_Ibcast_wrap_AsyncHandle<DATATYPE>* asyncHandle = static_cast<AMPI_Ibcast_wrap_AsyncHandle<DATATYPE>*>(handle);
    typename DATATYPE::Type* bufferSend = asyncHandle->bufferSend;
    typename DATATYPE::ModifiedType* bufferSendMod = asyncHandle->bufferSendMod;
    typename DATATYPE::Type* bufferRecv = asyncHandle->bufferRecv;
    typename DATATYPE::ModifiedType* bufferRecvMod = asyncHandle->bufferRecvMod;
    int count = asyncHandle->count;
    DATATYPE* datatype = asyncHandle->datatype;
    int root = asyncHandle->root;
    AMPI_Comm comm = asyncHandle->comm;
    AMPI_Request* request = asyncHandle->request;
    AMPI_Ibcast_wrap_AdjointHandle<DATATYPE>* h = asyncHandle->h;
    MEDI_UNUSED(bufferSend); // Unused generated to ignore warnings
    MEDI_UNUSED(bufferSendMod); // Unused generated to ignore warnings
    MEDI_UNUSED(bufferRecv); // Unused generated to ignore warnings
    MEDI_UNUSED(bufferRecvMod); // Unused generated to ignore warnings
    MEDI_UNUSED(count); // Unused generated to ignore warnings
    MEDI_UNUSED(datatype); // Unused generated to ignore warnings
    MEDI_UNUSED(root); // Unused generated to ignore warnings
    MEDI_UNUSED(comm); // Unused generated to ignore warnings
    MEDI_UNUSED(request); // Unused generated to ignore warnings
    MEDI_UNUSED(h); // Unused generated to ignore warnings

    delete asyncHandle;

    if(datatype->getADTool().isActiveType()) {

      datatype->getADTool().addToolAction(h);

      if(datatype->isModifiedBufferRequired()) {
        datatype->copyFromModifiedBuffer(bufferRecv, 0, bufferRecvMod, 0, count);
      }

      if(nullptr != h) {
        // handle the recv buffers
        datatype->registerValue(bufferRecv, 0, h->bufferRecvIndices, h->bufferRecvOldPrimals, 0, count);
      }

      datatype->getADTool().stopAssembly(h);

      if(root == getCommRank(comm)) {
        if(datatype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == bufferSend)) {
          datatype->deleteModifiedTypeBuffer(bufferSendMod);
        }
      }
      if(datatype->isModifiedBufferRequired() ) {
        datatype->deleteModifiedTypeBuffer(bufferRecvMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Igather_AdjointHandle : public HandleBase {
    int sendbufTotalSize;
    typename SENDTYPE::IndexType* sendbufIndices;
    typename SENDTYPE::PassiveType* sendbufPrimals;
    /* required for async */ typename SENDTYPE::PassiveType* sendbufAdjoints;
    int sendbufCount;
    int sendcount;
    SENDTYPE* sendtype;
    int recvbufTotalSize;
    typename RECVTYPE::IndexType* recvbufIndices;
    typename RECVTYPE::PassiveType* recvbufPrimals;
    typename RECVTYPE::PassiveType* recvbufOldPrimals;
    /* required for async */ typename RECVTYPE::PassiveType* recvbufAdjoints;
    int recvbufCount;
    int recvcount;
    RECVTYPE* recvtype;
    int root;
    AMPI_Comm comm;
    AMPI_Request requestReverse;

    ~AMPI_Igather_AdjointHandle () {
      if(nullptr != sendbufIndices) {
        sendtype->getADTool().deleteIndexTypeBuffer(sendbufIndices);
        sendbufIndices = nullptr;
      }
      if(nullptr != sendbufPrimals) {
        sendtype->getADTool().deletePassiveTypeBuffer(sendbufPrimals);
        sendbufPrimals = nullptr;
      }
      if(nullptr != recvbufIndices) {
        recvtype->getADTool().deleteIndexTypeBuffer(recvbufIndices);
        recvbufIndices = nullptr;
      }
      if(nullptr != recvbufPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufPrimals);
        recvbufPrimals = nullptr;
      }
      if(nullptr != recvbufOldPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufOldPrimals);
        recvbufOldPrimals = nullptr;
      }
    }
  };

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Igather_AsyncHandle : public HandleBase {
    const  typename SENDTYPE::Type* sendbuf;
    typename SENDTYPE::ModifiedType* sendbufMod;
    int sendcount;
    SENDTYPE* sendtype;
    typename RECVTYPE::Type* recvbuf;
    typename RECVTYPE::ModifiedType* recvbufMod;
    int recvcount;
    RECVTYPE* recvtype;
    int root;
    AMPI_Comm comm;
    AMPI_Request* request;
    AMPI_Igather_AdjointHandle<SENDTYPE, RECVTYPE>* h;
  };

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Igather_b(HandleBase* handle) {
    AMPI_Igather_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Igather_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);

    h->recvbufAdjoints = nullptr;
    if(h->root == getCommRank(h->comm)) {
      h->recvtype->getADTool().createAdjointTypeBuffer(h->recvbufAdjoints, h->recvbufTotalSize );
      // Adjoint buffers are always linear in space so we can accesses them in one sweep
      h->recvtype->getADTool().getAdjoints(h->recvbufIndices, h->recvbufAdjoints, h->recvbufTotalSize);

    }
    if(h->recvtype->getADTool().isOldPrimalsRequired()) {
      if(h->root == getCommRank(h->comm)) {
        h->recvtype->getADTool().setReverseValues(h->recvbufIndices, h->recvbufOldPrimals, h->recvbufTotalSize);
      }
    }
    h->sendbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->sendbufAdjoints, h->sendbufTotalSize );

    AMPI_Igather_adj<SENDTYPE, RECVTYPE>(h->sendbufAdjoints, h->sendbufCount, h->sendcount, h->sendtype, h->recvbufAdjoints,
                                         h->recvbufCount, h->recvcount, h->recvtype, h->root, h->comm, &h->requestReverse);

  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Igather_b_finish(HandleBase* handle) {

    AMPI_Igather_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Igather_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);
    MPI_Wait(&h->requestReverse.request, MPI_STATUS_IGNORE);

    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->recvtype->getADTool().updateAdjoints(h->sendbufIndices, h->sendbufAdjoints, h->sendbufTotalSize);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->sendbufAdjoints);
    if(h->root == getCommRank(h->comm)) {
      h->recvtype->getADTool().deleteAdjointTypeBuffer(h->recvbufAdjoints);
    }
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Igather_finish(HandleBase* handle);
  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Igather(const typename SENDTYPE::Type* sendbuf, int sendcount, SENDTYPE* sendtype,
                   typename RECVTYPE::Type* recvbuf, int recvcount, RECVTYPE* recvtype, int root, AMPI_Comm comm, AMPI_Request* request) {
    int rStatus;
    if(!recvtype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Igather(sendbuf, sendcount, sendtype->getMpiType(), recvbuf, recvcount, recvtype->getMpiType(), root,
                            comm, &request->request);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Igather_AdjointHandle<SENDTYPE, RECVTYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(recvtype->getADTool().isHandleRequired()) {
        h = new AMPI_Igather_AdjointHandle<SENDTYPE, RECVTYPE>();
      }
      recvtype->getADTool().startAssembly(h);
      typename SENDTYPE::ModifiedType* sendbufMod = nullptr;
      int sendbufElements = 0;

      // compute the total size of the buffer
      if(AMPI_IN_PLACE != sendbuf) {
        sendbufElements = sendcount;
      } else {
        sendbufElements = recvcount;
      }

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->createModifiedTypeBuffer(sendbufMod, sendbufElements);
      } else {
        sendbufMod = reinterpret_cast<typename SENDTYPE::ModifiedType*>(const_cast<typename SENDTYPE::Type*>(sendbuf));
      }
      typename RECVTYPE::ModifiedType* recvbufMod = nullptr;
      int recvbufElements = 0;

      if(root == getCommRank(comm)) {
        // compute the total size of the buffer
        recvbufElements = recvcount * getCommSize(comm);

        if(recvtype->isModifiedBufferRequired() ) {
          recvtype->createModifiedTypeBuffer(recvbufMod, recvbufElements);
        } else {
          recvbufMod = reinterpret_cast<typename RECVTYPE::ModifiedType*>(const_cast<typename RECVTYPE::Type*>(recvbuf));
        }
      }

      if(sendtype->isModifiedBufferRequired()) {
        if(AMPI_IN_PLACE != sendbuf) {
          sendtype->copyIntoModifiedBuffer(sendbuf, 0, sendbufMod, 0, sendcount);
        } else {
          recvtype->copyIntoModifiedBuffer(recvbuf, recvcount * getCommRank(comm), recvbufMod, recvcount * getCommRank(comm),
                                           recvcount);
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(AMPI_IN_PLACE != sendbuf) {
          h->sendbufCount = sendtype->computeActiveElements(sendcount);
        } else {
          h->sendbufCount = recvtype->computeActiveElements(recvcount);
        }
        h->sendbufTotalSize = sendtype->computeActiveElements(sendbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->sendbufIndices, h->sendbufTotalSize);
        if(root == getCommRank(comm)) {
          h->recvbufCount = recvtype->computeActiveElements(recvcount);
          h->recvbufTotalSize = recvtype->computeActiveElements(recvbufElements);
          recvtype->getADTool().createIndexTypeBuffer(h->recvbufIndices, h->recvbufTotalSize);
        }


        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(recvtype->getADTool().isOldPrimalsRequired()) {
          if(root == getCommRank(comm)) {
            recvtype->getADTool().createPassiveTypeBuffer(h->recvbufOldPrimals, h->recvbufTotalSize);
            if(root == getCommRank(comm)) {
              recvtype->getValues(recvbuf, 0, h->recvbufOldPrimals, 0, recvcount * getCommSize(comm));
            }
          }
        }


        if(AMPI_IN_PLACE != sendbuf) {
          sendtype->getIndices(sendbuf, 0, h->sendbufIndices, 0, sendcount);
        } else {
          recvtype->getIndices(recvbuf, recvcount * getCommRank(comm), h->sendbufIndices, 0, recvcount);
        }

        if(root == getCommRank(comm)) {
          if(!recvtype->isModifiedBufferRequired()) {
            recvtype->clearIndices(recvbuf, 0, recvcount * getCommSize(comm));
          }
        }

        // pack all the variables in the handle
        h->func = AMPI_Igather_b<SENDTYPE, RECVTYPE>;
        h->sendcount = sendcount;
        h->sendtype = sendtype;
        h->recvcount = recvcount;
        h->recvtype = recvtype;
        h->root = root;
        h->comm = comm;
      }

      rStatus = MPI_Igather(sendbufMod, sendcount, sendtype->getModifiedMpiType(), recvbufMod, recvcount,
                            recvtype->getModifiedMpiType(), root, comm, &request->request);

      AMPI_Igather_AsyncHandle<SENDTYPE, RECVTYPE>* asyncHandle = new AMPI_Igather_AsyncHandle<SENDTYPE, RECVTYPE>();
      asyncHandle->sendbuf = sendbuf;
      asyncHandle->sendbufMod = sendbufMod;
      asyncHandle->sendcount = sendcount;
      asyncHandle->sendtype = sendtype;
      asyncHandle->recvbuf = recvbuf;
      asyncHandle->recvbufMod = recvbufMod;
      asyncHandle->recvcount = recvcount;
      asyncHandle->recvtype = recvtype;
      asyncHandle->root = root;
      asyncHandle->comm = comm;
      asyncHandle->h = h;
      request->handle = asyncHandle;
      request->func = (ContinueFunction)AMPI_Igather_finish<SENDTYPE, RECVTYPE>;

      // create adjoint wait
      if(nullptr != h) {
        WaitHandle* waitH = new WaitHandle((ContinueFunction)AMPI_Igather_b_finish<SENDTYPE, RECVTYPE>, h);
        recvtype->getADTool().addToolAction(waitH);
      }
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Igather_finish(HandleBase* handle) {
    int rStatus = 0;

    AMPI_Igather_AsyncHandle<SENDTYPE, RECVTYPE>* asyncHandle = static_cast<AMPI_Igather_AsyncHandle<SENDTYPE, RECVTYPE>*>
        (handle);
    const  typename SENDTYPE::Type* sendbuf = asyncHandle->sendbuf;
    typename SENDTYPE::ModifiedType* sendbufMod = asyncHandle->sendbufMod;
    int sendcount = asyncHandle->sendcount;
    SENDTYPE* sendtype = asyncHandle->sendtype;
    typename RECVTYPE::Type* recvbuf = asyncHandle->recvbuf;
    typename RECVTYPE::ModifiedType* recvbufMod = asyncHandle->recvbufMod;
    int recvcount = asyncHandle->recvcount;
    RECVTYPE* recvtype = asyncHandle->recvtype;
    int root = asyncHandle->root;
    AMPI_Comm comm = asyncHandle->comm;
    AMPI_Request* request = asyncHandle->request;
    AMPI_Igather_AdjointHandle<SENDTYPE, RECVTYPE>* h = asyncHandle->h;
    MEDI_UNUSED(sendbuf); // Unused generated to ignore warnings
    MEDI_UNUSED(sendbufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(sendcount); // Unused generated to ignore warnings
    MEDI_UNUSED(sendtype); // Unused generated to ignore warnings
    MEDI_UNUSED(recvbuf); // Unused generated to ignore warnings
    MEDI_UNUSED(recvbufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(recvcount); // Unused generated to ignore warnings
    MEDI_UNUSED(recvtype); // Unused generated to ignore warnings
    MEDI_UNUSED(root); // Unused generated to ignore warnings
    MEDI_UNUSED(comm); // Unused generated to ignore warnings
    MEDI_UNUSED(request); // Unused generated to ignore warnings
    MEDI_UNUSED(h); // Unused generated to ignore warnings

    delete asyncHandle;

    if(recvtype->getADTool().isActiveType()) {

      recvtype->getADTool().addToolAction(h);

      if(root == getCommRank(comm)) {
        if(recvtype->isModifiedBufferRequired()) {
          recvtype->copyFromModifiedBuffer(recvbuf, 0, recvbufMod, 0, recvcount * getCommSize(comm));
        }
      }

      if(nullptr != h) {
        // handle the recv buffers
        if(root == getCommRank(comm)) {
          recvtype->registerValue(recvbuf, 0, h->recvbufIndices, h->recvbufOldPrimals, 0, recvcount * getCommSize(comm));
        }
      }

      recvtype->getADTool().stopAssembly(h);

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->deleteModifiedTypeBuffer(sendbufMod);
      }
      if(root == getCommRank(comm)) {
        if(recvtype->isModifiedBufferRequired() ) {
          recvtype->deleteModifiedTypeBuffer(recvbufMod);
        }
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Igatherv_AdjointHandle : public HandleBase {
    int sendbufTotalSize;
    typename SENDTYPE::IndexType* sendbufIndices;
    typename SENDTYPE::PassiveType* sendbufPrimals;
    /* required for async */ typename SENDTYPE::PassiveType* sendbufAdjoints;
    int sendbufCount;
    int sendcount;
    SENDTYPE* sendtype;
    int recvbufTotalSize;
    typename RECVTYPE::IndexType* recvbufIndices;
    typename RECVTYPE::PassiveType* recvbufPrimals;
    typename RECVTYPE::PassiveType* recvbufOldPrimals;
    /* required for async */ typename RECVTYPE::PassiveType* recvbufAdjoints;
    int* recvbufCount;
    int* recvbufDispls;
    const  int* recvcounts;
    const  int* displs;
    RECVTYPE* recvtype;
    int root;
    AMPI_Comm comm;
    AMPI_Request requestReverse;

    ~AMPI_Igatherv_AdjointHandle () {
      if(nullptr != sendbufIndices) {
        sendtype->getADTool().deleteIndexTypeBuffer(sendbufIndices);
        sendbufIndices = nullptr;
      }
      if(nullptr != sendbufPrimals) {
        sendtype->getADTool().deletePassiveTypeBuffer(sendbufPrimals);
        sendbufPrimals = nullptr;
      }
      if(nullptr != recvbufIndices) {
        recvtype->getADTool().deleteIndexTypeBuffer(recvbufIndices);
        recvbufIndices = nullptr;
      }
      if(nullptr != recvbufPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufPrimals);
        recvbufPrimals = nullptr;
      }
      if(nullptr != recvbufOldPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufOldPrimals);
        recvbufOldPrimals = nullptr;
      }
      if(nullptr != recvbufCount) {
        delete [] recvbufCount;
        recvbufCount = nullptr;
      }
      if(nullptr != recvbufDispls) {
        delete [] recvbufDispls;
        recvbufDispls = nullptr;
      }
    }
  };

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Igatherv_AsyncHandle : public HandleBase {
    const  typename SENDTYPE::Type* sendbuf;
    typename SENDTYPE::ModifiedType* sendbufMod;
    int sendcount;
    SENDTYPE* sendtype;
    typename RECVTYPE::Type* recvbuf;
    typename RECVTYPE::ModifiedType* recvbufMod;
    const int* displsMod;
    const  int* recvcounts;
    const  int* displs;
    RECVTYPE* recvtype;
    int root;
    AMPI_Comm comm;
    AMPI_Request* request;
    AMPI_Igatherv_AdjointHandle<SENDTYPE, RECVTYPE>* h;
  };

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Igatherv_b(HandleBase* handle) {
    AMPI_Igatherv_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Igatherv_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);

    h->recvbufAdjoints = nullptr;
    if(h->root == getCommRank(h->comm)) {
      h->recvtype->getADTool().createAdjointTypeBuffer(h->recvbufAdjoints, h->recvbufTotalSize );
      // Adjoint buffers are always linear in space so we can accesses them in one sweep
      h->recvtype->getADTool().getAdjoints(h->recvbufIndices, h->recvbufAdjoints, h->recvbufTotalSize);

    }
    if(h->recvtype->getADTool().isOldPrimalsRequired()) {
      if(h->root == getCommRank(h->comm)) {
        h->recvtype->getADTool().setReverseValues(h->recvbufIndices, h->recvbufOldPrimals, h->recvbufTotalSize);
      }
    }
    h->sendbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->sendbufAdjoints, h->sendbufTotalSize );

    AMPI_Igatherv_adj<SENDTYPE, RECVTYPE>(h->sendbufAdjoints, h->sendbufCount, h->sendcount, h->sendtype,
                                          h->recvbufAdjoints, h->recvbufCount, h->recvbufDispls, h->recvcounts, h->displs, h->recvtype, h->root, h->comm,
                                          &h->requestReverse);

  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Igatherv_b_finish(HandleBase* handle) {

    AMPI_Igatherv_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Igatherv_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);
    MPI_Wait(&h->requestReverse.request, MPI_STATUS_IGNORE);

    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->recvtype->getADTool().updateAdjoints(h->sendbufIndices, h->sendbufAdjoints, h->sendbufTotalSize);
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->sendbufAdjoints);
    if(h->root == getCommRank(h->comm)) {
      h->recvtype->getADTool().deleteAdjointTypeBuffer(h->recvbufAdjoints);
    }
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Igatherv_finish(HandleBase* handle);
  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Igatherv(const typename SENDTYPE::Type* sendbuf, int sendcount, SENDTYPE* sendtype,
                    typename RECVTYPE::Type* recvbuf, const int* recvcounts, const int* displs, RECVTYPE* recvtype, int root,
                    AMPI_Comm comm, AMPI_Request* request) {
    int rStatus;
    if(!recvtype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Igatherv(sendbuf, sendcount, sendtype->getMpiType(), recvbuf, recvcounts, displs, recvtype->getMpiType(),
                             root, comm, &request->request);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Igatherv_AdjointHandle<SENDTYPE, RECVTYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(recvtype->getADTool().isHandleRequired()) {
        h = new AMPI_Igatherv_AdjointHandle<SENDTYPE, RECVTYPE>();
      }
      recvtype->getADTool().startAssembly(h);
      const int* displsMod = displs;
      int displsTotalSize = 0;
      if(nullptr != displs) {
        displsTotalSize = computeDisplacementsTotalSize(recvcounts, getCommSize(comm));
        if(recvtype->isModifiedBufferRequired()) {
          displsMod = createLinearDisplacements(recvcounts, getCommSize(comm));
        }
      }
      typename SENDTYPE::ModifiedType* sendbufMod = nullptr;
      int sendbufElements = 0;

      // compute the total size of the buffer
      if(AMPI_IN_PLACE != sendbuf) {
        sendbufElements = sendcount;
      } else {
        sendbufElements = recvcounts[getCommRank(comm)];
      }

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->createModifiedTypeBuffer(sendbufMod, sendbufElements);
      } else {
        sendbufMod = reinterpret_cast<typename SENDTYPE::ModifiedType*>(const_cast<typename SENDTYPE::Type*>(sendbuf));
      }
      typename RECVTYPE::ModifiedType* recvbufMod = nullptr;
      int recvbufElements = 0;

      if(root == getCommRank(comm)) {
        // compute the total size of the buffer
        recvbufElements = displsTotalSize;

        if(recvtype->isModifiedBufferRequired() ) {
          recvtype->createModifiedTypeBuffer(recvbufMod, recvbufElements);
        } else {
          recvbufMod = reinterpret_cast<typename RECVTYPE::ModifiedType*>(const_cast<typename RECVTYPE::Type*>(recvbuf));
        }
      }

      if(sendtype->isModifiedBufferRequired()) {
        if(AMPI_IN_PLACE != sendbuf) {
          sendtype->copyIntoModifiedBuffer(sendbuf, 0, sendbufMod, 0, sendcount);
        } else {
          {
            const int rank = getCommRank(comm);
            recvtype->copyIntoModifiedBuffer(recvbuf, displs[rank], recvbufMod, displsMod[rank], recvcounts[rank]);
          }
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(AMPI_IN_PLACE != sendbuf) {
          h->sendbufCount = sendtype->computeActiveElements(sendcount);
        } else {
          h->sendbufCount = recvtype->computeActiveElements(recvcounts[getCommRank(comm)]);
        }
        h->sendbufTotalSize = sendtype->computeActiveElements(sendbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->sendbufIndices, h->sendbufTotalSize);
        if(root == getCommRank(comm)) {
          createLinearIndexDisplacements(h->recvbufCount, h->recvbufDispls, recvcounts, getCommSize(comm), recvtype);
          h->recvbufTotalSize = recvtype->computeActiveElements(recvbufElements);
          recvtype->getADTool().createIndexTypeBuffer(h->recvbufIndices, h->recvbufTotalSize);
        }


        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(recvtype->getADTool().isOldPrimalsRequired()) {
          if(root == getCommRank(comm)) {
            recvtype->getADTool().createPassiveTypeBuffer(h->recvbufOldPrimals, h->recvbufTotalSize);
            if(root == getCommRank(comm)) {
              for(int i = 0; i < getCommSize(comm); ++i) {
                recvtype->getValues(recvbuf, displs[i], h->recvbufOldPrimals, displsMod[i], recvcounts[i]);
              }
            }
          }
        }


        if(AMPI_IN_PLACE != sendbuf) {
          sendtype->getIndices(sendbuf, 0, h->sendbufIndices, 0, sendcount);
        } else {
          {
            const int rank = getCommRank(comm);
            recvtype->getIndices(recvbuf, displs[rank], h->sendbufIndices, 0, recvcounts[rank]);
          }
        }

        if(root == getCommRank(comm)) {
          if(!recvtype->isModifiedBufferRequired()) {
            for(int i = 0; i < getCommSize(comm); ++i) {
              recvtype->clearIndices(recvbuf, displs[i], recvcounts[i]);
            }
          }
        }

        // pack all the variables in the handle
        h->func = AMPI_Igatherv_b<SENDTYPE, RECVTYPE>;
        h->sendcount = sendcount;
        h->sendtype = sendtype;
        h->recvcounts = recvcounts;
        h->displs = displs;
        h->recvtype = recvtype;
        h->root = root;
        h->comm = comm;
      }

      rStatus = MPI_Igatherv(sendbufMod, sendcount, sendtype->getModifiedMpiType(), recvbufMod, recvcounts, displsMod,
                             recvtype->getModifiedMpiType(), root, comm, &request->request);

      AMPI_Igatherv_AsyncHandle<SENDTYPE, RECVTYPE>* asyncHandle = new AMPI_Igatherv_AsyncHandle<SENDTYPE, RECVTYPE>();
      asyncHandle->sendbuf = sendbuf;
      asyncHandle->sendbufMod = sendbufMod;
      asyncHandle->sendcount = sendcount;
      asyncHandle->sendtype = sendtype;
      asyncHandle->recvbuf = recvbuf;
      asyncHandle->recvbufMod = recvbufMod;
      asyncHandle->displsMod = displsMod;
      asyncHandle->recvcounts = recvcounts;
      asyncHandle->displs = displs;
      asyncHandle->recvtype = recvtype;
      asyncHandle->root = root;
      asyncHandle->comm = comm;
      asyncHandle->h = h;
      request->handle = asyncHandle;
      request->func = (ContinueFunction)AMPI_Igatherv_finish<SENDTYPE, RECVTYPE>;

      // create adjoint wait
      if(nullptr != h) {
        WaitHandle* waitH = new WaitHandle((ContinueFunction)AMPI_Igatherv_b_finish<SENDTYPE, RECVTYPE>, h);
        recvtype->getADTool().addToolAction(waitH);
      }
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Igatherv_finish(HandleBase* handle) {
    int rStatus = 0;

    AMPI_Igatherv_AsyncHandle<SENDTYPE, RECVTYPE>* asyncHandle =
      static_cast<AMPI_Igatherv_AsyncHandle<SENDTYPE, RECVTYPE>*>(handle);
    const  typename SENDTYPE::Type* sendbuf = asyncHandle->sendbuf;
    typename SENDTYPE::ModifiedType* sendbufMod = asyncHandle->sendbufMod;
    int sendcount = asyncHandle->sendcount;
    SENDTYPE* sendtype = asyncHandle->sendtype;
    typename RECVTYPE::Type* recvbuf = asyncHandle->recvbuf;
    typename RECVTYPE::ModifiedType* recvbufMod = asyncHandle->recvbufMod;
    const int* displsMod = asyncHandle->displsMod;
    const  int* recvcounts = asyncHandle->recvcounts;
    const  int* displs = asyncHandle->displs;
    RECVTYPE* recvtype = asyncHandle->recvtype;
    int root = asyncHandle->root;
    AMPI_Comm comm = asyncHandle->comm;
    AMPI_Request* request = asyncHandle->request;
    AMPI_Igatherv_AdjointHandle<SENDTYPE, RECVTYPE>* h = asyncHandle->h;
    MEDI_UNUSED(sendbuf); // Unused generated to ignore warnings
    MEDI_UNUSED(sendbufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(sendcount); // Unused generated to ignore warnings
    MEDI_UNUSED(sendtype); // Unused generated to ignore warnings
    MEDI_UNUSED(recvbuf); // Unused generated to ignore warnings
    MEDI_UNUSED(recvbufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(displsMod); // Unused generated to ignore warnings
    MEDI_UNUSED(recvcounts); // Unused generated to ignore warnings
    MEDI_UNUSED(displs); // Unused generated to ignore warnings
    MEDI_UNUSED(recvtype); // Unused generated to ignore warnings
    MEDI_UNUSED(root); // Unused generated to ignore warnings
    MEDI_UNUSED(comm); // Unused generated to ignore warnings
    MEDI_UNUSED(request); // Unused generated to ignore warnings
    MEDI_UNUSED(h); // Unused generated to ignore warnings

    delete asyncHandle;

    if(recvtype->getADTool().isActiveType()) {

      recvtype->getADTool().addToolAction(h);

      if(root == getCommRank(comm)) {
        if(recvtype->isModifiedBufferRequired()) {
          for(int i = 0; i < getCommSize(comm); ++i) {
            recvtype->copyFromModifiedBuffer(recvbuf, displs[i], recvbufMod, displsMod[i], recvcounts[i]);
          }
        }
      }

      if(nullptr != h) {
        // handle the recv buffers
        if(root == getCommRank(comm)) {
          for(int i = 0; i < getCommSize(comm); ++i) {
            recvtype->registerValue(recvbuf, displs[i], h->recvbufIndices, h->recvbufOldPrimals, displsMod[i], recvcounts[i]);
          }
        }
      }

      recvtype->getADTool().stopAssembly(h);
      if(recvtype->isModifiedBufferRequired()) {
        delete [] displsMod;
      }

      if(sendtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        sendtype->deleteModifiedTypeBuffer(sendbufMod);
      }
      if(root == getCommRank(comm)) {
        if(recvtype->isModifiedBufferRequired() ) {
          recvtype->deleteModifiedTypeBuffer(recvbufMod);
        }
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename DATATYPE>
  struct AMPI_Ireduce_global_AdjointHandle : public HandleBase {
    int sendbufTotalSize;
    typename DATATYPE::IndexType* sendbufIndices;
    typename DATATYPE::PassiveType* sendbufPrimals;
    /* required for async */ typename DATATYPE::PassiveType* sendbufAdjoints;
    int sendbufCount;
    int recvbufTotalSize;
    typename DATATYPE::IndexType* recvbufIndices;
    typename DATATYPE::PassiveType* recvbufPrimals;
    typename DATATYPE::PassiveType* recvbufOldPrimals;
    /* required for async */ typename DATATYPE::PassiveType* recvbufAdjoints;
    int recvbufCount;
    int count;
    DATATYPE* datatype;
    AMPI_Op op;
    int root;
    AMPI_Comm comm;
    AMPI_Request requestReverse;

    ~AMPI_Ireduce_global_AdjointHandle () {
      if(nullptr != sendbufIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(sendbufIndices);
        sendbufIndices = nullptr;
      }
      if(nullptr != sendbufPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(sendbufPrimals);
        sendbufPrimals = nullptr;
      }
      if(nullptr != recvbufIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(recvbufIndices);
        recvbufIndices = nullptr;
      }
      if(nullptr != recvbufPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(recvbufPrimals);
        recvbufPrimals = nullptr;
      }
      if(nullptr != recvbufOldPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(recvbufOldPrimals);
        recvbufOldPrimals = nullptr;
      }
    }
  };

  template<typename DATATYPE>
  struct AMPI_Ireduce_global_AsyncHandle : public HandleBase {
    const  typename DATATYPE::Type* sendbuf;
    typename DATATYPE::ModifiedType* sendbufMod;
    typename DATATYPE::Type* recvbuf;
    typename DATATYPE::ModifiedType* recvbufMod;
    int count;
    DATATYPE* datatype;
    AMPI_Op op;
    int root;
    AMPI_Comm comm;
    AMPI_Request* request;
    AMPI_Ireduce_global_AdjointHandle<DATATYPE>* h;
  };

  template<typename DATATYPE>
  void AMPI_Ireduce_global_b(HandleBase* handle) {
    AMPI_Ireduce_global_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Ireduce_global_AdjointHandle<DATATYPE>*>(handle);

    h->recvbufAdjoints = nullptr;
    if(h->root == getCommRank(h->comm)) {
      h->datatype->getADTool().createAdjointTypeBuffer(h->recvbufAdjoints, h->recvbufTotalSize );
      // Adjoint buffers are always linear in space so we can accesses them in one sweep
      h->datatype->getADTool().getAdjoints(h->recvbufIndices, h->recvbufAdjoints, h->recvbufTotalSize);

      h->op.preAdjointOperation(h->recvbufAdjoints, h->recvbufPrimals, h->recvbufCount);
    }
    if(h->datatype->getADTool().isOldPrimalsRequired()) {
      if(h->root == getCommRank(h->comm)) {
        h->datatype->getADTool().setReverseValues(h->recvbufIndices, h->recvbufOldPrimals, h->recvbufTotalSize);
      }
    }
    h->sendbufAdjoints = nullptr;
    h->datatype->getADTool().createAdjointTypeBuffer(h->sendbufAdjoints, h->sendbufTotalSize );

    AMPI_Ireduce_global_adj<DATATYPE>(h->sendbufAdjoints, h->sendbufCount, h->recvbufAdjoints, h->recvbufCount, h->count,
                                      h->datatype, h->op, h->root, h->comm, &h->requestReverse);

  }

  template<typename DATATYPE>
  void AMPI_Ireduce_global_b_finish(HandleBase* handle) {

    AMPI_Ireduce_global_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Ireduce_global_AdjointHandle<DATATYPE>*>(handle);
    MPI_Wait(&h->requestReverse.request, MPI_STATUS_IGNORE);

    // the primals of the recive buffer are always given to the function. The operator should ignore them if not needed.
    // The wrapper functions make sure that for operators that need the primals an all* action is perfomed (e.g. Allreduce instead of Reduce)
    h->op.postAdjointOperation(h->sendbufAdjoints, h->sendbufPrimals, h->recvbufPrimals, h->sendbufTotalSize);
    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->datatype->getADTool().updateAdjoints(h->sendbufIndices, h->sendbufAdjoints, h->sendbufTotalSize);
    h->datatype->getADTool().deleteAdjointTypeBuffer(h->sendbufAdjoints);
    if(h->root == getCommRank(h->comm)) {
      h->datatype->getADTool().deleteAdjointTypeBuffer(h->recvbufAdjoints);
    }
  }

  template<typename DATATYPE>
  int AMPI_Ireduce_global_finish(HandleBase* handle);
  template<typename DATATYPE>
  int AMPI_Ireduce_global(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count,
                          DATATYPE* datatype, AMPI_Op op, int root, AMPI_Comm comm, AMPI_Request* request) {
    int rStatus;
    if(!datatype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Ireduce(sendbuf, recvbuf, count, datatype->getMpiType(), op.primalFunction, root, comm,
                            &request->request);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Ireduce_global_AdjointHandle<DATATYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(datatype->getADTool().isHandleRequired()) {
        h = new AMPI_Ireduce_global_AdjointHandle<DATATYPE>();
      }
      datatype->getADTool().startAssembly(h);
      typename DATATYPE::ModifiedType* sendbufMod = nullptr;
      int sendbufElements = 0;

      // compute the total size of the buffer
      if(AMPI_IN_PLACE != sendbuf) {
        sendbufElements = count;
      } else {
        sendbufElements = count;
      }

      if(datatype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        datatype->createModifiedTypeBuffer(sendbufMod, sendbufElements);
      } else {
        sendbufMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(sendbuf));
      }
      typename DATATYPE::ModifiedType* recvbufMod = nullptr;
      int recvbufElements = 0;

      if(root == getCommRank(comm)) {
        // compute the total size of the buffer
        recvbufElements = count;

        if(datatype->isModifiedBufferRequired() ) {
          datatype->createModifiedTypeBuffer(recvbufMod, recvbufElements);
        } else {
          recvbufMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(recvbuf));
        }
      }

      if(datatype->isModifiedBufferRequired()) {
        if(AMPI_IN_PLACE != sendbuf) {
          datatype->copyIntoModifiedBuffer(sendbuf, 0, sendbufMod, 0, count);
        } else {
          datatype->copyIntoModifiedBuffer(recvbuf, 0, recvbufMod, 0, count);
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(AMPI_IN_PLACE != sendbuf) {
          h->sendbufCount = datatype->computeActiveElements(count);
        } else {
          h->sendbufCount = datatype->computeActiveElements(count);
        }
        h->sendbufTotalSize = datatype->computeActiveElements(sendbufElements);
        datatype->getADTool().createIndexTypeBuffer(h->sendbufIndices, h->sendbufTotalSize);
        if(root == getCommRank(comm)) {
          h->recvbufCount = datatype->computeActiveElements(count);
          h->recvbufTotalSize = datatype->computeActiveElements(recvbufElements);
          datatype->getADTool().createIndexTypeBuffer(h->recvbufIndices, h->recvbufTotalSize);
        }

        // extract the primal values for the operator if required
        if(op.requiresPrimal) {
          datatype->getADTool().createPassiveTypeBuffer(h->sendbufPrimals, h->sendbufTotalSize);
          if(AMPI_IN_PLACE != sendbuf) {
            datatype->getValues(sendbuf, 0, h->sendbufPrimals, 0, count);
          } else {
            datatype->getValues(recvbuf, 0, h->sendbufPrimals, 0, count);
          }
        }

        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(datatype->getADTool().isOldPrimalsRequired()) {
          if(root == getCommRank(comm)) {
            datatype->getADTool().createPassiveTypeBuffer(h->recvbufOldPrimals, h->recvbufTotalSize);
            if(root == getCommRank(comm)) {
              datatype->getValues(recvbuf, 0, h->recvbufOldPrimals, 0, count);
            }
          }
        }


        if(AMPI_IN_PLACE != sendbuf) {
          datatype->getIndices(sendbuf, 0, h->sendbufIndices, 0, count);
        } else {
          datatype->getIndices(recvbuf, 0, h->sendbufIndices, 0, count);
        }

        if(root == getCommRank(comm)) {
          if(!datatype->isModifiedBufferRequired()) {
            datatype->clearIndices(recvbuf, 0, count);
          }
        }

        // pack all the variables in the handle
        h->func = AMPI_Ireduce_global_b<DATATYPE>;
        h->count = count;
        h->datatype = datatype;
        h->op = op;
        h->root = root;
        h->comm = comm;
      }

      rStatus = MPI_Ireduce(sendbufMod, recvbufMod, count, datatype->getModifiedMpiType(), op.modifiedPrimalFunction, root,
                            comm, &request->request);

      AMPI_Ireduce_global_AsyncHandle<DATATYPE>* asyncHandle = new AMPI_Ireduce_global_AsyncHandle<DATATYPE>();
      asyncHandle->sendbuf = sendbuf;
      asyncHandle->sendbufMod = sendbufMod;
      asyncHandle->recvbuf = recvbuf;
      asyncHandle->recvbufMod = recvbufMod;
      asyncHandle->count = count;
      asyncHandle->datatype = datatype;
      asyncHandle->op = op;
      asyncHandle->root = root;
      asyncHandle->comm = comm;
      asyncHandle->h = h;
      request->handle = asyncHandle;
      request->func = (ContinueFunction)AMPI_Ireduce_global_finish<DATATYPE>;

      // create adjoint wait
      if(nullptr != h) {
        WaitHandle* waitH = new WaitHandle((ContinueFunction)AMPI_Ireduce_global_b_finish<DATATYPE>, h);
        datatype->getADTool().addToolAction(waitH);
      }
    }

    return rStatus;
  }

  template<typename DATATYPE>
  int AMPI_Ireduce_global_finish(HandleBase* handle) {
    int rStatus = 0;

    AMPI_Ireduce_global_AsyncHandle<DATATYPE>* asyncHandle = static_cast<AMPI_Ireduce_global_AsyncHandle<DATATYPE>*>
        (handle);
    const  typename DATATYPE::Type* sendbuf = asyncHandle->sendbuf;
    typename DATATYPE::ModifiedType* sendbufMod = asyncHandle->sendbufMod;
    typename DATATYPE::Type* recvbuf = asyncHandle->recvbuf;
    typename DATATYPE::ModifiedType* recvbufMod = asyncHandle->recvbufMod;
    int count = asyncHandle->count;
    DATATYPE* datatype = asyncHandle->datatype;
    AMPI_Op op = asyncHandle->op;
    int root = asyncHandle->root;
    AMPI_Comm comm = asyncHandle->comm;
    AMPI_Request* request = asyncHandle->request;
    AMPI_Ireduce_global_AdjointHandle<DATATYPE>* h = asyncHandle->h;
    MEDI_UNUSED(sendbuf); // Unused generated to ignore warnings
    MEDI_UNUSED(sendbufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(recvbuf); // Unused generated to ignore warnings
    MEDI_UNUSED(recvbufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(count); // Unused generated to ignore warnings
    MEDI_UNUSED(datatype); // Unused generated to ignore warnings
    MEDI_UNUSED(op); // Unused generated to ignore warnings
    MEDI_UNUSED(root); // Unused generated to ignore warnings
    MEDI_UNUSED(comm); // Unused generated to ignore warnings
    MEDI_UNUSED(request); // Unused generated to ignore warnings
    MEDI_UNUSED(h); // Unused generated to ignore warnings

    delete asyncHandle;

    if(datatype->getADTool().isActiveType()) {

      datatype->getADTool().addToolAction(h);

      if(root == getCommRank(comm)) {
        if(datatype->isModifiedBufferRequired()) {
          datatype->copyFromModifiedBuffer(recvbuf, 0, recvbufMod, 0, count);
        }
      }

      if(nullptr != h) {
        // handle the recv buffers
        if(root == getCommRank(comm)) {
          datatype->registerValue(recvbuf, 0, h->recvbufIndices, h->recvbufOldPrimals, 0, count);
        }
      }
      // extract the primal values for the operator if required
      if(nullptr != h && op.requiresPrimal) {
        if(root == getCommRank(comm)) {
          datatype->getADTool().createPassiveTypeBuffer(h->recvbufPrimals, h->recvbufTotalSize);
          if(root == getCommRank(comm)) {
            datatype->getValues(recvbuf, 0, h->recvbufPrimals, 0, count);
          }
        }
      }

      datatype->getADTool().stopAssembly(h);

      if(datatype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        datatype->deleteModifiedTypeBuffer(sendbufMod);
      }
      if(root == getCommRank(comm)) {
        if(datatype->isModifiedBufferRequired() ) {
          datatype->deleteModifiedTypeBuffer(recvbufMod);
        }
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Iscatter_AdjointHandle : public HandleBase {
    int sendbufTotalSize;
    typename SENDTYPE::IndexType* sendbufIndices;
    typename SENDTYPE::PassiveType* sendbufPrimals;
    /* required for async */ typename SENDTYPE::PassiveType* sendbufAdjoints;
    int sendbufCount;
    int sendcount;
    SENDTYPE* sendtype;
    int recvbufTotalSize;
    typename RECVTYPE::IndexType* recvbufIndices;
    typename RECVTYPE::PassiveType* recvbufPrimals;
    typename RECVTYPE::PassiveType* recvbufOldPrimals;
    /* required for async */ typename RECVTYPE::PassiveType* recvbufAdjoints;
    int recvbufCount;
    int recvcount;
    RECVTYPE* recvtype;
    int root;
    AMPI_Comm comm;
    AMPI_Request requestReverse;

    ~AMPI_Iscatter_AdjointHandle () {
      if(nullptr != sendbufIndices) {
        sendtype->getADTool().deleteIndexTypeBuffer(sendbufIndices);
        sendbufIndices = nullptr;
      }
      if(nullptr != sendbufPrimals) {
        sendtype->getADTool().deletePassiveTypeBuffer(sendbufPrimals);
        sendbufPrimals = nullptr;
      }
      if(nullptr != recvbufIndices) {
        recvtype->getADTool().deleteIndexTypeBuffer(recvbufIndices);
        recvbufIndices = nullptr;
      }
      if(nullptr != recvbufPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufPrimals);
        recvbufPrimals = nullptr;
      }
      if(nullptr != recvbufOldPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufOldPrimals);
        recvbufOldPrimals = nullptr;
      }
    }
  };

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Iscatter_AsyncHandle : public HandleBase {
    typename SENDTYPE::Type* sendbuf;
    typename SENDTYPE::ModifiedType* sendbufMod;
    int sendcount;
    SENDTYPE* sendtype;
    typename RECVTYPE::Type* recvbuf;
    typename RECVTYPE::ModifiedType* recvbufMod;
    int recvcount;
    RECVTYPE* recvtype;
    int root;
    AMPI_Comm comm;
    AMPI_Request* request;
    AMPI_Iscatter_AdjointHandle<SENDTYPE, RECVTYPE>* h;
  };

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Iscatter_b(HandleBase* handle) {
    AMPI_Iscatter_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Iscatter_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);

    h->recvbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->recvbufAdjoints, h->recvbufTotalSize );
    // Adjoint buffers are always linear in space so we can accesses them in one sweep
    h->recvtype->getADTool().getAdjoints(h->recvbufIndices, h->recvbufAdjoints, h->recvbufTotalSize);

    if(h->recvtype->getADTool().isOldPrimalsRequired()) {
      h->recvtype->getADTool().setReverseValues(h->recvbufIndices, h->recvbufOldPrimals, h->recvbufTotalSize);
    }
    h->sendbufAdjoints = nullptr;
    if(h->root == getCommRank(h->comm)) {
      h->recvtype->getADTool().createAdjointTypeBuffer(h->sendbufAdjoints, h->sendbufTotalSize );
    }

    AMPI_Iscatter_adj<SENDTYPE, RECVTYPE>(h->sendbufAdjoints, h->sendbufCount, h->sendcount, h->sendtype,
                                          h->recvbufAdjoints, h->recvbufCount, h->recvcount, h->recvtype, h->root, h->comm, &h->requestReverse);

  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Iscatter_b_finish(HandleBase* handle) {

    AMPI_Iscatter_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Iscatter_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);
    MPI_Wait(&h->requestReverse.request, MPI_STATUS_IGNORE);

    if(h->root == getCommRank(h->comm)) {
      // Adjoint buffers are always linear in space so they can be accessed in one sweep
      h->recvtype->getADTool().updateAdjoints(h->sendbufIndices, h->sendbufAdjoints, h->sendbufTotalSize);
      h->recvtype->getADTool().deleteAdjointTypeBuffer(h->sendbufAdjoints);
    }
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->recvbufAdjoints);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Iscatter_finish(HandleBase* handle);
  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Iscatter(typename SENDTYPE::Type* sendbuf, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::Type* recvbuf,
                    int recvcount, RECVTYPE* recvtype, int root, AMPI_Comm comm, AMPI_Request* request) {
    int rStatus;
    if(!recvtype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Iscatter(sendbuf, sendcount, sendtype->getMpiType(), recvbuf, recvcount, recvtype->getMpiType(), root,
                             comm, &request->request);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Iscatter_AdjointHandle<SENDTYPE, RECVTYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(recvtype->getADTool().isHandleRequired()) {
        h = new AMPI_Iscatter_AdjointHandle<SENDTYPE, RECVTYPE>();
      }
      recvtype->getADTool().startAssembly(h);
      typename SENDTYPE::ModifiedType* sendbufMod = nullptr;
      int sendbufElements = 0;

      if(root == getCommRank(comm)) {
        // compute the total size of the buffer
        sendbufElements = sendcount * getCommSize(comm);

        if(sendtype->isModifiedBufferRequired() ) {
          sendtype->createModifiedTypeBuffer(sendbufMod, sendbufElements);
        } else {
          sendbufMod = reinterpret_cast<typename SENDTYPE::ModifiedType*>(const_cast<typename SENDTYPE::Type*>(sendbuf));
        }
      }
      typename RECVTYPE::ModifiedType* recvbufMod = nullptr;
      int recvbufElements = 0;

      // compute the total size of the buffer
      if(AMPI_IN_PLACE != recvbuf) {
        recvbufElements = recvcount;
      } else {
        recvbufElements = sendcount;
      }

      if(recvtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == recvbuf)) {
        recvtype->createModifiedTypeBuffer(recvbufMod, recvbufElements);
      } else {
        recvbufMod = reinterpret_cast<typename RECVTYPE::ModifiedType*>(const_cast<typename RECVTYPE::Type*>(recvbuf));
      }

      if(root == getCommRank(comm)) {
        if(sendtype->isModifiedBufferRequired()) {
          sendtype->copyIntoModifiedBuffer(sendbuf, 0, sendbufMod, 0, sendcount * getCommSize(comm));
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(root == getCommRank(comm)) {
          h->sendbufCount = sendtype->computeActiveElements(sendcount);
          h->sendbufTotalSize = sendtype->computeActiveElements(sendbufElements);
          recvtype->getADTool().createIndexTypeBuffer(h->sendbufIndices, h->sendbufTotalSize);
        }
        if(AMPI_IN_PLACE != recvbuf) {
          h->recvbufCount = recvtype->computeActiveElements(recvcount);
        } else {
          h->recvbufCount = sendtype->computeActiveElements(sendcount);
        }
        h->recvbufTotalSize = recvtype->computeActiveElements(recvbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->recvbufIndices, h->recvbufTotalSize);


        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(recvtype->getADTool().isOldPrimalsRequired()) {
          recvtype->getADTool().createPassiveTypeBuffer(h->recvbufOldPrimals, h->recvbufTotalSize);
          if(AMPI_IN_PLACE != recvbuf) {
            recvtype->getValues(recvbuf, 0, h->recvbufOldPrimals, 0, recvcount);
          } else {
            sendtype->getValues(sendbuf, sendcount * getCommRank(comm), h->recvbufOldPrimals, 0, sendcount);
          }
        }


        if(root == getCommRank(comm)) {
          sendtype->getIndices(sendbuf, 0, h->sendbufIndices, 0, sendcount * getCommSize(comm));
        }

        if(!recvtype->isModifiedBufferRequired()) {
          if(AMPI_IN_PLACE != recvbuf) {
            recvtype->clearIndices(recvbuf, 0, recvcount);
          } else {
            sendtype->clearIndices(sendbuf, sendcount * getCommRank(comm), sendcount);
          }
        }

        // pack all the variables in the handle
        h->func = AMPI_Iscatter_b<SENDTYPE, RECVTYPE>;
        h->sendcount = sendcount;
        h->sendtype = sendtype;
        h->recvcount = recvcount;
        h->recvtype = recvtype;
        h->root = root;
        h->comm = comm;
      }

      rStatus = MPI_Iscatter(sendbufMod, sendcount, sendtype->getModifiedMpiType(), recvbufMod, recvcount,
                             recvtype->getModifiedMpiType(), root, comm, &request->request);

      AMPI_Iscatter_AsyncHandle<SENDTYPE, RECVTYPE>* asyncHandle = new AMPI_Iscatter_AsyncHandle<SENDTYPE, RECVTYPE>();
      asyncHandle->sendbuf = sendbuf;
      asyncHandle->sendbufMod = sendbufMod;
      asyncHandle->sendcount = sendcount;
      asyncHandle->sendtype = sendtype;
      asyncHandle->recvbuf = recvbuf;
      asyncHandle->recvbufMod = recvbufMod;
      asyncHandle->recvcount = recvcount;
      asyncHandle->recvtype = recvtype;
      asyncHandle->root = root;
      asyncHandle->comm = comm;
      asyncHandle->h = h;
      request->handle = asyncHandle;
      request->func = (ContinueFunction)AMPI_Iscatter_finish<SENDTYPE, RECVTYPE>;

      // create adjoint wait
      if(nullptr != h) {
        WaitHandle* waitH = new WaitHandle((ContinueFunction)AMPI_Iscatter_b_finish<SENDTYPE, RECVTYPE>, h);
        recvtype->getADTool().addToolAction(waitH);
      }
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Iscatter_finish(HandleBase* handle) {
    int rStatus = 0;

    AMPI_Iscatter_AsyncHandle<SENDTYPE, RECVTYPE>* asyncHandle =
      static_cast<AMPI_Iscatter_AsyncHandle<SENDTYPE, RECVTYPE>*>(handle);
    typename SENDTYPE::Type* sendbuf = asyncHandle->sendbuf;
    typename SENDTYPE::ModifiedType* sendbufMod = asyncHandle->sendbufMod;
    int sendcount = asyncHandle->sendcount;
    SENDTYPE* sendtype = asyncHandle->sendtype;
    typename RECVTYPE::Type* recvbuf = asyncHandle->recvbuf;
    typename RECVTYPE::ModifiedType* recvbufMod = asyncHandle->recvbufMod;
    int recvcount = asyncHandle->recvcount;
    RECVTYPE* recvtype = asyncHandle->recvtype;
    int root = asyncHandle->root;
    AMPI_Comm comm = asyncHandle->comm;
    AMPI_Request* request = asyncHandle->request;
    AMPI_Iscatter_AdjointHandle<SENDTYPE, RECVTYPE>* h = asyncHandle->h;
    MEDI_UNUSED(sendbuf); // Unused generated to ignore warnings
    MEDI_UNUSED(sendbufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(sendcount); // Unused generated to ignore warnings
    MEDI_UNUSED(sendtype); // Unused generated to ignore warnings
    MEDI_UNUSED(recvbuf); // Unused generated to ignore warnings
    MEDI_UNUSED(recvbufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(recvcount); // Unused generated to ignore warnings
    MEDI_UNUSED(recvtype); // Unused generated to ignore warnings
    MEDI_UNUSED(root); // Unused generated to ignore warnings
    MEDI_UNUSED(comm); // Unused generated to ignore warnings
    MEDI_UNUSED(request); // Unused generated to ignore warnings
    MEDI_UNUSED(h); // Unused generated to ignore warnings

    delete asyncHandle;

    if(recvtype->getADTool().isActiveType()) {

      recvtype->getADTool().addToolAction(h);

      if(recvtype->isModifiedBufferRequired()) {
        if(AMPI_IN_PLACE != recvbuf) {
          recvtype->copyFromModifiedBuffer(recvbuf, 0, recvbufMod, 0, recvcount);
        } else {
          sendtype->copyFromModifiedBuffer(sendbuf, sendcount * getCommRank(comm), sendbufMod, sendcount * getCommRank(comm),
                                           sendcount);
        }
      }

      if(nullptr != h) {
        // handle the recv buffers
        if(AMPI_IN_PLACE != recvbuf) {
          recvtype->registerValue(recvbuf, 0, h->recvbufIndices, h->recvbufOldPrimals, 0, recvcount);
        } else {
          sendtype->registerValue(sendbuf, sendcount * getCommRank(comm), h->recvbufIndices, h->recvbufOldPrimals, 0, sendcount);
        }
      }

      recvtype->getADTool().stopAssembly(h);

      if(root == getCommRank(comm)) {
        if(sendtype->isModifiedBufferRequired() ) {
          sendtype->deleteModifiedTypeBuffer(sendbufMod);
        }
      }
      if(recvtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == recvbuf)) {
        recvtype->deleteModifiedTypeBuffer(recvbufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Iscatterv_AdjointHandle : public HandleBase {
    int sendbufTotalSize;
    typename SENDTYPE::IndexType* sendbufIndices;
    typename SENDTYPE::PassiveType* sendbufPrimals;
    /* required for async */ typename SENDTYPE::PassiveType* sendbufAdjoints;
    int* sendbufCount;
    int* sendbufDispls;
    const  int* sendcounts;
    const  int* displs;
    SENDTYPE* sendtype;
    int recvbufTotalSize;
    typename RECVTYPE::IndexType* recvbufIndices;
    typename RECVTYPE::PassiveType* recvbufPrimals;
    typename RECVTYPE::PassiveType* recvbufOldPrimals;
    /* required for async */ typename RECVTYPE::PassiveType* recvbufAdjoints;
    int recvbufCount;
    int recvcount;
    RECVTYPE* recvtype;
    int root;
    AMPI_Comm comm;
    AMPI_Request requestReverse;

    ~AMPI_Iscatterv_AdjointHandle () {
      if(nullptr != sendbufIndices) {
        sendtype->getADTool().deleteIndexTypeBuffer(sendbufIndices);
        sendbufIndices = nullptr;
      }
      if(nullptr != sendbufPrimals) {
        sendtype->getADTool().deletePassiveTypeBuffer(sendbufPrimals);
        sendbufPrimals = nullptr;
      }
      if(nullptr != sendbufCount) {
        delete [] sendbufCount;
        sendbufCount = nullptr;
      }
      if(nullptr != sendbufDispls) {
        delete [] sendbufDispls;
        sendbufDispls = nullptr;
      }
      if(nullptr != recvbufIndices) {
        recvtype->getADTool().deleteIndexTypeBuffer(recvbufIndices);
        recvbufIndices = nullptr;
      }
      if(nullptr != recvbufPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufPrimals);
        recvbufPrimals = nullptr;
      }
      if(nullptr != recvbufOldPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufOldPrimals);
        recvbufOldPrimals = nullptr;
      }
    }
  };

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Iscatterv_AsyncHandle : public HandleBase {
    typename SENDTYPE::Type* sendbuf;
    typename SENDTYPE::ModifiedType* sendbufMod;
    const int* displsMod;
    const  int* sendcounts;
    const  int* displs;
    SENDTYPE* sendtype;
    typename RECVTYPE::Type* recvbuf;
    typename RECVTYPE::ModifiedType* recvbufMod;
    int recvcount;
    RECVTYPE* recvtype;
    int root;
    AMPI_Comm comm;
    AMPI_Request* request;
    AMPI_Iscatterv_AdjointHandle<SENDTYPE, RECVTYPE>* h;
  };

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Iscatterv_b(HandleBase* handle) {
    AMPI_Iscatterv_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Iscatterv_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);

    h->recvbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->recvbufAdjoints, h->recvbufTotalSize );
    // Adjoint buffers are always linear in space so we can accesses them in one sweep
    h->recvtype->getADTool().getAdjoints(h->recvbufIndices, h->recvbufAdjoints, h->recvbufTotalSize);

    if(h->recvtype->getADTool().isOldPrimalsRequired()) {
      h->recvtype->getADTool().setReverseValues(h->recvbufIndices, h->recvbufOldPrimals, h->recvbufTotalSize);
    }
    h->sendbufAdjoints = nullptr;
    if(h->root == getCommRank(h->comm)) {
      h->recvtype->getADTool().createAdjointTypeBuffer(h->sendbufAdjoints, h->sendbufTotalSize );
    }

    AMPI_Iscatterv_adj<SENDTYPE, RECVTYPE>(h->sendbufAdjoints, h->sendbufCount, h->sendbufDispls, h->sendcounts, h->displs,
                                           h->sendtype, h->recvbufAdjoints, h->recvbufCount, h->recvcount, h->recvtype, h->root, h->comm, &h->requestReverse);

  }

  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Iscatterv_b_finish(HandleBase* handle) {

    AMPI_Iscatterv_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Iscatterv_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);
    MPI_Wait(&h->requestReverse.request, MPI_STATUS_IGNORE);

    if(h->root == getCommRank(h->comm)) {
      // Adjoint buffers are always linear in space so they can be accessed in one sweep
      h->recvtype->getADTool().updateAdjoints(h->sendbufIndices, h->sendbufAdjoints, h->sendbufTotalSize);
      h->recvtype->getADTool().deleteAdjointTypeBuffer(h->sendbufAdjoints);
    }
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->recvbufAdjoints);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Iscatterv_finish(HandleBase* handle);
  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Iscatterv(typename SENDTYPE::Type* sendbuf, const int* sendcounts, const int* displs, SENDTYPE* sendtype,
                     typename RECVTYPE::Type* recvbuf, int recvcount, RECVTYPE* recvtype, int root, AMPI_Comm comm, AMPI_Request* request) {
    int rStatus;
    if(!recvtype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Iscatterv(sendbuf, sendcounts, displs, sendtype->getMpiType(), recvbuf, recvcount, recvtype->getMpiType(),
                              root, comm, &request->request);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Iscatterv_AdjointHandle<SENDTYPE, RECVTYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(recvtype->getADTool().isHandleRequired()) {
        h = new AMPI_Iscatterv_AdjointHandle<SENDTYPE, RECVTYPE>();
      }
      recvtype->getADTool().startAssembly(h);
      const int* displsMod = displs;
      int displsTotalSize = 0;
      if(nullptr != displs) {
        displsTotalSize = computeDisplacementsTotalSize(sendcounts, getCommSize(comm));
        if(recvtype->isModifiedBufferRequired()) {
          displsMod = createLinearDisplacements(sendcounts, getCommSize(comm));
        }
      }
      typename SENDTYPE::ModifiedType* sendbufMod = nullptr;
      int sendbufElements = 0;

      if(root == getCommRank(comm)) {
        // compute the total size of the buffer
        sendbufElements = displsTotalSize;

        if(sendtype->isModifiedBufferRequired() ) {
          sendtype->createModifiedTypeBuffer(sendbufMod, sendbufElements);
        } else {
          sendbufMod = reinterpret_cast<typename SENDTYPE::ModifiedType*>(const_cast<typename SENDTYPE::Type*>(sendbuf));
        }
      }
      typename RECVTYPE::ModifiedType* recvbufMod = nullptr;
      int recvbufElements = 0;

      // compute the total size of the buffer
      if(AMPI_IN_PLACE != recvbuf) {
        recvbufElements = recvcount;
      } else {
        recvbufElements = sendcounts[getCommRank(comm)];
      }

      if(recvtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == recvbuf)) {
        recvtype->createModifiedTypeBuffer(recvbufMod, recvbufElements);
      } else {
        recvbufMod = reinterpret_cast<typename RECVTYPE::ModifiedType*>(const_cast<typename RECVTYPE::Type*>(recvbuf));
      }

      if(root == getCommRank(comm)) {
        if(sendtype->isModifiedBufferRequired()) {
          for(int i = 0; i < getCommSize(comm); ++i) {
            sendtype->copyIntoModifiedBuffer(sendbuf, displs[i], sendbufMod, displsMod[i], sendcounts[i]);
          }
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(root == getCommRank(comm)) {
          createLinearIndexDisplacements(h->sendbufCount, h->sendbufDispls, sendcounts, getCommSize(comm), sendtype);
          h->sendbufTotalSize = sendtype->computeActiveElements(sendbufElements);
          recvtype->getADTool().createIndexTypeBuffer(h->sendbufIndices, h->sendbufTotalSize);
        }
        if(AMPI_IN_PLACE != recvbuf) {
          h->recvbufCount = recvtype->computeActiveElements(recvcount);
        } else {
          h->recvbufCount = sendtype->computeActiveElements(sendcounts[getCommRank(comm)]);
        }
        h->recvbufTotalSize = recvtype->computeActiveElements(recvbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->recvbufIndices, h->recvbufTotalSize);


        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(recvtype->getADTool().isOldPrimalsRequired()) {
          recvtype->getADTool().createPassiveTypeBuffer(h->recvbufOldPrimals, h->recvbufTotalSize);
          if(AMPI_IN_PLACE != recvbuf) {
            recvtype->getValues(recvbuf, 0, h->recvbufOldPrimals, 0, recvcount);
          } else {
            {
              const int rank = getCommRank(comm);
              sendtype->getValues(sendbuf, displs[rank], h->recvbufOldPrimals, 0, sendcounts[rank]);
            }
          }
        }


        if(root == getCommRank(comm)) {
          for(int i = 0; i < getCommSize(comm); ++i) {
            sendtype->getIndices(sendbuf, displs[i], h->sendbufIndices, displsMod[i], sendcounts[i]);
          }
        }

        if(!recvtype->isModifiedBufferRequired()) {
          if(AMPI_IN_PLACE != recvbuf) {
            recvtype->clearIndices(recvbuf, 0, recvcount);
          } else {
            {
              const int rank = getCommRank(comm);
              sendtype->clearIndices(sendbuf, displs[rank], sendcounts[rank]);
            }
          }
        }

        // pack all the variables in the handle
        h->func = AMPI_Iscatterv_b<SENDTYPE, RECVTYPE>;
        h->sendcounts = sendcounts;
        h->displs = displs;
        h->sendtype = sendtype;
        h->recvcount = recvcount;
        h->recvtype = recvtype;
        h->root = root;
        h->comm = comm;
      }

      rStatus = MPI_Iscatterv(sendbufMod, sendcounts, displsMod, sendtype->getModifiedMpiType(), recvbufMod, recvcount,
                              recvtype->getModifiedMpiType(), root, comm, &request->request);

      AMPI_Iscatterv_AsyncHandle<SENDTYPE, RECVTYPE>* asyncHandle = new AMPI_Iscatterv_AsyncHandle<SENDTYPE, RECVTYPE>();
      asyncHandle->sendbuf = sendbuf;
      asyncHandle->sendbufMod = sendbufMod;
      asyncHandle->displsMod = displsMod;
      asyncHandle->sendcounts = sendcounts;
      asyncHandle->displs = displs;
      asyncHandle->sendtype = sendtype;
      asyncHandle->recvbuf = recvbuf;
      asyncHandle->recvbufMod = recvbufMod;
      asyncHandle->recvcount = recvcount;
      asyncHandle->recvtype = recvtype;
      asyncHandle->root = root;
      asyncHandle->comm = comm;
      asyncHandle->h = h;
      request->handle = asyncHandle;
      request->func = (ContinueFunction)AMPI_Iscatterv_finish<SENDTYPE, RECVTYPE>;

      // create adjoint wait
      if(nullptr != h) {
        WaitHandle* waitH = new WaitHandle((ContinueFunction)AMPI_Iscatterv_b_finish<SENDTYPE, RECVTYPE>, h);
        recvtype->getADTool().addToolAction(waitH);
      }
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Iscatterv_finish(HandleBase* handle) {
    int rStatus = 0;

    AMPI_Iscatterv_AsyncHandle<SENDTYPE, RECVTYPE>* asyncHandle =
      static_cast<AMPI_Iscatterv_AsyncHandle<SENDTYPE, RECVTYPE>*>(handle);
    typename SENDTYPE::Type* sendbuf = asyncHandle->sendbuf;
    typename SENDTYPE::ModifiedType* sendbufMod = asyncHandle->sendbufMod;
    const int* displsMod = asyncHandle->displsMod;
    const  int* sendcounts = asyncHandle->sendcounts;
    const  int* displs = asyncHandle->displs;
    SENDTYPE* sendtype = asyncHandle->sendtype;
    typename RECVTYPE::Type* recvbuf = asyncHandle->recvbuf;
    typename RECVTYPE::ModifiedType* recvbufMod = asyncHandle->recvbufMod;
    int recvcount = asyncHandle->recvcount;
    RECVTYPE* recvtype = asyncHandle->recvtype;
    int root = asyncHandle->root;
    AMPI_Comm comm = asyncHandle->comm;
    AMPI_Request* request = asyncHandle->request;
    AMPI_Iscatterv_AdjointHandle<SENDTYPE, RECVTYPE>* h = asyncHandle->h;
    MEDI_UNUSED(sendbuf); // Unused generated to ignore warnings
    MEDI_UNUSED(sendbufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(displsMod); // Unused generated to ignore warnings
    MEDI_UNUSED(sendcounts); // Unused generated to ignore warnings
    MEDI_UNUSED(displs); // Unused generated to ignore warnings
    MEDI_UNUSED(sendtype); // Unused generated to ignore warnings
    MEDI_UNUSED(recvbuf); // Unused generated to ignore warnings
    MEDI_UNUSED(recvbufMod); // Unused generated to ignore warnings
    MEDI_UNUSED(recvcount); // Unused generated to ignore warnings
    MEDI_UNUSED(recvtype); // Unused generated to ignore warnings
    MEDI_UNUSED(root); // Unused generated to ignore warnings
    MEDI_UNUSED(comm); // Unused generated to ignore warnings
    MEDI_UNUSED(request); // Unused generated to ignore warnings
    MEDI_UNUSED(h); // Unused generated to ignore warnings

    delete asyncHandle;

    if(recvtype->getADTool().isActiveType()) {

      recvtype->getADTool().addToolAction(h);

      if(recvtype->isModifiedBufferRequired()) {
        if(AMPI_IN_PLACE != recvbuf) {
          recvtype->copyFromModifiedBuffer(recvbuf, 0, recvbufMod, 0, recvcount);
        } else {
          {
            const int rank = getCommRank(comm);
            sendtype->copyFromModifiedBuffer(sendbuf, displs[rank], sendbufMod, displsMod[rank], sendcounts[rank]);
          }
        }
      }

      if(nullptr != h) {
        // handle the recv buffers
        if(AMPI_IN_PLACE != recvbuf) {
          recvtype->registerValue(recvbuf, 0, h->recvbufIndices, h->recvbufOldPrimals, 0, recvcount);
        } else {
          {
            const int rank = getCommRank(comm);
            sendtype->registerValue(sendbuf, displs[rank], h->recvbufIndices, h->recvbufOldPrimals, 0, sendcounts[rank]);
          }
        }
      }

      recvtype->getADTool().stopAssembly(h);
      if(recvtype->isModifiedBufferRequired()) {
        delete [] displsMod;
      }

      if(root == getCommRank(comm)) {
        if(sendtype->isModifiedBufferRequired() ) {
          sendtype->deleteModifiedTypeBuffer(sendbufMod);
        }
      }
      if(recvtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == recvbuf)) {
        recvtype->deleteModifiedTypeBuffer(recvbufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename DATATYPE>
  struct AMPI_Reduce_global_AdjointHandle : public HandleBase {
    int sendbufTotalSize;
    typename DATATYPE::IndexType* sendbufIndices;
    typename DATATYPE::PassiveType* sendbufPrimals;
    /* required for async */ typename DATATYPE::PassiveType* sendbufAdjoints;
    int sendbufCount;
    int recvbufTotalSize;
    typename DATATYPE::IndexType* recvbufIndices;
    typename DATATYPE::PassiveType* recvbufPrimals;
    typename DATATYPE::PassiveType* recvbufOldPrimals;
    /* required for async */ typename DATATYPE::PassiveType* recvbufAdjoints;
    int recvbufCount;
    int count;
    DATATYPE* datatype;
    AMPI_Op op;
    int root;
    AMPI_Comm comm;

    ~AMPI_Reduce_global_AdjointHandle () {
      if(nullptr != sendbufIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(sendbufIndices);
        sendbufIndices = nullptr;
      }
      if(nullptr != sendbufPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(sendbufPrimals);
        sendbufPrimals = nullptr;
      }
      if(nullptr != recvbufIndices) {
        datatype->getADTool().deleteIndexTypeBuffer(recvbufIndices);
        recvbufIndices = nullptr;
      }
      if(nullptr != recvbufPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(recvbufPrimals);
        recvbufPrimals = nullptr;
      }
      if(nullptr != recvbufOldPrimals) {
        datatype->getADTool().deletePassiveTypeBuffer(recvbufOldPrimals);
        recvbufOldPrimals = nullptr;
      }
    }
  };


  template<typename DATATYPE>
  void AMPI_Reduce_global_b(HandleBase* handle) {
    AMPI_Reduce_global_AdjointHandle<DATATYPE>* h = static_cast<AMPI_Reduce_global_AdjointHandle<DATATYPE>*>(handle);

    h->recvbufAdjoints = nullptr;
    if(h->root == getCommRank(h->comm)) {
      h->datatype->getADTool().createAdjointTypeBuffer(h->recvbufAdjoints, h->recvbufTotalSize );
      // Adjoint buffers are always linear in space so we can accesses them in one sweep
      h->datatype->getADTool().getAdjoints(h->recvbufIndices, h->recvbufAdjoints, h->recvbufTotalSize);

      h->op.preAdjointOperation(h->recvbufAdjoints, h->recvbufPrimals, h->recvbufCount);
    }
    if(h->datatype->getADTool().isOldPrimalsRequired()) {
      if(h->root == getCommRank(h->comm)) {
        h->datatype->getADTool().setReverseValues(h->recvbufIndices, h->recvbufOldPrimals, h->recvbufTotalSize);
      }
    }
    h->sendbufAdjoints = nullptr;
    h->datatype->getADTool().createAdjointTypeBuffer(h->sendbufAdjoints, h->sendbufTotalSize );

    AMPI_Reduce_global_adj<DATATYPE>(h->sendbufAdjoints, h->sendbufCount, h->recvbufAdjoints, h->recvbufCount, h->count,
                                     h->datatype, h->op, h->root, h->comm);

    // the primals of the recive buffer are always given to the function. The operator should ignore them if not needed.
    // The wrapper functions make sure that for operators that need the primals an all* action is perfomed (e.g. Allreduce instead of Reduce)
    h->op.postAdjointOperation(h->sendbufAdjoints, h->sendbufPrimals, h->recvbufPrimals, h->sendbufTotalSize);
    // Adjoint buffers are always linear in space so they can be accessed in one sweep
    h->datatype->getADTool().updateAdjoints(h->sendbufIndices, h->sendbufAdjoints, h->sendbufTotalSize);
    h->datatype->getADTool().deleteAdjointTypeBuffer(h->sendbufAdjoints);
    if(h->root == getCommRank(h->comm)) {
      h->datatype->getADTool().deleteAdjointTypeBuffer(h->recvbufAdjoints);
    }
  }

  template<typename DATATYPE>
  int AMPI_Reduce_global(const typename DATATYPE::Type* sendbuf, typename DATATYPE::Type* recvbuf, int count,
                         DATATYPE* datatype, AMPI_Op op, int root, AMPI_Comm comm) {
    int rStatus;
    if(!datatype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Reduce(sendbuf, recvbuf, count, datatype->getMpiType(), op.primalFunction, root, comm);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Reduce_global_AdjointHandle<DATATYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(datatype->getADTool().isHandleRequired()) {
        h = new AMPI_Reduce_global_AdjointHandle<DATATYPE>();
      }
      datatype->getADTool().startAssembly(h);
      typename DATATYPE::ModifiedType* sendbufMod = nullptr;
      int sendbufElements = 0;

      // compute the total size of the buffer
      if(AMPI_IN_PLACE != sendbuf) {
        sendbufElements = count;
      } else {
        sendbufElements = count;
      }

      if(datatype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        datatype->createModifiedTypeBuffer(sendbufMod, sendbufElements);
      } else {
        sendbufMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(sendbuf));
      }
      typename DATATYPE::ModifiedType* recvbufMod = nullptr;
      int recvbufElements = 0;

      if(root == getCommRank(comm)) {
        // compute the total size of the buffer
        recvbufElements = count;

        if(datatype->isModifiedBufferRequired() ) {
          datatype->createModifiedTypeBuffer(recvbufMod, recvbufElements);
        } else {
          recvbufMod = reinterpret_cast<typename DATATYPE::ModifiedType*>(const_cast<typename DATATYPE::Type*>(recvbuf));
        }
      }

      if(datatype->isModifiedBufferRequired()) {
        if(AMPI_IN_PLACE != sendbuf) {
          datatype->copyIntoModifiedBuffer(sendbuf, 0, sendbufMod, 0, count);
        } else {
          datatype->copyIntoModifiedBuffer(recvbuf, 0, recvbufMod, 0, count);
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(AMPI_IN_PLACE != sendbuf) {
          h->sendbufCount = datatype->computeActiveElements(count);
        } else {
          h->sendbufCount = datatype->computeActiveElements(count);
        }
        h->sendbufTotalSize = datatype->computeActiveElements(sendbufElements);
        datatype->getADTool().createIndexTypeBuffer(h->sendbufIndices, h->sendbufTotalSize);
        if(root == getCommRank(comm)) {
          h->recvbufCount = datatype->computeActiveElements(count);
          h->recvbufTotalSize = datatype->computeActiveElements(recvbufElements);
          datatype->getADTool().createIndexTypeBuffer(h->recvbufIndices, h->recvbufTotalSize);
        }

        // extract the primal values for the operator if required
        if(op.requiresPrimal) {
          datatype->getADTool().createPassiveTypeBuffer(h->sendbufPrimals, h->sendbufTotalSize);
          if(AMPI_IN_PLACE != sendbuf) {
            datatype->getValues(sendbuf, 0, h->sendbufPrimals, 0, count);
          } else {
            datatype->getValues(recvbuf, 0, h->sendbufPrimals, 0, count);
          }
        }

        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(datatype->getADTool().isOldPrimalsRequired()) {
          if(root == getCommRank(comm)) {
            datatype->getADTool().createPassiveTypeBuffer(h->recvbufOldPrimals, h->recvbufTotalSize);
            if(root == getCommRank(comm)) {
              datatype->getValues(recvbuf, 0, h->recvbufOldPrimals, 0, count);
            }
          }
        }


        if(AMPI_IN_PLACE != sendbuf) {
          datatype->getIndices(sendbuf, 0, h->sendbufIndices, 0, count);
        } else {
          datatype->getIndices(recvbuf, 0, h->sendbufIndices, 0, count);
        }

        if(root == getCommRank(comm)) {
          if(!datatype->isModifiedBufferRequired()) {
            datatype->clearIndices(recvbuf, 0, count);
          }
        }

        // pack all the variables in the handle
        h->func = AMPI_Reduce_global_b<DATATYPE>;
        h->count = count;
        h->datatype = datatype;
        h->op = op;
        h->root = root;
        h->comm = comm;
      }

      rStatus = MPI_Reduce(sendbufMod, recvbufMod, count, datatype->getModifiedMpiType(), op.modifiedPrimalFunction, root,
                           comm);
      datatype->getADTool().addToolAction(h);

      if(root == getCommRank(comm)) {
        if(datatype->isModifiedBufferRequired()) {
          datatype->copyFromModifiedBuffer(recvbuf, 0, recvbufMod, 0, count);
        }
      }

      if(nullptr != h) {
        // handle the recv buffers
        if(root == getCommRank(comm)) {
          datatype->registerValue(recvbuf, 0, h->recvbufIndices, h->recvbufOldPrimals, 0, count);
        }
      }
      // extract the primal values for the operator if required
      if(nullptr != h && op.requiresPrimal) {
        if(root == getCommRank(comm)) {
          datatype->getADTool().createPassiveTypeBuffer(h->recvbufPrimals, h->recvbufTotalSize);
          if(root == getCommRank(comm)) {
            datatype->getValues(recvbuf, 0, h->recvbufPrimals, 0, count);
          }
        }
      }

      datatype->getADTool().stopAssembly(h);

      if(datatype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == sendbuf)) {
        datatype->deleteModifiedTypeBuffer(sendbufMod);
      }
      if(root == getCommRank(comm)) {
        if(datatype->isModifiedBufferRequired() ) {
          datatype->deleteModifiedTypeBuffer(recvbufMod);
        }
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Scatter_AdjointHandle : public HandleBase {
    int sendbufTotalSize;
    typename SENDTYPE::IndexType* sendbufIndices;
    typename SENDTYPE::PassiveType* sendbufPrimals;
    /* required for async */ typename SENDTYPE::PassiveType* sendbufAdjoints;
    int sendbufCount;
    int sendcount;
    SENDTYPE* sendtype;
    int recvbufTotalSize;
    typename RECVTYPE::IndexType* recvbufIndices;
    typename RECVTYPE::PassiveType* recvbufPrimals;
    typename RECVTYPE::PassiveType* recvbufOldPrimals;
    /* required for async */ typename RECVTYPE::PassiveType* recvbufAdjoints;
    int recvbufCount;
    int recvcount;
    RECVTYPE* recvtype;
    int root;
    AMPI_Comm comm;

    ~AMPI_Scatter_AdjointHandle () {
      if(nullptr != sendbufIndices) {
        sendtype->getADTool().deleteIndexTypeBuffer(sendbufIndices);
        sendbufIndices = nullptr;
      }
      if(nullptr != sendbufPrimals) {
        sendtype->getADTool().deletePassiveTypeBuffer(sendbufPrimals);
        sendbufPrimals = nullptr;
      }
      if(nullptr != recvbufIndices) {
        recvtype->getADTool().deleteIndexTypeBuffer(recvbufIndices);
        recvbufIndices = nullptr;
      }
      if(nullptr != recvbufPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufPrimals);
        recvbufPrimals = nullptr;
      }
      if(nullptr != recvbufOldPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufOldPrimals);
        recvbufOldPrimals = nullptr;
      }
    }
  };


  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Scatter_b(HandleBase* handle) {
    AMPI_Scatter_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Scatter_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);

    h->recvbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->recvbufAdjoints, h->recvbufTotalSize );
    // Adjoint buffers are always linear in space so we can accesses them in one sweep
    h->recvtype->getADTool().getAdjoints(h->recvbufIndices, h->recvbufAdjoints, h->recvbufTotalSize);

    if(h->recvtype->getADTool().isOldPrimalsRequired()) {
      h->recvtype->getADTool().setReverseValues(h->recvbufIndices, h->recvbufOldPrimals, h->recvbufTotalSize);
    }
    h->sendbufAdjoints = nullptr;
    if(h->root == getCommRank(h->comm)) {
      h->recvtype->getADTool().createAdjointTypeBuffer(h->sendbufAdjoints, h->sendbufTotalSize );
    }

    AMPI_Scatter_adj<SENDTYPE, RECVTYPE>(h->sendbufAdjoints, h->sendbufCount, h->sendcount, h->sendtype, h->recvbufAdjoints,
                                         h->recvbufCount, h->recvcount, h->recvtype, h->root, h->comm);

    if(h->root == getCommRank(h->comm)) {
      // Adjoint buffers are always linear in space so they can be accessed in one sweep
      h->recvtype->getADTool().updateAdjoints(h->sendbufIndices, h->sendbufAdjoints, h->sendbufTotalSize);
      h->recvtype->getADTool().deleteAdjointTypeBuffer(h->sendbufAdjoints);
    }
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->recvbufAdjoints);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Scatter(typename SENDTYPE::Type* sendbuf, int sendcount, SENDTYPE* sendtype, typename RECVTYPE::Type* recvbuf,
                   int recvcount, RECVTYPE* recvtype, int root, AMPI_Comm comm) {
    int rStatus;
    if(!recvtype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Scatter(sendbuf, sendcount, sendtype->getMpiType(), recvbuf, recvcount, recvtype->getMpiType(), root,
                            comm);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Scatter_AdjointHandle<SENDTYPE, RECVTYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(recvtype->getADTool().isHandleRequired()) {
        h = new AMPI_Scatter_AdjointHandle<SENDTYPE, RECVTYPE>();
      }
      recvtype->getADTool().startAssembly(h);
      typename SENDTYPE::ModifiedType* sendbufMod = nullptr;
      int sendbufElements = 0;

      if(root == getCommRank(comm)) {
        // compute the total size of the buffer
        sendbufElements = sendcount * getCommSize(comm);

        if(sendtype->isModifiedBufferRequired() ) {
          sendtype->createModifiedTypeBuffer(sendbufMod, sendbufElements);
        } else {
          sendbufMod = reinterpret_cast<typename SENDTYPE::ModifiedType*>(const_cast<typename SENDTYPE::Type*>(sendbuf));
        }
      }
      typename RECVTYPE::ModifiedType* recvbufMod = nullptr;
      int recvbufElements = 0;

      // compute the total size of the buffer
      if(AMPI_IN_PLACE != recvbuf) {
        recvbufElements = recvcount;
      } else {
        recvbufElements = sendcount;
      }

      if(recvtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == recvbuf)) {
        recvtype->createModifiedTypeBuffer(recvbufMod, recvbufElements);
      } else {
        recvbufMod = reinterpret_cast<typename RECVTYPE::ModifiedType*>(const_cast<typename RECVTYPE::Type*>(recvbuf));
      }

      if(root == getCommRank(comm)) {
        if(sendtype->isModifiedBufferRequired()) {
          sendtype->copyIntoModifiedBuffer(sendbuf, 0, sendbufMod, 0, sendcount * getCommSize(comm));
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(root == getCommRank(comm)) {
          h->sendbufCount = sendtype->computeActiveElements(sendcount);
          h->sendbufTotalSize = sendtype->computeActiveElements(sendbufElements);
          recvtype->getADTool().createIndexTypeBuffer(h->sendbufIndices, h->sendbufTotalSize);
        }
        if(AMPI_IN_PLACE != recvbuf) {
          h->recvbufCount = recvtype->computeActiveElements(recvcount);
        } else {
          h->recvbufCount = sendtype->computeActiveElements(sendcount);
        }
        h->recvbufTotalSize = recvtype->computeActiveElements(recvbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->recvbufIndices, h->recvbufTotalSize);


        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(recvtype->getADTool().isOldPrimalsRequired()) {
          recvtype->getADTool().createPassiveTypeBuffer(h->recvbufOldPrimals, h->recvbufTotalSize);
          if(AMPI_IN_PLACE != recvbuf) {
            recvtype->getValues(recvbuf, 0, h->recvbufOldPrimals, 0, recvcount);
          } else {
            sendtype->getValues(sendbuf, sendcount * getCommRank(comm), h->recvbufOldPrimals, 0, sendcount);
          }
        }


        if(root == getCommRank(comm)) {
          sendtype->getIndices(sendbuf, 0, h->sendbufIndices, 0, sendcount * getCommSize(comm));
        }

        if(!recvtype->isModifiedBufferRequired()) {
          if(AMPI_IN_PLACE != recvbuf) {
            recvtype->clearIndices(recvbuf, 0, recvcount);
          } else {
            sendtype->clearIndices(sendbuf, sendcount * getCommRank(comm), sendcount);
          }
        }

        // pack all the variables in the handle
        h->func = AMPI_Scatter_b<SENDTYPE, RECVTYPE>;
        h->sendcount = sendcount;
        h->sendtype = sendtype;
        h->recvcount = recvcount;
        h->recvtype = recvtype;
        h->root = root;
        h->comm = comm;
      }

      rStatus = MPI_Scatter(sendbufMod, sendcount, sendtype->getModifiedMpiType(), recvbufMod, recvcount,
                            recvtype->getModifiedMpiType(), root, comm);
      recvtype->getADTool().addToolAction(h);

      if(recvtype->isModifiedBufferRequired()) {
        if(AMPI_IN_PLACE != recvbuf) {
          recvtype->copyFromModifiedBuffer(recvbuf, 0, recvbufMod, 0, recvcount);
        } else {
          sendtype->copyFromModifiedBuffer(sendbuf, sendcount * getCommRank(comm), sendbufMod, sendcount * getCommRank(comm),
                                           sendcount);
        }
      }

      if(nullptr != h) {
        // handle the recv buffers
        if(AMPI_IN_PLACE != recvbuf) {
          recvtype->registerValue(recvbuf, 0, h->recvbufIndices, h->recvbufOldPrimals, 0, recvcount);
        } else {
          sendtype->registerValue(sendbuf, sendcount * getCommRank(comm), h->recvbufIndices, h->recvbufOldPrimals, 0, sendcount);
        }
      }

      recvtype->getADTool().stopAssembly(h);

      if(root == getCommRank(comm)) {
        if(sendtype->isModifiedBufferRequired() ) {
          sendtype->deleteModifiedTypeBuffer(sendbufMod);
        }
      }
      if(recvtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == recvbuf)) {
        recvtype->deleteModifiedTypeBuffer(recvbufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }

  template<typename SENDTYPE, typename RECVTYPE>
  struct AMPI_Scatterv_AdjointHandle : public HandleBase {
    int sendbufTotalSize;
    typename SENDTYPE::IndexType* sendbufIndices;
    typename SENDTYPE::PassiveType* sendbufPrimals;
    /* required for async */ typename SENDTYPE::PassiveType* sendbufAdjoints;
    int* sendbufCount;
    int* sendbufDispls;
    const  int* sendcounts;
    const  int* displs;
    SENDTYPE* sendtype;
    int recvbufTotalSize;
    typename RECVTYPE::IndexType* recvbufIndices;
    typename RECVTYPE::PassiveType* recvbufPrimals;
    typename RECVTYPE::PassiveType* recvbufOldPrimals;
    /* required for async */ typename RECVTYPE::PassiveType* recvbufAdjoints;
    int recvbufCount;
    int recvcount;
    RECVTYPE* recvtype;
    int root;
    AMPI_Comm comm;

    ~AMPI_Scatterv_AdjointHandle () {
      if(nullptr != sendbufIndices) {
        sendtype->getADTool().deleteIndexTypeBuffer(sendbufIndices);
        sendbufIndices = nullptr;
      }
      if(nullptr != sendbufPrimals) {
        sendtype->getADTool().deletePassiveTypeBuffer(sendbufPrimals);
        sendbufPrimals = nullptr;
      }
      if(nullptr != sendbufCount) {
        delete [] sendbufCount;
        sendbufCount = nullptr;
      }
      if(nullptr != sendbufDispls) {
        delete [] sendbufDispls;
        sendbufDispls = nullptr;
      }
      if(nullptr != recvbufIndices) {
        recvtype->getADTool().deleteIndexTypeBuffer(recvbufIndices);
        recvbufIndices = nullptr;
      }
      if(nullptr != recvbufPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufPrimals);
        recvbufPrimals = nullptr;
      }
      if(nullptr != recvbufOldPrimals) {
        recvtype->getADTool().deletePassiveTypeBuffer(recvbufOldPrimals);
        recvbufOldPrimals = nullptr;
      }
    }
  };


  template<typename SENDTYPE, typename RECVTYPE>
  void AMPI_Scatterv_b(HandleBase* handle) {
    AMPI_Scatterv_AdjointHandle<SENDTYPE, RECVTYPE>* h = static_cast<AMPI_Scatterv_AdjointHandle<SENDTYPE, RECVTYPE>*>
        (handle);

    h->recvbufAdjoints = nullptr;
    h->recvtype->getADTool().createAdjointTypeBuffer(h->recvbufAdjoints, h->recvbufTotalSize );
    // Adjoint buffers are always linear in space so we can accesses them in one sweep
    h->recvtype->getADTool().getAdjoints(h->recvbufIndices, h->recvbufAdjoints, h->recvbufTotalSize);

    if(h->recvtype->getADTool().isOldPrimalsRequired()) {
      h->recvtype->getADTool().setReverseValues(h->recvbufIndices, h->recvbufOldPrimals, h->recvbufTotalSize);
    }
    h->sendbufAdjoints = nullptr;
    if(h->root == getCommRank(h->comm)) {
      h->recvtype->getADTool().createAdjointTypeBuffer(h->sendbufAdjoints, h->sendbufTotalSize );
    }

    AMPI_Scatterv_adj<SENDTYPE, RECVTYPE>(h->sendbufAdjoints, h->sendbufCount, h->sendbufDispls, h->sendcounts, h->displs,
                                          h->sendtype, h->recvbufAdjoints, h->recvbufCount, h->recvcount, h->recvtype, h->root, h->comm);

    if(h->root == getCommRank(h->comm)) {
      // Adjoint buffers are always linear in space so they can be accessed in one sweep
      h->recvtype->getADTool().updateAdjoints(h->sendbufIndices, h->sendbufAdjoints, h->sendbufTotalSize);
      h->recvtype->getADTool().deleteAdjointTypeBuffer(h->sendbufAdjoints);
    }
    h->recvtype->getADTool().deleteAdjointTypeBuffer(h->recvbufAdjoints);
  }

  template<typename SENDTYPE, typename RECVTYPE>
  int AMPI_Scatterv(typename SENDTYPE::Type* sendbuf, const int* sendcounts, const int* displs, SENDTYPE* sendtype,
                    typename RECVTYPE::Type* recvbuf, int recvcount, RECVTYPE* recvtype, int root, AMPI_Comm comm) {
    int rStatus;
    if(!recvtype->getADTool().isActiveType()) {
      // call the regular function if the type is not active
      rStatus = MPI_Scatterv(sendbuf, sendcounts, displs, sendtype->getMpiType(), recvbuf, recvcount, recvtype->getMpiType(),
                             root, comm);
    } else {

      // the type is an AD type so handle the buffers
      AMPI_Scatterv_AdjointHandle<SENDTYPE, RECVTYPE>* h = nullptr;
      // the handle is created if a reverse action should be recorded, h != nullptr => tape is active
      if(recvtype->getADTool().isHandleRequired()) {
        h = new AMPI_Scatterv_AdjointHandle<SENDTYPE, RECVTYPE>();
      }
      recvtype->getADTool().startAssembly(h);
      const int* displsMod = displs;
      int displsTotalSize = 0;
      if(nullptr != displs) {
        displsTotalSize = computeDisplacementsTotalSize(sendcounts, getCommSize(comm));
        if(recvtype->isModifiedBufferRequired()) {
          displsMod = createLinearDisplacements(sendcounts, getCommSize(comm));
        }
      }
      typename SENDTYPE::ModifiedType* sendbufMod = nullptr;
      int sendbufElements = 0;

      if(root == getCommRank(comm)) {
        // compute the total size of the buffer
        sendbufElements = displsTotalSize;

        if(sendtype->isModifiedBufferRequired() ) {
          sendtype->createModifiedTypeBuffer(sendbufMod, sendbufElements);
        } else {
          sendbufMod = reinterpret_cast<typename SENDTYPE::ModifiedType*>(const_cast<typename SENDTYPE::Type*>(sendbuf));
        }
      }
      typename RECVTYPE::ModifiedType* recvbufMod = nullptr;
      int recvbufElements = 0;

      // compute the total size of the buffer
      if(AMPI_IN_PLACE != recvbuf) {
        recvbufElements = recvcount;
      } else {
        recvbufElements = sendcounts[getCommRank(comm)];
      }

      if(recvtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == recvbuf)) {
        recvtype->createModifiedTypeBuffer(recvbufMod, recvbufElements);
      } else {
        recvbufMod = reinterpret_cast<typename RECVTYPE::ModifiedType*>(const_cast<typename RECVTYPE::Type*>(recvbuf));
      }

      if(root == getCommRank(comm)) {
        if(sendtype->isModifiedBufferRequired()) {
          for(int i = 0; i < getCommSize(comm); ++i) {
            sendtype->copyIntoModifiedBuffer(sendbuf, displs[i], sendbufMod, displsMod[i], sendcounts[i]);
          }
        }
      }

      if(nullptr != h) {
        // gather the information for the reverse sweep

        // create the index buffers
        if(root == getCommRank(comm)) {
          createLinearIndexDisplacements(h->sendbufCount, h->sendbufDispls, sendcounts, getCommSize(comm), sendtype);
          h->sendbufTotalSize = sendtype->computeActiveElements(sendbufElements);
          recvtype->getADTool().createIndexTypeBuffer(h->sendbufIndices, h->sendbufTotalSize);
        }
        if(AMPI_IN_PLACE != recvbuf) {
          h->recvbufCount = recvtype->computeActiveElements(recvcount);
        } else {
          h->recvbufCount = sendtype->computeActiveElements(sendcounts[getCommRank(comm)]);
        }
        h->recvbufTotalSize = recvtype->computeActiveElements(recvbufElements);
        recvtype->getADTool().createIndexTypeBuffer(h->recvbufIndices, h->recvbufTotalSize);


        // extract the old primal values from the recv buffer if the AD tool
        // needs the primal values reset
        if(recvtype->getADTool().isOldPrimalsRequired()) {
          recvtype->getADTool().createPassiveTypeBuffer(h->recvbufOldPrimals, h->recvbufTotalSize);
          if(AMPI_IN_PLACE != recvbuf) {
            recvtype->getValues(recvbuf, 0, h->recvbufOldPrimals, 0, recvcount);
          } else {
            {
              const int rank = getCommRank(comm);
              sendtype->getValues(sendbuf, displs[rank], h->recvbufOldPrimals, 0, sendcounts[rank]);
            }
          }
        }


        if(root == getCommRank(comm)) {
          for(int i = 0; i < getCommSize(comm); ++i) {
            sendtype->getIndices(sendbuf, displs[i], h->sendbufIndices, displsMod[i], sendcounts[i]);
          }
        }

        if(!recvtype->isModifiedBufferRequired()) {
          if(AMPI_IN_PLACE != recvbuf) {
            recvtype->clearIndices(recvbuf, 0, recvcount);
          } else {
            {
              const int rank = getCommRank(comm);
              sendtype->clearIndices(sendbuf, displs[rank], sendcounts[rank]);
            }
          }
        }

        // pack all the variables in the handle
        h->func = AMPI_Scatterv_b<SENDTYPE, RECVTYPE>;
        h->sendcounts = sendcounts;
        h->displs = displs;
        h->sendtype = sendtype;
        h->recvcount = recvcount;
        h->recvtype = recvtype;
        h->root = root;
        h->comm = comm;
      }

      rStatus = MPI_Scatterv(sendbufMod, sendcounts, displsMod, sendtype->getModifiedMpiType(), recvbufMod, recvcount,
                             recvtype->getModifiedMpiType(), root, comm);
      recvtype->getADTool().addToolAction(h);

      if(recvtype->isModifiedBufferRequired()) {
        if(AMPI_IN_PLACE != recvbuf) {
          recvtype->copyFromModifiedBuffer(recvbuf, 0, recvbufMod, 0, recvcount);
        } else {
          {
            const int rank = getCommRank(comm);
            sendtype->copyFromModifiedBuffer(sendbuf, displs[rank], sendbufMod, displsMod[rank], sendcounts[rank]);
          }
        }
      }

      if(nullptr != h) {
        // handle the recv buffers
        if(AMPI_IN_PLACE != recvbuf) {
          recvtype->registerValue(recvbuf, 0, h->recvbufIndices, h->recvbufOldPrimals, 0, recvcount);
        } else {
          {
            const int rank = getCommRank(comm);
            sendtype->registerValue(sendbuf, displs[rank], h->recvbufIndices, h->recvbufOldPrimals, 0, sendcounts[rank]);
          }
        }
      }

      recvtype->getADTool().stopAssembly(h);
      if(recvtype->isModifiedBufferRequired()) {
        delete [] displsMod;
      }

      if(root == getCommRank(comm)) {
        if(sendtype->isModifiedBufferRequired() ) {
          sendtype->deleteModifiedTypeBuffer(sendbufMod);
        }
      }
      if(recvtype->isModifiedBufferRequired()  && !(AMPI_IN_PLACE == recvbuf)) {
        recvtype->deleteModifiedTypeBuffer(recvbufMod);
      }

      // handle is deleted by the AD tool
    }

    return rStatus;
  }


  inline int AMPI_Buffer_attach(void* buffer, int size) {
    return MPI_Buffer_attach(buffer, size);
  }

  inline int AMPI_Buffer_detach(void* buffer_addr, int* size) {
    return MPI_Buffer_detach(buffer_addr, size);
  }

  inline int AMPI_Cancel(AMPI_Request* request) {
    return MPI_Cancel(&request->request);
  }

  template<typename DATATYPE>
  inline int AMPI_Get_count(const AMPI_Status* status, DATATYPE* datatype, int* count) {
    return MPI_Get_count(status, datatype->getModifiedMpiType(), count);
  }

  inline int AMPI_Improbe(int source, int tag, AMPI_Comm comm, int* flag, AMPI_Message* message, AMPI_Status* status) {
    return MPI_Improbe(source, tag, comm, flag, message, status);
  }

  inline int AMPI_Iprobe(int source, int tag, AMPI_Comm comm, int* flag, AMPI_Status* status) {
    return MPI_Iprobe(source, tag, comm, flag, status);
  }

  inline int AMPI_Mprobe(int source, int tag, AMPI_Comm comm, AMPI_Message* message, AMPI_Status* status) {
    return MPI_Mprobe(source, tag, comm, message, status);
  }

  inline int AMPI_Probe(int source, int tag, AMPI_Comm comm, AMPI_Status* status) {
    return MPI_Probe(source, tag, comm, status);
  }

  inline int AMPI_Request_get_status(AMPI_Request request, int* flag, AMPI_Status* status) {
    return MPI_Request_get_status(request.request, flag, status);
  }

  inline int AMPI_Test_cancelled(const AMPI_Status* status, int* flag) {
    return MPI_Test_cancelled(status, flag);
  }

  inline int AMPI_Barrier(AMPI_Comm comm) {
    return MPI_Barrier(comm);
  }

  inline int AMPI_Comm_compare(AMPI_Comm comm1, AMPI_Comm comm2, int* result) {
    return MPI_Comm_compare(comm1, comm2, result);
  }

  inline int AMPI_Comm_create(AMPI_Comm comm, AMPI_Group group, AMPI_Comm* newcomm) {
    return MPI_Comm_create(comm, group, newcomm);
  }

  inline int AMPI_Comm_create_group(AMPI_Comm comm, AMPI_Group group, int tag, AMPI_Comm* newcomm) {
    return MPI_Comm_create_group(comm, group, tag, newcomm);
  }

  inline int AMPI_Comm_create_keyval(AMPI_Comm_copy_attr_function* comm_copy_attr_fn,
                                     AMPI_Comm_delete_attr_function* comm_delete_attr_fn, int* comm_keyval, void* extra_state) {
    return MPI_Comm_create_keyval(comm_copy_attr_fn, comm_delete_attr_fn, comm_keyval, extra_state);
  }

  inline int AMPI_Comm_delete_attr(AMPI_Comm comm, int comm_keyval) {
    return MPI_Comm_delete_attr(comm, comm_keyval);
  }

  inline int AMPI_Comm_dup(AMPI_Comm comm, AMPI_Comm* newcomm) {
    return MPI_Comm_dup(comm, newcomm);
  }

  inline int AMPI_Comm_dup_with_info(AMPI_Comm comm, AMPI_Info info, AMPI_Comm* newcomm) {
    return MPI_Comm_dup_with_info(comm, info, newcomm);
  }

  inline int AMPI_Comm_free(AMPI_Comm* comm) {
    return MPI_Comm_free(comm);
  }

  inline int AMPI_Comm_free_keyval(int* comm_keyval) {
    return MPI_Comm_free_keyval(comm_keyval);
  }

  inline int AMPI_Comm_get_attr(AMPI_Comm comm, int comm_keyval, void* attribute_val, int* flag) {
    return MPI_Comm_get_attr(comm, comm_keyval, attribute_val, flag);
  }

  inline int AMPI_Comm_get_info(AMPI_Comm comm, AMPI_Info* info_used) {
    return MPI_Comm_get_info(comm, info_used);
  }

  inline int AMPI_Comm_get_name(AMPI_Comm comm, char* comm_name, int* resultlen) {
    return MPI_Comm_get_name(comm, comm_name, resultlen);
  }

  inline int AMPI_Comm_group(AMPI_Comm comm, AMPI_Group* group) {
    return MPI_Comm_group(comm, group);
  }

  inline int AMPI_Comm_idup(AMPI_Comm comm, AMPI_Comm* newcomm, AMPI_Request* request) {
    return MPI_Comm_idup(comm, newcomm, &request->request);
  }

  inline int AMPI_Comm_rank(AMPI_Comm comm, int* rank) {
    return MPI_Comm_rank(comm, rank);
  }

  inline int AMPI_Comm_remote_group(AMPI_Comm comm, AMPI_Group* group) {
    return MPI_Comm_remote_group(comm, group);
  }

  inline int AMPI_Comm_remote_size(AMPI_Comm comm, int* size) {
    return MPI_Comm_remote_size(comm, size);
  }

  inline int AMPI_Comm_set_attr(AMPI_Comm comm, int comm_keyval, void* attribute_val) {
    return MPI_Comm_set_attr(comm, comm_keyval, attribute_val);
  }

  inline int AMPI_Comm_set_info(AMPI_Comm comm, AMPI_Info info) {
    return MPI_Comm_set_info(comm, info);
  }

  inline int AMPI_Comm_set_name(AMPI_Comm comm, const char* comm_name) {
    return MPI_Comm_set_name(comm, comm_name);
  }

  inline int AMPI_Comm_size(AMPI_Comm comm, int* size) {
    return MPI_Comm_size(comm, size);
  }

  inline int AMPI_Comm_split(AMPI_Comm comm, int color, int key, AMPI_Comm* newcomm) {
    return MPI_Comm_split(comm, color, key, newcomm);
  }

  inline int AMPI_Comm_split_type(AMPI_Comm comm, int split_type, int key, AMPI_Info info, AMPI_Comm* newcomm) {
    return MPI_Comm_split_type(comm, split_type, key, info, newcomm);
  }

  inline int AMPI_Comm_test_inter(AMPI_Comm comm, int* flag) {
    return MPI_Comm_test_inter(comm, flag);
  }

  inline int AMPI_Group_compare(AMPI_Group group1, AMPI_Group group2, int* result) {
    return MPI_Group_compare(group1, group2, result);
  }

  inline int AMPI_Group_difference(AMPI_Group group1, AMPI_Group group2, AMPI_Group* newgroup) {
    return MPI_Group_difference(group1, group2, newgroup);
  }

  inline int AMPI_Group_excl(AMPI_Group group, int n, const int* ranks, AMPI_Group* newgroup) {
    return MPI_Group_excl(group, n, ranks, newgroup);
  }

  inline int AMPI_Group_free(AMPI_Group* group) {
    return MPI_Group_free(group);
  }

  inline int AMPI_Group_incl(AMPI_Group group, int n, const int* ranks, AMPI_Group* newgroup) {
    return MPI_Group_incl(group, n, ranks, newgroup);
  }

  inline int AMPI_Group_intersection(AMPI_Group group1, AMPI_Group group2, AMPI_Group* newgroup) {
    return MPI_Group_intersection(group1, group2, newgroup);
  }

  inline int AMPI_Group_range_excl(AMPI_Group group, int n, Range* ranges, AMPI_Group* newgroup) {
    return MPI_Group_range_excl(group, n, ranges, newgroup);
  }

  inline int AMPI_Group_range_incl(AMPI_Group group, int n, Range* ranges, AMPI_Group* newgroup) {
    return MPI_Group_range_incl(group, n, ranges, newgroup);
  }

  inline int AMPI_Group_rank(AMPI_Group group, int* rank) {
    return MPI_Group_rank(group, rank);
  }

  inline int AMPI_Group_size(AMPI_Group group, int* size) {
    return MPI_Group_size(group, size);
  }

  inline int AMPI_Group_translate_ranks(AMPI_Group group1, int n, const int* ranks1, AMPI_Group group2, int* ranks2) {
    return MPI_Group_translate_ranks(group1, n, ranks1, group2, ranks2);
  }

  inline int AMPI_Group_union(AMPI_Group group1, AMPI_Group group2, AMPI_Group* newgroup) {
    return MPI_Group_union(group1, group2, newgroup);
  }

  inline int AMPI_Intercomm_create(AMPI_Comm local_comm, int local_leader, AMPI_Comm peer_comm, int remote_leader,
                                   int tag, AMPI_Comm* newintercomm) {
    return MPI_Intercomm_create(local_comm, local_leader, peer_comm, remote_leader, tag, newintercomm);
  }

  inline int AMPI_Intercomm_merge(AMPI_Comm intercomm, int high, AMPI_Comm* newintracomm) {
    return MPI_Intercomm_merge(intercomm, high, newintracomm);
  }

  inline int AMPI_Type_create_keyval(AMPI_Type_copy_attr_function* type_copy_attr_fn,
                                     AMPI_Type_delete_attr_function* type_delete_attr_fn, int* type_keyval, void* extra_state) {
    return MPI_Type_create_keyval(type_copy_attr_fn, type_delete_attr_fn, type_keyval, extra_state);
  }

  template<typename DATATYPE>
  inline int AMPI_Type_delete_attr(DATATYPE* datatype, int type_keyval) {
    return MPI_Type_delete_attr(datatype->getModifiedMpiType(), type_keyval);
  }

  inline int AMPI_Type_free_keyval(int* type_keyval) {
    return MPI_Type_free_keyval(type_keyval);
  }

  template<typename DATATYPE>
  inline int AMPI_Type_get_attr(DATATYPE* datatype, int type_keyval, void* attribute_val, int* flag) {
    return MPI_Type_get_attr(datatype->getModifiedMpiType(), type_keyval, attribute_val, flag);
  }

  template<typename DATATYPE>
  inline int AMPI_Type_get_name(DATATYPE* datatype, char* type_name, int* resultlen) {
    return MPI_Type_get_name(datatype->getModifiedMpiType(), type_name, resultlen);
  }

  template<typename DATATYPE>
  inline int AMPI_Type_set_attr(DATATYPE* datatype, int type_keyval, void* attribute_val) {
    return MPI_Type_set_attr(datatype->getModifiedMpiType(), type_keyval, attribute_val);
  }

  template<typename DATATYPE>
  inline int AMPI_Type_set_name(DATATYPE* datatype, const char* type_name) {
    return MPI_Type_set_name(datatype->getModifiedMpiType(), type_name);
  }

  inline int AMPI_Win_create_keyval(AMPI_Win_copy_attr_function* win_copy_attr_fn,
                                    AMPI_Win_delete_attr_function* win_delete_attr_fn, int* win_keyval, void* extra_state) {
    return MPI_Win_create_keyval(win_copy_attr_fn, win_delete_attr_fn, win_keyval, extra_state);
  }

  inline int AMPI_Win_delete_attr(AMPI_Win win, int win_keyval) {
    return MPI_Win_delete_attr(win, win_keyval);
  }

  inline int AMPI_Win_free_keyval(int* win_keyval) {
    return MPI_Win_free_keyval(win_keyval);
  }

  inline int AMPI_Win_get_attr(AMPI_Win win, int win_keyval, void* attribute_val, int* flag) {
    return MPI_Win_get_attr(win, win_keyval, attribute_val, flag);
  }

  inline int AMPI_Win_get_name(AMPI_Win win, char* win_name, int* resultlen) {
    return MPI_Win_get_name(win, win_name, resultlen);
  }

  inline int AMPI_Win_set_attr(AMPI_Win win, int win_keyval, void* attribute_val) {
    return MPI_Win_set_attr(win, win_keyval, attribute_val);
  }

  inline int AMPI_Win_set_name(AMPI_Win win, const char* win_name) {
    return MPI_Win_set_name(win, win_name);
  }

  inline double AMPI_Wtick() {
    return MPI_Wtick();
  }

  inline double AMPI_Wtime() {
    return MPI_Wtime();
  }

  inline int AMPI_Abort(AMPI_Comm comm, int errorcode) {
    return MPI_Abort(comm, errorcode);
  }

  inline int AMPI_Add_error_class(int* errorclass) {
    return MPI_Add_error_class(errorclass);
  }

  inline int AMPI_Add_error_code(int errorclass, int* errorcode) {
    return MPI_Add_error_code(errorclass, errorcode);
  }

  inline int AMPI_Add_error_string(int errorcode, const char* string) {
    return MPI_Add_error_string(errorcode, string);
  }

  inline int AMPI_Alloc_mem(AMPI_Aint size, AMPI_Info info, void* baseptr) {
    return MPI_Alloc_mem(size, info, baseptr);
  }

  inline int AMPI_Comm_call_errhandler(AMPI_Comm comm, int errorcode) {
    return MPI_Comm_call_errhandler(comm, errorcode);
  }

  inline int AMPI_Comm_create_errhandler(AMPI_Comm_errhandler_function* comm_errhandler_fn, AMPI_Errhandler* errhandler) {
    return MPI_Comm_create_errhandler(comm_errhandler_fn, errhandler);
  }

  inline int AMPI_Comm_get_errhandler(AMPI_Comm comm, AMPI_Errhandler* errhandler) {
    return MPI_Comm_get_errhandler(comm, errhandler);
  }

  inline int AMPI_Comm_set_errhandler(AMPI_Comm comm, AMPI_Errhandler errhandler) {
    return MPI_Comm_set_errhandler(comm, errhandler);
  }

  inline int AMPI_Errhandler_free(AMPI_Errhandler* errhandler) {
    return MPI_Errhandler_free(errhandler);
  }

  inline int AMPI_Error_class(int errorcode, int* errorclass) {
    return MPI_Error_class(errorcode, errorclass);
  }

  inline int AMPI_Error_string(int errorcode, char* string, int* resultlen) {
    return MPI_Error_string(errorcode, string, resultlen);
  }

  inline int AMPI_File_call_errhandler(AMPI_File fh, int errorcode) {
    return MPI_File_call_errhandler(fh, errorcode);
  }

  inline int AMPI_File_create_errhandler(AMPI_File_errhandler_function* file_errhandler_fn, AMPI_Errhandler* errhandler) {
    return MPI_File_create_errhandler(file_errhandler_fn, errhandler);
  }

  inline int AMPI_File_get_errhandler(AMPI_File file, AMPI_Errhandler* errhandler) {
    return MPI_File_get_errhandler(file, errhandler);
  }

  inline int AMPI_File_set_errhandler(AMPI_File file, AMPI_Errhandler errhandler) {
    return MPI_File_set_errhandler(file, errhandler);
  }

  inline int AMPI_Finalize() {
    return MPI_Finalize();
  }

  inline int AMPI_Finalized(int* flag) {
    return MPI_Finalized(flag);
  }

  inline int AMPI_Free_mem(void* base) {
    return MPI_Free_mem(base);
  }

  inline int AMPI_Get_library_version(char* version, int* resultlen) {
    return MPI_Get_library_version(version, resultlen);
  }

  inline int AMPI_Get_processor_name(char* name, int* resultlen) {
    return MPI_Get_processor_name(name, resultlen);
  }

  inline int AMPI_Get_version(int* version, int* subversion) {
    return MPI_Get_version(version, subversion);
  }

  inline int AMPI_Initialized(int* flag) {
    return MPI_Initialized(flag);
  }

  inline int AMPI_Win_call_errhandler(AMPI_Win win, int errorcode) {
    return MPI_Win_call_errhandler(win, errorcode);
  }

  inline int AMPI_Win_create_errhandler(AMPI_Win_errhandler_function* win_errhandler_fn, AMPI_Errhandler* errhandler) {
    return MPI_Win_create_errhandler(win_errhandler_fn, errhandler);
  }

  inline int AMPI_Win_get_errhandler(AMPI_Win win, AMPI_Errhandler* errhandler) {
    return MPI_Win_get_errhandler(win, errhandler);
  }

  inline int AMPI_Win_set_errhandler(AMPI_Win win, AMPI_Errhandler errhandler) {
    return MPI_Win_set_errhandler(win, errhandler);
  }


}
