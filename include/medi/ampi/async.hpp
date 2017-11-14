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

#include "ampiMisc.h"

#include "../../../generated/medi/ampiDefinitions.h"

/**
 * @brief Global namespace for MeDiPack - Message Differentiation Package
 */
namespace medi {

  typedef void (*DeleteReverseData)(void* data);

  struct AMPI_Request {
      MPI_Request request;
      HandleBase* handle;
      ContinueFunction func;

      // required for reverse communication that needs to create data
      void* reverseData;
      DeleteReverseData deleteDataFunc;

      AMPI_Request() :
        request(MPI_REQUEST_NULL),
        handle(NULL),
        func(NULL),
        reverseData(NULL),
        deleteDataFunc(NULL){}

      inline void setReverseData(void* data, DeleteReverseData func) {
        this->reverseData = data;
        this->deleteDataFunc = func;
      }

      inline void deleteReverseData() {
        if(NULL != reverseData) {
           this->deleteDataFunc(this->reverseData);
        }
      }
  };

  inline bool operator ==(const AMPI_Request& a, const AMPI_Request& b) {
    return a.request == b.request;
  }

  inline bool operator !=(const AMPI_Request& a, const AMPI_Request& b) {
    return a.request != b.request;
  }

  extern const AMPI_Request AMPI_REQUEST_NULL;

  inline void AMPI_Wait_b(HandleBase* handle);
  struct WaitHandle : HandleBase {
      ReverseFunction finishFunc;
      HandleBase* handle;

      WaitHandle(ReverseFunction finishFunc, HandleBase* handle) :
        finishFunc(finishFunc),
        handle(handle) {
         this->func = (ReverseFunction)AMPI_Wait_b;
        this->deleteType = ManualDeleteType::Wait;
        this->handle->deleteType = ManualDeleteType::Async;
      }
  };

  inline void AMPI_Wait_b(HandleBase* handle) {
    WaitHandle* h = static_cast<WaitHandle*>(handle);

    h->finishFunc(h->handle);
  }

  inline void performReverseAction(AMPI_Request *request) {
    if(nullptr != request->func) {
      request->func(request->handle);

      request->deleteReverseData();
    }

    *request = AMPI_REQUEST_NULL;
  }

  inline MPI_Request* convertToMPI(AMPI_Request* array, int count) {
    MPI_Request* converted = new MPI_Request[count];

    for(int i = 0; i < count; ++i) {
      converted[i] = array[i].request;
    }

    return converted;
  }

#if MEDI_MPI_VERSION_1_0 <= MEDI_MPI_TARGET
  inline int AMPI_Wait(AMPI_Request *request, AMPI_Status *status) {
    if(AMPI_REQUEST_NULL == *request) {
      return 0;
    }

    int rStatus = MPI_Wait(&request->request, status);

    performReverseAction(request);

    return rStatus;
  }
#endif

#if MEDI_MPI_VERSION_1_0 <= MEDI_MPI_TARGET
  inline int AMPI_Test(AMPI_Request *request, int* flag, AMPI_Status *status) {
    if(AMPI_REQUEST_NULL == *request) {
      *flag = (int)true;
      return 0;
    }

    int rStatus = MPI_Test(&request->request, flag, status);

    if(true == *flag) {
      performReverseAction(request);
    }

    return rStatus;
  }
#endif

#if MEDI_MPI_VERSION_1_0 <= MEDI_MPI_TARGET
  inline int AMPI_Request_free(AMPI_Request *request) {
    if(AMPI_REQUEST_NULL == *request) {
      return 0;
    }

    int rStatus = MPI_Request_free(&request->request);
    *request = AMPI_REQUEST_NULL;

    return rStatus;
  }
#endif

#if MEDI_MPI_VERSION_1_0 <= MEDI_MPI_TARGET
  inline int AMPI_Waitany(int count, AMPI_Request* array_of_requests, int *index, AMPI_Status *status) {
    MPI_Request* array = convertToMPI(array_of_requests, count);

    int rStatus = MPI_Waitany(count, array, index, status);

    if(MPI_UNDEFINED != *index) {
      performReverseAction(&array_of_requests[*index]);
    }

    delete [] array;

    return rStatus;
  }
#endif

#if MEDI_MPI_VERSION_1_0 <= MEDI_MPI_TARGET
  inline int AMPI_Testany(int count, AMPI_Request* array_of_requests, int *index, int *flag, AMPI_Status *status) {
    MPI_Request* array = convertToMPI(array_of_requests, count);

    int rStatus = MPI_Testany(count, array, index, flag, status);

    if(true == *flag) {
      if(MPI_UNDEFINED != *index) {
        performReverseAction(&array_of_requests[*index]);
      }
    }

    delete [] array;

    return rStatus;
  }
#endif

#if MEDI_MPI_VERSION_1_0 <= MEDI_MPI_TARGET
  inline int AMPI_Waitall(int count, AMPI_Request* array_of_requests, AMPI_Status* array_of_statuses) {
    MPI_Request* array = convertToMPI(array_of_requests, count);

    int rStatus = MPI_Waitall(count, array, array_of_statuses);

    for(int i = 0; i < count; ++i) {
      if(AMPI_REQUEST_NULL != array_of_requests[i]) {
        performReverseAction(&array_of_requests[i]);
      }
    }

    delete [] array;

    return rStatus;
  }
#endif

#if MEDI_MPI_VERSION_1_0 <= MEDI_MPI_TARGET
  inline int AMPI_Testall(int count, AMPI_Request* array_of_requests, int *flag, AMPI_Status* array_of_statuses) {
    MPI_Request* array = convertToMPI(array_of_requests, count);

    int rStatus = MPI_Testall(count, array, flag, array_of_statuses);

    if(true == *flag) {
      for(int i = 0; i < count; ++i) {
        if(AMPI_REQUEST_NULL != array_of_requests[i]) {
          performReverseAction(&array_of_requests[i]);
        }
      }
    }

    delete [] array;

    return rStatus;
  }
#endif

#if MEDI_MPI_VERSION_1_0 <= MEDI_MPI_TARGET
  inline int AMPI_Waitsome(int incount, AMPI_Request* array_of_requests, int *outcount, int* array_of_indices, AMPI_Status* array_of_statuses) {
    MPI_Request* array = convertToMPI(array_of_requests, incount);

    int rStatus = MPI_Waitsome(incount, array, outcount, array_of_indices, array_of_statuses);

    for(int i = 0; i < *outcount; ++i) {
      int index = array_of_indices[i];
      if(AMPI_REQUEST_NULL != array_of_requests[index]) {
        performReverseAction(&array_of_requests[index]);
      }
    }

    delete [] array;

    return rStatus;
  }
#endif

#if MEDI_MPI_VERSION_1_0 <= MEDI_MPI_TARGET
  inline int AMPI_Testsome(int incount, AMPI_Request* array_of_requests, int *outcount, int* array_of_indices, AMPI_Status* array_of_statuses) {
    MPI_Request* array = convertToMPI(array_of_requests, incount);

    int rStatus = MPI_Testsome(incount, array, outcount, array_of_indices, array_of_statuses);

    for(int i = 0; i < *outcount; ++i) {
      int index = array_of_indices[i];
      if(AMPI_REQUEST_NULL != array_of_requests[index]) {
        performReverseAction(&array_of_requests[index]);
      }
    }

    delete [] array;

    return rStatus;
  }
#endif

#if MEDI_MPI_VERSION_3_0 <= MEDI_MPI_TARGET
  inline int AMPI_Ibarrier(AMPI_Comm comm, AMPI_Request *request) {

    int rStatus = MPI_Ibarrier(comm, &request->request);
    request->func = nullptr;

    return rStatus;
  }
#endif
}
