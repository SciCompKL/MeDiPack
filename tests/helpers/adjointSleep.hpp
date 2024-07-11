#pragma once

#include <iostream>
#include <unistd.h>

#ifdef CODI_MAJOR_VERSION
template<typename Tape>
void sleep_rev(Tape* t, void* d, codi::VectorAccessInterface<typename Tape::Real, typename Tape::Identifier>* va) {
  int* data = (int*)d;

  usleep(*data);
}

template<typename Tape>
void sleep_del(Tape* t, void* d) {
  int* data = (int*)d;

  delete data;
}

template<typename Real>
void addAdjointSleep(int microseconds) {
  using Tape = typename Real::Tape;
  Tape& tape = Real::getTape();

  int* data = new int(microseconds);
  tape.pushExternalFunction(codi::ExternalFunction<Tape>::create(sleep_rev<Tape>, data, sleep_del<Tape>));
}
#else
template<typename Real>
void addAdjointSleep(int microseconds) {
  std::cerr << "Adjoint sleep not implemented for tool." << std::endl;
}

#endif
