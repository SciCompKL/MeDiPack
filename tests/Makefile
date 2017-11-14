#
# MeDiPack, a Message Differentiation Package
#
# Copyright (C) 2017 Chair for Scientific Computing (SciComp), TU Kaiserslautern
# Homepage: http://www.scicomp.uni-kl.de
# Contact:  Prof. Nicolas R. Gauger (codi@scicomp.uni-kl.de)
#
# Lead developers: Max Sagebaum (SciComp, TU Kaiserslautern)
#
# This file is part of MeDiPack (http://www.scicomp.uni-kl.de/software/codi).
#
# MeDiPack is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# MeDiPack is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU General Public License for more details.
# You should have received a copy of the GNU
# General Public License along with MeDiPack.
# If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Max Sagebaum (SciComp, TU Kaiserslautern)
#

# names of the basic deriectories
TEST_DIR = tests
BUILD_DIR = build
DRIVER_DIR = drivers
RESULT_BASE_DIR = results_base
RESULT_DIR = results

#list all source files in TEST_DIR
TEST_FILES   = $(wildcard $(TEST_DIR)/Test**.cpp)
TEST_FILES  += $(wildcard $(TEST_DIR)/**/Test**.cpp)
TEST_FILES  += $(wildcard $(TEST_DIR)/**/**/Test**.cpp)

#list all dependency files in BUILD_DIR
DEP_FILES   = $(wildcard $(BUILD_DIR)/Test**.d)
DEP_FILES  += $(wildcard $(BUILD_DIR)/**/Test**.d)
DEP_FILES  += $(wildcard $(BUILD_DIR)/**/**/Test**.d)

FLAGS = -Wall -pedantic -std=c++11 -I../include -I../src

# The default is to run all drives
DRIVERS?=ALL

ifeq ($(OPT), yes)
  CXX_FLAGS := -O3 $(FLAGS)
else
  CXX_FLAGS := -O0 -g $(FLAGS)
endif

COMPILER ?= gnu

ifeq ($(COMPILER), gnu)
  MPICXX ?= mpic++
  MPIRUN ?= mpiexec
  MPIOUT ?= --output-filename $@
endif
ifeq ($(COMPILER), intel)
  MPICXX ?= mpiicpc -wd3802
  MPIRUN ?= mpiexec.hydra
  MPIOUT ?= --outfile-pattern=$@.%r
endif

# Complete list of test files
TESTS = $(patsubst $(TEST_DIR)/%.cpp,$(RESULT_DIR)/%.test,$(TEST_FILES))

# set default rule
all:

# disable the deletion of secondary targets
.SECONDARY:

# define general sets for tests
BASIC_TESTS = $(wildcard $(TEST_DIR)/misc/Test**.cpp) $(wildcard $(TEST_DIR)/datatypes/Test**.cpp) $(wildcard $(TEST_DIR)/collective/Test**.cpp) $(wildcard $(TEST_DIR)/collective/inplace/Test**.cpp) $(wildcard $(TEST_DIR)/pointToPoint/Test**.cpp) $(wildcard $(TEST_DIR)/wait_test/Test**.cpp)
FORWARD_TESTS = $(wildcard $(TEST_DIR)/forward/Test**.cpp)

# The build rules for all drivers.
define DRIVER_RULE
$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : $(TEST_DIR)/%.cpp
	@mkdir -p $(@D)
	$(MPICXX) $(CXX_FLAGS) $< -o $@ $(DRIVER)
	@$(MPICXX) $(CXX_FLAGS) $(DRIVER) $< -MM -MP -MT $@ -MF $@.d
DRIVER_TEST_FILES := $(patsubst $(TEST_DIR)/%.cpp,$(RESULT_DIR)/%.test,$(DRIVER_TESTS))
$(DRIVER_TEST_FILES): $(RESULT_DIR)/%.test: $(RESULT_DIR)/%_$(DRIVER_NAME).out
endef

# The general template for each driver. It checks if the driver should be run
define DRIVER_INST
  ifeq (${DRIVER_NAME}, $(filter ${DRIVER_NAME}, ${DRIVERS}))
    $(eval $(value DRIVER_RULE))
  else
    ifeq (ALL, $(filter ALL, ${DRIVERS}))
      $(eval $(value DRIVER_RULE))
    endif
  endif
endef

# The first 3 lines of each driver define the basic parameters for this driver.
# DRIVER_NAME:    The short name for the driver. This used to create the specific files for the driver
# DRIVER_TESTS:   The full list of tests which are run for this driver. See the general test list for details.
# DRIVER:         The special flags for this driver. It needs to be defined inside the build rule.
# The other lines create the rules and dependencies to run the tests for this driver.

# Driver for RealReverse
DRIVER_NAME  := CoDi
DRIVER_TESTS := $(BASIC_TESTS)
$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER = $(DRIVER_DIR)/codi/codiDriver.cpp -I$(CODI_DIR)/include -I$(CODI_DIR)/source -I$(DRIVER_DIR)/codi -DCODI_TYPE=codi::RealReverse
$(eval $(value DRIVER_INST))

# Driver for RealReverse
DRIVER_NAME  := CoDiVec
DRIVER_TESTS := $(BASIC_TESTS)
$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER = $(DRIVER_DIR)/codi/codiDriver.cpp -I$(CODI_DIR)/include -I$(CODI_DIR)/source -I$(DRIVER_DIR)/codi -DCODI_TYPE="codi::RealReverseVec<2>" -DVECTOR
$(eval $(value DRIVER_INST))

# Driver for RealReverseIndex
DRIVER_NAME  := CoDiIndex
DRIVER_TESTS := $(BASIC_TESTS)
$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER = $(DRIVER_DIR)/codi/codiDriver.cpp -I$(CODI_DIR)/include -I$(CODI_DIR)/source -I$(DRIVER_DIR)/codi -DCODI_TYPE=codi::RealReverseIndex
$(eval $(value DRIVER_INST))

# Driver for RealReversePrimal
DRIVER_NAME  := CoDiPrimal
DRIVER_TESTS := $(BASIC_TESTS)
$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER = $(DRIVER_DIR)/codi/codiDriver.cpp -I$(CODI_DIR)/include -I$(CODI_DIR)/source -I$(DRIVER_DIR)/codi -DCODI_TYPE=codi::RealReversePrimal
$(eval $(value DRIVER_INST))

# Driver for RealReversePrimalIndex
DRIVER_NAME  := CoDiPrimalIndex
DRIVER_TESTS := $(BASIC_TESTS)
$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER = $(DRIVER_DIR)/codi/codiDriver.cpp -I$(CODI_DIR)/include -I$(CODI_DIR)/source -I$(DRIVER_DIR)/codi -DCODI_TYPE=codi::RealReversePrimalIndex -DPRIMAL_RESTORE
$(eval $(value DRIVER_INST))

# Driver for RealReverse with untyped interface
DRIVER_NAME  := CoDiUntyped
DRIVER_TESTS := $(BASIC_TESTS)
$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER = $(DRIVER_DIR)/codiUntyped/codiDriver.cpp -I$(CODI_DIR)/include -I$(CODI_DIR)/source -I$(DRIVER_DIR)/codiUntyped
$(eval $(value DRIVER_INST))

# Driver for RealReverse with untyped interface
DRIVER_NAME  := CoDiForward
DRIVER_TESTS := $(FORWARD_TESTS)
$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER = $(DRIVER_DIR)/codiForward/codiDriver.cpp -I$(CODI_DIR)/include -I$(CODI_DIR)/source -I$(DRIVER_DIR)/codiForward -DCODI_TYPE=codi::RealForward
$(eval $(value DRIVER_INST))

# Driver for ADOL-c
DRIVER_NAME  := ADOL-c
DRIVER_TESTS := $(BASIC_TESTS)
$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER = $(DRIVER_DIR)/adolc/adolcDriver.cpp -I$(ADOLC_DIR)/include -L$(ADOLC_DIR)/lib64 -ladolc -I$(DRIVER_DIR)/adolc
$(eval $(value DRIVER_INST))

# rules for generating the test files
$(RESULT_DIR)/%.out : $(BUILD_DIR)/%_bin
	@mkdir -p $(@D)
	$(MPIRUN) -n 2 $(MPIOUT) $<
	@cat $@.* > $@
	@rm $@.*

# rule for printing the results (dependencies are generated by the drivers)
$(RESULT_DIR)/%.test:
	@./compare.sh -n $* -b $(RESULT_BASE_DIR)/$*.out $^

all: $(TESTS)
	@mkdir -p $(BUILD_DIR)

.PHONY: clean
clean:
	rm -fr $(BUILD_DIR)
	rm -fr $(RESULT_DIR)

-include $(DEP_FILES)
