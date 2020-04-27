#
# MeDiPack, a Message Differentiation Package
#
# Copyright (C) 2017-2020 Chair for Scientific Computing (SciComp), TU Kaiserslautern
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
# Authors: Max Sagebaum, Tim Albring (SciComp, TU Kaiserslautern)
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
  MPIOUT ?= --output-filename $@.dir
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
BASIC_TESTS = $(wildcard $(TEST_DIR)/misc/Test**.cpp) $(wildcard $(TEST_DIR)/datatypes/Test**.cpp) $(wildcard $(TEST_DIR)/collective/Test**.cpp) $(wildcard $(TEST_DIR)/collective/inplace/Test**.cpp) $(wildcard $(TEST_DIR)/pointToPoint/Test**.cpp) $(wildcard $(TEST_DIR)/pointToPoint/init/Test**.cpp) $(wildcard $(TEST_DIR)/wait_test/Test**.cpp)
FORWARD_TESTS = $(wildcard $(TEST_DIR)/forward/Test**.cpp)
PRIMAL_TESTS = $(wildcard $(TEST_DIR)/primal/Test**.cpp)

# The build rules for all drivers.
define DRIVER_RULE
$(BUILD_DIR)/$(DRIVER_NAME).o : $(DRIVER_SRC)
	@mkdir -p $(@D)
	$(MPICXX) $(CXX_FLAGS) $(DRIVER_INC) $< -c -o $@
	@$(MPICXX) $(CXX_FLAGS) $(DRIVER_INC) $< -MM -MP -MT $@ -MF $@.d

$(BUILD_DIR)/%_$(DRIVER_NAME).o : $(TEST_DIR)/%.cpp
	@mkdir -p $(@D)
	$(MPICXX) $(CXX_FLAGS) $(DRIVER_INC) $< -c -o $@
	@$(MPICXX) $(CXX_FLAGS) $(DRIVER_INC) $< -MM -MP -MT $@ -MF $@.d

$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : $(BUILD_DIR)/%_$(DRIVER_NAME).o $(BUILD_DIR)/$(DRIVER_NAME).o
	@mkdir -p $(@D)
	$(MPICXX) $(CXX_FLAGS) $(DRIVER_LIB) $^ -o $@

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
DRIVER_SRC = $(DRIVER_DIR)/codi/codiDriver.cpp
$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER_INC = -I$(CODI_DIR)/include -I$(CODI_DIR)/source -I$(DRIVER_DIR)/codi -DCODI_TYPE=codi::RealReverse
$(eval $(value DRIVER_INST))

# Driver for RealReverse
DRIVER_NAME  := CoDiVec
DRIVER_TESTS := $(BASIC_TESTS)
DRIVER_SRC = $(DRIVER_DIR)/codi/codiDriver.cpp
$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER_INC = -I$(CODI_DIR)/include -I$(CODI_DIR)/source -I$(DRIVER_DIR)/codi -DCODI_TYPE="codi::RealReverseVec<2>" -DVECTOR
$(eval $(value DRIVER_INST))

# Driver for RealReverseIndex
DRIVER_NAME  := CoDiIndex
DRIVER_TESTS := $(BASIC_TESTS)
DRIVER_SRC = $(DRIVER_DIR)/codi/codiDriver.cpp
$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER_INC = -I$(CODI_DIR)/include -I$(CODI_DIR)/source -I$(DRIVER_DIR)/codi -DCODI_TYPE=codi::RealReverseIndex
$(eval $(value DRIVER_INST))

# Driver for RealReversePrimal
DRIVER_NAME  := CoDiPrimal
DRIVER_TESTS := $(BASIC_TESTS)
DRIVER_SRC = $(DRIVER_DIR)/codi/codiDriver.cpp
$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER_INC = -I$(CODI_DIR)/include -I$(CODI_DIR)/source -I$(DRIVER_DIR)/codi -DCODI_TYPE=codi::RealReversePrimal
$(eval $(value DRIVER_INST))

# Driver for RealReversePrimalIndex
DRIVER_NAME  := CoDiPrimalIndex
DRIVER_TESTS := $(BASIC_TESTS)
DRIVER_SRC = $(DRIVER_DIR)/codi/codiDriver.cpp
$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER_INC = -I$(CODI_DIR)/include -I$(CODI_DIR)/source -I$(DRIVER_DIR)/codi -DCODI_TYPE=codi::RealReversePrimalIndex
$(eval $(value DRIVER_INST))

# Driver for RealReverse with untyped interface
DRIVER_NAME  := CoDiUntyped
DRIVER_TESTS := $(BASIC_TESTS)
DRIVER_SRC = $(DRIVER_DIR)/codi/codiDriver.cpp
$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER_INC = -I$(CODI_DIR)/include -I$(CODI_DIR)/source -I$(DRIVER_DIR)/codi -DUNTYPED -DCODI_TYPE=codi::RealReverse

$(eval $(value DRIVER_INST))

# Driver for RealForward
DRIVER_NAME  := CoDiForward
DRIVER_TESTS := $(FORWARD_TESTS)
DRIVER_SRC = $(DRIVER_DIR)/codi/codiForwardDriver.cpp
$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER_INC = -I$(CODI_DIR)/include -I$(CODI_DIR)/source -I$(DRIVER_DIR)/codi -DCODI_TYPE=codi::RealForward
$(eval $(value DRIVER_INST))

# Driver for RealReverse with a forward tape evaluation
DRIVER_NAME  := CoDiTapeForward
DRIVER_TESTS := $(FORWARD_TESTS)
DRIVER_SRC = $(DRIVER_DIR)/codi/codiDriver.cpp
$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER_INC = -I$(CODI_DIR)/include -I$(CODI_DIR)/source -I$(DRIVER_DIR)/codi -DCODI_TYPE=codi::RealReverse -DFORWARD_TAPE
$(eval $(value DRIVER_INST))

# Driver for RealReverse with a primal tape evaluation
DRIVER_NAME  := CoDiTapePrimal
DRIVER_TESTS := $(PRIMAL_TESTS)
DRIVER_SRC = $(DRIVER_DIR)/codi/codiDriver.cpp
$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER_INC = -I$(CODI_DIR)/include -I$(CODI_DIR)/source -I$(DRIVER_DIR)/codi -DCODI_TYPE=codi::RealReversePrimal -DPRIMAL_TAPE
$(eval $(value DRIVER_INST))

# Driver for 2nd order
DRIVER_NAME  := CoDi2nd
DRIVER_TESTS := $(BASIC_TESTS)
DRIVER_SRC = $(DRIVER_DIR)/codi/codiDriver.cpp
$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER_INC = -I$(CODI_DIR)/include -I$(CODI_DIR)/source -I$(DRIVER_DIR)/codi -DCODI_TYPE="codi::RealReverseGen<codi::RealForward>"
$(eval $(value DRIVER_INST))

## Driver for ADOL-c
#DRIVER_NAME  := ADOL-c
#DRIVER_TESTS := $(BASIC_TESTS)
#DRIVER_SRC = $(DRIVER_DIR)/adolc/adolcDriver.cpp
#$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER_INC = -I$(ADOLC_DIR)/include -L$(ADOLC_DIR)/lib64 -ladolc -I$(DRIVER_DIR)/adolc
#$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER_LIB = -L$(ADOLC_DIR)/lib64 -ladolc
#$(eval $(value DRIVER_INST))
#
## Driver for ADOL-c forward mode
#DRIVER_NAME  := ADOL-cForward
#DRIVER_TESTS := $(FORWARD_TESTS)
#DRIVER_SRC = $(DRIVER_DIR)/adolcForward/adolcDriver.cpp
#$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER_INC = -I$(ADOLC_DIR)/include -L$(ADOLC_DIR)/lib64 -ladolc -I$(DRIVER_DIR)/adolcForward
#$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER_LIB = -L$(ADOLC_DIR)/lib64 -ladolc
#$(eval $(value DRIVER_INST))
#
## Driver for ADOL-c primal mode
#DRIVER_NAME  := ADOL-cPrimal
#DRIVER_TESTS := $(PRIMAL_TESTS)
#DRIVER_SRC = $(DRIVER_DIR)/adolcPrimal/adolcDriver.cpp
#$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER_INC = -I$(ADOLC_DIR)/include -L$(ADOLC_DIR)/lib64 -ladolc -I$(DRIVER_DIR)/adolcPrimal
#$(BUILD_DIR)/%_$(DRIVER_NAME)_bin : DRIVER_LIB = -L$(ADOLC_DIR)/lib64 -ladolc
#$(eval $(value DRIVER_INST))

# rules for generating the test files
$(RESULT_DIR)/%.out : $(BUILD_DIR)/%_bin
	@mkdir -p $(@D)
	$(MPIRUN) -n 2 $(MPIOUT) $<
	@cat $@.dir/1/rank.*/stdout > $@
	@rm -r $@.dir

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
