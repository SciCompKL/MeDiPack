#
# MeDiPack, a Message Differentiation Package
#
# Copyright (C) 2017-2024 Chair for Scientific Computing (SciComp), University of Kaiserslautern-Landau
# Homepage: http://www.scicomp.uni-kl.de
# Contact:  Prof. Nicolas R. Gauger (codi@scicomp.uni-kl.de)
#
# Lead developers: Max Sagebaum (SciComp, University of Kaiserslautern-Landau)
#
# This file is part of MeDiPack (http://www.scicomp.uni-kl.de/software/codi).
#
# MeDiPack is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# MeDiPack is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU
# Lesser General Public License along with MeDiPack.
# If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Max Sagebaum, Tim Albring (SciComp, University of Kaiserslautern-Landau)
#

INCLUDE_DIR=include
SRC_DIR=src
TEMPL_DIR=templates
DEF_DIR=definitions
BUILD_DIR = build
DOC_DIR   = doc
MEDI_DIR := .

MAJOR_VERSION = $(shell grep -oP 'define MEDI_MAJOR_VERSION \K\d+' include/medi/medi.hpp)
MINOR_VERSION = $(shell grep -oP 'define MEDI_MINOR_VERSION \K\d+' include/medi/medi.hpp)
BUILD_VERSION = $(shell grep -oP 'define MEDI_BUILD_VERSION \K\d+' include/medi/medi.hpp)
MEDI_VERSION = $(MAJOR_VERSION).$(MINOR_VERSION).$(BUILD_VERSION)

GEN_DIR=include/medi/generated

GENERATED_FILES= \
  $(GEN_DIR)/ampiFunctions.hpp \
  $(GEN_DIR)/ampiDefinitions.cpp \
  $(GEN_DIR)/ampiDefinitions.h

ASTYLE_FILE=template.style

ASTYLE:=$(shell command -v astyle 2> /dev/null)
ifdef ASTYLE
  ASTYLE_CMD=astyle --options=$(ASTYLE_FILE)
else
	ASTYLE_CMD=cat
endif

#list all source files in DOC_DIR
DOC_FILES   = $(wildcard $(DOC_DIR)/*.cpp)
#list all dependency files in BUILD_DIR
DEP_FILES   = $(wildcard $(BUILD_DIR)/*.d)
# Complete list of test files
TUTORIALS = $(patsubst $(DOC_DIR)/%.cpp,$(BUILD_DIR)/%.exe,$(DOC_FILES))

FLAGS = -Wall -pedantic -std=c++11 -I$(CODI_DIR)/include -I$(MEDI_DIR)/include -I$(MEDI_DIR)/src

ifeq ($(OPT), yes)
  CXX_FLAGS := -O3 $(FLAGS)
else
  CXX_FLAGS := -O0 -g $(FLAGS)
endif

ifeq ($(MPICXX), )
	MPICXX := mpic++
else
	MPICXX := $(MPICXX)
endif

all: $(GEN_DIR) $(GENERATED_FILES)

# define the dependencies for all the files
$(GEN_DIR)/ampiFunctions.hpp: 	$(TEMPL_DIR)/medi/ampiFunctions_hpp.gsl   $(DEF_DIR)/mpiFunctions.xml
$(GEN_DIR)/ampiDefinitions.cpp: $(TEMPL_DIR)/medi/ampiDefinitions_cpp.gsl $(DEF_DIR)/mpiDefinitions.xml
$(GEN_DIR)/ampiDefinitions.h:   $(TEMPL_DIR)/medi/ampiDefinitions_h.gsl   $(DEF_DIR)/mpiDefinitions.xml

# directory generation rules
$(GEN_DIR):
	mkdir -p $(GEN_DIR)

# the generation rules
$(GEN_DIR)/%.hpp:$(TEMPL_DIR)/medi/%_hpp.gsl
	gsl -script:$< -a $(filter-out $<,$^) $@.tmp
	$(ASTYLE_CMD) < $@.tmp > $@
	@rm $@.tmp

$(GEN_DIR)/%.h:$(TEMPL_DIR)/medi/%_h.gsl
	gsl -script:$< -a $(filter-out $<,$^) $@.tmp
	$(ASTYLE_CMD) < $@.tmp > $@
	@rm $@.tmp

$(GEN_DIR)/%.cpp:$(TEMPL_DIR)/medi/%_cpp.gsl
	gsl -script:$< -a $(filter-out $<,$^) $@.tmp
	$(ASTYLE_CMD) < $@.tmp > $@
	@rm $@.tmp

#rules for the tutorial files
$(BUILD_DIR)/%.exe : $(DOC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(MPICXX) $(CXX_FLAGS) $< -o $@
	@$(MPICXX) $(CXX_FLAGS) $< -MM -MP -MT $@ -MF $@.d

tutorials: $(TUTORIALS)
	@mkdir -p $(BUILD_DIR)

.PHONY: doc
doc:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)/documentation
	MEDI_VERSION=$(MEDI_VERSION) doxygen

.PHONY: clean
clean:
	rm -fr $(GEN_DIR)/*
	rm -fr $(BUILD_DIR)

-include $(DEP_FILES)
