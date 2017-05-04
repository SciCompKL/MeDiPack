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

INCLUDE_DIR=include
SRC_DIR=src
TEMPL_DIR=templates
DEF_DIR=definitions

GEN_DIR=generated

GENERATED_FILES= \
  $(GEN_DIR)/medi/ampiFunctions.hpp \
  $(GEN_DIR)/medi/ampiDefinitions.cpp \
  $(GEN_DIR)/medi/ampiDefinitions.h

ASTYLE_FILE=template.style

all: $(GEN_DIR)/medi $(GENERATED_FILES)

# define the dependencies for all the files
$(GEN_DIR)/medi/ampiFunctions.hpp: 	 $(TEMPL_DIR)/medi/ampiFunctions_hpp.gsl   $(DEF_DIR)/mpiFunctions.xml
$(GEN_DIR)/medi/ampiDefinitions.cpp: $(TEMPL_DIR)/medi/ampiDefinitions_cpp.gsl $(DEF_DIR)/mpiDefinitions.xml
$(GEN_DIR)/medi/ampiDefinitions.h:   $(TEMPL_DIR)/medi/ampiDefinitions_h.gsl   $(DEF_DIR)/mpiDefinitions.xml

# directory generation rules
$(GEN_DIR):
	mkdir $(GEN_DIR)

$(GEN_DIR)/medi: $(GEN_DIR)
	mkdir $(GEN_DIR)/medi

# the generation rules
$(GEN_DIR)/%.hpp:$(TEMPL_DIR)/%_hpp.gsl
	gsl -script:$< -a $(filter-out $<,$^) $@.tmp
	astyle --options=$(ASTYLE_FILE) < $@.tmp > $@
	@rm $@.tmp

$(GEN_DIR)/%.h:$(TEMPL_DIR)/%_h.gsl
	gsl -script:$< -a $(filter-out $<,$^) $@.tmp
	astyle --options=$(ASTYLE_FILE) < $@.tmp > $@
	@rm $@.tmp

$(GEN_DIR)/%.cpp:$(TEMPL_DIR)/%_cpp.gsl
	gsl -script:$< -a $(filter-out $<,$^) $@.tmp
	astyle --options=$(ASTYLE_FILE) < $@.tmp > $@
	@rm $@.tmp

.PHONY: clean
clean:
	rm -fr $(GEN_DIR)/*

.PHONY: test
test:
	mpic++ -std=c++11 test.cpp -o test -I../CoDi/include -I. -Iinclude
