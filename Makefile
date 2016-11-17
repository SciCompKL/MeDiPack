INCLUDE_DIR=include
SRC_DIR=src
TEMPL_DIR=templates
DEF_DIR=definitions

GEN_DIR=generated

GENERATED_FILES= \
  $(GEN_DIR)/medi/ampiFunctions.hpp \
  $(GEN_DIR)/medi/miscAmpiFunctions.hpp \
  $(GEN_DIR)/medi/ampiDatatypes.h \
  $(GEN_DIR)/medi/ampiDatatypes.cpp

ASTYLE_FILE=template.style

all: $(GEN_DIR)/medi $(GENERATED_FILES)

# define the dependencies for all the files
$(GEN_DIR)/medi/ampiFunctions.hpp: $(TEMPL_DIR)/medi/ampiFunctions_hpp.gsl $(DEF_DIR)/mpiFunctions.xml
$(GEN_DIR)/medi/miscAmpiFunctions.hpp: $(TEMPL_DIR)/medi/miscAmpiFunctions_hpp.gsl $(DEF_DIR)/miscFunctions.xml
$(GEN_DIR)/medi/ampiDatatypes.h: $(TEMPL_DIR)/medi/ampiDatatypes_h.gsl $(DEF_DIR)/mpiDatatypes.xml
$(GEN_DIR)/medi/ampiDatatypes.cpp: $(TEMPL_DIR)/medi/ampiDatatypes_cpp.gsl $(DEF_DIR)/mpiDatatypes.xml

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
