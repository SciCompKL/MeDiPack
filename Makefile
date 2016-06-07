INCLUDE_DIR=include
SRC_DIR=src
TEMPL_DIR=templates
DEF_DIR=definitions

GEN_DIR=generated

GENERATED_FILES= \
  $(GEN_DIR)/tampiFunctions.hpp

ASTYLE_FILE=template.style

all: $(GENERATED_FILES)

# define the dependencies for all the files
$(GEN_DIR)/tampiFunctions.hpp: $(TEMPL_DIR)/tampiFunctions_hpp.gsl $(DEF_DIR)/mpiFunctions.xml

# the generation rules
$(GEN_DIR)/%.hpp:$(TEMPL_DIR)/%_hpp.gsl
	gsl -script:$< -a $(filter-out $<,$^) $@.tmp
	astyle --options=$(ASTYLE_FILE) < $@.tmp > $@
	@rm $@.tmp

.PHONY: clean
clean:
	rm -fr $(GEN_DIR)/*

.PHONY: test
test:
	mpic++ -std=c++11 test.cpp -o test -I../CoDi/include -I. -Iinclude
