SHELL = /bin/sh

# Top-level directory
TOPDIR = .
# Source directory
MODULEDIR = pyxflow
EQNDIR = $(MODULEDIR)/lib
all: build

.PHONY: build
build:
	(cd $(MODULEDIR); ./build.py)

.PHONY: style
style:
	(cd $(MODULEDIR); make style)

.PHONY: clean
clean:
	(cd $(MODULEDIR); rm _pyxflow.so)
