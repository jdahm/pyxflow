SHELL = /bin/sh

# Top-level directory
TOPDIR = .
# Source directory
BUILDDIR = pyxflow
EQNDIR = $(BUILDDIR)/lib
all: build

.PHONY: build
build:
	(cd $(BUILDDIR); ./build.py)

.PHONY: clean
clean:
	(cd $(
	(cd $(BUILDDIR); rm _pyxflow.so)
