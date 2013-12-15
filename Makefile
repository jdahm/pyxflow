SHELL = /bin/sh

# Top-level directory
TOPDIR = .
# Source directory
BUILDDIR = pyxflow

all: build

.PHONY: build
build:
	(cd $(BUILDDIR); ./build.py)

.PHONY: clean
clean:
	(cd $(BUILDDIR); rm _pyxflow.so)
