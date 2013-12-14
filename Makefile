SHELL = /bin/sh

# Top-level directory
TOPDIR = .
BUILDDIR = pyxflow

all: build

.PHONY: build
build:
	(cd $(BUILDDIR); ./build.py)
