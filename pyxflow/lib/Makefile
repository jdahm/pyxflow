-include Makefile.in

ADDDIR = $(RDYNAMIC) -L$(XF_HOME)/lib

all: Scalar CompressibleNS Acoustics ShallowWater RadHydro

Scalar:
	gcc -shared $(ADDDIR) -o libpxScalar.so -lxfSerial -lScalar

CompressibleNS:
	gcc -shared $(ADDDIR) -o libpxCompressibleNS.so -lxfSerial -lCompressibleNS

Acoustics:
	gcc -shared $(ADDDIR) -o libpxAcoustics.so -lxfSerial -lAcoustics

ShallowWater:
	gcc -shared $(ADDDIR) -o libpxShallowWater.so -lxfSerial -lShallowWater

RadHydro:
	gcc -shared $(ADDDIR) -o libpxRadHydro.so -lxfSerial -lRadHydro
