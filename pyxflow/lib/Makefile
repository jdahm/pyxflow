-include Makefile.in

ADDDIR = $(RDYNAMIC) -L$(XF_HOME)/lib

all: Scalar CompressibleNS Acoustics ShallowWater RadHydro

Scalar:
	gcc -shared $(ADDDIR) -o libpxScalar.so $(OBJ_LIST) -lxfSerial -lScalar

CompressibleNS:
	gcc -shared $(ADDDIR) -o libpxCompressibleNS.so $(OBJ_LIST) -lxfSerial -lCompressibleNS

Acoustics:
	gcc -shared $(ADDDIR) -o libpxAcoustics.so $(OBJ_LIST) -lxfSerial -lAcoustics

ShallowWater:
	gcc -shared $(ADDDIR) -o libpxShallowWater.so $(OBJ_LIST) -lxfSerial -lShallowWater

RadHydro:
	gcc -shared $(ADDDIR) -o libpxRadHydro.so $(OBJ_LIST) -lxfSerial -lRadHydro
