-include Makefile.in

ADDDIR = -L$(XF_HOME)/lib
DYNLDFLAGS = $(LDFLAGS)

all: Scalar CompressibleNS Acoustics ShallowWater RadHydro

Scalar:
	gcc $(ADDDIR) -o libpxScalar.so $(DYNLDFLAGS) -lxfSerial -lScalar

CompressibleNS:
	gcc $(ADDDIR) -o libpxCompressibleNS.so $(DYNLDFLAGS) -lxfSerial -lCompressibleNS

Acoustics:
	gcc $(ADDDIR) -o libpxAcoustics.so $(DYNLDFLAGS) -lxfSerial -lAcoustics

ShallowWater:
	gcc $(ADDDIR) -o libpxShallowWater.so $(DYNLDFLAGS) -lxfSerial -lShallowWater

RadHydro:
	gcc $(ADDDIR) -o libpxRadHydro.so $(DYNLDFLAGS) -lxfSerial -lRadHydro
