-include Makefile.in

ADDDIR = $(LDFLAGS) -L$(XF_HOME)/lib

all: Scalar CompressibleNS Acoustics ShallowWater RadHydro

HOOK_OBJ = px_Hook.o

$(HOOK_OBJ): px_Hook.c
	$(CC) -I$(XF_HOME)/include -c -o $@ $^

Scalar: $(HOOK_OBJ)
	gcc -shared $(ADDDIR) -o libpxScalar.so $(HOOK_OBJ) -lxfSerial -lScalar

CompressibleNS: $(HOOK_OBJ)
	gcc -shared $(ADDDIR) -o libpxCompressibleNS.so $(HOOK_OBJ) -lxfSerial -lCompressibleNS

Acoustics: $(HOOK_OBJ)
	gcc -shared $(ADDDIR) -o libpxAcoustics.so $(HOOK_OBJ) -lxfSerial -lAcoustics

ShallowWater: $(HOOK_OBJ)
	gcc -shared $(ADDDIR) -o libpxShallowWater.so $(HOOK_OBJ) -lxfSerial -lShallowWater

RadHydro: $(HOOK_OBJ)
	gcc -shared $(ADDDIR) -o libpxRadHydro.so $(HOOK_OBJ) -lxfSerial -lRadHydro
