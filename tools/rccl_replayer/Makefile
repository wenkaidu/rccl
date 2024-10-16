ROCM_DIR ?= /opt/rocm
RCCL_DIR ?= ../../build/release
MPI_DIR  ?= /opt/
MPI_INC_DIR ?= /usr/include/mpich-x86_64
MPI_LIB_DIR ?= /usr/lib64/mpich/lib

INCLUDES = -I$(MPI_INC_DIR) -I$(MPI_DIR)/include -I$(RCCL_DIR)/include
LDFLAGS  = -L$(MPI_LIB_DIR) -L$(MPI_DIR)/lib -L$(RCCL_DIR) -lmpich -lrccl

main: rcclReplayer.cpp
	$(ROCM_DIR)/bin/hipcc rcclReplayer.cpp -O1 -g -o rcclReplayer $(INCLUDES) $(LDFLAGS)

clean:
	rm -f ./rcclReplayer
