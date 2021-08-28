#
# Makefile
#


CUCSRSRC=spmm_csr_driver.cu
CUCSCSRC=spmm_csc_driver.cu
CUOPTSRC=spmm_opt_driver.cu

CPPSRC=mm_helper.cpp

EXECCSR=spmm_csr_driver
EXECCSC=spmm_csc_driver
EXECOPT=spmm_opt_driver

OBJSCSR=$(CUCSRSRC:.cu=.o)
OBJSCSR+=$(CPPSRC:.cpp=.o)

OBJSCSC=$(CUCSCSRC:.cu=.o)
OBJSCSC+=$(CPPSRC:.cpp=.o)

OBJSOPT=$(CUOPTSRC:.cu=.o)
OBJSOPT+=$(CPPSRC:.cpp=.o)

NVCCFLAGS= -O3

CC=nvcc
HCC=gcc

all: $(EXECCSC) 
	@echo "Change NVCCFLAGS to set the correct arch version"
	@echo "Change line the above line in Makefile to \"all: \$$(EXECSR) \$$(EXECSC)\" to build both CSR and CSC versions"
	@echo "Change line the above line in Makefile to \"all: \$$(EXECSR) \$$(EXECSC) \$$(EXEOPT)\" to build both CSR, CSC and OPT versions"

$(EXECCSR): $(OBJSCSR)
	$(CC) $(NVCCFLAGS) $^ -o $@

$(EXECCSC): $(OBJSCSC)
	$(CC) $(NVCCFLAGS) $^ -o $@

$(EXECOPT): $(OBJSOPT)
	$(CC) $(NVCCFLAGS) $^ -o $@

%.o : %.cu
	$(CC)  $(NVCCFLAGS) -c $< -o $@

%.o : %.cpp
	$(HCC) -O3 -c $< -o $@

clean:
	rm -f $(EXECCSR) $(OBJSCSR) $(EXECCSC) $(OBJSCSC) $(EXECOPT) $(OBJSOPT) 


# vim:ft=make
#
