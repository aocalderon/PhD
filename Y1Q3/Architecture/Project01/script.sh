#!/bin/bash
# Number of instructions fetched...
./sim-outorder -fetch:ifqsize 1 ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_F1.txt
./sim-outorder -fetch:ifqsize 2 ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_F2.txt
./sim-outorder -fetch:ifqsize 4 ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_F4.txt
./sim-outorder -fetch:ifqsize 8 ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_F8.txt

# Number of instructions decoded...
./sim-outorder -decode:width 1 ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_D1.txt
./sim-outorder -decode:width 2 ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_D2.txt
./sim-outorder -decode:width 4 ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_D4.txt
./sim-outorder -decode:width 8 ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_D8.txt

# Number of instructions issued...
./sim-outorder -issue:width 1 ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_I1.txt
./sim-outorder -issue:width 2 ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_I2.txt
./sim-outorder -issue:width 4 ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_I4.txt
./sim-outorder -issue:width 8 ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_I8.txt

# Number of instructions commited...
./sim-outorder -commit:width 1 ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_C1.txt
./sim-outorder -commit:width 2 ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_C2.txt
./sim-outorder -commit:width 4 ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_C4.txt
./sim-outorder -commit:width 8 ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_C8.txt

# Branch prediction...
./sim-outorder -bpred nottaken ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_BPNT.txt
./sim-outorder -bpred taken ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_BPT.txt
./sim-outorder -bpred bimod ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_BPBM.txt
./sim-outorder -bpred 2lev ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_BP2L.txt

# Cache block size...
./sim-outorder -cache:dl2 ul2:128:32:4:l ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_CacheBS32.txt
./sim-outorder -cache:dl2 ul2:128:64:4:l ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_CacheBS64.txt
./sim-outorder -cache:dl2 ul2:128:128:4:l ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_CacheBS128.txt
./sim-outorder -cache:dl2 ul2:128:256:4:l ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_CacheBS256.txt

# Cache replacement policy...
./sim-outorder -cache:dl2 ul2:128:32:4:f ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_CacheRPF.txt
./sim-outorder -cache:dl2 ul2:128:32:4:l ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_CacheRPL.txt
./sim-outorder -cache:dl2 ul2:128:32:4:r ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_CacheRPR.txt

# Cache associativity...
./sim-outorder -cache:dl2 ul2:128:32:1:l ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_CacheA1.txt
./sim-outorder -cache:dl2 ul2:128:32:4:l ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_CacheA4.txt
./sim-outorder -cache:dl2 ul2:128:32:8:l ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_CacheA8.txt
./sim-outorder -cache:dl2 ul2:128:32:64:l ../benchmarks/cc1.alpha ../benchmarks/1stmt.i 2> results_CacheA64.txt

