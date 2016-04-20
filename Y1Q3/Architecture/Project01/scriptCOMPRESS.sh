#!/bin/bash
# Number of instructions fetched...
./sim-outorder -fetch:ifqsize 1 ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_F1.txt
./sim-outorder -fetch:ifqsize 2 ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_F2.txt
./sim-outorder -fetch:ifqsize 4 ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_F4.txt
./sim-outorder -fetch:ifqsize 8 ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_F8.txt

# Number of instructions decoded...
./sim-outorder -decode:width 1 ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_D1.txt
./sim-outorder -decode:width 2 ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_D2.txt
./sim-outorder -decode:width 4 ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_D4.txt
./sim-outorder -decode:width 8 ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_D8.txt

# Number of instructions issued...
./sim-outorder -issue:width 1 ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_I1.txt
./sim-outorder -issue:width 2 ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_I2.txt
./sim-outorder -issue:width 4 ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_I4.txt
./sim-outorder -issue:width 8 ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_I8.txt

# Number of instructions commited...
./sim-outorder -commit:width 1 ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_C1.txt
./sim-outorder -commit:width 2 ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_C2.txt
./sim-outorder -commit:width 4 ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_C4.txt
./sim-outorder -commit:width 8 ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_C8.txt

# Branch prediction...
./sim-outorder -bpred nottaken ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_BPNT.txt
./sim-outorder -bpred taken ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_BPT.txt
./sim-outorder -bpred bimod ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_BPBM.txt
./sim-outorder -bpred 2lev ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_BP2L.txt

# Cache block size...
./sim-outorder -cache:dl2 ul2:128:32:4:l ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_CacheBS32.txt
./sim-outorder -cache:dl2 ul2:128:64:4:l ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_CacheBS64.txt
./sim-outorder -cache:dl2 ul2:128:128:4:l ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_CacheBS128.txt
./sim-outorder -cache:dl2 ul2:128:256:4:l ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_CacheBS256.txt

# Cache replacement policy...
./sim-outorder -cache:dl2 ul2:128:32:4:f ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_CacheRPF.txt
./sim-outorder -cache:dl2 ul2:128:32:4:l ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_CacheRPL.txt
./sim-outorder -cache:dl2 ul2:128:32:4:r ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_CacheRPR.txt

# Cache associativity...
./sim-outorder -cache:dl2 ul2:128:32:1:l ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_CacheA1.txt
./sim-outorder -cache:dl2 ul2:128:32:4:l ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_CacheA4.txt
./sim-outorder -cache:dl2 ul2:128:32:8:l ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_CacheA8.txt
./sim-outorder -cache:dl2 ul2:128:32:64:l ../benchmarks/compress95.alpha ../benchmarks/1stmt.i 2> Results/results_C_CacheA64.txt

