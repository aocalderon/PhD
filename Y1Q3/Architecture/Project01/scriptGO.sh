#!/bin/bash
# Number of instructions fetched...
./sim-outorder -fetch:ifqsize 1 ../benchmarks/go.alpha 2> Results/results_G_F1.txt
./sim-outorder -fetch:ifqsize 2 ../benchmarks/go.alpha 2> Results/results_G_F2.txt
./sim-outorder -fetch:ifqsize 4 ../benchmarks/go.alpha 2> Results/results_G_F4.txt
./sim-outorder -fetch:ifqsize 8 ../benchmarks/go.alpha 2> Results/results_G_F8.txt

# Number of instructions decoded...
./sim-outorder -decode:width 1 ../benchmarks/go.alpha 2> Results/results_G_D1.txt
./sim-outorder -decode:width 2 ../benchmarks/go.alpha 2> Results/results_G_D2.txt
./sim-outorder -decode:width 4 ../benchmarks/go.alpha 2> Results/results_G_D4.txt
./sim-outorder -decode:width 8 ../benchmarks/go.alpha 2> Results/results_G_D8.txt

# Number of instructions issued...
./sim-outorder -issue:width 1 ../benchmarks/go.alpha 2> Results/results_G_I1.txt
./sim-outorder -issue:width 2 ../benchmarks/go.alpha 2> Results/results_G_I2.txt
./sim-outorder -issue:width 4 ../benchmarks/go.alpha 2> Results/results_G_I4.txt
./sim-outorder -issue:width 8 ../benchmarks/go.alpha 2> Results/results_G_I8.txt

# Number of instructions commited...
./sim-outorder -commit:width 1 ../benchmarks/go.alpha 2> Results/results_G_C1.txt
./sim-outorder -commit:width 2 ../benchmarks/go.alpha 2> Results/results_G_C2.txt
./sim-outorder -commit:width 4 ../benchmarks/go.alpha 2> Results/results_G_C4.txt
./sim-outorder -commit:width 8 ../benchmarks/go.alpha 2> Results/results_G_C8.txt

# Branch prediction...
./sim-outorder -bpred nottaken ../benchmarks/go.alpha 2> Results/results_G_BPNT.txt
./sim-outorder -bpred taken ../benchmarks/go.alpha 2> Results/results_G_BPT.txt
./sim-outorder -bpred bimod ../benchmarks/go.alpha 2> Results/results_G_BPBM.txt
./sim-outorder -bpred 2lev ../benchmarks/go.alpha 2> Results/results_G_BP2L.txt

# Cache block size...
./sim-outorder -cache:dl2 ul2:128:32:4:l ../benchmarks/go.alpha 2> Results/results_G_CacheBS32.txt
./sim-outorder -cache:dl2 ul2:128:64:4:l ../benchmarks/go.alpha 2> Results/results_G_CacheBS64.txt
./sim-outorder -cache:dl2 ul2:128:128:4:l ../benchmarks/go.alpha 2> Results/results_G_CacheBS128.txt
./sim-outorder -cache:dl2 ul2:128:256:4:l ../benchmarks/go.alpha 2> Results/results_G_CacheBS256.txt

# Cache replacement policy...
./sim-outorder -cache:dl2 ul2:128:32:4:f ../benchmarks/go.alpha 2> Results/results_G_CacheRPF.txt
./sim-outorder -cache:dl2 ul2:128:32:4:l ../benchmarks/go.alpha 2> Results/results_G_CacheRPL.txt
./sim-outorder -cache:dl2 ul2:128:32:4:r ../benchmarks/go.alpha 2> Results/results_G_CacheRPR.txt

# Cache associativity...
./sim-outorder -cache:dl2 ul2:128:32:1:l ../benchmarks/go.alpha 2> Results/results_G_CacheA1.txt
./sim-outorder -cache:dl2 ul2:128:32:4:l ../benchmarks/go.alpha 2> Results/results_G_CacheA4.txt
./sim-outorder -cache:dl2 ul2:128:32:8:l ../benchmarks/go.alpha 2> Results/results_G_CacheA8.txt
./sim-outorder -cache:dl2 ul2:128:32:64:l ../benchmarks/go.alpha 2> Results/results_G_CacheA64.txt

