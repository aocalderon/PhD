#!/bin/bash
# Number of instructions fetched...
./sim-outorder -fetch:ifqsize 1 ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_F1.txt
./sim-outorder -fetch:ifqsize 2 ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_F2.txt
./sim-outorder -fetch:ifqsize 4 ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_F4.txt
./sim-outorder -fetch:ifqsize 8 ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_F8.txt

# Number of instructions decoded...
./sim-outorder -decode:width 1 ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_D1.txt
./sim-outorder -decode:width 2 ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_D2.txt
./sim-outorder -decode:width 4 ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_D4.txt
./sim-outorder -decode:width 8 ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_D8.txt

# Number of instructions issued...
./sim-outorder -issue:width 1 ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_I1.txt
./sim-outorder -issue:width 2 ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_I2.txt
./sim-outorder -issue:width 4 ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_I4.txt
./sim-outorder -issue:width 8 ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_I8.txt

# Number of instructions commited...
./sim-outorder -commit:width 1 ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_C1.txt
./sim-outorder -commit:width 2 ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_C2.txt
./sim-outorder -commit:width 4 ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_C4.txt
./sim-outorder -commit:width 8 ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_C8.txt

# Branch prediction...
./sim-outorder -bpred nottaken ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_BPNT.txt
./sim-outorder -bpred taken ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_BPT.txt
./sim-outorder -bpred bimod ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_BPBM.txt
./sim-outorder -bpred 2lev ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_BP2L.txt

# Cache block size...
./sim-outorder -cache:dl2 ul2:128:32:4:l ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_CacheBS32.txt
./sim-outorder -cache:dl2 ul2:128:64:4:l ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_CacheBS64.txt
./sim-outorder -cache:dl2 ul2:128:128:4:l ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_CacheBS128.txt
./sim-outorder -cache:dl2 ul2:128:256:4:l ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_CacheBS256.txt

# Cache replacement policy...
./sim-outorder -cache:dl2 ul2:128:32:4:f ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_CacheRPF.txt
./sim-outorder -cache:dl2 ul2:128:32:4:l ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_CacheRPL.txt
./sim-outorder -cache:dl2 ul2:128:32:4:r ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_CacheRPR.txt

# Cache associativity...
./sim-outorder -cache:dl2 ul2:128:32:1:l ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_CacheA1.txt
./sim-outorder -cache:dl2 ul2:128:32:4:l ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_CacheA4.txt
./sim-outorder -cache:dl2 ul2:128:32:8:l ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_CacheA8.txt
./sim-outorder -cache:dl2 ul2:128:32:64:l ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in 2> Results/results_A_CacheA64.txt

