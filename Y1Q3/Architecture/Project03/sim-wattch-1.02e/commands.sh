The following commands can be used to run the simulation for the GCC application.  Note that the output is redirected to a specific file.

./sim-outorder -redir:sim gcc_baseline.txt -cache:victim none -cache:isbuffer none -cache:dsbuffer none ../benchmarks/cc1.alpha ../benchmarks/1stmt.i

./sim-outorder -redir:sim gcc_victim_cache.txt -cache:isbuffer none -cache:dsbuffer none ../benchmarks/cc1.alpha ../benchmarks/1stmt.i

./sim-outorder -redir:sim gcc_stream_buffers.txt -cache:victim none ../benchmarks/cc1.alpha ../benchmarks/1stmt.i

./sim-outorder -redir:sim gcc_plru.txt -cache:dl1 dl1:128:32:4:p  -cache:victim none -cache:isbuffer none -cache:dsbuffer none ../benchmarks/cc1.alpha ../benchmarks/1stmt.i

The following commands can be used to run the simulation for the Anagram application.  Note that the output is redirected to a specific file.

./sim-outorder -redir:sim anagram_baseline.txt -cache:victim none -cache:isbuffer none -cache:dsbuffer none ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in

./sim-outorder -redir:sim anagram_victim_cache.txt -cache:isbuffer none -cache:dsbuffer none ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in

./sim-outorder -redir:sim anagram_stream_buffers.txt -cache:victim none ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in

./sim-outorder -redir:sim anagram_plru.txt -cache:dl1 dl1:128:32:4:p  -cache:victim none -cache:isbuffer none -cache:dsbuffer none ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in


