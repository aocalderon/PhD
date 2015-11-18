## Text model saving vocabulary
./word2vec -train text8_small -output vectors_small_50.txt -cbow 1 -size 50 -window 5 -negative 0 -hs 25 -threads 1 -iter 4 -binary 0 -save-vocab vocab.txt
## Text model with just 3 dimensions
./word2vec -train text8_small -output vectors_small_3.txt -cbow 1 -size 3 -window 5 -negative 0 -hs 25 -threads 1 -iter 4 -binary 0 
## See the results...
echo "Text model size 50..."
head -n 5 vectors_small_50.txt
echo "Vocabulary..."
head -n 5 vocab.txt 
echo "Text model size 3..."
head -n 5 vectors_small_3.txt 
