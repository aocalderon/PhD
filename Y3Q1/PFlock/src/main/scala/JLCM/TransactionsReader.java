package JLCM;
/*
	This file is part of jLCM - see https://github.com/martinkirch/jlcm/

	Copyright 2013,2014 Martin Kirchgessner, Vincent Leroy, Alexandre Termier, Sihem Amer-Yahia, Marie-Christine Rousset, Université Joseph Fourier and CNRS

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

	 http://www.apache.org/licenses/LICENSE-2.0

	or see the LICENSE.txt file joined with this program.

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.
*/

import fr.liglab.jlcm.internals.TransactionReader;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;

 final class TransactionsIterator implements Iterator<TransactionReader> {
    private static final int COPY_PAGES_SIZE = 1024*1024;
    private final ArrayList<int[]> pages = new ArrayList<>();
    private Iterator<int[]> pagesIterator;
    private int[] currentPage;
    private int currentPageIndex;
    private int currentTransIdx;
    private int currentTransLen;
    private int[] renaming = null;
    private final CopyReader copyReader = new CopyReader();
    private CopyReader nextCopyReader = new CopyReader();
    private BufferedReader inBuffer;
    private final LineReader lineReader = new LineReader();
    private int nextChar = 0;

    TransactionsIterator(String transactionsBuffer) {
        try {
            inBuffer = new BufferedReader(new StringReader(transactionsBuffer));
            nextChar = inBuffer.read();
            newPage();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private void newPage() {
        currentPage = new int[COPY_PAGES_SIZE];
        pages.add(currentPage);
        currentPageIndex = 1;
        currentTransIdx = 0;
        currentTransLen = 0;
    }

    private void writeNewTransactionToNextPage() {
        if (currentTransLen+1 >= COPY_PAGES_SIZE) {
            if (currentTransIdx == 0) {
                throw new RuntimeException("Out of buffer bounds - please check the input file format: only LF line terminators are expected, even at EOF.");
            } else {
                throw new RuntimeException("Inputted transactions are too long ! Try increasing FileReader.COPY_PAGES_SIZE");
            }
        }
        int[] previousPage = currentPage;
        currentPage = new int[COPY_PAGES_SIZE];
        pages.add(currentPage);
        previousPage[currentTransIdx] = -1;
        System.arraycopy(previousPage, currentTransIdx+1, currentPage, 1, currentTransLen);
        currentTransIdx = 0;
        currentPageIndex = currentTransLen+1;
    }

    public void close() {
        close(null);
    }

    public void close(int[] renamingMap) {
        try {
            inBuffer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        inBuffer = null;
        renaming = renamingMap;
        currentPageIndex--;
        currentPage[currentPageIndex] = -1;
        pagesIterator = pages.iterator();
        currentPage = null;
        prepareNextCopyReader();
    }

    private void prepareNextCopyReader() {
        if (currentPage == null || currentPageIndex == COPY_PAGES_SIZE ||
                currentPage[currentPageIndex] == -1) {
            if (pagesIterator.hasNext()) {
                currentPage = pagesIterator.next();
                currentPageIndex = 0;
                if (currentPage[currentPageIndex] == -1) { // yes, it may happen !
                    nextCopyReader = null;
                    return;
                }

            } else {
                nextCopyReader = null;
                return;
            }
        }
        currentTransIdx = currentPageIndex;
        currentTransLen = currentPage[currentTransIdx];
        currentTransIdx++;
        if (renaming != null) {
            int filteredI = currentTransIdx;
            for (int i = currentTransIdx; i < currentTransIdx + currentTransLen; i++) {
                int renamed = renaming[currentPage[i]];
                if (renamed >= 0) {
                    currentPage[filteredI++] = renamed;
                }
            }
            Arrays.sort(currentPage, currentTransIdx, filteredI);
            this.nextCopyReader.setup(currentPage, currentTransIdx, filteredI);
        } else {
            this.nextCopyReader.setup(currentPage, currentTransIdx, currentTransIdx + currentTransLen);
        }
        currentPageIndex = currentTransIdx + currentTransLen;
    }

    public boolean hasNext() {
        if (inBuffer == null) {
            return nextCopyReader != null;
        } else {
            skipNewLines();
            return nextChar != -1;
        }
    }

    public TransactionReader next() {
        if (inBuffer == null) {
            if (nextCopyReader != null) {
                copyReader.setup(nextCopyReader.source, nextCopyReader.i, nextCopyReader.end);
                prepareNextCopyReader();
            }
            return copyReader;
        } else {
            skipNewLines();
            return this.lineReader;
        }
    }

    public void remove() {
        throw new UnsupportedOperationException();
    }

    private void skipNewLines() {
        try {
            while (nextChar == '\n') {
                nextChar = inBuffer.read();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private final class LineReader implements TransactionReader {
        @Override
        public int getTransactionSupport() {
            return 1;
        }
        @Override
        public int next() {
            int nextInt = -1;
            try {
                while (nextChar == ' ')
                    nextChar = inBuffer.read();

                while('0' <= nextChar && nextChar <= '9') {
                    if (nextInt < 0) {
                        nextInt = nextChar - '0';
                    } else {
                        nextInt = (10*nextInt) + (nextChar - '0');
                    }
                    nextChar = inBuffer.read();
                }
                while (nextChar == ' ')
                    nextChar = inBuffer.read();
            } catch (IOException e) {
                e.printStackTrace();
            }
            if (currentPageIndex == COPY_PAGES_SIZE) {
                writeNewTransactionToNextPage();
            }
            currentPage[currentPageIndex++] = nextInt;
            currentTransLen++;
            if (nextChar == '\n') {
                currentPage[currentTransIdx] = currentTransLen;
                if (currentPageIndex == COPY_PAGES_SIZE) {
                    newPage();
                } else {
                    currentTransIdx = currentPageIndex++;
                    currentTransLen = 0;
                }
            }
            return nextInt;
        }
        @Override
        public boolean hasNext() {
            return (nextChar != '\n');
        }
    }

    private final class CopyReader implements TransactionReader {
        private int[] source;
        private int i;
        private int end;

        private void setup(int[] array, int from, int to){
            source = array;
            i = from;
            end = to;
        }
        @Override
        public int getTransactionSupport() {
            return 1;
        }
        @Override
        public int next() {
            return source[i++];
        }
        @Override
        public boolean hasNext() {
            return i < end;
        }
    }
}

public final class TransactionsReader implements Iterable<TransactionReader> {
    private String transactionsBuffer;

    public TransactionsReader(String transactionsBuffer) {
        this.transactionsBuffer = transactionsBuffer;
    }
    @Override
    public Iterator<TransactionReader> iterator() {
        return new TransactionsIterator(this.transactionsBuffer);
    }
}
