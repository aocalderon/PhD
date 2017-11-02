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

import fr.liglab.jlcm.io.PatternsWriter;
import java.util.ArrayList;

public class ListCollector extends PatternsWriter {
    private ArrayList<ArrayList<Integer>> closed;
    private long collected = 0;
    private long collectedLength = 0;

    public ListCollector(){
        closed =  new ArrayList<>();
    }

    @Override
    synchronized public void collect(int support, int[] pattern, int length) {
		if(support == 1){
			ArrayList<Integer> p = new ArrayList<>(length);

			for (int i = 0; i < length; i++) {
				p.add(pattern[i]);
			}
			closed.add(p);
			this.collected++;
			this.collectedLength += pattern.length;
		}
    }

    public ArrayList<ArrayList<Integer>> getClosed() {
        return closed;
    }

    public long close() {
        closed.clear();
        return this.collected;
    }

    public int getAveragePatternLength() {
        if (this.collected == 0) {
            return 0;
        } else {
            return (int) (this.collectedLength / this.collected);
        }
    }
}

