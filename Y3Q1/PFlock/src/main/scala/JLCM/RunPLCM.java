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

import java.util.ArrayList;
import fr.liglab.jlcm.internals.ExplorationStep;
import fr.liglab.jlcm.io.PatternsCollector;
import fr.liglab.jlcm.PLCM;

public class RunPLCM {
	public static void main(String[] args) {
		long chrono = System.currentTimeMillis();
		int minsup = 5;
		String transactionsBuffer = new String("1 2 5 7 9\n1 3 5 7 9\n1 2 3 4\n1 3 4 5 6\n1 2\n1 2\n1 2 3 4 5 7 8 9\n1 2 6\n4 5 6\n2 5 8\n1 2 9\n1 2 4 8 9\n");
		TransactionsReader transactions = new TransactionsReader(transactionsBuffer);
		ExplorationStep initState;
		initState = new ExplorationStep(minsup, transactions);
		long loadingTime = System.currentTimeMillis() - chrono;
		int nbThreads = Runtime.getRuntime().availableProcessors() - 1;
		PatternsCollector collector = new ListCollector();
		PLCM miner = new PLCM(collector, nbThreads);
		chrono = System.currentTimeMillis();
		miner.lcm(initState);
		chrono = System.currentTimeMillis() - chrono;
		final ArrayList<ArrayList<Integer>> closed = ((ListCollector)collector).getClosed();
		for(ArrayList<Integer> pattern : closed){
			System.out.println(pattern);
		}
		System.out.println(String.format("Done in %dms", chrono));
	}
}
