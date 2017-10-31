package Misc;

import java.util.ArrayList;
import fr.liglab.jlcm.internals.ExplorationStep;
import fr.liglab.jlcm.io.PatternsCollector;
import fr.liglab.jlcm.PLCM;

public class RunPLCM {
	public static void main(String[] args) {
		long chrono = System.currentTimeMillis();
		int minsup = 5;
		StringBuilder transactionsBuffer = new StringBuilder("1 2 5 7 9\n1 3 5 7 9\n1 2 3 4\n1 3 4 5 6\n1 2\n1 2\n1 2 3 4 5 7 8 9\n1 2 6\n4 5 6\n2 5 8\n1 2 9\n1 2 4 8 9\n");
		Transactions transactions = new Transactions(transactionsBuffer);
		ExplorationStep initState;
		initState = new ExplorationStep(minsup, transactions);
		long loadingTime = System.currentTimeMillis() - chrono;

		System.out.println("Dataset loaded in " + loadingTime + "ms");

		int nbThreads = Runtime.getRuntime().availableProcessors();

		PatternsCollector collector = new ListCollector();

		PLCM miner = new PLCM(collector, nbThreads);

		chrono = System.currentTimeMillis();
		miner.lcm(initState);
		chrono = System.currentTimeMillis() - chrono;

		final ArrayList<ArrayList<Integer>> closed = ((ListCollector) collector).getClosed();
		for(ArrayList<Integer> pattern : closed){
			System.out.println(pattern);
		}
		System.out.println(String.format("Done in %dms", chrono));
	}
}
