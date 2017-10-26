package SPMF;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;
import java.util.Map;

public class Tester {
    public static void main(String[] arg) throws IOException {
		// Setting variables...
		Map< Integer, List<List<Integer>> > map = new TreeMap<>();
        // Reading file...
        // String input = "/home/and/Documents/PhD/Code/Y3Q1/Datasets/CandidatesFrame_B60K_E40.0_M12_P1024.csv";
        String input = arg[0];
        System.out.println(input);
        System.out.println(arg[1]);
        BufferedReader reader = new BufferedReader(new FileReader(input));
        String line;
        while (((line = reader.readLine()) != null)) {
			// Extracting key string...
			String[] KeyAndTransaction = line.split(";");
			Integer key = Integer.parseInt(KeyAndTransaction[0]);
			// Looking for previous transaction set...
			List<List<Integer>> transactions = map.computeIfAbsent(key, k -> new ArrayList<>());
			// Extracting transaction string...
			String transactionString  = KeyAndTransaction[1]; 
			// Splitting the transaction into items...
			String[] itemsString = transactionString.split(",");
			// Converting the item and adding to transaction...
			List<Integer> transaction = new ArrayList<>();
			for (String itemString : itemsString) {
				Integer item = Integer.parseInt(itemString);
				transaction.add(item);
			}
			// increase the transaction count
			transactions.add(transaction);
		}
		int minsup = 1;
		int mu = 12;
		for(Integer key : map.keySet()){
			List<List<Integer>> transactions = map.get(key);
			Integer size = transactions.size();
			Integer n;
			Double sum = 0.0;
			Integer max = 0;
			Integer min = Integer.MAX_VALUE;
			for(List<Integer> transaction : transactions){
				// System.out.println(transaction.toString());
				n = transaction.size();
				if(n > max){ max = n; }
				if(n < min){ min = n; }
				sum += n;
			}
			Double avg = sum / size;

			if(arg[1].equals("LCM")){
				AlgoLCM algoLCM = new AlgoLCM();
				Transactions data = new Transactions(transactions);
				Itemsets closedLCM = algoLCM.runAlgorithm(minsup, data);
				// algoLCM.printStats();
				AlgoCharmLCM algoCharmLCM = new AlgoCharmLCM();
				Itemsets maximalsLCM = algoCharmLCM.runAlgorithm(null, closedLCM);
				// algoCharmLCM.printStats();
				ArrayList<ArrayList<Integer>> maximalsLCMPruned = maximalsLCM.getItemsets(mu);
				String reportLCM = String.format("%5d, %5.2f, %3d, %3d, %.3f, %s, %s, %4d",
						key, avg, min, max, algoLCM.getTime() + algoCharmLCM.getTime(),
						algoLCM.getStats(), algoCharmLCM.getStats(), maximalsLCMPruned.size());
				System.out.println(reportLCM);
			}

			if(arg[1].equals("FPMax")){
				AlgoFPMax fpMax = new AlgoFPMax();
				Itemsets maximalsFPMax = fpMax.runAlgorithm(transactions, minsup);
				// fpMax.printStats();
				// maximalsFPMax.printItemsets();
				ArrayList<ArrayList<Integer>> maximalsFPMaxPruned = maximalsFPMax.getItemsets(mu);
				String reportFPMax = String.format("%5d, %5.2f, %3d, %3d, %.3f, %s, %4d",
						key, avg, min, max, fpMax.getTime(), fpMax.getStats(),
						maximalsFPMaxPruned.size());
				System.out.println(reportFPMax);
			}
		}
    }
}
