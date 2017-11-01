package SPMF;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Tester {
    public static void main(String[] arg) throws IOException {
		// Setting variables...
		Map< Integer, Set<List<Integer>> > map = new TreeMap<>();
        // Reading file...
        String input = "/home/and/Documents/PhD/Code/Y3Q1/Datasets/TI568.dat";
        //String input = arg[0];
        System.out.println(input);
        //String algorithm = arg[1];
		String algorithm = "LCM";
        System.out.println(algorithm);
        BufferedReader reader = new BufferedReader(new FileReader(input));
        String line;
        while (((line = reader.readLine()) != null)) {
			// Extracting key string...
			String[] KeyAndTransaction = line.split(";");
			Integer key = Integer.parseInt(KeyAndTransaction[0]);
			// Looking for previous transaction set...
			Set<List<Integer>> transactions = map.computeIfAbsent(key, k -> new HashSet<>());
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
			Set<List<Integer>> transactions = map.get(key);
			Integer size = transactions.size();
			HashSet<List<Integer>> transactionsSet = new HashSet<>(transactions);
			transactions.clear();
			transactions.addAll(transactionsSet);
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

			if(algorithm.equals("LCM")){
				AlgoLCM algoLCM = new AlgoLCM();
				Transactions data = new Transactions(transactions);
				Itemsets closedLCM = algoLCM.runAlgorithm(minsup, data);
				// algoLCM.printStats();
				AlgoCharmLCM algoCharmLCM = new AlgoCharmLCM();
				Itemsets maximalsLCM = algoCharmLCM.runAlgorithm(closedLCM);
				// algoCharmLCM.printStats();
				ArrayList<ArrayList<Integer>> maximalsLCMPruned = maximalsLCM.getItemsets(mu);
				String reportLCM = String.format("%5d, %5.2f, %3d, %3d, %.3f, %s, %s, %4d",
						key, avg, min, max, algoLCM.getTime() + algoCharmLCM.getTime(),
						algoLCM.getStats(), algoCharmLCM.getStats(), maximalsLCMPruned.size());
				System.out.println(reportLCM);
				for(List<Integer> pattern : maximalsLCMPruned){
					System.out.println(pattern.toString());
				}
			}
		}
    }
}
