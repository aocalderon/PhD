package SPMF;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.TreeMap;
import java.util.Map;

public class Tester {
    public static void main(String[] arg) throws IOException {
		// Setting variables...
		Map< Integer, List<List<Integer>> > map = new TreeMap< Integer, List<List<Integer>> >();
        // Reading file...
        String input = "/home/and/Documents/PhD/Code/Y3Q1/Datasets/CandidatesInside_B60K_E40.0_M12_P1024.csv";
        System.out.println(input);
        BufferedReader reader = new BufferedReader(new FileReader(input));
        String line;
        while (((line = reader.readLine()) != null)) {
			// Extracting key string...
			String[] KeyAndTransaction = line.split(";");
			Integer key = Integer.parseInt(KeyAndTransaction[0]);
			// Looking for previous transaction set...
			List<List<Integer>> transactions = map.get(key);
			// Checking if it exists, if not create a new one...
			if(transactions == null){
				transactions = new ArrayList<>();
				map.put(key, transactions);
			}
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
			Integer n = 0;
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
			AlgoFPMax fpMax = new AlgoFPMax();
			Itemsets itemsets = fpMax.runAlgorithm(transactions, minsup);
			// fpMax.printStats();
			ArrayList<ArrayList<Integer>> maximals = itemsets.getItemsets(mu);
			// itemsets.printItemsets();
			String report = String.format("%5d, %5.2f, %3d, %3d, %s, %s, %4d", key, avg, min, max, fpMax.getTime(), fpMax.getStats(), maximals.size());
			System.out.println(report);
		}
    }
}
/*
568 or 569
*/
