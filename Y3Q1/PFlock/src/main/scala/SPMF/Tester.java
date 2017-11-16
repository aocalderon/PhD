package SPMF;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.logging.Logger;

public class Tester {
    private static Logger logger = Logger.getLogger(Tester.class.getName());

    public static void main(String[] arg) throws IOException {
        //System.setProperty("java.util.logging.SimpleFormatter.format", "%1$tF %1$tT %4$s %2$s %5$s%6$s%n");
        System.setProperty("java.util.logging.SimpleFormatter.format", "%1$tF_%1$tT -> %5$s%6$s%n");
		// Setting variables...
		Map< Integer, Set<List<Integer>> > map = new TreeMap<>();
        // Reading file...
        String input = "/tmp/DB80K_E100.0_M50_Maximals.txt";
        //String input = arg[0];
        logger.info(input);
        //String algorithm = arg[1];
		String algorithm = "LCM";
        logger.info(algorithm);
        BufferedReader reader = new BufferedReader(new FileReader(input));
        String line;
        long timer = System.currentTimeMillis();
        while (((line = reader.readLine()) != null)) {
			// Extracting key string...
			String[] KeyAndTransaction = line.split(";");
			Integer key = Integer.parseInt(KeyAndTransaction[0]);
			// Looking for previous transaction set...
			Set<List<Integer>> transactions = map.computeIfAbsent(key, k -> new HashSet<>());
			// Extracting transaction string...
			String transactionString  = KeyAndTransaction[1]; 
			// Splitting the transaction into items...
			String[] itemsString = transactionString.split(" ");
			// Converting the item and adding to transaction...
			List<Integer> transaction = new ArrayList<>();
			for (String itemString : itemsString) {
				Integer item = Integer.parseInt(itemString);
				transaction.add(item);
			}
			// increase the transaction count
			transactions.add(transaction);
		}
		int nMap = map.keySet().size();
		double clockTime = (System.currentTimeMillis() - timer) / 1000.0;
        logger.info(String.format("Reading file %s... [%.3fms][%d results]", input, clockTime, nMap));
		int minsup = 1;
		int mu = 50;
		for(Integer key : map.keySet()){
            //timer = System.currentTimeMillis();
			Set<List<Integer>> transactions = map.get(key);
			Integer size = transactions.size();
			HashSet<List<Integer>> transactionsSet = new HashSet<>(transactions);
			transactions.clear();
			transactions.addAll(transactionsSet);
            //int nTransactions = transactions.size();
            //clockTime = (System.currentTimeMillis() - timer) / 1000.0;
            //logger.info(String.format("Extracting partition's transactions... [%.3fms][%d results]", clockTime, nTransactions));
            //timer = System.currentTimeMillis();
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
            //clockTime = (System.currentTimeMillis() - timer) / 1000.0;
            //logger.info(String.format("Extracting partition's statistics... [%.3fms][%d results]", clockTime, nTransactions));

			if(algorithm.equals("LCM")){
                timer = System.currentTimeMillis();
				AlgoLCM algoLCM = new AlgoLCM();
				Transactions data = new Transactions(transactions);
				Itemsets closedLCM = algoLCM.runAlgorithm(minsup, data);
				// algoLCM.printStats();
				// AlgoCharmLCM algoCharmLCM = new AlgoCharmLCM();
				// Itemsets maximalsLCM = algoCharmLCM.runAlgorithm(closedLCM);
				// algoCharmLCM.printStats();
				//ArrayList<ArrayList<Integer>> maximalsLCMPruned = maximalsLCM.getItemsets(mu);
				ArrayList<ArrayList<Integer>> maximalsLCMPruned = closedLCM.getMaximalItemsets1(mu);
                clockTime = (System.currentTimeMillis() - timer) / 1000.0;
                int nMaximalsLCMPrunned = maximalsLCMPruned.size();
                logger.info(String.format("Running LCM algorithm... [%.2fms][%d results]", clockTime, nMaximalsLCMPrunned));
				String reportLCM = String.format("===,%5d, %5.2f, %3d, %3d, %.3f, %s, %4d",
						key, avg, min, max, algoLCM.getTime() /*+ algoCharmLCM.getTime()*/,
						algoLCM.getStats(), /*algoCharmLCM.getStats(),*/ maximalsLCMPruned.size());
				logger.info(reportLCM);
				//for(List<Integer> pattern : maximalsLCMPruned){
					//System.out.println(pattern.toString());
				//}
			}
		}
    }
}
