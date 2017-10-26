package ca.pfv.spmf.algorithms.frequentpatterns.lcm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import ca.pfv.spmf.algorithms.frequentpatterns.charm.AlgoCharmLCM;
import ca.pfv.spmf.algorithms.frequentpatterns.fpgrowth.AlgoFPMax;
import ca.pfv.spmf.patterns.itemset_array_integers_with_count.Itemsets;
/* This file is copyright (c) 2012-2014 Alan Souza
* 
* This file is part of the SPMF DATA MINING SOFTWARE
* (http://www.philippe-fournier-viger.com/spmf).
* 
* SPMF is free software: you can redistribute it and/or modify it under the
* terms of the GNU General Public License as published by the Free Software
* Foundation, either version 3 of the License, or (at your option) any later
* version.
* SPMF is distributed in the hope that it will be useful, but WITHOUT ANY
* WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
* A PARTICULAR PURPOSE. See the GNU General Public License for more details.
* You should have received a copy of the GNU General Public License along with
* SPMF. If not, see <http://www.gnu.org/licenses/>.
*/
/**
 * Example of how to use LCM algorithm from the source code.
 * @author Alan Souza <apsouza@inf.ufrgs.br> 
 */
public class MainTestLCM_saveToMemory {
	public static void main(String [] arg) throws IOException{
		int transactionCount = 0;
		String input = "/home/and/Documents/PhD/Code/Y3Q1/PFlock/src/main/scala/SPMF/input3.txt";
		BufferedReader reader = new BufferedReader(new FileReader(input));
		String line;
		// for each line (transaction) until the end of file
		List<List<Integer>> transactions = new ArrayList<>();
		while (((line = reader.readLine()) != null)) {
			// split the line into items
			String[] lineSplited = line.split(",");
			// for each item
			List<Integer> transaction = new ArrayList<>();
			for (String itemString : lineSplited) {
				// increase the support count of the item
				Integer item = Integer.parseInt(itemString);
				transaction.add(item);
			}
			// increase the transaction count
			transactions.add(transaction);
			transactionCount++;
		}
		int support = 1;
		Dataset dataset = new Dataset(transactions);

		AlgoLCM algoLCM = new AlgoLCM();
		Itemsets closed = algoLCM.runAlgorithm(support, dataset);
		algoLCM.printStats();
		//closed.printItemsets();
		AlgoCharmLCM algoCharmLCM  = new AlgoCharmLCM();
		algoCharmLCM.runAlgorithm(null, closed);
		algoCharmLCM.printStats();
		Itemsets maximals = algoCharmLCM.getItemsets();
		maximals.printItemsets();

		AlgoFPMax fpMax2 = new AlgoFPMax();
		Itemsets itemsets = fpMax2.runAlgorithm(transactions, support);
		fpMax2.printStats();
		itemsets.printItemsets();
	}
	
	public static String fileToPath(String filename) throws UnsupportedEncodingException{
		URL url = MainTestLCM_saveToMemory.class.getResource(filename);
		 return java.net.URLDecoder.decode(url.getPath(),"UTF-8");
	}
}
