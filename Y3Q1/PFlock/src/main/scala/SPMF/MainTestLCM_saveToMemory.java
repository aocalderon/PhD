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
package SPMF;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Example of how to use LCM algorithm from the source code.
 * @author Alan Souza <apsouza@inf.ufrgs.br> 
 */
public class MainTestLCM_saveToMemory {
	public static void main(String [] arg) throws IOException{
		String input = "/opt/LCM/T568.dat";
		String separator = ",";
		BufferedReader reader = new BufferedReader(new FileReader(input));
		String line;
		Set<List<Integer>> transactions = new HashSet<>();
		while (((line = reader.readLine()) != null)) {
			String[] lineSplited = line.split(separator);
			List<Integer> transaction = new ArrayList<>();
			for (String itemString : lineSplited) {
				Integer item = Integer.parseInt(itemString);
				transaction.add(item);
			}
			transactions.add(transaction);
		}
		int support = 1;
		Transactions dataset = new Transactions(transactions);

		AlgoLCM algoLCM = new AlgoLCM();
		Itemsets closed = algoLCM.runAlgorithm(support, dataset);
		for(List<Integer> maximal : closed.getMaximalItemsets1(1)){
			System.out.println(maximal.toString().replace("[", "").replace("]", "").replace(",", "") + " (1)");
		}
	}
}
