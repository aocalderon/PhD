package ca.pfv.spmf.test;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URL;

import ca.pfv.spmf.algorithms.frequentpatterns.charm.AlgoCharmLCM;
import ca.pfv.spmf.algorithms.frequentpatterns.lcm.AlgoLCM;
import ca.pfv.spmf.algorithms.frequentpatterns.lcm.Dataset;
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

		String input = fileToPath("contextPasquier99.txt");
		
		int minsup = 1;
		Dataset dataset = new Dataset(input);
		
		// Applying the algorithm
		AlgoLCM algo = new AlgoLCM();
		// if true in next line it will find only closed itemsets, otherwise, all frequent itemsets
		Itemsets closed = algo.runAlgorithm(minsup, dataset);
		algo.printStats();
		
		//itemsets.printItemsets(dataset.getTransactions().size());

		AlgoCharmLCM algo2 = new AlgoCharmLCM();
		algo2.runAlgorithm(null, closed);
		algo2.printStats();
		Itemsets maximals = algo2.getItemsets();
		maximals.printItemsets();
	}
	
	public static String fileToPath(String filename) throws UnsupportedEncodingException{
		URL url = MainTestLCM_saveToMemory.class.getResource(filename);
		 return java.net.URLDecoder.decode(url.getPath(),"UTF-8");
	}
}
