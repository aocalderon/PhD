/* This file is copyright (c) 2008-2013 Philippe Fournier-Viger
 *
 * This file is part of the SPMF DATA MINING SOFTWARE
 * (http://www.philippe-fournier-viger.com/spmf).
 *
 * SPMF is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * SPMF is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License along with
 * SPMF. If not, see <http://www.gnu.org/licenses/>.
 */
package SPMF;

import java.util.Iterator;
import java.util.List;

public class AlgoCharmLCM {
    private long startTimestamp;
    private long endTimestamp;
    private int count;

    public AlgoCharmLCM() {
    }

    public Itemsets runAlgorithm(Itemsets frequentClosed){
        startTimestamp = System.currentTimeMillis();
        // get the size of the largest closed itemset.
        if(frequentClosed != null) {
            int maxItemsetLength = frequentClosed.getLevels().size();
            // For closed itemsets of size i=1 to the largest size
            for (int i = 1; i < maxItemsetLength - 1; i++) {
                // Get the itemsets of size i
                List<Itemset> ti = frequentClosed.getLevels().get(i);
                // For closed itemsets of size j = i+1 to the largest size
                for (int j = i + 1; j < maxItemsetLength; j++) {
                    // get itemsets of size j
                    List<Itemset> tip1 = frequentClosed.getLevels().get(j);
                    // Check which itemsets are maximals by comparing itemsets
                    // of size i and i+1
                    findMaximal(ti, tip1, frequentClosed);
                }
            }
            this.count = frequentClosed.getItemsetsCount();
        }
        endTimestamp = System.currentTimeMillis();

        return frequentClosed;
    }

    private void findMaximal(List<Itemset> ti, List<Itemset> tip1, Itemsets frequentClosed) {
        // for each itemset of J
        for (Itemset itemsetJ : tip1) {
            // iterates over the itemsets of size I
            Iterator<Itemset> iter = ti.iterator();
            while (iter.hasNext()) {
                Itemset itemsetI = iter.next();
                // if the current itemset of size I is contained in the current itemset of size J
                if (itemsetJ.containsAll(itemsetI) ) {
                    // Then, it means that the itemset of size I is not maximal so we remove it
                    iter.remove();
                    // We decrease the current number of maximal itemsets.
                    frequentClosed.decreaseItemsetCount();
                }
            }
        }
    }

    public double getTime() {
        return (endTimestamp - startTimestamp)/1000.0;
    }

    public String getStats() {
        return String.format("%d, %.2f", count, MemoryLogger.getInstance().getMaxMemory());
    }

    public void printStats() {
        System.out.println("=============  Charm_LCM-MFI - STATS =============");
        long temps = endTimestamp - startTimestamp;
        System.out.println(" Frequent maximal itemsets count : " + this.count);
        System.out.println(" Total time ~ " + temps + " ms");
        System.out.println("===================================================");
    }
}

