package SPMF;
/* This file is copyright (c) 2008-2012 Philippe Fournier-Viger
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

import java.util.ArrayList;
import java.util.List;

public class Itemsets {
    private final List<List<Itemset>> levels = new ArrayList<>();
    private int itemsetsCount = 0;

    Itemsets() {
        levels.add(new ArrayList<>()); // We create an empty level 0 by default.
    }

    public void printItemsets() {
        for (List<Itemset> level : levels) {
            for (Itemset itemset : level) {
                System.out.format("%s[%d]%n", itemset.toString(), itemset.getAbsoluteSupport());
            }
        }
    }

    public ArrayList<ArrayList<Integer>> getItemsets(int mu) {
        ArrayList<ArrayList<Integer>> itemsets = new ArrayList<>();
        for (List<Itemset> level : levels) {
            if (level.size() != 0) {
                if (level.get(0).size() >= mu) {
                    for (Itemset aLevel : level) {
                        int[] array = aLevel.getItems();
                        ArrayList<Integer> list = new ArrayList<>(array.length);
                        for (int anArray : array) list.add(anArray);
                        itemsets.add(list);
                    }
                }
            }
        }
        return itemsets;
    }

    public ArrayList<ArrayList<Integer>> getMaximalItemsets1(int mu) {
        ArrayList<ArrayList<Integer>> itemsets = new ArrayList<>();
        for (List<Itemset> level : levels) {
            if (level.size() != 0) {
                if (level.get(0).size() >= mu) {
                    for (Itemset aLevel : level) {
                        if (aLevel.support == 1) {
                            int[] array = aLevel.getItems();
                            ArrayList<Integer> list = new ArrayList<>(array.length);
                            for (int anArray : array) {
                                list.add(anArray);
                            }
                            itemsets.add(list);
                        }
                    }
                }
            }
        }
        return itemsets;
    }

    public void addItemset(Itemset itemset, int k) {
        while (levels.size() <= k) {
            levels.add(new ArrayList<>());
        }
        levels.get(k).add(itemset);
        itemsetsCount++;
    }

    public List<List<Itemset>> getLevels() {
        return levels;
    }

    public int getItemsetsCount() {
        return itemsetsCount;
    }

    public void decreaseItemsetCount() {
        itemsetsCount--;
    }
}
