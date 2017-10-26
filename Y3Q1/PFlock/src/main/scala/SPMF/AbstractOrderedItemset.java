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


/**
 * This is an abstract class indicating general methods
 * that an ordered itemset should have, and is designed for ordered itemsets where items are sorted
 * by lexical order and no item can appear twice.
 *
 * @author Philippe Fournier-Viger
 * @see AbstractItemset
 */
public abstract class AbstractOrderedItemset extends AbstractItemset {

    AbstractOrderedItemset() {
        super();
    }

    /**
     * Get the support of this itemset
     *
     * @return the support of this itemset
     */
    public abstract int getAbsoluteSupport();

    /**
     * Get the size of this itemset
     *
     * @return the size of this itemset
     */
    public abstract int size();

    /**
     * Get the item at a given position of this itemset
     *
     * @param position the position of the item to be returned
     * @return the item
     */
    public abstract Integer get(int position);

    /**
     * Get this itemset as a string
     *
     * @return a string representation of this itemset
     */
    public String toString() {
        if (size() == 0) {
            return "EMPTYSET";
        }
        // use a string buffer for more efficiency
        StringBuilder r = new StringBuilder();
        // for each item, append it to the StringBuilder
        for (int i = 0; i < size(); i++) {
            r.append(get(i));
            r.append(' ');
        }
        return r.toString(); // return the tring
    }


    /**
     * Check if this itemset contains a given item.
     *
     * @param item the item
     * @return true if the item is contained in this itemset
     */
    public boolean contains(Integer item) {
        for (int i = 0; i < size(); i++) {
            if (get(i).equals(item)) {
                return true;
            } else if (get(i) > item) {
                return false;
            }
        }
        return false;
    }

    /**
     * This methods checks if another itemset is contained in this one.
     * The method assumed that items are lexically ordered in itemsets.
     *
     * @param itemset2 the other itemset
     * @return true if it is contained
     */
    public boolean containsAll(AbstractOrderedItemset itemset2){
        // first we check the size
        if(size() < itemset2.size()){
            return false;
        }

        // we will use this variable to remember where we are in this itemset
        int i = 0;

        // for each item in itemset2, we will try to find it in this itemset
        for(int j =0; j < itemset2.size(); j++){
            boolean found = false; // flag to remember if we have find the item at position j

            // we search in this itemset starting from the current position i
            while(!found && i< size()){
                // if we found the current item from itemset2, we stop searching
                if(get(i).equals(itemset2.get(j))){
                    found = true;
                }// if the current item in this itemset is larger than
                // the current item from itemset2, we return false
                // because the itemsets are assumed to be lexically ordered.
                else if(get(i) > itemset2.get(j)){
                    return false;
                }

                i++; // continue searching from position  i++
            }
            // if the item was not found in the previous loop, return false
            if(!found){
                return false;
            }
        }
        return true; // if all items were found, return true
    }

}