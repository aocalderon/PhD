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

import java.util.*;

public class Transactions {
    private List<Transaction> transactions;
    Set<Integer> uniqueItems = new HashSet<>();
	private int maxItem = 0;

    public Transactions(Set<List<Integer>> ts){
        this.transactions = new ArrayList<>(ts.size());

        for (List<Integer> t: ts) {
            this.transactions.add(createTransaction(t));
        }
        // sort transactions by increasing last item (optimization)
        transactions.sort(Comparator.comparingInt(arg0 ->
                arg0.getItems()[arg0.getItems().length - 1])
        );
    }

    private Transaction createTransaction(List<Integer> t) {
        int size = t.size();
        uniqueItems.addAll(t);
        // update max item by checking the last item of the transaction
        int lastItem = t.get(size - 1);
        if(lastItem > maxItem) {
            maxItem = lastItem;
        }
        return new Transaction(t.toArray(new Integer[size]));
    }

     public List<Transaction> getTransactions() {
        return transactions;
    }

    public Set<Integer> getUniqueItems() {
		return uniqueItems;
	}

    public int getMaxItem() {
        return maxItem;
    }

    @Override
    public String toString() {
        StringBuilder db = new StringBuilder();

        for(Transaction transaction : transactions) {
            db.append(transaction);
            db.append("\n");
        }
        return db.toString();
    }
}
