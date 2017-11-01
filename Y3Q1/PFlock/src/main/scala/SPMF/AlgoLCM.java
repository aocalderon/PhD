package SPMF;
/*
 * This is an implementation of the LCM algorithm for
 * mining frequent closed itemsets from a transaction database.
 * More information on the LCM algorithm can be found in papers by
 * T. Uno, such as: <br/><br/>
 *
 * T. Uno, M. Kiyomi, and H. Arimura. Lcm ver. 2:
 * Efficient mining algorithms for
 * frequent/closed/maximal itemsets. In FIMI, 2004
 *
 * This implementation of LCM was made by Alan Souza and was
 * modified by Philippe Fournier-Viger to add optimizations.. <br/>
 *
 * The implementation is similar to LCM version 2 with some differences.
 * For example, transaction merging is not performed yet and
 * items in transactions are not sorted in descending order of frequency.
 *
 * This file is copyright (c) 2012-2014 Alan Souza
 */

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class AlgoLCM {
    private Itemsets closedFrequentItemsets;
	// the number of frequent itemsets found (for
	// statistics)
	private int frequentCount; 
	// the start time and end time of the last algorithm execution
	private long startTimestamp;
	private long endTimestamp;
	private int minsupRelative;
	// Buckets for occurence delivery
	// Recall that each bucket correspond to an item
    // and contains the transactions where the items appears.
	private List<Transaction>[] buckets;

	public Itemsets runAlgorithm(int minimumSupport, Transactions dataset){
		// record the start time
		startTimestamp = System.currentTimeMillis();
		this.closedFrequentItemsets = new Itemsets();
		// reset the number of itemset found
		frequentCount = 0;
		// reset the memory usage checking utility
		MemoryLogger.getInstance().reset();
		// convert from an absolute minsup to a relative minsup by multiplying
		// by the database size
		this.minsupRelative = minimumSupport;
		// Create the initial occurrence array for the dataset
		performFirstOccurenceDelivery(dataset);
		// Remove infrequent items from transactions by using support calculated using
		// the buckets. Recall that each bucket correspond to an item
		// and contains the transactions where the items appears.
		for(Transaction transaction : dataset.getTransactions()) {
			transaction.removeInfrequentItems(buckets, minsupRelative);
		}
		// Create the array of all frequent items.
		List<Integer> allItems = new ArrayList<>();
		for(Integer item : dataset.getUniqueItems()) {
			if(buckets[item].size() >= minsupRelative) {
				allItems.add(item);
			}
		}
		// Sort all items
		Collections.sort(allItems);
		// Call the recursive method witht the empty set as prefix.
		// Since it is the empty set, we will have all transactions and no frequency count
		backtrackingLCM(null, dataset.getTransactions(), allItems, -1);
		// record the end time
		endTimestamp = System.currentTimeMillis();
		MemoryLogger.getInstance().checkMemory();

		return closedFrequentItemsets;
	}

	private void backtrackingLCM(List<Integer> p, List<Transaction> transactionsOfP, List<Integer> frequentItems, int tailPosInP) {
        // for each frequent item e
		for (int j = 0; j < frequentItems.size(); j++) {
			Integer e = frequentItems.get(j);
			// if the item is not already in p  before the current tail position
			// we will consider it to form a new closed itemset
			if(p != null && containsByBinarySearch(p, e, tailPosInP)) {
				continue;
			}
			// Calculate transactions containing P U e
			// At the same time truncate the transactions to keep what appears after "e"
			List<Transaction> transactionsPe = intersectTransactions(transactionsOfP, e); //ok
			// Check if PU{e...} is a ppc extension
			if (isPPCExtension(p, transactionsPe, e)) {
				// Create a closed itemset using PU{e...}
				// First add all items from PU{e}
		    	List<Integer> itemset = new ArrayList<>();
		    	if(p != null) {
			        //add every item i of p  such that i < e to the  itemset
			        for (int m = 0; m < p.size() && p.get(m) < e; m++) {
			        	itemset.add(p.get(m));
			        }
		    	}
		    	itemset.add(e);
		    	int tailPositionInPe = itemset.size()-1;
		    	for (int k = j+1; k < frequentItems.size(); k++) {
					Integer itemk = frequentItems.get(k);
		            // for every item i > e add if it is in all transactions of T(P U e)
		            if(isItemInAllTransactions(transactionsPe, itemk)) {
		            	itemset.add(itemk);
		            }
		        }
		        // save the frequent closed itemset
		    	int supportPe = transactionsPe.size();
				output(itemset, supportPe);
				// perform database reduction
				anyTimeDatabaseReductionClosed(transactionsPe, j, frequentItems, e);
		    	// Find frequent items in transactions containing P
		        // Get all frequent items e such that e > tailOfP  
		    	// (i.e. "e" appears after the position of the tail item in the list of all items)
				List<Integer> newFrequentItems = new ArrayList<>();
		    	for (int k = j+1; k < frequentItems.size(); k++) {
		        	Integer itemK =  frequentItems.get(k);
		        	int supportK = buckets[itemK].size();
		            if(supportK >= minsupRelative) {
		            	newFrequentItems.add(itemK);
		            }
		        }
				// recursive call
				backtrackingLCM(itemset, transactionsPe, newFrequentItems, tailPositionInPe);
			}
		}
		MemoryLogger.getInstance().checkMemory();
    }

	private void performFirstOccurenceDelivery(Transactions dataset) {
		buckets = new List[dataset.getMaxItem() + 1];
		for (Integer item : dataset.uniqueItems) buckets[item] = new ArrayList<>();
		for (Transaction transaction : dataset.getTransactions()) {
			for (Integer item : transaction.getItems()) {
				// for each item get its bucket and add the current transaction
				buckets[item].add(transaction);
			}
		}
	}

    private void anyTimeDatabaseReductionClosed(List<Transaction> transactionsPe, int j, List<Integer> frequentItems, Integer e) {
		// We just reset the buckets for item  > e instead of all buckets
		for (int i = j+1; i < frequentItems.size(); i++) {
			Integer item = frequentItems.get(i);
			buckets[item] = new ArrayList<>();
		}
       // for each transaction
		for(Transaction transaction : transactionsPe) {
			// we consider each item I  of the transaction such that  itemI > e
			for(int i = transaction.getItems().length-1; i >transaction.getOffset(); i--) {
				Integer item = transaction.getItems()[i];
				if(item > e && frequentItems.contains(item)) {
					// we add the transaction to the bucket of the itemI
					buckets[item].add(transaction);
				}
			}
		}
	}
    
	private boolean containsByBinarySearch(List<Integer> items, Integer item, int searchAfterPosition) {
		if(items.size() == 0 || item > items.get(items.size() -1)) {
			return false;
		}
		int low = searchAfterPosition +1;
		int high = items.size() - 1;

		while (high >= low) {
			int middle = ( low + high ) >>> 1; // divide by 2
			if (items.get(middle).equals(item)) {
				return true;
			}
			if (items.get(middle) < item) {
				low = middle + 1;
			}
			if (items.get(middle) > item) {
				high = middle - 1;
			}
		}
		return false;
	}
	
	private boolean containsByBinarySearch(List<Integer> items, Integer item) {
		if(items.size() == 0 || item > items.get(items.size() -1)) {
			return false;
		}
		int low = 0;
		int high = items.size() - 1;

		while (high >= low) {
			int middle = ( low + high ) >>> 1; // divide by 2
			if (items.get(middle).equals(item)) {
				return true;
			}
			if (items.get(middle) < item) {
				low = middle + 1;
			}
			if (items.get(middle) > item) {
				high = middle - 1;
			}
		}
		return false;
	}
	
	private List<Transaction> intersectTransactions(List<Transaction> transactionsOfP, Integer e) {
        List<Transaction> transactionsPe = new ArrayList<>();

        // transactions of P U e
        for(Transaction transaction : transactionsOfP) {
        	// we remember the position where e appears.
        	// we will call this position an "offset"
        	int posE = transaction.containsByBinarySearch(e);
            if (posE != -1) { // T(P U e)
                transactionsPe.add(new Transaction(transaction, posE));
            }
        }
        return transactionsPe;
    }


    /**
     * Check if a given itemset PUe is a PPC extension according to
     * the set of transactions containing PUe.
     * @param p the itemset p
     * @param e the item e
     * @param transactionsPe  the transactions containing P U e
     * @return true if it is a PPC extension
     */
    private boolean isPPCExtension(List<Integer> p, List<Transaction> transactionsPe, Integer e) {
    	// We do a loop on each item i of the first transaction 
    	if(transactionsPe.size() == 0) { return false; }
    	Transaction firstTrans = transactionsPe.get(0);
    	Integer[] firstTransaction = firstTrans.getItems();
        for (int i = 0; i < firstTrans.getOffset(); i++) {
        	Integer item = firstTransaction[i];
            // if p does not contain item i < e and item i is present in all transactions, 
        	// then it PUe is not a ppc
            if(item < e && (p == null || !containsByBinarySearch(p,item))
                    && isItemInAllTransactionsExceptFirst(transactionsPe, item)) {
                return false;
            }
        }
        return true;
    }
    
    private boolean isItemInAllTransactionsExceptFirst(List<Transaction> transactions, Integer item) {
    	for(int i=1; i < transactions.size(); i++) {
            if(!transactions.get(i).containsByBinarySearchOriginalTransaction(item)) {
                return false;
            }
        }
        return true;
    }

    private boolean isItemInAllTransactions(List<Transaction> transactions, Integer item) {
        for(Transaction transaction : transactions) {
            if(transaction.containsByBinarySearch(item) == -1) {
                return false;
            }
        }
        return true;
    }

    private void output(List<Integer> itemset, int support){
    	// if not the empty set
        if(!itemset.isEmpty()) {
            frequentCount++;
       		closedFrequentItemsets.addItemset(new Itemset(itemset, support), itemset.size());
        }
    }

	public double getTime() {
		return (endTimestamp - startTimestamp)/1000.0;
	}

	public String getStats() {
		return String.format("%d, %.2f", frequentCount, MemoryLogger.getInstance().getMaxMemory());
	}

	public void printStats() {
		System.out.println("========== LCM - STATS ============");
		System.out.println(" Freq. closed itemsets count: " + frequentCount);			
		System.out.println(" Total time ~: " + (endTimestamp - startTimestamp) + " ms");
		System.out.println(" Max memory:" + MemoryLogger.getInstance().getMaxMemory());
		System.out.println("=====================================");
	}
}
