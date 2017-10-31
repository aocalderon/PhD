package Misc;

import fr.liglab.jlcm.io.PatternsWriter;

import java.util.ArrayList;

public class ListCollector extends PatternsWriter {
    private ArrayList<ArrayList<Integer>> closed;
    private long collected = 0;
    private long collectedLength = 0;

    ListCollector(){
        closed =  new ArrayList<>();
    }

    @Override
    synchronized public void collect(int support, int[] pattern, int length) {
        ArrayList<Integer> p = new ArrayList<>();

        for (int i = 0; i < length; i++) {
            p.add(pattern[i]);
        }
        closed.add(p);
        this.collected++;
        this.collectedLength += pattern.length;
    }

    public ArrayList<ArrayList<Integer>> getClosed() {
        return closed;
    }

    public long close() {
        closed.clear();
        return this.collected;
    }

    public int getAveragePatternLength() {
        if (this.collected == 0) {
            return 0;
        } else {
            return (int) (this.collectedLength / this.collected);
        }
    }
}

