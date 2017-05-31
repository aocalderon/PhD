package main.java;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

//import java.io.UnsupportedEncodingException;
//import java.net.URL;
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
 *
 * @author Alan Souza <apsouza@inf.ufrgs.br>
 */
public class Tester {

    public static void main(String[] arg) throws IOException {
        //String input = fileToPath("test.dat");

        ArrayList<Integer> t1 = new ArrayList<>();
        t1.add(1);
        t1.add(2);
        t1.add(5);
        t1.add(7);
        t1.add(9);
        Collections.sort(t1);
        ArrayList<Integer> t2 = new ArrayList<>();
        t2.add(1);
        t2.add(3);
        t2.add(5);
        t2.add(7);
        t2.add(9);
        Collections.sort(t2);
        ArrayList<Integer> t3 = new ArrayList<>();
        t3.add(2);
        t3.add(4);
        t3.add(1);
        t3.add(3);
        Collections.sort(t3);
        ArrayList<Integer> t4 = new ArrayList<>();
        t4.add(1);
        t4.add(3);
        t4.add(4);
        t4.add(5);
        t4.add(6);
        Collections.sort(t4);
        ArrayList<Integer> t5 = new ArrayList<>();
        t5.add(1);
        t5.add(2);
        Collections.sort(t5);
        ArrayList<Integer> t6 = new ArrayList<>();
        t6.add(2);
        t6.add(1);
        Collections.sort(t6);
        ArrayList<Integer> t7 = new ArrayList<>();
        t7.add(1);
        t7.add(7);
        t7.add(2);
        t7.add(3);
        t7.add(4);
        t7.add(5);
        t7.add(8);
        t7.add(9);
        Collections.sort(t7);
        ArrayList<Integer> t8 = new ArrayList<>();
        t8.add(6);
        t8.add(1);
        t8.add(2);
        Collections.sort(t8);
        ArrayList<Integer> t9 = new ArrayList<>();
        t9.add(4);
        t9.add(5);
        t9.add(6);
        Collections.sort(t9);
        ArrayList<Integer> t10 = new ArrayList<>();
        t10.add(8);
        t10.add(2);
        t10.add(5);
        Collections.sort(t10);
        ArrayList<Integer> t11 = new ArrayList<>();
        t11.add(9);
        t11.add(2);
        t11.add(1);
        Collections.sort(t11);
        ArrayList<Integer> t12 = new ArrayList<>();
        t12.add(1);
        t12.add(2);
        t12.add(4);
        t12.add(8);
        t12.add(9);
        Collections.sort(t12);

        ArrayList<ArrayList<Integer>> ts = new ArrayList<>();
        ts.add(t1);
        ts.add(t2);
        ts.add(t3);
        ts.add(t4);
        ts.add(t5);
        ts.add(t6);
        ts.add(t7);
        ts.add(t8);
        ts.add(t9);
        ts.add(t10);
        ts.add(t11);
        ts.add(t12);
        AlgoFPMax fpMax = new AlgoFPMax();
        //Itemsets itemsets = fpMax.runAlgorithm(ts, 2);
        //fpMax.printStats();

        //itemsets.printItemsets();

    }

//	public static String fileToPath(String filename) throws UnsupportedEncodingException{
//		URL url = Tester.class.getResource(filename);
//		 return java.net.URLDecoder.decode(url.getPath(),"UTF-8");
//	}
}
