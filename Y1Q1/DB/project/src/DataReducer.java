import java.io.IOException;
import java.io.DataInput;
import java.io.DataOutput;
import java.util.*;
import java.text.DateFormatSymbols;
import java.math.BigDecimal;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;
import org.apache.hadoop.util.*;
import org.apache.hadoop.fs.FileSystem;

public class DataReducer extends Configured implements Tool{ 
	
	/* Class to map station data  */
	public static class StationMapper extends Mapper<LongWritable, Text, Text, Text> {
     	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
       		String line = value.toString();
       		String columns[] = line.split("\",\"");
			// Take data just for US and where state label is not empty.  Also discard header...
			if(columns.length >= 5 && !columns[0].equals("USAF") && columns[3].equals("US") && !columns[4].equals("")){
				// If station id is different to 999999...
				if(!columns[0].equals("\"999999")){
					// Map de station id as key and the state as value.  Mark states with a S...
					context.write(new Text(columns[0].substring(1)), new Text("S~"+columns[4]));
				}
			}
		}
	}	
	
	/* Class to map temperature data from files */
	public static class DataMapper extends Mapper<LongWritable, Text, Text, Text> {
     	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
       		String line = value.toString();
       		StringBuilder buffer = new StringBuilder();
       		String header = line.substring(0, 6);		// Capture the station id...
			// Discard multiple headers...
			if(!header.equals("STN---")){
				buffer.append(line.substring(18, 20)).append(","); 	// Capture date...
				buffer.append(line.substring(24, 30)); 			// Capture temperature...
				// Map the station id as key and date and temperature as value.  Mark this info with a D...
				context.write(new Text(header), new Text("D~"+buffer.toString()));
			}
		}
	}	
	
	/* Class to map temporary files to be passed to the next reducer */
	public static class FileMapper extends Mapper<LongWritable, Text, Text, Text> {
     	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
       		String line = value.toString();
       		String columns[] = line.split("\t");
			context.write(new Text(columns[0]), new Text(columns[1]));
     	}
   	}
	
	/* Class to sort and format the final result */
	public static class SortMapper extends Mapper<LongWritable, Text, SortableKey, Text> {
     	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
       		String line = value.toString();
       		String columns[] = line.split("\t");
			String max_month = new DateFormatSymbols().getMonths()[Integer.parseInt(columns[1])-1];
			String min_month = new DateFormatSymbols().getMonths()[Integer.parseInt(columns[3])-1];
			String output = String.format("%10s\t%5s\t%10s\t%5s\t%5s", max_month, columns[2], min_month, columns[4], columns[5]);
			context.write(new SortableKey(columns[0], new Float(columns[5])), new Text(output));
     	}
   	}
	
	/* Reduce job to join the data */
	public static class JoinReducer extends Reducer<Text, Text, Text, Text> {
		private ArrayList<String> data;
		private String state;
		private String value;
		private String month;
		private String temperature;
		private String[] records;

		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException{
			data = new ArrayList<String>();
			state = null;
			// For each station id collect its state and data (month and temperature) separately...
			for (Text v : values) {
				value = v.toString();
				records = value.split("~");
				if(records[0].equals("D")){
					data.add(records[1]);
				} else {
					state = records[1];
				}
			}
			// Just if there are state and data available...
			if(state != null && data.size() > 0){
				// For each state and month combination store their temperature...
				for(String sample : data){
					records = sample.split(",");
					month = records[0];
					temperature = records[1];
					// Map the state and month as key and temperature as values...
					context.write(new Text(state + "-" + month), new Text(temperature));
				}
			}
		}
	}
	
	/* Reduce job to compute the average for each state-month key */
	public static class AverageReducer extends Reducer<Text, Text, Text, Text> {
		private float temperature;
		private float sum;
		private int count;
		private float avg;
		
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException{
			sum = 0;
			count = 0;	
			for(Text value : values){
				temperature = Float.parseFloat(value.toString());
				sum += temperature;
				count++;
			}
			avg = sum / count;
			String[] parts = key.toString().split("-");
			// Map the state as key and month and its average as value...
			context.write(new Text(parts[0]), new Text(parts[1] + "," + avg));
		}
	}
	
	/* Reduce job to select the months with maximum and minimum temperature averages */
	public static class MaxMinReducer extends Reducer<Text, Text, Text, Text> {
		private float avg;
		private float min;
		private float max;
		private float dif;
		private String min_month;
		private String max_month;
		
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException{
			min = Float.MAX_VALUE;
			max = Float.MIN_VALUE;
			for(Text value : values){
				String[] parts = value.toString().split(",");
				avg = Float.parseFloat(parts[1]);
				if(max < avg){ 
					max = avg;  
					max_month = parts[0];
				}
				if(min > avg){
					min = avg;
					min_month = parts[0];
				}
			}
			dif = max - min;
			// Map keeping the same key (state) and the collected info as value...
			context.write(key, new Text(max_month + "\t" + round(max, 2) + "\t" + min_month + "\t" + round(min, 2) + "\t" + round(dif, 2)));
		}
	}

	/* Just a fancy class to print by state but sort by difference */
	public static class SortableKey implements WritableComparable<SortableKey>{
		private String state;
		private Float dif;

		public SortableKey(){
		}		
		
		public SortableKey(String state, Float dif){
			this.state = state;
			this.dif = dif;
		}
		
		@Override
		public String toString(){
			return this.state; // Show just the state...
		}

		@Override
		public int compareTo(SortableKey o){
			return dif.compareTo(o.dif); // Sort by difference...
		}

		@Override
		public void readFields(DataInput in) throws IOException{
			this.state = WritableUtils.readString(in);
			this.dif = in.readFloat();
		}
		
		@Override
		public void write(DataOutput out) throws IOException{
			WritableUtils.writeString(out, state);
			out.writeFloat(this.dif);
		}
	}

	@Override
	public int run(String[] args) throws Exception {
		String PATH_LOC = args[0];
		String PATH_REC = args[1];
		String PATH_OUT = args[2];
		String PATH1 =  PATH_OUT + "temp1";
		String PATH2 =  PATH_OUT + "temp2";
		String PATH3 =  PATH_OUT + "temp3";
		String PATH4 =  PATH_OUT + "final";

		System.out.println("Reading from... " + args[0]);	
	
		Configuration conf = new Configuration();
		
		// First job read the files and call Station and Data Mappers...
		// The output will be joined by JoinReducer and saved to PATH1...
		Job job1 = new Job(conf, "Job1");
		job1.setJarByClass(DataReducer.class);
        MultipleInputs.addInputPath(job1, new Path(PATH_LOC + "WeatherStationLocations.csv"), TextInputFormat.class, StationMapper.class);
       	MultipleInputs.addInputPath(job1, new Path(PATH_REC + "2006.txt"), TextInputFormat.class, DataMapper.class);
       	MultipleInputs.addInputPath(job1, new Path(PATH_REC + "2007.txt"), TextInputFormat.class, DataMapper.class);
       	MultipleInputs.addInputPath(job1, new Path(PATH_REC + "2008.txt"), TextInputFormat.class, DataMapper.class);
       	MultipleInputs.addInputPath(job1, new Path(PATH_REC + "2009.txt"), TextInputFormat.class, DataMapper.class);
		job1.setReducerClass(JoinReducer.class);
		job1.setOutputValueClass(Text.class);
		job1.setOutputKeyClass(Text.class);
		TextOutputFormat.setOutputPath(job1, new Path(PATH1));
			
		job1.waitForCompletion(true);

		// Read PATH1 and compute the average temperature grouped by state and month... 			
		Job job2 = new Job(new Configuration(), "Job2");
		job2.setJarByClass(DataReducer.class);
		job2.setMapperClass(FileMapper.class);
		job2.setReducerClass(AverageReducer.class);
		job2.setOutputKeyClass(Text.class);
		job2.setOutputValueClass(Text.class);
		TextInputFormat.addInputPath(job2, new Path(PATH1));
		TextOutputFormat.setOutputPath(job2, new Path(PATH2));
		
		job2.waitForCompletion(true);
       		
		// Job 3 call MaxMinReducer to extract maximum and minimum month temperature for each state...
		Job job3 = new Job(new Configuration(), "Job3");
		job3.setJarByClass(DataReducer.class);
		job3.setMapperClass(FileMapper.class);
		job3.setReducerClass(MaxMinReducer.class);
		job3.setOutputKeyClass(Text.class);
		job3.setOutputValueClass(Text.class);
		TextInputFormat.addInputPath(job3, new Path(PATH2));
		TextOutputFormat.setOutputPath(job3, new Path(PATH3));
               				
		job3.waitForCompletion(true);
		
		// Job 4 just read the last output, map it using SortableKey and let the default reducer to sort the data...
		Job job4 = new Job(new Configuration(), "Job4");
		job4.setJarByClass(DataReducer.class);
		job4.setMapperClass(SortMapper.class);
		job4.setOutputKeyClass(SortableKey.class);
		job4.setOutputValueClass(Text.class);
		TextInputFormat.addInputPath(job4, new Path(PATH3));
		TextOutputFormat.setOutputPath(job4, new Path(PATH4));
			
		return job4.waitForCompletion(true) ? 0 : 1;
	}
           	
	public static BigDecimal round(float d, int decimalPlace) {
		BigDecimal bd = new BigDecimal(Float.toString(d));
		bd = bd.setScale(decimalPlace, BigDecimal.ROUND_HALF_UP);       
		return bd;
	}
                                                             	
	public static void main(String[] args) throws Exception {
		int r = ToolRunner.run(new DataReducer(), args);
		if(r == 0){
			System.out.println("Done! See the result at " + args[1] + "final/part-r-0000");
		}
	}	
}
