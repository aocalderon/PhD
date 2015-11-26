import java.util.*;

public class Test{
	public static void main(String[] args){
		String str = args[0];
		
		for (String column: str.split("\\s+")){
			System.out.println(column);
		}
	}
}
