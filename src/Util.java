import java.util.ArrayList;
import java.util.List;

public class Util {
	public static <T> void printArrays(T[] array) {
		for (int i = 0; i < array.length; ++i) {
			System.out.print(array[i] + ", ");
		}
		System.out.println();
	}
	
	public static  void printArrays(int[] array) {
		for (int i = 0; i < array.length; ++i) {
			System.out.print(array[i] + ", ");
		}
		System.out.println();
	}
	
	public static  void printArrays(double[] array) {
		for (int i = 0; i < array.length; ++i) {
			System.out.print(array[i] + ", ");
		}
		System.out.println();
	}

	public static List<Integer> arrasyToList(int[] array) {
		List<Integer> list = new ArrayList<>();
		for (int num : array) {
			list.add(num);
		}
		return list;
	}
}
