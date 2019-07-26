import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.function.Function;

public class TimeMap {

    private class Node {
        final String mValue;
        final int mTimestamp;
        Node(String value, int timestamp) {
            mValue = value;
            mTimestamp = timestamp;
        }
    }

    private final HashMap<String, List<Node>> mMap = new HashMap<>();

    public TimeMap() {

    }

    public void set(String key, String value, int timestamp) {
        mMap.computeIfAbsent(key, new Function<String, List<Node>>() {
            @Override
            public List<Node> apply(String s) {
                return new ArrayList<>();
            }
        }).add(new Node(value, timestamp));
    }

    public String get(String key, int timestamp) {
        final List<Node> arrays = mMap.get(key);
        return binarySearch(arrays, timestamp);
    }

    private String binarySearch(List<Node> arrays, int timestamp) {
        if (arrays == null || arrays.size() <= 0) {
            return "";
        }

        int left = 0;
        int right = arrays.size() - 1;

        while (left <= right) {
            int middle = left + (right - left) / 2;
            Node node = arrays.get(middle);

            if (node.mTimestamp == timestamp) {
                return node.mValue;
            } else if (node.mTimestamp > timestamp) {
                right = middle - 1;
            } else {
                left = middle + 1;
            }
        }

        return right == -1 ? "" : arrays.get(right).mValue;
    }


}
