import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;

public class FreqStack {
    PriorityQueue<int[]> pq;
    Map<Integer, Integer> freq;
    int index;
    public FreqStack() {
        index = 0;
        freq = new HashMap<>();
        pq = new PriorityQueue<>((a, b) -> a[2] != b[2] ? b[2] - a[2] : b[1] - a[1]);
    }

    public void push(int x) {
        freq.put(x, freq.getOrDefault(x, 0) + 1);
        pq.offer(new int[]{x, index, freq.get(x)});
        index++;
    }

    public int pop() {
        int v = pq.poll()[0];
        freq.put(v, freq.get(v) - 1);
        return v;
    }
}
