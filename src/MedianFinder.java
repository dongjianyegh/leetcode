import java.util.Comparator;
import java.util.PriorityQueue;

public class MedianFinder {
	/** initialize your data structure here. */

	private int mCnt;
	private final PriorityQueue<Integer> mMinHeap;
	private final PriorityQueue<Integer> mMaxHeap;
    public MedianFinder() {
    		mCnt = 0;
    		mMinHeap = new PriorityQueue<>();
	    	mMaxHeap = new PriorityQueue<>(new Comparator<Integer>() {
	    		@Override
	    		public int compare(Integer o1, Integer o2) {
	    			// TODO Auto-generated method stub
	    			return o2 - o1;
	    		}
	    	});
    }
    
    public void addNum(int num) {
        mCnt++;
        int left = (1 + mCnt) / 2;
        
        if (mCnt == 1) {
        		mMaxHeap.add(num);
        		return;
        }
        
        int max = mMaxHeap.peek();
        if (num >= max) {
        		mMinHeap.add(num);
        } else {
        		mMaxHeap.add(num);
        }
        
        while (mMaxHeap.size() < left) {
        		mMaxHeap.add(mMinHeap.poll());
        }
        
        while (mMaxHeap.size() > left) {
        		mMinHeap.add(mMaxHeap.poll());
        }
    }
    
    public double findMedian() {
        if ((mCnt & 1) == 0) {
        		return (mMaxHeap.peek() + mMinHeap.peek()) / 2.0f;
        } else {
        		return mMaxHeap.peek();
        }
    }
}
