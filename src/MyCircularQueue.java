public class MyCircularQueue {

    private final int mCapacity;
    private int mSize;
    /** Initialize your data structure here. Set the size of the queue to be k. */
    public MyCircularQueue(int k) {
        mCapacity = k;
        mSize = 0;
    }

    /** Insert an element into the circular queue. Return true if the operation is successful. */
    public boolean enQueue(int value) {
        return true;
    }

    /** Delete an element from the circular queue. Return true if the operation is successful. */
    public boolean deQueue() {
        return true;
    }

    /** Get the front item from the queue. */
    public int Front() {
        return 0;
    }

    /** Get the last item from the queue. */
    public int Rear() {
        return 0;
    }

    /** Checks whether the circular queue is empty or not. */
    public boolean isEmpty() {
        return mSize <= 0;
    }

    /** Checks whether the circular queue is full or not. */
    public boolean isFull() {
        return mSize >= mCapacity;
    }
}
