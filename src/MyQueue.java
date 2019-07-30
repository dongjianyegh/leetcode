import java.util.Stack;

public class MyQueue {
    private final Stack<Integer> mPush;
    private final Stack<Integer> mPop;
    /** Initialize your data structure here. */
    public MyQueue() {
        mPush = new Stack<>();
        mPop = new Stack<>();
    }

    /** Push element x to the back of queue. */
    public void push(int x) {
        mPush.push(x);
    }

    /** Removes the element from in front of queue and returns that element. */
    public int pop() {
        if (mPop.isEmpty()) {
            while (!mPush.isEmpty()) {
                mPop.push(mPush.pop());
            }
        }
        return mPop.pop();
    }

    /** Get the front element. */
    public int peek() {
        if (mPop.isEmpty()) {
            while (!mPush.isEmpty()) {
                mPop.push(mPush.pop());
            }
        }
        return mPop.peek();
    }

    /** Returns whether the queue is empty. */
    public boolean empty() {
        return mPop.isEmpty() && mPush.isEmpty();
    }
}
