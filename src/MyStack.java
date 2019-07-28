import java.util.ArrayDeque;
import java.util.Queue;

public class MyStack {

    private Queue<Integer> mPush;

    /** Initialize your data structure here. */
    public MyStack() {
        mPush = new ArrayDeque<>();
    }

    /** Push element x onto stack. */
    public void push(int x) {
        mPush.add(x);
        int size  = mPush.size();
        while (size > 1) {
            mPush.add(mPush.poll());
            size--;
        }
    }

    /** Removes the element on top of the stack and returns that element. */
    public int pop() {
        return mPush.poll();
    }

    /** Get the top element. */
    public int top() {
        return mPush.peek();
    }

    /** Returns whether the stack is empty. */
    public boolean empty() {
        return mPush.isEmpty();
    }
}
