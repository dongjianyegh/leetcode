import java.util.Stack;

public class StockSpanner {

    private final Stack<int[]> mStack;
    private int mIdx = 0;
    private int mLastResult = 0;
    private int mLastPrice = -1;
    public StockSpanner() {
        mStack = new Stack<>();
    }

    public int next(int price) {
        if (mLastPrice != -1) {
            if (mLastPrice == price) {
                mStack.peek()[1] = mIdx++;
                return ++mLastPrice;
            } else {
                mLastPrice = price;
            }
        } else {
            mLastPrice = price;
        }

        int[] oldItem = null;
        while (!mStack.isEmpty() && mStack.peek()[0] <= price) {
            oldItem = mStack.pop();
        }

        final int[] item = oldItem != null ? oldItem :new int[2];
        item[0] = price;
        item[1] = mIdx++;

        final int result = mStack.isEmpty() ? mIdx : (mIdx - mStack.peek()[1] - 1);
        mStack.push(item);

        mLastResult = result;

        return result;
    }
}
