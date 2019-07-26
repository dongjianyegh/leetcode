import java.util.*;

public class NestedIterator implements Iterator<Integer> {

    private final Deque<NestedInteger> mQueue;
    public NestedIterator(List<NestedInteger> nestedList) {
        mQueue = new LinkedList<>();
        for (NestedInteger nestedInteger : nestedList) {
            mQueue.addLast(nestedInteger);
        }
    }

    @Override
    public Integer next() {
        return mQueue.pollFirst().getInteger();
    }

    @Override
    public boolean hasNext() {
        while (!mQueue.isEmpty()) {
            NestedInteger head = mQueue.peekFirst();
            if (head.isInteger()) {
                return true;
            }

            mQueue.poll();

            ListIterator<NestedInteger> iterator = head.getList().listIterator(head.getList().size()-1);
            while (iterator.hasPrevious()) {
                mQueue.addFirst(iterator.previous());
            }
        }

        return false;

    }
}
