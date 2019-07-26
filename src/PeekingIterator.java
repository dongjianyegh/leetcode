import java.util.Iterator;

public class PeekingIterator implements Iterator<Integer> {

    private final Iterator<Integer> mInterator;

    private Integer mCurrent = null;

    public PeekingIterator(Iterator<Integer> iterator) {
        // initialize any member here.
        mInterator = iterator;
    }

    // Returns the next element in the iteration without advancing the iterator.
    public Integer peek() {
        if (mCurrent == null) {
            mCurrent = mInterator.next();
        }

        return mCurrent;
    }

    // hasNext() and next() should behave the same as in the Iterator interface.
    // Override them if needed.
    @Override
    public Integer next() {
        if (mCurrent != null) {
            final Integer oldNext = mCurrent;
            mCurrent = null;
            return oldNext;
        } else {
            return mInterator.next();
        }
    }

    @Override
    public boolean hasNext() {
        if (mCurrent != null) {
            return true;
        } else {
            return mInterator.hasNext();
        }
    }
}
