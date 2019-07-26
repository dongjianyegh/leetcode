import java.util.TreeMap;

public class RLEIterator {

    private final TreeMap<Long, Integer> mNums;
    private final long mSize;
    private long mCur = 0;

    public RLEIterator(int[] A) {
        mNums = new TreeMap<>();
        if (A == null) {
            mSize = 0L;
        } else {
            long cur = 0;
            for (int i = 0; i < A.length; i += 2) {
                if (A[i] > 0) {
                    cur += A[i];
                    mNums.put(cur, A[i + 1]);
                }
            }
            mSize = cur;
        }
    }

    public int next(int n) {
        mCur += n;

        if (mCur > mSize) {
            return -1;
        } else {
            return mNums.ceilingEntry(mCur).getValue();
        }
    }
}
