public class SegmentTree {

    private final int[] mTree;
    private final int[] mLazy;
    private final int mCapacity;

    public SegmentTree(final int capacity) {
        mCapacity = capacity;
        mTree = new int[capacity << 1 + 1];
        mLazy = new int[capacity << 1 + 1];
    }

    public void update(int left, int right, int value) {
        updateLazy(0, 0, mCapacity - 1, left, right, value);
    }

    private void updateLazy(int idx, int low, int high, int left, int right, int value) {
        if (left > high || right < low) {
            return;
        }

        if (left <= low && right >= high) {
            mTree[idx] = Math.max(mTree[idx], value);
            mLazy[idx] = Math.max(mLazy[idx], value);

            return;
        }


        pushDown(idx);

        final int mid = low + (high - low) / 2;

        updateLazy(idx * 2 + 1, low, mid, left, right, value);
        updateLazy(idx * 2 + 2, mid + 1, high, left, right, value);

        mTree[idx] = Math.max(mTree[idx * 2 + 1], mTree[idx * 2 + 2]);
    }

    public int query(int left, int right) {
        return queryLazy(0, 0, mCapacity - 1, left, right);
    }

    private int queryLazy(int idx, int low, int high, int left, int right) {
        if (left > high || right < low) {
            return 0;
        }

        if (left <= low && right >= high) {
            return mTree[idx];
        }

        pushDown(idx);

        final int mid = low + (high - low) / 2;

        return Math.max(queryLazy(idx * 2 + 1, low, mid, left, right),
                queryLazy(idx * 2 + 2, mid + 1, high, left, right));

    }

    private void pushDown(int rootIdx) {
        if (mLazy[rootIdx] != 0) {

            final int leftChild = (rootIdx << 1) + 1;
            final int rightChild = (rootIdx << 1) + 2;
            mLazy[leftChild] = Math.max(mLazy[rootIdx], mLazy[leftChild]);
            mLazy[rightChild] = Math.max(mLazy[rootIdx], mLazy[rightChild]);

            mTree[leftChild] = Math.max(mLazy[leftChild], mTree[leftChild]);
            mTree[rightChild] = Math.max(mTree[rightChild], mLazy[rightChild]);

            mLazy[rootIdx] = 0;
        }
    }
}
