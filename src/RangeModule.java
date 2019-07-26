public class RangeModule {

    private final SegmentTreeNode mRoot;
    public RangeModule() {
        mRoot = new SegmentTreeNode(0, 1000000000 - 1);
    }

    public void addRange(int left, int right) {
        mRoot.update(left, right - 1, 1);
    }

    public boolean queryRange(int left, int right) {
        return right - left == mRoot.query(left, right - 1);
    }

    public void removeRange(int left, int right) {
        mRoot.update(left, right - 1, 0);
    }

    private static class SegmentTreeNode {

        private final int mStart;
        private final int mEnd;

        private SegmentTreeNode mLeft;
        private SegmentTreeNode mRight;

        private int mValue;
        private boolean mLazy;
        private int mLazyValue;

        private SegmentTreeNode(int start, int end) {
            mStart = start;
            mEnd = end;

            mValue = 0;
            mLazy = false;
            mLazyValue = 0;

            mLeft = null;
            mRight = null;
        }

        private void update(int L, int R, int value) {
            if (L > mEnd || R < mStart) {
                return;
            }

            if (L <= mStart && R >= mEnd) {
                mValue = (mEnd - mStart + 1) * value;
                mLazy = true;
                mLazyValue = value;

                return;
            }

            int mid = mStart + (mEnd - mStart) / 2;

            if (mLeft == null) {
                mLeft = new SegmentTreeNode(mStart, mid);
            }

            if (mRight == null) {
                mRight = new SegmentTreeNode(mid + 1, mEnd);
            }

            pushDown();

            mLeft.update(L, R, value);
            mRight.update(L, R, value);

            mValue = mLeft.mValue + mRight.mValue;
        }

        private void pushDown() {
            if (mLazy) {
                mLeft.mLazy = true;
                mLeft.mLazyValue = mLazyValue;

                mLeft.mValue = (mLeft.mEnd - mLeft.mStart + 1) * mLazyValue;

                mRight.mLazy = true;
                mRight.mLazyValue = mLazyValue;

                mRight.mValue = (mRight.mEnd - mRight.mStart + 1) * mLazyValue;

                mLazy = false;
            }
        }

        private int query(int L, int R) {
            if (L > mEnd || R < mStart) {
                return 0;
            }

            if (L <= mStart && R >= mEnd) {
                return mValue;
            }

            int mid = mStart + (mEnd - mStart) / 2;

            if (mLeft == null) {
                mLeft = new SegmentTreeNode(mStart, mid);
            }

            if (mRight == null) {
                mRight = new SegmentTreeNode(mid + 1, mEnd);
            }

            pushDown();

            return mLeft.query(L, R) + mRight.query(L, R);
        }
    }
}
