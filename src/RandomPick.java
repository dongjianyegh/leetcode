import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class RandomPick {

    private class Interval {
        int start;
        int end;
        int cntWithPre;

        public Interval(int start, int end, int cntWithPre) {
            this.start = start;
            this.end = end;
            this.cntWithPre = cntWithPre;
        }
    }

    private final ArrayList<Interval> mIntervals;
    private final int mTotal;
    private final Random mRandom;

    public RandomPick(int N, int[] blacklist) {
        mRandom = new Random();

        if (blacklist == null || blacklist.length <= 0) {
            mIntervals = new ArrayList<>(1);
            mIntervals.add(new Interval(0, N - 1, N));
            mTotal = N;
        } else {
            mIntervals = new ArrayList<>();
            Arrays.sort(blacklist);

            int start = -1;
            for (int i = 0; i < blacklist.length + 1; ++i) {
                final int end = i < blacklist.length ? blacklist[i] : N;
                if (end > start + 1) {
                    Interval interval;
                    if (mIntervals.isEmpty()) {
                        interval = new Interval(start + 1, end - 1, end - start - 1);
                    } else {
                        interval = new Interval(start + 1, end - 1, end - start - 1 + mIntervals.get(mIntervals.size() - 1).cntWithPre);
                    }

                    mIntervals.add(interval);
                }

                if (i < blacklist.length) {
                    start = blacklist[i];
                }

            }
            mTotal = mIntervals.get(mIntervals.size() - 1).cntWithPre;
        }
    }

    public int pick() {
        int find = mRandom.nextInt(mTotal) + 1;

        int left = 0;
        int right = mIntervals.size() - 1;

        int idx = -1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            Interval interval = mIntervals.get(mid);

            if (interval.cntWithPre == find) {
                idx = mid;
                break;
            } else if (interval.cntWithPre < find) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        if (idx == -1) {
            idx = left;
        }

        return mIntervals.get(idx).end - (mIntervals.get(idx).cntWithPre - find);
    }
}
