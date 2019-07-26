import java.util.HashMap;

public class TopVotedCandidate {

    private final HashMap<Integer, Integer> mMap;

    private final int[] mResult;
    private final int[] mTimes;
    public TopVotedCandidate(int[] persons, int[] times) {
        mMap = new HashMap<>();
        mResult = new int[times.length];
        mTimes = times;

        int lastMaxPerson = -1;
        int lastMaxVote = 0;
        for (int i = 0; i < persons.length; ++i) {
            if (i == 0) {
                mMap.put(persons[0], 1);
                lastMaxPerson = persons[0];
                lastMaxVote = 1;
            } else if (persons[i] == lastMaxPerson){
                mMap.put(lastMaxPerson, ++lastMaxVote);
            } else {
                final int newVote = mMap.getOrDefault(persons[i], 0) + 1;
                if (newVote >= lastMaxVote) {
                    lastMaxPerson = persons[i];
                    lastMaxVote = newVote;
                }
                mMap.put(persons[i], newVote);
            }

            mResult[i] = lastMaxPerson;
        }
    }

    public int q(int t) {
        // 二分查找，latest <= t

        if (t >= mTimes[mTimes.length - 1]) {
            return mResult[mTimes.length - 1];
        }

        int left = 0;
        int right = mTimes.length - 1;

        while (left <= right) {
            int middle = left + (right - left) / 2;
            if (mTimes[middle] == t) {
                return mResult[middle];
            } else if (mTimes[middle] < t) {
                left = middle + 1;
            } else {
                right = middle - 1;
            }
        }

        return mResult[right];
    }

}
