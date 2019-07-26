import java.util.*;

public class RandomizedCollection {

    private final Map<Integer, Set<Integer>> mMap = new HashMap<>();
    private final ArrayList<Integer> mArray = new ArrayList<>();

    private int mSize = 0;

    /** Initialize your data structure here. */
    public RandomizedCollection() {

    }

    /** Inserts a value to the collection. Returns true if the collection did not already contain the specified element. */
    public boolean insert(int val) {
        mArray.add(mSize, val);
        Set<Integer> set = mMap.get(val);

        final boolean hasOld;
        if (set == null) {
            hasOld = false;
            set = new HashSet<>();
            mMap.put(val, set);
        } else {
            hasOld = true;
        }

        set.add(mSize);

        mSize++;

        return !hasOld;
    }

    /** Removes a value from the collection. Returns true if the collection contained the specified element. */
    public boolean remove(int val) {
        final Set<Integer> set = mMap.get(val);
        if (set == null || mSize <= 0) {
            return false;
        }

        final int lastValue = mArray.get(mSize - 1);
        if (lastValue == val) {
            mArray.remove(mSize - 1);
            set.remove(mSize - 1);

            if (set.isEmpty()) {
                mMap.remove(val);
            }

            mSize--;
            return true;
        }

        final Iterator<Integer> iterator = set.iterator();
        final int valOneIndex = iterator.next();
        iterator.remove();

        if (!iterator.hasNext()) {
            mMap.remove(val);
        }

        mArray.set(valOneIndex, lastValue);

        final Set<Integer> lastSet = mMap.get(lastValue);
        lastSet.remove(mSize - 1);
        lastSet.add(valOneIndex);

        mArray.remove(mSize - 1);

        mSize--;

        return true;
    }

    /** Get a random element from the collection. */
    public int getRandom() {
        return mArray.get(new Random().nextInt(mSize));
    }
}
