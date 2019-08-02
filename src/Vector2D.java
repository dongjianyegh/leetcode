import java.util.Iterator;
import java.util.List;

public class Vector2D implements Iterator<Integer> {

    private final Iterator<List<Integer>> mRow;
    private Iterator<Integer> mColumn;

    public Vector2D(List<List<Integer>> vec2d) {
        // Initialize your data structure here
        mRow = vec2d == null ? null : vec2d.iterator();
        if (mRow != null && mRow.hasNext()) {
            mColumn = mRow.next().iterator();
        }
    }

    @Override
    public Integer next() {
        // Write your code here
        return mColumn.next();
    }

    @Override
    public boolean hasNext() {
        // Write your code here
        if (mRow == null || mColumn == null) {
            return false;
        }

        do {
            if (mColumn.hasNext()) {
                return true;
            }
            if (!mRow.hasNext()) {
                return false;
            }

            mColumn = mRow.next().iterator();

        } while (true);
    }

    @Override
    public void remove() {
        mColumn.remove();
    }
}
