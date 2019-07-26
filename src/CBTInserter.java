import java.util.LinkedList;
import java.util.Queue;

public class CBTInserter {

    private TreeNode mRoot;
    private TreeNode mLastParent;
    private final Queue<TreeNode> mQueue;
    private boolean mInsertLeft;

    public CBTInserter(TreeNode root) {
        mQueue = new LinkedList<>();
        if (root != null) {
            mRoot = root;
            mQueue.add(mRoot);

            while (!mQueue.isEmpty()) {
                TreeNode head = mQueue.poll();

                if (head.left != null && head.right != null) {
                    mQueue.add(head.left);
                    mQueue.add(head.right);
                } else if (head.left == null) {
                    mLastParent = head;
                    mInsertLeft = true;
                    break;
                } else {
                    mQueue.add(head.left);
                    mLastParent = head;
                    mInsertLeft = false;
                    break;
                }
            }
        }
    }

    public int insert(int v) {
        TreeNode inserted = new TreeNode(v);

        if (mLastParent == null) {
            mLastParent = mQueue.poll();
        }

        final int result = mLastParent.val;

        if (mInsertLeft) {
            mLastParent.left = inserted;
            mQueue.add(inserted);
            mInsertLeft = false;
        } else {
            mLastParent.right = inserted;
            mQueue.add(inserted);
            mInsertLeft = true;
            mLastParent = null;
        }

        return result;
    }

    public TreeNode get_root() {
        return mRoot;
    }
}
