import java.util.Stack;

public class BSTIterator {

    private Stack<TreeNode> mStack = new Stack<>();

    private TreeNode mCur = null;
    public BSTIterator(TreeNode root) {
        while (root != null) {
            mStack.push(root);
            root = root.left;
        }
    }

    /** @return the next smallest number */
    public int next() {
        TreeNode next = mStack.pop();
        TreeNode right = next.right;
        while(right != null) {
            mStack.push(right);
            right = right.left;
        }

        return next.val;
    }

    /** @return whether we have a next smallest number */
    public boolean hasNext() {
        return !mStack.isEmpty();
    }
}
