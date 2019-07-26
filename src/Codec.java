import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class Codec {
	private static final String EMPTY = "null";
	private static final String SPLIT = ",";
	// Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        // 先序
//    		if (root == null)
//    			return EMPTY;
//    		
//    		return String.valueOf(root.val) + SPLIT + serialize(root.left) + SPLIT + serialize(root.right);
    	
    		// 中序
//    		if (root == null)
//    			return EMPTY;
//    		
//    		return serialize(root.left) + SPLIT + String.valueOf(root.val) + SPLIT + serialize(root.right);
    	
    		// 后序
    		if (root == null)
    			return EMPTY;
    		return serialize(root.left) + SPLIT + serialize(root.right)  + SPLIT + String.valueOf(root.val);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        List<String> list = new LinkedList<>();
        list.addAll(Arrays.asList(data.split(SPLIT)));
        return deserialize(list);
    }
    
//    private TreeNode deserialize(List<String> preorder) {
//    		if (preorder.isEmpty())
//    			return null;
//    		
//    		String first = preorder.remove(0);
//    		if (first.equals(EMPTY)) {
//    			return null;
//    		}
//    		
//    		TreeNode root = new TreeNode(Integer.valueOf(first));
//    		
//    		root.left = deserialize(preorder);
//    		root.right = deserialize(preorder);
//    		
//    		return root;
//    }
    
//    private TreeNode deserialize(List<String> inorder) {
//    		if (inorder.isEmpty())
//    			return null;
//    		
//    		TreeNode left = deserialize(inorder);
//    		
//    		String r = inorder.remove(0);
//    		if (EMPTY.equals(r))
//    			return null;
//    		
//    		TreeNode root = new TreeNode(Integer.valueOf(r));
//    		
//    		TreeNode right = deserialize(inorder);
//    		
//    		root.left = left;
//    		root.right = right;
//    		
//    		return root;
//    }
     
    private TreeNode deserialize(List<String> postorder) {
    		if (postorder.isEmpty())
    			return null;
    		
    		String laString = postorder.remove(postorder.size() - 1);
    		if (EMPTY.equals(laString))
    			return null;
    		
    		TreeNode root = new TreeNode(Integer.valueOf(laString));
    		
    		root.right = deserialize(postorder);
    		root.left = deserialize(postorder);
    		
    		return root;
    }
}
