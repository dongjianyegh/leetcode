import java.util.HashMap;
import java.util.Map;

public class LRUCache {
	static class Node {
		public int value;
		public int key;
		public Node next;
		public Node pre;
		
		public Node(int key, int value) {
			this.value = value;
			this.key = key;
		}
	}
	
	
	private final int mCapacity;
	private final Node mHead;
	private Node mTail;
	private final Map<Integer, Node> mMap;
	private int mSize;
	public LRUCache(int capacity) {
        mCapacity = capacity;
        mHead = new Node(-1, -1);
        mMap = new HashMap<>();
        mTail = mHead;
    }
	
	private void insertToHead(Node node) {
		if (node == null) {
			return;
		}
		
		if (mHead == mTail) {
			// 为空
			mHead.next = node;
			node.pre = mHead;
			node.next = null;
			mTail = node;
		} else {
			// 不为空，至少有一个或者两个
			Node next = mHead.next;
			
			node.next = next;
			node.pre = mHead;
			mHead.next = node;
			next.pre = node;
		}
	}
	
	private Node deleteNode(Node delete) {
		if (delete == null) {
			return null;
		}
		
		if (mHead == mTail) {
			return null;
		}
		
		if (mTail == delete) {
			
			// 正好在队尾
			Node pre = mTail.pre;
			pre.next = null;
			mTail = pre;
			
		} else {
			
			// 在中间
			Node pre = delete.pre;
			Node next = delete.next;
			
			if (pre != null)
				pre.next = next;
			if (next != null)
				next.pre = pre;
		}
		
		// gc free
		delete.next = null;
		delete.pre = null;
		
		return delete;
	}
    
    public int get(int key) {
        if (mMap.containsKey(key)) {
        	Node node = mMap.get(key);
        	deleteNode(node);
        	insertToHead(node);
        	// 将key插入到最前面
        	return node.value;
        } else {
        	return -1;
        }
    }
    
    public void put(int key, int value) {
        if (mMap.containsKey(key)) {
        	// 更改node的值，然后插入到最前面
        	Node node = mMap.get(key);
        	node.value = value;
        	
        	deleteNode(node);
        	insertToHead(node);
        } else {
        	if (mSize >= mCapacity) {
        		mMap.remove(mTail.key);
        		deleteNode(mTail);
        		mSize--;
        	}
        	
        	Node node = new Node(key, value);
        	
        	// 将node插入到最前面
        	insertToHead(node);
        	
        	mMap.put(key, node);
        	
        	mSize++;
        }
    }

	public String reverseWords(String s) {
		if (s == null || s.length() <= 0) {
			return "";
		}

		int end = -1;

		StringBuilder result = new StringBuilder();

		for (int i = s.length() - 1; i >= -1; --i) {
			if (i == -1 || s.charAt(i) == ' ') {
				if (end != -1) {
					result.append(s.substring(i+1, end + 1)).append(" ");
					end = -1;
				}
			} else if (end == -1) {
				end = i;
			}
		}

		if (result.length() > 0) {
			return result.deleteCharAt(result.length() - 1).toString();
		} else {
			return result.toString();
		}
	}
}
