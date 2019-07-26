import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

public class AllOne {
	
	private final Map<String, OneNode> mMapNodes;
	private final Map<Integer, HeadNode> mMapValues;
	// 从大到小排序
	private final DoubleLinkedHead mListHeads;
	
	/** Initialize your data structure here. */
    public AllOne() {
        mMapNodes = new HashMap<>();
        mListHeads = new DoubleLinkedHead();
        mMapValues = new HashMap<>();
    }
    
    /** Inserts a new key <Key> with value 1. Or increments an existing key by 1. */
    public void inc(String key) {
        OneNode oneNode = mMapNodes.get(key);
        HeadNode after = null;
        if (oneNode == null) {
        		oneNode = new OneNode(key);
        		mMapNodes.put(key, oneNode);
        		
        } else {
        		int value = oneNode.mValue;
        		after = mMapValues.get(value);
        		after.mHead.deleteNode(oneNode);
        		oneNode.mValue++;
        }
        
        HeadNode insert = mMapValues.get(oneNode.mValue);
		if (insert == null) {
			insert = new HeadNode(new DoubleLinkedNode());
			mMapValues.put(oneNode.mValue, insert);
			mListHeads.insertAfter(after, insert);
		}
		
		insert.mHead.addHead(oneNode);
		if (after != null && after.mHead.isEmpty()) {
			mMapValues.remove(oneNode.mValue - 1);
			mListHeads.deleteNode(after);
		}
        
    }
    
    /** Decrements an existing key by 1. If Key's value is 1, remove it from the data structure. */
    public void dec(String key) {
        OneNode node = mMapNodes.get(key);
        if (node == null)
        		return;
        
        final int value = node.mValue;
       
        HeadNode before = mMapValues.get(value);
        before.mHead.deleteNode(node);
        
        if (value == 1) {
    			mMapNodes.remove(key);
        } else {
        		node.mValue--;
        		HeadNode insert = mMapValues.get(node.mValue);
        		if (insert == null) {
        			insert = new HeadNode(new DoubleLinkedNode());
        			mMapValues.put(node.mValue, insert);
        			mListHeads.insertBefore(before, insert);
        		}
        		insert.mHead.addHead(node);
        }
        
        if (before.mHead.isEmpty()) {
			mListHeads.deleteNode(before);
			mMapValues.remove(value);
		}
    }
    
    /** Returns one of the keys with maximal value. */
    public String getMaxKey() {
    		HeadNode last = mListHeads.getLast();
    		if (last == null)
    			return "";
    		else 
    			return last.mHead.getFirst().mKey;
    }
    
    /** Returns one of the keys with Minimal value. */
    public String getMinKey() {
    		HeadNode last = mListHeads.getFirst();
		if (last == null)
			return "";
		else 
			return last.mHead.getFirst().mKey;
    }
    
    private class HeadNode {
    		final DoubleLinkedNode mHead;
    		HeadNode mPre;
    		HeadNode mNext;
    		
    		public HeadNode(DoubleLinkedNode head) {
    			mHead = head;
    		}
    		
    		public HeadNode() {
    			mHead = null;
    		}
    }
    
    private class DoubleLinkedHead {
    		final HeadNode mHead;
    		final HeadNode mTail;
    		
    		public DoubleLinkedHead() {
    			mHead = new HeadNode();
    			mTail = new HeadNode();
    			mHead.mNext = mTail;
    			mTail.mPre = mHead;
    		}
    		
    		public void deleteNode(HeadNode node) {
    			HeadNode pre = node.mPre;
    			pre.mNext = node.mNext;
    			node.mNext.mPre = pre;
    		}
    		
    		public void insertAfter(HeadNode after, HeadNode insert) {
    			after = after == null ? mHead : after;
    			
    			HeadNode first = after.mNext;
				
    			after.mNext = insert;
			insert.mPre = after;
			
			insert.mNext = first;
			first.mPre = insert;
    		}
    		
    		public void insertBefore(HeadNode before, HeadNode insert) {
    			
    			HeadNode pre = before.mPre;
    			
    			pre.mNext = insert;
    			insert.mPre = pre;
    			
    			insert.mNext = before;
    			before.mPre = insert;
    		}
    		
    		public boolean isEmpty() {
    			return mHead.mNext == mTail;
    		}
    		
    		public HeadNode getFirst() {
    			if (isEmpty())
    				return null;
    			else
    				return mHead.mNext;
    		}
    		
    		public HeadNode getLast() {
    			if (isEmpty())
    				return null;
    			else 
    				return mTail.mPre;
    		}
    		
    }
    
    private class OneNode {
    		private final String mKey;
    		int mValue = 1;
    		OneNode mPre;
    		OneNode mNext;
    		public OneNode(String key) {
    			mKey = key;
    		}
    }
    
    private class DoubleLinkedNode {
    		private OneNode mHead;
    		private OneNode mTail;
    		
    		public DoubleLinkedNode() {
    			mHead = new OneNode(null);
    			mTail = new OneNode(null);
    			mHead.mNext = mTail;
    			mTail.mPre = mHead;
    		}
    		
    		public boolean isEmpty() {
    			return mHead.mNext == mTail;
    		}
    		
    		public OneNode deleteNode(OneNode node) {
    			if (node == null || isEmpty())
    				return null;
    			
    			OneNode pre = node.mPre;
    			pre.mNext = node.mNext;
    			node.mNext.mPre = pre;
    			
    			return node;
    		}
    		
    		public OneNode addHead(OneNode node) {
    			OneNode first = mHead.mNext;
    			
    			node.mNext = first;
    			first.mPre = node;
    			
    			mHead.mNext = node;
    			node.mPre = mHead;
    			
    			return node;
    		}
    		
    		public OneNode getFirst() {
    			if (isEmpty())
    				return null;
    			else
    				return mHead.mNext;
    		}
    }
    
    
}
