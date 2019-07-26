import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

public class LFUCache {
	private final int mCapacity;

	// <Key, Node>
	private final Map<Integer, Node> mMapCache;
	// <Frequency, LinkedList>
	private final Map<Integer, LinkedList> mMapContent;
	// <Frequency>
	private final Set<Integer> mSetFrequency;

	private int mSize;

	public LFUCache(int capacity) {
		mCapacity = capacity;

		mMapCache = new HashMap<>();
		mMapContent = new HashMap<>();

		mSetFrequency = new TreeSet<>();
	}

	public int get(int key) {
		Node node = mMapCache.get(key);
		if (node == null) {
			return -1;
		}

		// 更新node所在位置
		deleteNode(node);

		node.frequency++;

		// 插入头部
		insertNode(node);

		return node.value;
	}

	public void put(int key, int value) {
		Node node = mMapCache.get(key);
		if (node != null) {
			deleteNode(node);
			node.frequency++;
			node.value = value;
			insertNode(node);

			return;
		}

		node = new Node(key, value);
		node.frequency = 1;

		if (mSize >= mCapacity) {
			// 需要知道删除哪个map里面的值，但是怎么能迅速找到这个呢
			int minFrequency = 0;
			for (int frequency : mSetFrequency) {
				minFrequency = frequency;
				break;
			}

			LinkedList list = mMapContent.get(minFrequency);
			if (list != null) {
				Node tail = list.deleteTail();
				if (tail != null) {
					mMapCache.remove(tail.key);

					if (list.isEmpty()) {
						mSetFrequency.remove(tail.frequency);
						mMapContent.remove(tail.frequency);
					}

					mSize--;
				}

			}
		}

		if (mSize < mCapacity)
			insertNode(node);

	}

	private void insertNode(Node insert) {
		if (insert == null) {
			return;
		}

		LinkedList list = mMapContent.get(insert.frequency);
		if (list == null) {
			list = new LinkedList();
			mMapContent.put(insert.frequency, list);
			mSetFrequency.add(insert.frequency);
		}

		list.addHead(insert);

		mMapCache.put(insert.key, insert);

		mSize++;
	}

	private void deleteNode(Node delete) {
		if (delete == null) {
			return;
		}

		int frequency = delete.frequency;

		LinkedList list = mMapContent.get(frequency);
		if (list == null || list.isEmpty()) {
			return;
		}

		list.deleteNode(delete);

		mMapCache.remove(delete.key);

		if (list.isEmpty()) {
			mMapContent.remove(frequency);
			mSetFrequency.remove(frequency);
		}

		mSize--;
	}

	static class Node {
		public int key;
		public int value;
		public int frequency;
		Node next;
		Node pre;

		public Node(int key, int value) {
			this.key = key;
			this.value = value;
		}
	}

	static class LinkedList {

		public final Node mHead;
		public final Node mTail;

		public LinkedList() {
			mHead = new Node(-1, -1);
			mTail = new Node(-1, -1);

			mHead.next = mTail;
			mTail.pre = mHead;
		}

		public boolean isEmpty() {
			return mHead.next == mTail;
		}

		public void addHead(Node node) {
			if (node == null) {
				return;
			}

			Node next = mHead.next;

			node.pre = mHead;
			node.next = next;

			mHead.next = node;
			next.pre = node;
		}

		public Node deleteTail() {
			if (isEmpty()) {
				return null;
			}

			Node pre = mTail.pre;
			if (pre == null) {
				return null;
			}

			Node prepre = pre.pre;
			if (prepre != null) {
				prepre.next = mTail;
				mTail.pre = prepre;
			}

			pre.next = null;
			pre.pre = null;

			return pre;
		}

		public Node deleteNode(Node delete) {
			if (delete == null) {
				return null;
			}

			if (isEmpty()) {
				return delete;
			}

			Node pre = delete.pre;
			Node next = delete.next;
			if (pre == null || next == null) {
				return delete;
			}

			pre.next = next;
			next.pre = pre;

			delete.pre = null;
			delete.next = null;

			return delete;
		}
	}

}
