

public class Trie {
	
	private final TrieNode mRoot;
	/** Initialize your data structure here. */
    public Trie() {
        mRoot = new TrieNode();
    }
    
    /** Inserts a word into the trie. */
    public void insert(String word) {
		TrieNode root = mRoot;
		if (root == null) return;

		for (int i = 0; i < word.length(); ++i) {
			int idx = word.charAt(i) - 'a';
			TrieNode cur = root.mNext[idx];
			if (cur == null) {
				cur = new TrieNode();
				root.mNext[idx] = cur;
			}

			root = cur;
			if (i == word.length() - 1) {
				cur.mWord = true;
			}
		}
    }
    
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
    		TrieNode root = mRoot;
    		if (root == null) return false;
    		
    		for (int i = 0; i < word.length(); ++i) {
			int idx = word.charAt(i) - 'a';
			TrieNode node = root.mNext[idx];
			if (node == null)
				return false;
			else if (i == word.length() - 1)
				return node.mWord;
			else
				root = node;
		}
		
		return false;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
    		TrieNode root = mRoot;
		if (root == null) return false;
		if (prefix == null || prefix.length() <= 0) return false;
		
		for (int i = 0; i < prefix.length(); ++i) {
			int idx = prefix.charAt(i) - 'a';
			TrieNode node = root.mNext[idx];
			if (node == null)
				return false;
			else
				root = node;
		}
	
		return true;
    }
    
    private class TrieNode {
    		boolean mWord;
    		TrieNode[] mNext = new TrieNode[26];
    }
}
