

public class WordDictionary {
	
	private final TrieNode mRoot;
	/** Initialize your data structure here. */
    public WordDictionary() {
    		mRoot = new TrieNode();
    }
    
    /** Adds a word into the data structure. */
    public void addWord(String word) {
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
    
    /** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
    public boolean search(String word) {
    		TrieNode root = mRoot;
		if (root == null) return false;
		
		return search(mRoot, word, 0);
    }
    
    private boolean search(TrieNode root, CharSequence word, int start) {
    		if (start >= word.length())
    			return false;
    		
    		final char c = word.charAt(start);
    		if (c == '.') {
    			for (int i = 0; i < root.mNext.length; ++i) {
    				if (root.mNext[i] == null)
    					continue;
    				if (start == word.length() - 1 && root.mNext[i].mWord) {
    					return true;
    				}
    				
    				if (search(root.mNext[i], word, start + 1))
    					return true;
    			}
    			
    			return false;
    		} else {
    			TrieNode cur = root.mNext[c-'a'];
    			if (cur == null)
    				return false;
    			if (start == word.length() - 1)
    				return cur.mWord;
    			return search(cur, word, start + 1);
    		}
    }
    
    private class TrieNode {
		boolean mWord;
		TrieNode[] mNext = new TrieNode[26];
    }
}
