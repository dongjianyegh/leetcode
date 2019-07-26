
public class MagicDictionary {
	
	private final TrieNode mRoot = new TrieNode();
	
	/** Initialize your data structure here. */
    public MagicDictionary() {
        
    }
    
    /** Build a dictionary through a list of words */
    public void buildDict(String[] dict) {
    		if (dict == null)
    			return;
    		
        for (String word : dict) {
        		addWord(word);
        }
    }
    
    /** Returns if there is any word in the trie that equals to the given word after modifying exactly one character */
    public boolean search(String word) {
        return search(word, 0, false, mRoot);
    }
    
    private boolean search(CharSequence word, int start, boolean modify, TrieNode root) {
    		final char c = word.charAt(start);
    		final int idx = c - 'a';
    		
    		if (modify) {
    			if (root.mNext[idx] == null) {
    				return false;
    			}
    			if (start == word.length() - 1) {
    				return root.mNext[idx].mWord;
    			}
    			
    			return search(word, start + 1, modify, root.mNext[idx]);
    		} else {
    			boolean result = false;
    			for (int i = 0; i < root.mNext.length; ++i) {
    				
    				TrieNode next = root.mNext[i];
    				if (next == null) continue;
    				
    				// 不改变
    				if (idx == i) {
    					if (start != word.length() - 1) {
    						result = search(word, start + 1, modify, next);
    					}
    				} else {
    					if (start == word.length() - 1) {
    						return next.mWord;
    					}
    					
    					result = search(word, start + 1, true, next);
    				}
    				
    				if (result) 
    					return true;
    			}
    			
    			return result;
    		}
    		
    		
    }
    
    private class TrieNode {
    		private boolean mWord = false;
    		private TrieNode[] mNext = new TrieNode[26];
    }
    
    private void addWord(String word) {
    		if (!word.isEmpty()) {
    			addWord(word, 0, mRoot);
    		}
    }
    
    private void addWord(CharSequence word, int start, TrieNode root) {
    		if (start >= word.length()) {
    			root.mWord = true;
    			return;
    		}
    		final char c = word.charAt(start);
    		final int idx = c - 'a';
    		if (root.mNext[idx] == null) {
    			root.mNext[idx] = new TrieNode();
    		}
    		
    		addWord(word, start + 1, root.mNext[idx]);
    }
}
