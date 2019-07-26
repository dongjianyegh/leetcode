import java.util.LinkedList;
import java.util.List;

public class StreamChecker {
    TrieNode root;
    List<Character> dq;
    int max;

    public StreamChecker(String[] words) {
        root = new TrieNode();
        dq = new LinkedList<>();
        max = 0;
        for (String word : words) {
            insert(root, word.toCharArray());
            max = Math.max(max, word.length());
        }
    }

    public boolean query(char letter) {
        dq.add(letter);
        if (dq.size() > max) {
            dq.remove(0);
        }
        TrieNode cur = root;
        int i = dq.size() - 1;
        while (i >= 0) {
            int idx = dq.get(i--) - 'a';
            if (cur.kids[idx] == null) {
                return false;
            }
            cur = cur.kids[idx];
            if (cur.isWord) {
                return true;
            }
        }
        return false;
    }

    private void insert(TrieNode root, char[] c) {
        for (int i = c.length - 1; i >= 0; --i) {
            int idx = c[i] - 'a';
            if (root.kids[idx] == null) {
                root.kids[idx] = new TrieNode();
            }
            root = root.kids[idx];
        }
        root.isWord = true;
    }

    private class TrieNode {
        TrieNode[] kids;
        boolean isWord;

        TrieNode() {
            kids = new TrieNode[26];
        }
    }
}
