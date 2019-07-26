import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class WordFilter {


    private class TrieNode {
        public TrieNode[] next;
        List<Integer> weights;
        public TrieNode() {
        }

        public void addWeight(int weight) {
            if (weights == null) {
                weights = new ArrayList<>();
            }
            weights.add(weight);
        }

    }

    private final TrieNode mPrefixRoot;
    private final TrieNode mSuffixRoot;

    public WordFilter(String[] words) {
        mPrefixRoot = new TrieNode();
        mSuffixRoot = new TrieNode();

        int i = 0;
        for (String word : words) {
            addWord(mPrefixRoot, word, 0, i, false);
            addWord(mSuffixRoot, word, word.length() - 1, i, true);
            i++;
        }
    }

    public void addWord(TrieNode root, CharSequence word, int idx, int weight, boolean reverse) {
        if ((!reverse && idx >= word.length()) || (reverse && idx < 0)) {
            return;
        }

        root.addWeight(weight);
        final char c = word.charAt(idx);
        if (root.next == null) {
            root.next = new TrieNode[26];
        }

        if (root.next[c-'a'] == null) {
            root.next[c - 'a'] = new TrieNode();
        }

        if (!reverse) {
            if (idx == word.length() - 1) {
                root.next[c - 'a'].addWeight(weight);
            } else {
                addWord(root.next[c - 'a'], word, idx + 1, weight, reverse);
            }
        } else {
            if (idx == 0) {
                root.next[c - 'a'].addWeight(weight);
            } else {
                addWord(root.next[c - 'a'], word, idx - 1, weight, reverse);
            }
        }
    }

    public List<Integer> searchPrefix(TrieNode root, CharSequence prefix, int idx, boolean reverse) {
        if (prefix.length() <= 0) {
            return root.weights;
        }

        final char c = prefix.charAt(idx);
        final TrieNode node = root.next[c - 'a'];

        if (node == null) {
            return null;
        } else if (!reverse) {
            if (idx == prefix.length() - 1) {
                return node.weights;
            } else {
                return searchPrefix(node, prefix, idx + 1, false);
            }
        } else {
            if (idx == 0) {
                return node.weights;
            } else {
                return searchPrefix(node, prefix, idx - 1, true);
            }
        }
    }

    public int f(String prefix, String suffix) {
        List<Integer> prefixWeights = searchPrefix(mPrefixRoot, prefix, 0, false);
        List<Integer> suffixWeights = searchPrefix(mSuffixRoot, suffix, suffix.length() - 1, true);

        if (prefixWeights == null || suffixWeights == null
                || prefixWeights.isEmpty() || suffixWeights.isEmpty()) {
            return -1;
        }

        int startPre = prefixWeights.size() - 1;
        int startSuf = suffixWeights.size() - 1;

        while (startPre >= 0 && startSuf >= 0) {
            final int l = prefixWeights.get(startPre);
            final int m = suffixWeights.get(startSuf);
            if (prefixWeights.get(startPre).equals(suffixWeights.get(startSuf))) {
                return prefixWeights.get(startPre);
            } else if (prefixWeights.get(startPre) > suffixWeights.get(startSuf)) {
                startPre--;
            } else {
                startSuf--;
            }
        }

        return -1;
    }
}
