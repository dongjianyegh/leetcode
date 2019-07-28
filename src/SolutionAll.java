import java.util.*;
import java.util.function.Function;

public class SolutionAll {


    public int[] twoSum(int[] nums, int target) {
        Integer[] idxes = new Integer[nums.length];
        for (int i = 0; i < nums.length; ++i) {
            idxes[i] = i;
        }

        Arrays.sort(idxes, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return nums[o1] - nums[o2];
            }
        });

        int[] result = new int[2];

        int i = 0;
        int j = nums.length - 1;

        while (i < j) {
            final int sum = nums[idxes[i]] + nums[idxes[j]];
            if (sum == target) {
                result[0] = Math.min(idxes[i], idxes[j]);
                result[1] = Math.max(idxes[i], idxes[j]);
                return result;
            } else if (sum < target) {
                i++;
            } else {
                j--;
            }
        }

        return null;
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {

        ListNode result = null;
        ListNode tail = null;

        ListNode temp1 = l1;
        ListNode temp2 = l2;

        int add = 0;
        while (temp1 != null && temp2 != null) {
            int sum = temp1.val + temp2.val + add;
            if (sum >= 10) {
                sum -= 10;
                add = 1;
            } else {
                add = 0;
            }

            if (result == null) {
                result = new ListNode(sum);
                tail = result;
            } else {
                tail.next = new ListNode(sum);
                tail = tail.next;
            }
            temp1 = temp1.next;
            temp2 = temp2.next;
        }

        while (temp1 != null) {
            int sum = add + temp1.val;
            if (sum >= 10) {
                sum -= 10;
                add = 1;
            } else {
                add = 0;
            }
            if (result == null) {
                result = new ListNode(sum);
                tail = result;
            } else {
                tail.next = new ListNode(sum);
                tail = tail.next;
            }
            temp1 = temp1.next;
        }

        while (temp2 != null) {
            int sum = add + temp2.val;
            if (sum >= 10) {
                sum -= 10;
                add = 1;
            } else {
                add = 0;
            }
            if (result == null) {
                result = new ListNode(sum);
                tail = result;
            } else {
                tail.next = new ListNode(sum);
                tail = tail.next;
            }
            temp2 = temp2.next;
        }

        if (add == 1) {
            tail.next = new ListNode(1);
        }

        return result;
    }

    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.length() <= 0) {
            return 0;
        }

        int[] hash = new int[26];

        int i = 0;
        int j = 0;

        int kinds = 0;
        int result = 0;
        while (j < s.length()) {
            final int idx = s.charAt(j) - 'a';
            if (hash[idx]++ == 0) {
                kinds++;
            }
            if (kinds == j - i + 1) {
                result = Math.max(result, kinds);
            }
            while (j - i + 1 > kinds && i < j) {
                if (--hash[s.charAt(i) - 'a'] == 0) {
                    kinds--;
                }
                i++;
            }

            j++;
        }

        return result;
    }

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        return 0f;
    }

    public String longestPalindrome(String s) {
        if (s == null || s.length() <= 0) {
            return "";
        }

        // 以中间的点向两边找
        int maxStart = 0;
        int maxEnd = 0;

        for (int i = 0; i < s.length(); ++i) {
            int j = 0;
            while (i - j >= 0 && i + j < s.length() && s.charAt(i-j) == s.charAt(i+j)) {
                j++;
            }
            if (2 * j - 1 > maxEnd - maxStart + 1) {
                maxStart = i - j + 1;
                maxEnd = i + j - 1;
            }
        }


        for (int i = 0; i + 1 < s.length(); ++i) {
            if (s.charAt(i) != s.charAt(i+1)) {
                continue;
            }

            int j = 0;
            while (i - j >= 0 && i + 1 + j < s.length() && s.charAt(i-j) == s.charAt(i+1+j)) {
                j++;
            }
            if (2 * j > maxEnd - maxStart + 1) {
                maxStart = i - j + 1;
                maxEnd = i + j;
            }
        }

        return s.substring(maxStart, maxEnd + 1);
    }

    public String convert(String s, int numRows) {
        if (numRows == 1) {
            return s;
        }

        StringBuilder builder = new StringBuilder();

        for (int i = 0; i < numRows; ++i) {
            boolean odd = true;
            int nextIdx = i;
            int preIdx = -1;
            while (nextIdx < s.length()) {
                if (preIdx != nextIdx) {
                    builder.append(s.charAt(nextIdx));
                }
                preIdx = nextIdx;
                nextIdx += odd ? (2 * numRows - 2 * (i + 1)) : 2 * i;
                odd = !odd;
            }
        }

        return builder.toString();
    }

    public int reverse(int x) {
        int result = 0;
        while (x != 0) {
            int left = x % 10;
            x /= 10;
            if (result > Integer.MAX_VALUE/10 || (result == Integer.MAX_VALUE / 10 && left > 7)) return 0;
            if (result < Integer.MIN_VALUE/10 || (result == Integer.MIN_VALUE / 10 && left < -8)) return 0;
            result = result * 10 + left;
        }

        return result;
    }

    public int myAtoi(String str) {
        if (str == null || str.length() <= 0) {
            return 0;
        }

        int result = 0;
        boolean hasNum = false;
        boolean trimSpace = true;
        boolean hasSign = false;
        boolean sign = true;
        for (int i = 0; i < str.length(); ++i) {
            final char c = str.charAt(i);
            if (c == ' ') {
                if (trimSpace) continue;
                else return sign ? result : -result;
            }

            trimSpace = false;
            if (c == '-' || c == '+') {
                if (!hasSign && !hasNum) {
                    hasSign = true;
                    sign = c == '+';
                } else {
                    return sign ? result : -result;
                }
            } else if (Character.isDigit(c)) {
                if (!hasSign) {
                    sign = true;
                }

                hasNum = true;
                if (!checkIsValid(result, c - '0', sign)) {
                    if (sign) {
                        return Integer.MAX_VALUE;
                    } else {
                        return Integer.MIN_VALUE;
                    }
                }
                result = result * 10 + c - '0';

            } else {
                return sign ? result : -result;
            }
        }

        return sign ? result : -result;
    }

    private boolean checkIsValid(int result, int left, boolean positive) {
        if (positive) {
            return !(result > Integer.MAX_VALUE/10 || (result == Integer.MAX_VALUE / 10 && left > 7));
        } else {
            result = -result;
            return !(result < Integer.MIN_VALUE/10 || (result == Integer.MIN_VALUE / 10 && left > 8));
        }
    }

    public boolean isPalindrome(int x) {
        if (x < 0 || (x % 10 == 0 && x != 0)) {
            return false;
        } else if (x == 0) {
            return true;
        }

        int revert = 0;
        while (x > revert) {
            revert = revert * 10 + x % 10;
            x /= 10;
        }

        return x == revert || revert / 10 == x;
    }

    public boolean isMatch(String s, String p) {
        boolean[][] dp = new boolean[p.length() + 1][s.length() + 1];

        dp[0][0] = true;

        for (int i = 1; i <= s.length(); i++) {
            dp[i][0] = dp[i-2][0] && p.charAt(i-1) == '*';
        }

        for (int i = 1; i <= p.length(); ++i) {
            for (int j = 1; j <= s.length(); ++j) {
                final char pc = p.charAt(i-1);
                final char sc = s.charAt(j - 1);
                if (pc != '*') {
                    dp[i][j] = isMatch(pc, sc) && dp[i-1][j-1];
                } else {
                    final char pprec = p.charAt(i - 2);
                    if (pprec == '.') {
                        dp[i][j] = dp[i-2][j] || dp[i][j-1];
                    } else if (isMatch(pprec, sc)){
                        dp[i][j] = dp[i-1][j] || dp[i][j-1];
                    } else {
                        dp[i][j] = dp[i-2][j];
                    }
                }
            }
        }

        return dp[p.length()][s.length()];
    }

    private boolean isMatch(char pc, char sc) {
        return pc == '.' || pc == sc;
    }

    public int maxArea(int[] height) {
        if (height == null || height.length <= 1) {
            return 0;
        }

        int maxArea = 0;

        int left = 0;
        int right = height.length - 1;
        while (left < right) {
            maxArea = Math.max(maxArea, (right - left) * Math.min(height[left], height[right]));
            if (height[left] > height[right]) {
                right--;
            } else {
                left++;
            }
        }

        return maxArea;
    }

    public String intToRoman(int num) {
        StringBuilder builder = new StringBuilder();
        int thousandQuotient = num / 1000;

        while (thousandQuotient-- > 0) {
            builder.append('M');
        }

        num %= 1000;

        int hunderdQuotient = num / 100;

        intToRoman(hunderdQuotient, new char[]{'C', 'D', 'M'}, builder);

        num %= 100;

        int tenQuotient = num / 10;

        intToRoman(tenQuotient, new char[]{'X', 'L', 'C'}, builder);

        num %= 10;

        int quotient = num;

        intToRoman(quotient, new char[]{'I', 'V', 'X'}, builder);

        return builder.toString();
    }

    private void intToRoman(int quotient, char[] romans, StringBuilder builder) {
        if (quotient <= 3) {
            while (quotient-- > 0) {
                builder.append(romans[0]);
            }
        } else if (quotient == 4) {
            builder.append(romans[0]).append(romans[1]);
        } else if (quotient == 5) {
            builder.append(romans[1]);
        } else if (quotient <= 8) {
            builder.append(romans[1]);
            while (quotient-- > 8) {
                builder.append(romans[0]);
            }
        } else {
            builder.append(romans[0]).append(romans[2]);
        }
    }

    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length <= 0) {
            return "";
        }

        StringBuilder builder = new StringBuilder();

        for (int index = 0; ; ++index) {

            if (index >= strs[0].length()) {
                break;
            } else {
                char c = strs[0].charAt(index);
                boolean isValid = true;
                for (int i = 1; i < strs.length; ++i) {
                    if (index >= strs[i].length() || strs[i].charAt(index) != c) {
                        isValid = false;
                        break;
                    }
                }

                if (!isValid) {
                    break;
                } else {
                    builder.append(c);
                }
            }
        }

        return builder.toString();
    }

    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();

        if (nums == null || nums.length < 3) {
            return result;
        }

        Arrays.sort(nums);

        for (int i = 0; i < nums.length - 2; ++i) {
            if (i == 0 || nums[i] != nums[i-1]) {
                final int toFind = -nums[i];

                int left = i + 1;
                int right = nums.length - 1;

                while (left < right) {
                    final int twoSum = nums[left] + nums[right];
                    if (twoSum == toFind) {
                        List<Integer> one = new ArrayList<>(3);
                        one.add(nums[i]);
                        one.add(nums[left]);
                        one.add(nums[right]);
                        result.add(one);

                        left++;
                        right--;

                        while (left < right && nums[left - 1] == nums[left]) {
                            left++;
                        }
                        while (left < right && nums[right + 1] == nums[right]) {
                            right--;
                        }
                    } else if (twoSum > toFind) {
                        right--;
                    } else {
                        left++;
                    }
                }
            }
        }

        return result;
     }

    public int threeSumClosest(int[] nums, int target) {
        if (nums == null || nums.length < 3) {
            return -1;
        }

        Arrays.sort(nums);

        int distance = Integer.MAX_VALUE;

        int result = 0;
        for (int i = 0; i < nums.length - 2; ++i) {
            final int toFind = target - nums[i];

            int left = i + 1;
            int right = nums.length - 1;


            while (left < right) {
                final int twoSum = nums[left] + nums[right];
                if (twoSum == toFind) {
                    return target;
                }

                if (Math.abs(twoSum - toFind) < distance) {
                    distance = Math.abs(twoSum - toFind);
                    result = nums[i] + twoSum;
                }

                if (twoSum > toFind) {
                    right--;
                } else {
                    left++;
                }
            }
        }

        return result;
    }

    public List<String> letterCombinations(String digits) {
        List<String> result = new LinkedList<>();
        letterCombinationsDfs(digits, 0, new LinkedList<>(), result);
        return result;
    }

    private final int[][] letterCombinationsRule = {
            {0, 2},
            {3, 5},
            {6, 8},
            {9, 11},
            {12, 14},
            {15, 18},
            {19, 21},
            {22, 25}
    };
    private void letterCombinationsDfs(String digits, int index, List<Character> one, List<String> result) {
        if (index >= digits.length()) {
            if (one.size() > 0) {
                StringBuilder builder = new StringBuilder();
                for (char c : one) {
                    builder.append(c);
                }
                result.add(builder.toString());

            }
            return;
        }

        final int digit = digits.charAt(index) - '0';

        for (int c = letterCombinationsRule[digit-2][0]; c <= letterCombinationsRule[digit-2][1]; ++c) {
            one.add((char) (c + 'a'));
            letterCombinationsDfs(digits, index + 1, one, result);
            one.remove(one.size() - 1);
        }
    }

    // 用两种方法，看一看哪个快
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> result = new ArrayList<>();

        if (nums == null || nums.length < 4) {
            return result;
        }

        Arrays.sort(nums);

        for (int i = 0; i < nums.length - 3; ++i) {
            if (i == 0 || nums[i] != nums[i-1]) {

                for (int j = i + 1; j < nums.length - 2; ++j) {
                    if (j == i + 1 || nums[j] != nums[j-1]) {
                        final int toFind = target - nums[j] - nums[i];

                        int left = j + 1;
                        int right = nums.length - 1;

                        while (left < right) {
                            final int twoSum = nums[left] + nums[right];
                            if (twoSum == toFind) {
                                result.add(Arrays.asList(nums[i], nums[j], nums[left], nums[right]));
                                left++;
                                right--;
                                while (left < right && nums[left] == nums[left-1]) {
                                    left++;
                                }
                                while (left < right && nums[right + 1] == nums[right]) {
                                    right--;
                                }
                            } else if (twoSum > toFind) {
                                right--;
                            } else {
                                left++;
                            }
                        }
                    }
                }
            }
        }

        return result;
    }

    private void fourSum(int[] nums, int target, int sum, int cnt, int k, int index, List<Integer> one, List<List<Integer>> result) {
        if (cnt == k) {
            if (sum == target) {
                result.add(new ArrayList<>(one));
            }

            return;
        }

        if (index >= nums.length) {
            return;
        }

        for (int i = index; i < nums.length; ++i) {
            one.add(nums[i]);
            fourSum(nums, target, sum + nums[i], cnt + 1, k, i + 1, one, result);

            while (i + 1 < nums.length && nums[i + 1] == nums[i]) {
                i++;
            }

            one.remove(nums[i]);
        }
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {

        if (head == null) {
            return head;
        }

        ListNode slow = head;
        ListNode fast = head;

        while (fast != null && n-- > 0) {
            fast = fast.next;
        }

        if (fast == null && n == 0) {
            return head.next;
        } else if (n > 0 && fast == null) {
            return head;
        }

        ListNode slowPre = null;

        while (fast != null) {
            fast = fast.next;
            slowPre = slow;
            slow = slow.next;
        }

        slowPre.next = slow.next;

        return head;
    }

    public boolean isValid(String s) {
        if (s == null || s.length() % 2 != 0) {
            return false;
        }

        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); ++i) {
            final char c = s.charAt(i);
            if (c == '(' || c == '{' || c == '[') {
                stack.push(c);
            } else if (c == ')'){
                if (stack.isEmpty() || stack.pop() != '(') {
                    return false;
                }
            } else if (c == ']'){
                if (stack.isEmpty() || stack.pop() != '[') {
                    return false;
                }
            } else {
                if (stack.isEmpty() || stack.pop() != '{') {
                    return false;
                }
            }
        }

        return stack.isEmpty();
    }

    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        } else if (l2 == null) {
            return l1;
        }

        ListNode temp1 = l1;
        ListNode temp2 = l2;

        ListNode head = new ListNode(-1);
        ListNode tail = head;
        while (temp1 != null && temp2 != null) {
            if (temp1.val < temp2.val) {
                tail.next = temp1;
                temp1 = temp1.next;
            } else {
                tail.next = temp2;
                temp2 = temp2.next;
            }
            tail = tail.next;
        }

        ListNode notNull = temp1 == null ? temp2 : temp1;
        while (notNull != null) {
            tail.next = notNull;
            notNull = notNull.next;
            tail = tail.next;
        }

        return head.next;
    }

    public List<String> generateParenthesis(int n) {
        List<String> result = new LinkedList<>();
        if (n == 0) {
            result.add("");
            return result;
        }

        for (int i = 0; i <= n - 1; ++i) {
            List<String> mids = generateParenthesis(i);
            List<String> afters = generateParenthesis(n - 1 - i);

            for (String mid : mids) {
                for (String after : afters) {
                    String one = "(" + mid + ")" + after;
                    result.add(one);
                }
            }
        }

        return result;
    }

    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length <= 0) {
            return null;
        }

        PriorityQueue<ListNode> minHeap = new PriorityQueue<>(new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                return o1.val - o2.val;
            }
        });

        for (ListNode node : lists) {
            if (node != null)
                minHeap.add(node);
        }

        ListNode head = null;
        ListNode tail = null;

        while (!minHeap.isEmpty()) {
            ListNode min = minHeap.poll();
            if (min.next != null) {
                minHeap.add(min.next);
            }

            if (head == null) {
                head = min;
                tail = min;
            } else {
                tail.next = min;
                tail = tail.next;
            }
        }
        if (tail != null)
            tail.next = null;

        return head;
    }

    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        ListNode newHead = new ListNode(-1);
        newHead.next = head;

        ListNode preTail = newHead;
        ListNode pair1 = newHead.next;

        while (pair1 != null && pair1.next != null) {
            final ListNode nextPair1 = pair1.next.next;
            pair1.next.next = pair1;
            preTail.next = pair1.next;
            pair1.next = nextPair1;
            preTail = pair1;
            pair1 = nextPair1;
        }

        return newHead.next;
    }

    public int removeDuplicates(int[] nums) {
        if (nums == null) return 0;
        else if (nums.length <= 1) {
            return nums.length;
        }

        int cnt = 1;
        int idx = 0;
        int pre = nums[0];

        for (int i = 1; i < nums.length; ++i) {
            if (pre != nums[i]) {
                nums[++idx] = nums[i];
                pre = nums[i];
                cnt++;
            }
        }

        return cnt;
    }

    public int removeElement(int[] nums, int val) {
        if (nums == null || nums.length <= 0) {
            return 0;
        }

        int left = 0;
        int right = nums.length - 1;

        int result = 0;
        while (left <= right) {
            while (left <= right && nums[left] != val) {
                result++;
                left++;
            }
            while (left <= right && nums[right] == val) {
                right--;
            }

            if (left < right) {
                int old = nums[left];
                nums[left] = nums[right];
                nums[right] = old;
                left++;
                right--;
                result++;
            }
        }

        return result;
    }

    public int divide(int dividend, int divisor) {
        if (dividend == 0) {
            return 0;
        }

        if (dividend == Integer.MIN_VALUE) {
            if (divisor == Integer.MIN_VALUE) {
                return 1;
            } else if (divisor == Integer.MAX_VALUE) {
                return -1;
            } else if (divisor == -1) {
                return Integer.MAX_VALUE;
            } else if (divisor == -2) {
                return dividePositive(Integer.MAX_VALUE, 2) + 1;
            } else if (divisor < 0) {
                return dividePositive(Integer.MAX_VALUE, -divisor);
            } else if (divisor == 1) {
                return Integer.MIN_VALUE;
            } else if (divisor == 2) {
                return -1 - dividePositive(Integer.MAX_VALUE, 2);
            } else {
                return -dividePositive(Integer.MAX_VALUE, divisor);
            }
        }

        if (divisor == Integer.MIN_VALUE) {
            return 0;
        }

        if (dividend > 0 && divisor > 0) {
            return dividePositive(dividend, divisor);
        } else if (dividend < 0 && divisor < 0) {
            return dividePositive(-dividend, -divisor);
        } else if (dividend > 0 && divisor < 0) {
            return -dividePositive(dividend, -divisor);
        } else {
            return -dividePositive(-dividend, divisor);
        }


    }

    private int dividePositive(int dividend, int divisor) {

        if (dividend == 0) {
            return 0;
        } else if (dividend == divisor) {
            return 1;
        } else if (dividend < divisor) {
            return 0;
        } else if (divisor == 1) {
            return dividend;
        } else if (divisor == 2) {
            return dividend >> 1;
        }

        int cnt = 0;
        int temp = divisor;

        do {
            int newTemp = temp << 1;
            if (newTemp > 0 && newTemp <= dividend) {
                cnt++;
                temp = newTemp;
            } else {
                break;
            }

        } while(true);


        return (1 << cnt) + dividePositive(dividend - temp, divisor);
    }


    public List<Integer> findSubstring(String s, String[] words) {
        List<Integer> result = new ArrayList<>();
        if (s.length() <= 0 || words.length <= 0) {
            return result;
        }
        Map<String, Integer> wordMap = new HashMap<>();
        int i = 0;
        int totalLength = 0;
        int[] wordsHash = new int[26];


        for (String word : words) {
            wordMap.put(word, wordMap.getOrDefault(word, 0) + 1);
            totalLength += word.length();

            for (int j = 0; j < word.length(); ++j) {
                wordsHash[word.charAt(j) - 'a']++;
            }
        }


        Map<String, Boolean> cache = new HashMap<>();

        int[] wordSlideHash = new int[26];
        for (i = 0; i <= s.length() - totalLength; ++i) {
            final String subString = s.substring(i, i + totalLength);
            if (i == 0) {
                for (int j = 0; j < subString.length(); ++j) {
                    wordSlideHash[subString.charAt(j) - 'a']++;
                }
            } else {
                wordSlideHash[s.charAt(i-1) - 'a']--;
                wordSlideHash[s.charAt(i + totalLength - 1) - 'a']++;
            }

            boolean hashEqual = true;
            for (int j = 0; j < 26; ++j) {
                if (wordsHash[j] != wordSlideHash[j]) {
                    hashEqual = false;
                    break;
                }
            }

            if (!hashEqual) {
                continue;
            }
            final Boolean cacheVal = cache.getOrDefault(subString, null);
            if (cacheVal != null) {
                if (cacheVal) {
                    result.add(i);
                }
            } else {
                if (findSubstringDfs(s, i, words[0].length(), words.length, 0, wordMap)) {
                    result.add(i);
                    cache.put(subString, true);
                } else {
                    cache.put(subString, false);
                }
            }

        }

        return result;

    }

    private boolean findSubstringDfs(String s, int index, int wordLength, int wordCnt, int cnt, Map<String, Integer> map) {
        if (cnt == wordCnt) {
            return true;
        }
        if (index >= s.length()) {
            return false;
        }

        String subString = s.substring(index, index + wordLength);

        final Integer node = map.get(subString);
        if (node == null || node <= 0) {
            return false;
        }

        map.put(subString, node - 1);

        boolean temp = findSubstringDfs(s, index + wordLength, wordLength, wordCnt, cnt + 1, map);

        map.put(subString, node);

        return temp;
    }

    public void nextPermutation(int[] nums) {
        if (nums == null || nums.length <= 1) {
            return;
        }

        // 从后面往前找，第一个非递增的index
        int index = -1;
        for (int i = nums.length - 1; i >= 1; --i) {
            if (nums[i] <= nums[i-1]) {
                continue;
            } else {
                index = i - 1;
                break;
            }
        }

        if (index == -1) {
            nextPermutationReverser(nums, 0, nums.length - 1);
            return;
        }

        for (int i = nums.length - 1; i >= index + 1; --i) {
            if (nums[i] > nums[index]) {
                int temp = nums[index];
                nums[index] = nums[i];
                nums[i] = temp;

                nextPermutationReverser(nums, index + 1, nums.length - 1);
                break;
            }
        }


    }

    private void nextPermutationReverser(int[] nums, int start, int end) {
        int left = start;
        int right = end;
        while (left < right) {
            int temp = nums[left];
            nums[left] = nums[right];
            nums[right] = temp;
            left++;
            right--;
        }
    }

    public int longestValidParentheses(String s) {
        if (s == null || s.length() <= 1) {
            return 0;
        }

        int result = 0;
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {
                if (stack.isEmpty()) {
                    stack.push(i);
                } else if (s.charAt(stack.peek()) == '(' ){
                    stack.pop();
                    result = Math.max(result, i - (stack.isEmpty() ? -1 : stack.peek()));
                }
            }
        }

        return result;
    }

    public int search(int[] nums, int target) {
        if (nums == null || nums.length <= 0) {
            return -1;
        }

        int left = 0;
        int right = nums.length - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            }

            if (nums[left] <= nums[mid] && nums[right] <= nums[mid]) {
                if (target > nums[mid]) {
                    left = mid + 1;
                } else if (target >= nums[left]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else if (nums[left] >= nums[mid] && nums[mid] <= nums[right]) {
                if (target < nums[mid]) {
                    right = mid - 1;
                } else if (target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            } else {
                if (target > nums[mid]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }

        return -1;
    }

    public int[] searchRange(int[] nums, int target) {
        int left = searchRangeLeft(nums, target);
        int right = searchRangeRight(nums, target);

        return new int[]{left, right};
    }

    private int searchRangeLeft(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] >= target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        return (left < nums.length && nums[left] == target) ? left : -1;
    }

    private int searchRangeRight(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        return (right >= 0 && nums[right] == target) ? right : -1;
    }

    public int searchInsert(int[] nums, int target) {
        if (nums == null || nums.length <= 0) {
            return 0;
        }

        int left = 0;
        int right = nums.length - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        return right == -1 ? 0 : left;
    }

    public void solveSudoku(char[][] board) {
        ArrayList<Integer> pointPos = new ArrayList<>();
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                if (board[i][j] == '.') {
                    pointPos.add(i * 9 + j);
                }
            }
        }

        if (pointPos.isEmpty()) {
            return;
        }

        solveSudokuDfs(board, pointPos, 0, new HashMap<>());
    }

    private void solveSudokuDfs(char[][] board, List<Integer> pointPos, int index, Map<Integer, Character> map) {
        System.out.println(index);
        if (index >= pointPos.size()) {
            for (Map.Entry<Integer, Character> entry : map.entrySet()) {
                board[entry.getKey() / 9][entry.getKey() % 9] = entry.getValue();
            }
            return;
        }

        final int row = pointPos.get(index) / 9;
        final int column = pointPos.get(index) % 9;

        for (char c = '1'; c <= '9'; ++c) {
            if (checkSudokuValidRow(board, row, column, map, c)
                    && checkSudokuValidColumn(board, row, column, map, c)
                    && checkSudokuValidSquare(board, row, column, map, c)) {
                map.put(pointPos.get(index), c);
                solveSudokuDfs(board, pointPos, index + 1, map);
                map.remove(pointPos.get(index));
            }
        }
    }

    private boolean checkSudokuValidRow(char[][] board, int row, int column, Map<Integer, Character> map, char c) {
        for (int i = 0; i < 9; ++i) {
            final char node;
            if (board[row][i] != '.') {
                node = board[row][i];
            } else {
                final Character value = map.getOrDefault(row * 9 + i, null);
                node = value == null ? '.' : value;
            }

            if (node == c) {
                return false;
            }
        }

        return true;
    }

    private boolean checkSudokuValidColumn(char[][] board, int row, int column, Map<Integer, Character> map, char c) {
        for (int i = 0; i < 9; ++i) {
            final char node;
            if (board[i][column] != '.') {
                node = board[i][column];
            } else {
                final Character value = map.getOrDefault(i * 9 + column, null);
                node = value == null ? '.' : value;
            }

            if (node == c) {
                return false;
            }
        }

        return true;
    }

    private boolean checkSudokuValidSquare(char[][] board, int row, int column, Map<Integer, Character> map, char c) {
        int squareRow = row / 3 * 3;
        int squareColumn = column / 3 * 3;


        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                row = squareRow + i;
                column = squareColumn + j;
                final char node;
                if (board[row][column] != '.') {
                    node = board[row][column];
                } else {
                    final Character value = map.getOrDefault(row * 9 + column, null);
                    node = value == null ? '.' : value;
                }

                if (node == c) {
                    return false;
                }
            }
        }

        return true;
    }

    public String countAndSay(int n) {
        if (n == 1) {
            return "1";
        } else {
            String pre = countAndSay(n - 1);

            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < pre.length(); ++i) {
                int oldi = i;
                while (i + 1 < pre.length() && pre.charAt(i+1) == pre.charAt(i)) {
                    i++;
                }

                builder.append(i - oldi + 1).append((int)(pre.charAt(i) - '0'));
            }

            return builder.toString();
        }
    }

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new LinkedList<>();

        List<Integer> one = new LinkedList<>();

        combinationSumDfs(candidates, 0, 0, target, one, result);

        return result;
    }

    private void combinationSumDfs(int[] candidates, int index, int sum, int target, List<Integer> one, List<List<Integer>> result) {
        if (sum == target) {
            List<Integer> oneRes = new LinkedList<>(one);
            result.add(oneRes);
            return;
        }

        if (index >= candidates.length) {
            return;
        }

        if (sum + candidates[index] <= target) {
            one.add(candidates[index]);
            combinationSumDfs(candidates, index, sum + candidates[index], target, one, result);
            one.remove(one.size() - 1);
        }

        combinationSumDfs(candidates, index + 1, sum, target, one, result);
    }

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);

        List<List<Integer>> result = new LinkedList<>();

        List<Integer> one = new LinkedList<>();

        combinationSum2Dfs(candidates, target, 0, 0, one, result);

        return result;
    }

    private void combinationSum2Dfs(int[] candidates, int target, int index, int sum, List<Integer> one, List<List<Integer>> result) {
        if (sum == target) {
            List<Integer> oneRes = new LinkedList<>(one);
            result.add(oneRes);
            return;
        }

        if (index >= candidates.length) {
            return;
        }

        for (int i = index; i < candidates.length; ++i) {
            if (sum + candidates[i] <= target) {
                one.add(candidates[i]);
                combinationSum2Dfs(candidates, target, i + 1, sum + candidates[i], one, result);
                one.remove(one.size() - 1);
            }

            while (i + 1 < candidates.length && candidates[i] == candidates[i + 1]) {
                i++;
            }
        }
    }

    public int firstMissingPositive(int[] nums) {
        if (nums == null || nums.length <= 0) {
            return 1;
        }

        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] <= 0 || nums[i] > nums.length || nums[i] == i + 1) {
                continue;
            }

            int nextIdx = nums[i] - 1;
            nums[i] = 0;

            do {
                final int oldNextValue = nums[nextIdx];
                nums[nextIdx] = nextIdx + 1;

                if (oldNextValue <= 0 || oldNextValue > nums.length || nums[oldNextValue - 1] == oldNextValue) {
                    break;
                } else {
                    nextIdx = oldNextValue - 1;
                }

            } while (true);
        }

        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }

        return nums.length + 1;
    }

    public int trap(int[] height) {
        if (height == null || height.length <= 2) {
            return 0;
        }

        int left = 0;
        int right = height.length - 1;

        int leftMax = 0;
        int rightMax = 0;

        int result = 0;

        while (left < right) {
            if (height[left] < height[right]) {
                if (height[left] < leftMax) {
                    result += leftMax - height[left];
                } else {
                    leftMax = height[left];
                }
                left++;
            } else {
                if (height[right] < rightMax) {
                    result += rightMax - height[right];
                } else {
                    rightMax = height[right];
                }
                right--;
            }
        }

        return result;
    }

    public String multiply(String num1, String num2) {
        final int len1 = num1.length();
        final int len2 = num2.length();
        int[] multis = new int[len1 + len2];

        for (int i = 0; i < len1; ++i) {
            for (int j = 0; j < len2; ++j) {
                final int c1 = num1.charAt(len1 - 1 - i) - '0';
                final int c2 = num2.charAt(len2 - 1 - j) - '0';
                multis[i + j] += c1 * c2;
            }
        }

        for (int i = 0; i < multis.length - 2; ++i) {
            multis[i + 1] += multis[i] / 10;
            multis[i] %= 10;
        }

        StringBuilder builder = new StringBuilder();
        boolean trimZero = true;
        for (int i = multis.length - 1; i >= 0; i--) {
            if (multis[i] == 0) {
                if (trimZero) {
                    continue;
                } else {
                    builder.append(0);
                }
            } else {
                trimZero = false;
                builder.append(multis[i]);
            }
        }
        if (builder.length() <= 0) {
            return "0";
        } else {
            return builder.toString();
        }
    }

    public boolean isMatchII(String s, String p) {
        boolean[][] dp = new boolean[p.length() + 1][s.length() + 1];
        dp[0][0] = true;

        for (int i = 1; i <= p.length(); ++i) {
            if (p.charAt(i-1) != '*') {
                break;
            } else {
                dp[i][0] = true;
            }
        }

        for (int i = 1; i <= p.length(); ++i) {
            for (int j = 1; j <= s.length(); ++j) {
                final char pc = p.charAt(i-1);
                final char sc = s.charAt(j-1);

                if (pc == '*') {
                    dp[i][j] = dp[i-1][j] || dp[i][j-1];
                } else {
                    dp[i][j] = (pc == '?' || (pc == sc)) && dp[i-1][j-1];
                }
            }
        }

        return dp[p.length()][s.length()];
    }

    public int jump(int[] nums) {
        if (nums == null || nums.length <= 1) {
            return 0;
        }

        int result = 0;

        int maxDestination = 0;

        int i = 0;

        while (maxDestination < nums.length - 1) {
            result++;

            int newMaxDes = maxDestination;
            for (; i <= maxDestination; ++i) {
                if (nums[i] + i > newMaxDes) {
                    newMaxDes = nums[i] + i;
                }
            }

            maxDestination = newMaxDes;
        }

        return result;
    }

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new LinkedList<>();

        permute(nums, 0, result);

        return result;
    }

    private void permute(int[] nums, int index, List<List<Integer>> result) {
        if (index == nums.length) {
            List<Integer> one = new ArrayList<>(nums.length);
            for (int num : nums) {
                one.add(num);
            }
            result.add(one);
            return;
        }

        for (int i = index; i <= nums.length - 1; ++i) {
            permuteSwap(nums, i, index);
            permute(nums, index + 1, result);
            permuteSwap(nums, i, index);
        }
    }

    private void permuteSwap(int[] nums, int from, int to) {
        int temp = nums[from];
        nums[from] = nums[to];
        nums[to] = temp;
    }

    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> result = new LinkedList<>();

        permuteUnique(nums, 0, result);

        return result;
    }

    private void permuteUnique(int[] nums, int index, List<List<Integer>> result) {
        if (index == nums.length) {
            List<Integer> one = new ArrayList<>(nums.length);
            for (int num : nums) {
                one.add(num);
                System.out.print(num);
            }
            System.out.println();
            result.add(one);
            return;
        }

        Set<Integer> visited = new HashSet<>();

        for (int i = index; i <= nums.length - 1; ++i) {
            if (i != index && nums[i] == nums[index]) continue;
            if (i != index && !visited.add(nums[i])) continue;

            permuteSwap(nums, i, index);
            permuteUnique(nums, index + 1, result);
            permuteSwap(nums, i, index);
        }
    }

    public void rotate(int[][] matrix) {
        if (matrix == null || matrix.length <= 0 || matrix[0] == null || matrix[0].length <= 0) {
            return;
        }

        int start = 0;
        int end = matrix.length - 1;

        while (start < end) {
            rotateLevel(matrix, start, start, end, end);
            start++;
            end--;
        }
    }

    private void rotateLevel(int[][] matrix, int sx, int sy, int ex, int ey) {
        for (int column = sy; column < ey; column++) {
            // 分为4步，
            int temp = matrix[sx + column - sy][ey];
            matrix[sx + column - sy][ey] = matrix[sx][column];

            int temp1 = matrix[ex][ey - (column - sy)];
            matrix[ex][ey - (column - sy)] = temp;

            int temp2 = matrix[ex - (column - sy)][sy];
            matrix[ex - (column - sy)][sy] = temp1;

            matrix[sx][column] = temp2;
        }
    }

    public double myPow(double x, int n) {
        if (n < 0) {
            return fastMyPow( 1 / x, -n);
        } else {
            return fastMyPow(x, n);
        }
    }

    private double fastMyPow(double x, int n) {
        if (n == 0) {
            return 1;
        } else {
            double temp = fastMyPow(x, n / 2);
            double result = temp * temp;

            if ((n & 1) == 1) {
                result *= x;
            }
            return result;
        }
    }

    public List<List<String>> solveNQueens(int n) {
        int[] one = new int[n];
        Arrays.fill(one, -1);

        List<List<String>> result = new LinkedList<>();
        solveNQueens(n, 0, one, result);
        return result;
    }

    private void solveNQueens(int n , int index, int[] one, List<List<String>> result) {
        if (index >= n) {
            List<String> oneList = new ArrayList<>(n);
            for (int pos : one) {
                oneList.add(solvNQueensOneResult(pos, n));
            }
            result.add(oneList);
            return;
        }

        for (int i = 0; i < n; i++) {
            if (solveNQueensValid(one, index, i)) {
                one[index] = i;
                solveNQueens(n, index + 1, one, result);
                one[index] = -1;
            }
        }
    }

    private boolean solveNQueensValid(int[] one, int index, int pos) {
        // 纵向检查
        boolean valid = true;
        for (int i = 0; i < index; ++i) {
            if (one[i] == pos) {
                valid = false;
                break;
            }

            if (Math.abs(i - index) == Math.abs(one[i] - pos)) {
                valid = false;
                break;
            }
        }

        return valid;
    }

    private String solvNQueensOneResult(int pos, int length) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < length; ++i) {
            builder.append(i == pos ? 'Q' : '.');
        }
        return builder.toString();
    }

    public int totalNQueens(int n) {
        int[] result = {0};
        int[] one = new int[n];
        Arrays.fill(one, -1);
        totalNQueens(n, 0, one, result);
        return result[0];
    }

    private void totalNQueens(int n , int index, int[] one, int[] result) {
        if (index >= n) {
            result[0]++;
            return;
        }

        for (int i = 0; i < n; i++) {
            if (solveNQueensValid(one, index, i)) {
                one[index] = i;
                totalNQueens(n, index + 1, one, result);
                one[index] = -1;
            }
        }
    }

    public int maxSubArray(int[] nums) {
        if (nums == null || nums.length <= 0) {
            return 0;
        }

        int result = nums[0];
        int preMax = result;

        for (int i = 1; i < nums.length; ++i) {
            if (preMax <= 0) {
                preMax = nums[i];
            } else {
                preMax = nums[i] + preMax;
            }
            result = Math.max(result, preMax);
        }

        return result;

    }

    public List<Integer> spiralOrder(int[][] matrix) {
        if (matrix == null || matrix.length <= 0 || matrix[0] == null || matrix[0].length <= 0) {
            return new ArrayList<>();
        }

        final int rows = matrix.length;
        final int columns = matrix[0].length;

        final List<Integer> result = new ArrayList<>(rows * columns);

        int sx = 0;
        int sy = 0;

        int ex = rows - 1;
        int ey = columns - 1;

        while (sx <= ex && sy <= ey) {
            if (sx == ex) {
                for (int i = sy; i <= ey; ++i) {
                    result.add(matrix[sx][i]);
                }
                continue;
            } else if (sy == ey) {
                for (int i = sx; i <= ex; ++i) {
                    result.add(matrix[i][sy]);
                }
                continue;
            }

            for (int i = sy; i <= ey; ++i) {
                result.add(matrix[sx][i]);
            }

            for (int i = sx + 1; i <= ex; ++i) {
                result.add(matrix[i][ey]);
            }

            for (int i = ey - 1; i >= sy; --i) {
                result.add(matrix[ex][i]);
            }

            for (int i = ex - 1; i > sx; --i) {
                result.add(matrix[i][sy]);
            }

            sx++;sy++;
            ex--;ey--;
        }

        return result;
    }

    public boolean canJump(int[] nums) {
        int maxPos = 0;

        for (int i = 0; i <= maxPos && i < nums.length; ++i) {
            if (maxPos >= nums.length - 1) {
                return true;
            }
            if (nums[i] + i > maxPos) {
                maxPos = nums[i] + i;
            }
        }

        return maxPos >= nums.length - 1;
    }

    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] != o2[0])
                    return o1[0] - o2[0];
                else
                    return o1[1] - o2[1];
            }
        });

        int start = -1;
        int end = -1;

        ArrayList<int[]> result = new ArrayList<>();

        for (int i = 0; i < intervals.length; ++i) {
            if (start == -1 ) {
                start = intervals[i][0];
                end = intervals[i][1];
            } else {
                if (intervals[i][0] > end) {
                    result.add(new int[]{start, end});
                    start = intervals[i][0];
                    end = intervals[i][1];
                } else {
                    end = Math.max(end, intervals[i][1]);
                }
            }
        }

        if (start != -1) {
            result.add(new int[]{start, end});
        }

        int[][] finalResult = new int[result.size()][];
        finalResult = result.toArray(finalResult);

        return finalResult;
    }

    public int[][] insert(int[][] intervals, int[] newInterval) {
        if(intervals == null || intervals.length == 0 || intervals[0].length == 0)
            return new int[][]{newInterval};
        int i = 0, n = intervals.length;
        List<int[]> res = new ArrayList<>();

        //adding all the interval end before the newInterval starts
        while(i < n && intervals[i][1] < newInterval[0]){
            res.add(intervals[i]);
            i++;
        }

        //merge all overlapping intervals
        while(i < n && intervals[i][0] <= newInterval[1]){
            newInterval[0] = Math.min(intervals[i][0], newInterval[0]);
            newInterval[1] = Math.max(intervals[i][1], newInterval[1]);
            i++;
        }

        res.add(newInterval);
        while(i < n){
            res.add(intervals[i]);
            i++;
        }
        return res.toArray(new int[res.size()][]);
    }

    public int lengthOfLastWord(String s) {
        if (s == null || s.length() <= 0) {
            return 0;
        }

        int i = s.length() - 1;
        while (i >= 0 && s.charAt(i) == ' ') {
            --i;
        }

        if (i < 0) {
            return 0;
        }

        int end = i;
        while (i >= 0 && s.charAt(i) != ' ') {
            --i;
        }

        return end - i;
    }

    public String getPermutation(int n, int k) {
        int[] nums = new int[n];
        for (int i = 0; i < n; ++i) {
            nums[i] = i + 1;
        }

        int[] perms = new int[n+1];
        perms[1] = 1;
        for (int i = 2; i <= n; ++i) {
            perms[i] = perms[i-1] * i;
        }

        StringBuilder builder = new StringBuilder();


        for (int i = 1; i <= n - 1; ++i) {
            int index = (k - 1) / perms[n - i];

            if (index != 0) {
                swapNums(nums, i - 1, i - 1 + index);
                Arrays.sort(nums, i, nums.length);
            }

            k = k - perms[n-i] * index;
        }

        for (int i = 0; i < n; ++i) {
            builder.append(nums[i]);
        }
        return builder.toString();
    }

    private void swapNums(int[] nums, int from, int to) {
        int temp = nums[from];
        nums[from] = nums[to];
        nums[to] = temp;
    }

    public int mySqrt(int x) {
        if (x == 0 || x == 1) {
            return x;
        }

        int left = 1;
        int right = x;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            int square = mid * mid;
            if (square == x) {
                return mid;
            } else if (square > x) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        return left;
    }

    public ListNode rotateRight(ListNode head, int k) {
        if (head == null || k == 0) {
            return head;
        }

        int totalLength = 0;
        ListNode temp = head;
        ListNode tail = null;

        while (temp != null) {
            totalLength++;
            tail = temp;
            temp = temp.next;
        }

        int left = k % totalLength;
        if (left == 0) {
            return head;
        }

        temp = head;
        ListNode pre = null;
        int index = 1;
        while (index++ <= totalLength - k) {
            pre = temp;
            temp = temp.next;
        }

        pre.next = null;
        tail.next = head;

        return temp;
    }

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;

        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            if (obstacleGrid[i][0] == 1) {
                break;
            } else {
                dp[i][0] = 1;
            }
        }
        for (int i = 0; i < n; i++) {
            if (obstacleGrid[0][i] == 1) {
                break;
            } else {
                dp[0][i] = 1;
            }
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = obstacleGrid[i][j] == 1 ? 0 : dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    public int minPathSum(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;

        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < m; i++) {
            dp[i][0] = dp[i-1][0] + grid[i][0];
        }
        for (int i = 1; i < n; i++) {
            dp[0][i] = dp[0][i-1] + grid[0][i];
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = grid[i][j] + Math.min(dp[i-1][j], dp[i][j-1]);
            }
        }
        return dp[m - 1][n - 1];
    }

    public boolean isNumber(String s) {
        if (s == null || s.length() <= 0) {
            return false;
        }

        int start = 0;
        int right = s.length() - 1;
        while (start < s.length() && s.charAt(start) == ' ') {
            start++;
        }

        while (right >= start && s.charAt(right) == ' ') {
            right--;
        }

        if (right < start) {
            return false;
        }

        boolean hasSignBeforeE = false;
        boolean hasSignAfterE = false;
        int ePos = -1;
        boolean hasNumBeforeE = false;
        boolean hasNumAfterE = false;
        boolean hasDot = false;

        for (int i = start; i <= right; ++i) {
            final char c = s.charAt(i);

            if (Character.isDigit(c)) {
                // 没有遇到e之前
                if (ePos == -1) {
                    hasNumBeforeE = true;
                } else {
                    hasNumAfterE = true;
                }
            } else if (c == '-' || c == '+') {
                if (ePos != -1) {
                    if (hasSignAfterE || i != ePos + 1 ) return false;
                    else hasSignAfterE = true;
                } else {
                    if (i != start || hasSignBeforeE) {
                        return false;
                    } else {
                        hasSignBeforeE = true;
                    }
                }
            } else if (c == 'e'){
                if (ePos != -1) {
                    return false;
                } else {
                    if (hasDot && !hasNumBeforeE) return false;
                    ePos = i;
                }
            } else if (c == '.') {
                if (ePos != -1 || hasDot) {
                    return false;
                } else {
                    hasDot = true;
                }
            } else {
                return false;
            }
        }

        if (hasDot) {
            if (!hasNumBeforeE) return false;
        }

        return ePos == -1 || hasNumAfterE;
    }

    public String simplifyPath(String path) {
        String[] split = path.split("/");
        Stack<String> stack = new Stack<>();

        for (int i = 0; i < split.length; ++i) {
            if (".".equals(split[i]) || split[i] == null || split[i].isEmpty()) {
                continue;
            } else if ("..".equals(split[i])) {
                if (!stack.isEmpty()) {
                    stack.pop();
                }
            } else {
                stack.push(split[i]);
            }
        }

        String result = "";
        while (!stack.isEmpty()) {
            result = "/" + stack.pop() + result;
        }
        if (result.isEmpty()) {
            result = "/";
        }

        return result;
    }

    public int minDistance(String word1, String word2) {
        int[][] dp = new int[word1.length() + 1][word2.length() + 1];
        for (int i = 1; i <= word1.length(); ++i) {
            dp[i][0] = i;
        }
        for (int i = 1; i <= word2.length(); ++i) {
            dp[0][i] = i;
        }

        for (int i = 1; i <= word1.length(); ++i)
            for (int j = 1; j <= word2.length(); ++j) {
                if (word1.charAt(i-1) == word2.charAt(j-1)) {
                    dp[i][j] = dp[i-1][j-1];
                } else {
                    dp[i][j] = 1 + Math.min(dp[i-1][j], Math.min(dp[i][j-1], dp[i-1][j-1]));
                }
            }

        return dp[word1.length()][word2.length()];
    }

    public void setZeroes(int[][] matrix) {

    }

    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length <= 0 || matrix[0] == null || matrix[0].length <= 0) {
            return false;
        }
        int left = 0;
        int right = matrix.length - 1;

        int toSearchRow = -1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (target < matrix[mid][0]) {
                right = mid - 1;
            } else if (target > matrix[mid][matrix[0].length - 1]) {
                left = mid + 1;
            } else {
                toSearchRow = mid;
                break;
            }
        }

        if (toSearchRow == -1) {
            return false;
        }

        left = 0;
        right = matrix[0].length - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (target < matrix[toSearchRow][mid]) {
                right = mid - 1;
            } else if (target > matrix[toSearchRow][mid]) {
                left = mid + 1;
            } else {
                return true;
            }
        }

        return false;
    }

    public void sortColors(int[] nums) {
        int front = 0;
        int back = nums.length - 1;

        // key thing to notice
        // we end on i <= back because back updates
        // back updates everytime we find a 2
        // everything past back is also sorted
        // just like how everything in before front is also sorted
        for(int i = 0; i <= back; i++){
            if(nums[i] == 0) swap(nums, front++, i);
            if(nums[i] == 2) swap(nums, back--, i--);
        }
    }



    public String minWindow(String s, String t) {
        int[] hashT = new int[128];
        int tCnt = t.length();
        for (int i = 0; i < t.length(); ++i) {
            hashT[t.charAt(i)]++;
        }

        int right = 0;
        int left = 0;

        int minLength = s.length();
        int minStart = -1;

        while (right < s.length()) {
            if (--hashT[s.charAt(right)] >= 0) {
                tCnt--;
            }

            while (tCnt == 0 && left <= right) {
                if (minLength > right - left + 1) {
                    minLength = right - left + 1;
                    minStart = left;
                }
                if (++hashT[s.charAt(left)] > 0) {
                    tCnt++;
                }
                left++;
            }
            right++;
        }

        if (minStart == -1) {
            return "";
        } else {
            return s.substring(minStart, minStart + minLength);
        }
    }

    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> result = new LinkedList<>();

        combineHelper(1, n, k, result, new ArrayList<>(k));

        return result;
    }

    private void combineHelper(int index, int n, int k, List<List<Integer>> result, List<Integer> one) {
        if (one.size() == k) {
            List<Integer> oneResult = new ArrayList<>(one);
            result.add(oneResult);
            return;
        }

        if (index > n) {
            return;
        }

        combineHelper(index + 1, n, k, result, one);

        one.add(index);
        combineHelper(index + 1, n, k, result, one);
        one.remove(one.size() - 1);
    }

    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new LinkedList<>();

        subsetsHelper(nums, 0, result, new ArrayList<>(nums.length));
        result.add(new ArrayList<>());
        return result;
    }

    private void subsetsHelper(int[] nums, int index, List<List<Integer>> result, List<Integer> one) {
        if (one.size() > 0) {
            List<Integer> oneResult = new ArrayList<>(one);
            result.add(oneResult);
        }

        if (index >= nums.length) {
            return;
        }

        for (int i = index; i < nums.length; ++i) {
            one.add(nums[index]);
            subsetsHelper(nums,i + 1, result, one);
            one.remove(one.size() - 1);
        }
    }

    public int removeDuplicatesII(int[] nums) {
        int start = 0;

        int right = 0;

        int cnt = 0;
        int pre = -1;

        while (right < nums.length) {
            if (nums[right] == pre) {
                cnt++;
            } else {
                cnt = 1;
                pre = nums[right];
            }

            if (cnt <= 2) {
                swap(nums, start, right);
                start++;
            }
            right++;
        }

        return start;
    }

    private void swap(int[] nums, int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public boolean searchII(int[] nums, int target) {
        return search(nums, 0, nums.length - 1, target);
    }

    public boolean search(int[] nums, int start, int end, int target) {
        if (nums == null || nums.length <= 0 || end < start) {
            return false;
        }

        int left = start;
        int right = end;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return true;
            }

            // 全部相等
            if (nums[left] == nums[right]) {
                if (nums[mid] == nums[left])
                    return search(nums, start, mid - 1, target) || search(nums, mid + 1, end, target);
                else if (nums[mid] > nums[left]) {
                    if (target > nums[mid]) {
                        left = mid + 1;
                    } else {
                        right = mid - 1;
                    }
                } else {
                    return search(nums, start, mid - 1, target) || search(nums, mid + 1, end, target);
                }
            } else if (nums[left] == nums[mid]) {
                if (nums[left] > nums[right])
                    return search(nums, start, mid - 1, target) || search(nums, mid + 1, end, target);
                else {
                    if (target > nums[mid]) {
                        left = mid + 1;
                    } else {
                        return search(nums, start, mid - 1, target) || search(nums, mid + 1, end, target);
                    }
                }

            } else if (nums[right] == nums[mid]) {
                if (nums[left] > nums[right]) {
                    if (target > nums[mid]) {
                        right = mid - 1;
                    } else {
                        return search(nums, start, mid - 1, target) || search(nums, mid + 1, end, target);
                    }
                } else {
                    return search(nums, start, mid - 1, target) || search(nums, mid + 1, end, target);
                }
            } else if (nums[left] <= nums[mid] && nums[right] <= nums[mid]) {
                if (target > nums[mid]) {
                    left = mid + 1;
                } else if (target >= nums[left]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }

            } else if (nums[left] >= nums[mid] && nums[mid] <= nums[right]) {
                if (target < nums[mid]) {
                    right = mid - 1;
                } else if (target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            } else {
                if (target > nums[mid]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }

        return false;
    }

    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        ListNode newHead = new ListNode(-1);
        newHead.next = head;

        ListNode pre = newHead;
        ListNode temp = head;


        while (temp != null) {
            int cnt = 1;
            while (temp != null && temp.next != null && temp.val == temp.next.val) {
                temp = temp.next;
                cnt++;
            }

            if (cnt >= 2) {
                pre.next = temp == null ? null : temp.next;
                temp = temp == null ? null : temp.next;
            } else {
                pre.next = temp;
                pre = temp;
                temp = temp.next;
            }
        }

        return newHead.next;
    }

    public ListNode deleteDuplicatesI(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        ListNode pre = head;
        ListNode cur = head.next;

        while (cur != null) {
            if (pre.val == cur.val) {
                cur = cur.next;
                pre.next = cur;
            } else {
                pre = cur;
                cur = cur.next;
            }
        }

        return head;
    }

    public int largestRectangleArea(int[] heights) {
        if (heights == null || heights.length <= 0) {
            return 0;
        }

        Deque<Integer> deque = new LinkedList<>();

        int result = 0;
        for (int i = 0; i <= heights.length; ++i) {
            int value = i == heights.length ? 0 : heights[i];
            while (!deque.isEmpty() && heights[deque.peekLast()] > value) {
                int cur = deque.pollLast();
                result = Math.max(result, (i - (deque.isEmpty() ? 0 : (deque.peekLast() + 1))) * heights[cur]);
            }
            deque.addLast(i);
        }

        return result;
    }

    public ListNode partition(ListNode head, int x) {
        if (head == null || head.next == null) {
            return head;
        }

        ListNode newHead = new ListNode(-1);
        newHead.next = head;

        ListNode pre = newHead;
        ListNode cur = head;

        while (cur != null && cur.val < x) {
            pre = cur;
            cur = cur.next;
        }

        if (cur == null) {
            return newHead.next;
        }

        ListNode insertAfter = pre;
        ListNode insertNext = cur;

        while (cur != null) {
            if (cur.val < x) {
                final ListNode next = cur.next;

                insertAfter.next = cur;
                cur.next = insertNext;
                insertAfter = insertAfter.next;

                pre.next = next;
                cur = next;
            } else {
                pre = cur;
                cur = cur.next;
            }
        }

        return newHead.next;

    }

    public boolean isScramble(String s1, String s2) {
        return isScramble(s1, 0, s2, 0, s1.length());
    }

    private boolean isScramble(String s1, int start1, String s2, int start2, int length) {
        int compare = scrambleEqual(s1, s2, start1, start2, length);
        if (compare == 0) {
            return true;
        } else if (compare == 1) {
            return false;
        }
        for (int len = 1; len <= length - 1; len++) {
            if (isScramble(s1, start1, s2, start2, len) && isScramble(s1, start1 + len, s2, start2 + len, length - len)) {
                return true;
            }
            if (isScramble(s1, start1, s2, start2 + length - len, len) && isScramble(s1, start1 + len, s2, start2, length - len)) {
                return true;
            }
        }

        return false;
    }

    private int scrambleEqual(String s1, String s2, int start1, int start2, int length) {
        boolean equal = true;
        int[] hash = new int[26];
        for (int i = 0; i < length; ++i) {
            final char c1 = s1.charAt(start1 + i);
            final char c2 = s2.charAt(start2 + i);
            if (c1 != c2) {
                equal = false;
            }

            hash[c1 - 'a']++;
            hash[c2 - 'a']--;
        }

        if (equal) {
            return 0;
        }

        for (int num : hash) {
            if (num != 0) {
                return 1;
            }
        }

        return -1;
    }

    public List<Integer> grayCode(int n) {
        List<Integer> result = new ArrayList<>();

        if (n == 0) {
            result.add(0);
            return result;
        }

        for (int i = 1; i <=n; ++i) {
            final int size = result.size();

            for (int j = size - 1; j >= 0; --j) {
                result.add((1 << (i - 1)) | result.get(j));
            }
        }

        return result;
    }

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> result = new LinkedList<>();
        subsetsWithDup(nums, 0, new ArrayList<>(), result);
        return result;
    }

    private void subsetsWithDup(int[] nums, int index, List<Integer> one, List<List<Integer>> result) {
        if (one.size() > 0) {
            List<Integer> temp = new ArrayList<>(one);
            result.add(temp);
        }

        for (int i = index; i < nums.length; ++i) {
            one.add(nums[index]);
            subsetsWithDup(nums, i + 1, one, result);
            one.remove(one.size() - 1);

            while (i + 1 < nums.length && nums[i] == nums[i+1]) {
                ++i;
            }
        }
    }

    public int numDecodings(String s) {
        if (s == null || s.length() <= 1) {
            return 1;
        }

        int[] dp = new int[s.length() + 1];
        dp[0] = 1;
        dp[1] = s.charAt(0) != '0' ? 1 : 0;

        for (int i = 2; i <= s.length(); ++i) {
            final char c = s.charAt(i-1);
            final char prec = s.charAt(i-2);

            if (c == '0') {
                dp[i] = (prec == '1' || prec == '2') ? dp[i-2] : 0;
            } else {
                dp[i] = dp[i-1];
                if (c > '6') {
                    if (prec == '1') {
                        dp[i] += dp[i-2];
                    }
                } else {
                    if (prec == '1' || prec == '2') {
                        dp[i] += dp[i-2];
                    }
                }
            }

        }

        return dp[s.length()];
    }

    public ListNode reverseBetween(ListNode head, int m, int n) {
        if (m == n || head == null) {
            return head;
        }

        ListNode newHead = new ListNode(-1);
        newHead.next = head;

        ListNode pre = newHead;
        ListNode cur = head;

        ListNode start = null;
        ListNode startPre = null;

        ListNode end = null;
        ListNode endNext = null;

        int index = 1;

        while (cur != null && index <= n) {
            if (index == m) {
                start = cur;
                startPre = pre;
            } else if (index == n) {
                end = cur;
                endNext = cur.next;
                break;
            }

            pre = cur;
            cur = cur.next;
            index++;
        }

        if (start == null) {
            return newHead.next;
        }

        pre = endNext;
        cur = start;

        while (cur != endNext) {
            ListNode temp = cur.next;
            cur.next = pre;

            pre = cur;
            cur = temp;
        }

        startPre.next = pre;

        return newHead.next;
    }

    public List<Integer> inorderTraversal(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();

        Stack<TreeNode> stack = new Stack<>();
        if (root == null) {
            return result;
        }

        TreeNode cur = root;

        while (!stack.isEmpty() || cur != null) {
            while (cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
            cur = stack.pop();
            result.add(cur.val);
            cur = cur.right;
        }

        return result;
    }

    public List<TreeNode> generateTrees(int n) {
        if (n == 0) {
            return new ArrayList<>();
        }
        return generateTrees(1, n);
    }

    private List<TreeNode> generateTrees(int start, int end) {
        List<TreeNode> result = new LinkedList<>();
        if (start > end) {
            result.add(null);
            return result;
        } else if (start == end) {
            result.add(new TreeNode(start));
            return result;
        }

        for (int i = start; i <= end; ++i) {
            List<TreeNode> lefts = generateTrees(start, i - 1);
            List<TreeNode> rights = generateTrees(i + 1, end);
            for (TreeNode left : lefts) {
                for (TreeNode right : rights) {
                    TreeNode root = new TreeNode(i);
                    root.left = left;
                    root.right = right;
                    result.add(root);
                }
            }
        }

        return result;
    }

    public int numTrees(int n) {
        if (n <= 2) {
            return n;
        }


        int[] dp = new int[n + 1];
        dp[0] = dp[1] = 1;

        for (int i = 2; i <= n; ++i) {
            for (int j = 1; j <= i; ++j) {
                int left = dp[j-1];
                int right = dp[i - j];
                dp[i] += left * right;
            }
        }

        return dp[n];
    }

    public boolean isValidBST(TreeNode root) {
        return isValidBST(root, Long.MAX_VALUE, Long.MIN_VALUE);
    }

    private boolean isValidBST(TreeNode root, long max, long min) {
        if (root == null) {
            return true;
        }

        if (root.val >= max || root.val <= min) {
            return false;
        }

        return isValidBST(root.left, root.val, min) && isValidBST(root.right, max, Math.max(root.val, min));
    }

    public void recoverTree(TreeNode root) {
        recoverValidBST(root, null, null);
    }

    private boolean hasRecovered = false;
    private void recoverValidBST(TreeNode root, TreeNode maxNode, TreeNode minNode) {
        if (root == null || hasRecovered) {
            return;
        }

        final long max = maxNode == null ? Long.MAX_VALUE : maxNode.val;
        final long min = minNode == null ? Long.MIN_VALUE : minNode.val;

        if (root.val > max) {
            final int temp = maxNode.val;
            maxNode.val = root.val;
            root.val = temp;
            hasRecovered = true;
            return;
        } else if (root.val > min) {
            final int temp = minNode.val;
            minNode.val = root.val;
            root.val = temp;
            hasRecovered = true;
            return;
        }

        recoverValidBST(root.left, root, minNode);

        if (!hasRecovered) {
            recoverValidBST(root.right, maxNode, root);
        }
    }

    public boolean isSymmetric(TreeNode root) {
        return root == null ? true : isSymmetric(root.left,root.right);
    }
    public boolean isSymmetric(TreeNode left,TreeNode right){
        if(left==null && right==null)return true;
        if(left==null || right==null)return false;
        if(left.val!=right.val)return false;
        return isSymmetric(left.left,right.right) && isSymmetric(left.right,right.left);
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Queue<TreeNode> queue = new LinkedList<>();

        queue.add(root);

        while (!queue.isEmpty()) {
            final int size = queue.size();
            List<Integer> level = new ArrayList<>(size);
            for (int i = 0; i < size; ++i) {
                TreeNode head = queue.poll();
                if (head.left != null) queue.add(head.left);
                if (head.right != null) queue.add(head.right);
                level.add(head.val);
            }
            result.add(level);

        }

        return result;
    }

    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Queue<TreeNode> queue = new LinkedList<>();

        queue.add(root);

        boolean increase = true;
        while (!queue.isEmpty()) {
            final int size = queue.size();
            List<Integer> level = new ArrayList<>(size);
            for (int i = 0; i < size; ++i) {
                TreeNode head = queue.poll();
                if (head.left != null) queue.add(head.left);
                if (head.right != null) queue.add(head.right);
                if (increase)
                    level.add(head.val);
                else
                    level.add(size - i - 1, head.val);
            }
            result.add(level);
            increase = !increase;
        }

        return result;
    }

    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        } else {
            return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
        }
    }

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder == null || inorder == null ||
                preorder.length <= 0 || inorder.length <= 0 || preorder.length != inorder.length) {
            return null;
        }

        return buildTree(preorder, inorder, 0, preorder.length - 1, 0, preorder.length - 1);
    }

    private TreeNode buildTree(int[] preorder, int[] inorder, int prestart, int preend, int instart, int inend) {
        if (prestart > preend) {
            return null;
        }
        TreeNode root = new TreeNode(preorder[prestart]);

        int pos = -1;
        for (int i = instart; i <= inend; ++i) {
            if (inorder[i] == preorder[prestart]) {
                pos = i;
                break;
            }
        }
        if (pos == -1) {
            return root;
        }

        root.left = buildTree(preorder, inorder, prestart + 1, prestart + pos - instart + 1, instart, pos - 1);
        root.right = buildTree(preorder, inorder, prestart + pos - instart + 2, preend, pos + 1, inend);

        return root;
    }

    public TreeNode sortedArrayToBST(int[] nums) {
        if (nums == null) return null;
        return sortedArrayToBST(nums, 0, nums.length - 1);
    }

    private TreeNode sortedArrayToBST(int[] nums, int left, int right) {
        if (left > right) {
            return null;
        }

        int mid = left + (right - left) / 2;
        TreeNode root = new TreeNode(nums[mid]);

        root.left = sortedArrayToBST(nums, left, mid - 1);
        root.right = sortedArrayToBST(nums, mid + 1, right);

        return root;
    }

    int minHeight = -1;
    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        minDepth(root, 0);
        return minHeight;
    }

    private void minDepth(TreeNode root, int height) {
        if (root == null) {
            return;
        }

        if (root.left == null && root.right == null) {
            if (minHeight == -1) {
                minHeight = height + 1;
            } else {
                minHeight = Math.min(height + 1, minHeight);
            }
        }

        if (minHeight != -1 && height >= minHeight) {
            return;
        }

        minDepth(root.left, 1 + height);
        minDepth(root.right, 1 + height);
    }

    public boolean hasPathSum(TreeNode root, int sum) {
        return hasPathSum(root, sum, 0);
    }

    private boolean hasPath = false;
    private boolean hasPathSum(TreeNode root, int sum, int temp) {
        if (hasPath) {
            return true;
        }

        if (root == null) {
            return false;
        }

        if (root.left == null && root.right == null) {
            int newSum = temp + root.val;
            if (newSum == sum) {
                hasPath = true;
                return true;
            } else {
                return false;
            }
        }

        return hasPathSum(root.left, sum, temp + root.val) || hasPathSum(root.right, sum, temp + root.val);
    }

    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> result = new LinkedList<>();

        pathSumDfs(root, new ArrayList<>(), 0, sum, result);

        return result;
    }

    private void pathSumDfs(TreeNode root, List<Integer> back, int temp, int sum, List<List<Integer>> result) {
        if (root == null) {
            return;
        }

        if (root.left == null && root.right == null) {
            if (temp + root.val == sum) {
                List<Integer> one = new ArrayList<>(back);
                one.add(root.val);
                result.add(one);
            }
            return;
        }

        back.add(root.val);

        pathSumDfs(root.left, back, temp + root.val, sum, result);
        pathSumDfs(root.right, back, temp + root.val, sum, result);

        back.remove(back.size() - 1);
    }

    private TreeNode preFlatten = null;
    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }

        final TreeNode left = root.left;
        final TreeNode right = root.right;

        if (preFlatten == null) {
            preFlatten = root;
        } else {
            preFlatten.right = root;
            preFlatten.left = null;
            preFlatten = root;
        }

        flatten(left);
        flatten(right);
    }

    public int numDistinct(String s, String t) {
        int[][] dp = new int[s.length() + 1][t.length() + 1];

        for (int i = 0; i <= t.length(); ++i) {
            dp[0][i] = 1;
        }

        for (int i = 1; i <= s.length(); ++i) {
            for (int j = 1; j <= t.length(); ++j) {
                if (s.charAt(i-1) == t.charAt(j-1)) {
                    dp[i][j] = dp[i-1][j-1] + dp[i][j-1];
                } else {
                    dp[i][j] = dp[i][j-1];
                }
            }
        }

        return dp[s.length()][t.length()];
    }

//    public Node connect(Node root) {
//        if (root == null || root.left == null) {
//            return root;
//        }
//
//        root.left.next = root.right.next;
//        if (root.next != null) {
//            root.right.next = root.next.left;
//        }
//
//        connect(root.left);
//        connect(root.right);
//
//        return root;
//    }

    public Node connect(Node root) {
        if (root == null || (root.left == null && root.right == null)) {
            return root;
        }

        Node next = root.next;
        Node toNext = null;
        while (next != null) {
            if (next.left != null) {
                toNext = next.left;
                break;
            }

            if (next.right != null) {
                toNext = next.right;
                break;
            }

            next = next.next;
        }


        if (root.left != null) {
            root.left.next = root.right != null ? root.right : toNext;
        }
        if (root.right != null) {
            root.right.next = toNext;
        }
        connect(root.left);
        connect(root.right);

        return root;
    }

    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> result = new ArrayList<>(numRows);
        for (int i = 1; i <= numRows; ++i) {
            List<Integer> one = new ArrayList<>(i);
            one.add(0, 1);
            one.add(i-1, 1);

            for (int j = 1; j < i - 1; ++j) {
                final List<Integer> pre = result.get(i-2);
                one.add(j, pre.get(j-1) + pre.get(j));
            }

            result.add(one);
        }

        return result;
    }

    public int minimumTotal(List<List<Integer>> triangle) {
        if (triangle == null || triangle.size() <= 0) {
            return 0;
        }

        final int rows = triangle.size();
        int[] dp = new int[rows];

        dp[0] = triangle.get(0).get(0);

        if (rows == 1) {
            return dp[0];
        }

        int result = Integer.MAX_VALUE;
        for (int i = 1; i < rows; ++i) {
            for (int j = i; j >= 0; --j) {
                if (j == i) {
                    dp[j] = dp[j-1] + triangle.get(i).get(j);
                } else if (j == 0) {
                    dp[j] = dp[0] + triangle.get(i).get(j);
                } else {
                    dp[j] = Math.min(dp[j], dp[j-1]) + triangle.get(i).get(j);
                }
                if (i == rows - 1) {
                    result = Math.min(result, dp[j]);
                }
            }
        }
        return result;
    }

    public int maxProfit(int[] prices) {
        if (prices == null || prices.length <= 1) {
            return 0;
        }

        int result = 0;
        int min = prices[0];
        for (int i = 1; i < prices.length; ++i) {
            result = Math.max(result, prices[i] - min);
            min = Math.min(min, prices[i]);
        }

        return result;
    }

    public int maxProfitII(int[] prices) {
        if (prices == null || prices.length <= 1) {
            return 0;
        }

        int result = 0;
        for (int i = 0; i < prices.length - 1; ++i) {
            result += Math.max(0, prices[i+1] - prices[i] );
        }

        return result;
    }

    public int maxProfitIV(int k, int[] prices) {
        if (prices == null || prices.length < 2) {
            return 0;
        }

        if (k >= prices.length / 2) {
            return maxProfitII(prices);
        }

        int[][] local = new int[prices.length][k + 1];
        int[][] global = new int[prices.length][k + 1];

        for (int i = 1; i < prices.length; ++i) {
            final int diff = prices[i] - prices[i - 1];

            for (int j = 1; j <= k; ++j) {
                local[i][j] = Math.max(global[i-1][j-1] + Math.max(diff, 0), local[i-1][j] + diff);
                global[i][j] = Math.max(global[i-1][j], local[i][j]);
            }
        }

        return global[prices.length - 1][k];
    }

    private int maxPathSumResult = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        if (root == null) {
            return 0;
        }

        int leftMax = maxPathSum(root.left);
        int rightMax = maxPathSum(root.right);

        leftMax = Math.max(0, leftMax);
        rightMax = Math.max(0,rightMax);

        maxPathSumResult = Math.max(maxPathSumResult, root.val + leftMax + rightMax);

        return root.val + Math.max(0, Math.max(leftMax, rightMax));
    }

    public boolean isPalindrome(String s) {
        if (s == null || s.length() <= 0) {
            return true;
        }

        int left = 0;
        int right = s.length() - 1;

        while (left <= right) {
            while (left <= right && !Character.isLetterOrDigit(s.charAt(left)) ) {
                left++;
            }

            while (left <= right && !Character.isLetterOrDigit(s.charAt(right))) {
                right--;
            }

            if (left <= right) {
                if (Character.toLowerCase(s.charAt(left)) != Character.toLowerCase(s.charAt(right))) {
                    return false;
                } else {
                    left++;
                    right--;
                }
            }
        }

        return true;
    }

    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        List<List<String>> result = new LinkedList<>();
        List<String> backstrace = new LinkedList<>();
        backstrace.add(beginWord);

        Set<String> wordSet = new HashSet<>(wordList);
        if (!wordSet.contains(endWord)) {
            return result;
        }

        findLadders(beginWord, endWord, wordSet, backstrace, result);

        return result;
    }

    private void findLadders(String beginWord, String endWord, Set<String> wordList, List<String> backstrace, List<List<String>> result) {

        if (beginWord.equals(endWord)) {
            List<String> one = new ArrayList<>(backstrace);
            result.add(one);
            return;
        }

        if (wordList.isEmpty()) {
            return;
        }


        for (int i = 0; i < beginWord.length(); ++i) {
            for (char j = 'a'; j <= 'z'; ++j) {
                if (beginWord.charAt(i) == j) {
                    continue;
                }

                final String newBegin = beginWord.substring(0, i) + j + beginWord.substring(i+1);
                if (!wordList.contains(newBegin)) {
                    continue;
                }

                backstrace.add(newBegin);

                wordList.remove(newBegin);
                findLadders(newBegin, endWord, wordList, backstrace, result);

                wordList.add(newBegin);
                backstrace.remove(backstrace.size() - 1);
            }
        }
    }

    private static class LadderNode {
        String word;
        LadderNode parent;
    }

    public List<List<String>> findLaddersShortestLength(String beginWord, String endWord, List<String> wordList) {
        List<List<String>> result = new LinkedList<>();
        Map<String, Integer> visited = new HashMap<>();
        visited.put(beginWord, 0);

        Set<String> wordSet = new HashSet<>(wordList);
        if (!wordSet.contains(endWord)) {
            return result;
        }

        Queue<LadderNode> queue = new LinkedList<>();
        LadderNode root = new LadderNode();
        root.word = beginWord;

        queue.add(root);

        boolean find = false;
        boolean oneFind = false;

        int level = 0;
        while (!queue.isEmpty() && !find) {
            final int size = queue.size();
            level++;

            for (int k = 0; k < size; ++k) {
                LadderNode head = queue.poll();

                final String headStr = head.word;

                oneFind = false;

                for (int i = 0; i < headStr.length(); ++i) {
                    for (char j = 'a'; j <= 'z'; ++j) {
                        if (headStr.charAt(i) == j) {
                            continue;
                        }

                        final String newBegin = headStr.substring(0, i) + j + headStr.substring(i + 1);

                        if (newBegin.equals(endWord)) {
                            find = true;
                            oneFind = true;
                            result.add(getWordLadderList(endWord, head));
                            break;
                        }

                        if (!wordSet.contains(newBegin)) {
                            continue;
                        }

                        if (!find) {
                            final Integer oldLevel = visited.get(newBegin);
                            if (oldLevel == null || oldLevel >= level) {
                                visited.put(newBegin, level);

                                wordSet.add(newBegin);

                                LadderNode newNode = new LadderNode();
                                newNode.word = newBegin;
                                newNode.parent = head;

                                queue.add(newNode);
                            }
                        }

                    }

                    if (oneFind) {
                        break;
                    }
                }

            }
        }
        return result;
    }

    private List<String> getWordLadderList(String endWord, LadderNode last) {
        LinkedList<String> result = new LinkedList<>();
        result.addFirst(endWord);

        while (last != null) {
            result.addFirst(last.word);
            last = last.parent;
        }

        return result;
    }

    public int longestConsecutive(int[] nums) {
        Set<Integer> num_set = new HashSet<Integer>();
        for (int num : nums) {
            num_set.add(num);
        }

        int longestStreak = 0;

        for (int num : num_set) {
            if (!num_set.contains(num-1)) {
                int currentNum = num;
                int currentStreak = 1;

                while (num_set.contains(currentNum+1)) {
                    currentNum += 1;
                    currentStreak += 1;
                }

                longestStreak = Math.max(longestStreak, currentStreak);
            }
        }

        return longestStreak;
    }

    public int sumNumbers(TreeNode root) {
        sumNumbers(root, 0);
        return sumTotal;
    }

    private int sumTotal = 0;
    private void sumNumbers(TreeNode root, int sum) {
        if (root == null) {
            return;
        }

        if (root.left == null && root.right == null) {
            sumTotal += sum * 10 + root.val;
            return;
        }

        sumNumbers(root.left, sum * 10 + root.val);
        sumNumbers(root.right, sum * 10 + root.val);
    }

    public void solve(char[][] board) {
        if (board == null || board.length <= 0 || board[0] == null || board[0].length <= 0) {
            return;
        }

        for (int i = 0; i < board[0].length; ++i) {
            if (board[0][i] == 'O')
                solveDfs(board, 0, i);
        }
        for (int i = 0; i < board[0].length; ++i) {
            if (board[board.length - 1][i] == 'O')
                solveDfs(board, board.length - 1, i);
        }
        for (int i = 0; i < board.length; ++i) {
            if (board[i][0] == 'O')
                solveDfs(board, i, 0);
        }
        for (int i = 0; i < board.length; ++i) {
            if (board[i][board[0].length-1] == 'O')
                solveDfs(board, i, board[0].length-1);
        }

        for (int i = 0; i < board.length; ++i) {
            for (int j = 0; j < board[0].length; ++j) {
                if (board[i][j] == '.') {
                    board[i][j] = 'O';
                } else if (board[i][j] != 'X') {
                    board[i][j] = 'X';
                }
            }
        }
    }

    private void solveDfs(char[][] board, int x, int y) {
        board[x][y] = '.';

        if (x - 1 >= 0 && board[x-1][y] == 'O') {
            solveDfs(board, x - 1, y);
        }
        if (x + 1 < board.length && board[x+1][y] == 'O') {
            solveDfs(board, x + 1, y);
        }
        if (y - 1 >= 0 && board[x][y-1] == 'O') {
            solveDfs(board, x, y - 1);
        }
        if (y + 1 < board[0].length && board[x][y+1] == 'O') {
            solveDfs(board, x, y + 1);
        }
    }

    public List<List<String>> partition(String s) {
        Map<Integer, List<List<String>>> dp = new HashMap<>();
        return partition(s, 0, dp);
    }

    private List<List<String>> partition(String s, int index, Map<Integer,List<List<String>>> dp) {

        List<List<String>> result = new LinkedList<>();

        if (index >= s.length()) {
            result.add(null);
            return result;
        }

        List<List<String>> cache = dp.get(index);
        if (cache != null) {
            return cache;
        }

        if (index == s.length() - 1) {
            List<String> one = new ArrayList<>(1);
            one.add(String.valueOf(s.charAt(index)));
            result.add(one);
            dp.put(index, result);

            return result;
        }

        for (int i = index; i < s.length(); ++i) {
            if (isPalindrome(s, index, i)) {
                final String pre = s.substring(index, i+1);
                List<List<String>> next = partition(s, i + 1, dp);
                for (List<String> one : next) {
                    List<String> newOne = new LinkedList<>();
                    newOne.add(pre);
                    if (one != null) {
                        newOne.addAll(one);
                    }
                    result.add(newOne);
                }
            }
        }

        dp.put(index, result);

        return result;
    }

    private boolean isPalindrome(String s, int start, int end) {
        while (start <= end) {
            if (s.charAt(start++) != s.charAt(end--)) {
                return false;
            }
        }
        return true;
    }

    public int minCut(String s) {
        boolean[][] palindromes = new boolean[s.length()][s.length()];

        for (int i = 0; i < s.length(); ++i) {
            for (int j = 0; i - j >= 0 && i + j < s.length(); ++j) {
                if (s.charAt(i-j) == s.charAt(i+j)) {
                    palindromes[i-j][i+j] = true;
                } else {
                    break;
                }
            }
        }

        for (int i = 0; i < s.length() - 1; ++i) {
            for (int j = 0; i - j >= 0 && i + 1 + j < s.length(); ++j) {
                if (s.charAt(i-j) == s.charAt(i + 1 + j)) {
                    palindromes[i-j][i+1 + j] = true;
                } else {
                    break;
                }
            }
        }

        int[][] dp = new int[s.length()][s.length()];
        for (int i = 0; i < s.length(); ++i) {
            for (int j = i; j < s.length(); ++j) {
                dp[i][j] = -1;
            }
        }

        return minCut(s, 0, s.length() - 1, palindromes, dp);
    }

    private int minCut(String s, int start, int end, boolean[][] palindromes, int[][] dp) {
        if (start >= end) {
            return 0;
        }

        if (dp[start][end] != -1) {
            return dp[start][end];
        }

        if (palindromes[start][end]) {
            dp[start][end] = 0;
            return 0;
        }

        int result = Integer.MAX_VALUE;
        for (int i = start; i < end; ++i) {
            result = Math.min(result, 1 + minCut(s, start, i, palindromes, dp) + minCut(s, i + 1, end, palindromes, dp));
        }

        dp[start][end] = result;

        return result;
    }

    public GraphNode cloneGraph(GraphNode node) {
        return cloneGraph(node, new HashMap<>());
    }

    public GraphNode cloneGraph(GraphNode node, Map<GraphNode, GraphNode> visited) {
        if (node == null) {
            return null;
        }

        GraphNode newNode = new GraphNode();
        newNode.val = node.val;

        if (node.neighbors == null) {
            return newNode;
        }


        visited.put(node, newNode);
        List<GraphNode> list = new ArrayList<>(node.neighbors.size());
        node.neighbors = list;

        for (GraphNode neighbor : node.neighbors) {
            final GraphNode vistedNode = visited.get(neighbor);
            if (vistedNode != null) {
                list.add(vistedNode);
            } else {
                GraphNode newNeighbor = cloneGraph(neighbor, visited);
                list.add(newNeighbor);
            }
        }

        visited.remove(node);

        return newNode;
    }

    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }

        if (head.next == head) {
            return true;
        }

        ListNode slow = head;
        ListNode fast = head;

        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;

            if (slow == fast) {
                return true;
            }
        }

        return false;
    }

    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null) {
            return null;
        }

        ListNode slow = head;
        ListNode fast = head;

        ListNode meet = null;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;

            if (slow == fast) {
                meet = slow;
                break;
            }
        }

        if (meet == null) {
            return null;
        }

        slow = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }

        return slow;
    }

    public void reorderList(ListNode head) {
        if (head == null || head.next == null) {
            return;
        }

        ListNode slow = head;
        ListNode fast = head;

        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }

        ListNode meet = slow;
        ListNode right = meet.next;
        meet.next = null;

        ListNode pre = null;
        ListNode cur = right;

        while (cur != null) {
            final ListNode next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }

        ListNode insert = head;
        ListNode toInsert = pre;

        while (toInsert != null) {
            final ListNode next = insert.next;
            final ListNode toInsertNext = toInsert.next;
            insert.next = toInsert;
            toInsert.next = next;
            toInsert = toInsertNext;
            insert = next;
        }
    }

    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();

        Stack<TreeNode> stack = new Stack<>();

        TreeNode cur = root;

        while (cur != null || stack.isEmpty()) {
            while (cur != null) {
                stack.push(cur);
                result.add(cur.val);
                cur = cur.left;
            }

            cur = stack.pop().right;
        }

        return result;
    }

    public List<Integer> preorderTraversalII(TreeNode root) {
        List<Integer> result = new ArrayList<>();

        Stack<TreeNode> stack = new Stack<>();

        if (root != null) {
            stack.push(root);
        }

        while (!stack.isEmpty()) {
            TreeNode top = stack.pop();
            result.add(top.val);

            if (top.right != null) {
                stack.push(top.right);
            }
            if (top.left != null) {
                stack.push(top.left);
            }
        }

        return result;
    }

    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();

        Stack<TreeNode> stack = new Stack<>();

        TreeNode cur = root;

        Set<TreeNode> toFirstPopNodes = new HashSet<>();

        while (cur != null || !stack.isEmpty()) {
            while (cur != null) {
                stack.push(cur);
                cur = cur.left;
            }

            TreeNode top = stack.peek();

            if (toFirstPopNodes.contains(top)) {
                result.add(top.val);
                stack.pop();
            } else {
                toFirstPopNodes.add(top);
                cur = top.right;
            }
        }

        return result;
    }

    public List<Integer> postorderTraversalII(TreeNode root) {
        List<Integer> result = new ArrayList<>();

        Stack<TreeNode> stack = new Stack<>();

        if (root != null) {
            stack.push(root);
        }

        while (!stack.isEmpty()) {
            TreeNode top = stack.pop();
            result.add(0, top.val);

            if (top.left != null) {
                stack.push(top.left);
            }
            if (top.right != null) {
                stack.push(top.right);
            }
        }

        return result;
    }

    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) {
            return null;
        }

        ListNode tempa = headA;
        int lengtha = 1;
        ListNode tempb = headB;
        int lengthb = 1;

        while (tempa.next != null) {
            tempa = tempa.next;
            lengtha++;
        }

        while (tempb.next != null) {
            tempb = tempb.next;
            lengthb++;
        }

        if (tempa != tempb) {
            return null;
        }

        tempa = headA;
        tempb = headB;
        int la = lengtha;
        int lb = lengthb;
        while (la > lb) {
            tempa = tempa.next;
            la--;
        }

        while (lb > la) {
            tempb = tempb.next;
            lb--;
        }

        while (tempa != tempb) {
            tempa = tempa.next;
            tempb = tempb.next;
        }

        return tempa;

    }

    public int findPeakElement(int[] nums) {
        int n = nums.length;
        if(0 == n) return -1;
        if(1 == n) return 0;
        if(2 == n) return nums[0] > nums[1] ? 0 : 1;
        int low = 0, high = n - 1;
        while(low + 2 <= high)
        {
            int mid = low + (high - low) / 2;
            if(nums[mid - 1] < nums[mid] && nums[mid] > nums[mid + 1])
                return mid;
            if(nums[mid - 1] > nums[mid])
                high = mid;
            else
                low = mid;
        }
        return nums[low] > nums[high] ? low : high;
    }

    public int compareVersion(String version1, String version2) {
        final int len1 = version1.length();
        final int len2 = version2.length();

        int i = 0;
        int j = 0;

        int v1 = 0;
        int v2 = 0;

        while (i < len1 || j < len2) {
            while (i < len1 && version1.charAt(i) != '.') {
                v1 = v1 * 10 + (version1.charAt(i) - '0');
                i++;
            }

            while (j < len2 && version2.charAt(j) != '.') {
                v2 = v2 * 10 + (version2.charAt(j) - '0');
            }

            if (v1 > v2) {
                return 1;
            } else if (v1 < v2) {
                return -1;
            } else {
                v1 = v2 = 0;
            }
            i++;
            j++;
        }

        return 0;
    }

    private String fractionToDecimalLong(long numerator, long denominator) {
        boolean nagative = numerator * denominator < 0;

        numerator = Math.abs(numerator);
        denominator = Math.abs(denominator);

        long maxCommon = gcd(numerator, denominator);

        numerator /= maxCommon;
        denominator /= maxCommon;

        StringBuilder builder = new StringBuilder();
        if (nagative) {
            builder.append('-');
        }
        if (denominator == 1) {
            return builder.append(numerator).toString();
        }

        builder.append(numerator / denominator).append(".");

        long left = numerator % denominator;

        Map<Long, Integer> map = new HashMap<>();

        map.put(left, builder.length() - 1);

        while (left != 0) {
            long shang = left * 10 / denominator;
            left = left * 10 - denominator * shang;
            builder.append(shang);

            if (map.containsKey(left)) {
                int index = map.get(left);
                builder.insert(index + 1, '(');

                builder.append(')');
                break;
            } else {
                map.put(left, builder.length() - 1);
            }
        }

        return builder.toString();
    }
    public String fractionToDecimal(int numerator, int denominator) {
        if (numerator == denominator) {
            return "1";
        }
        if (numerator == 0) {
            return "0";
        }


        return fractionToDecimalLong(numerator, denominator);
    }

    private long gcd(long a, long b) {
        if (a < b) {
            return gcd(b, a);
        }

        if (b == 0) {
            return a;
        }

        return gcd(b, a % b);
    }

    public int[] twoSumII(int[] numbers, int target) {
        int[] result = new int[2];

        int left = 0;
        int right = numbers.length - 1;

        while (left < right) {
            final int sum = numbers[left] + numbers[right];
            if (sum == target) {
                result[0] = left;
                result[1] = right;
                break;
            } else if (sum < target) {
                left++;
            } else {
                right--;
            }
        }

        return result;
    }

    public String convertToTitle(int n) {
        StringBuilder builder = new StringBuilder();

        while (n > 26) {
            int yu = n % 26;
            builder.append((char) (yu == 0 ? 'Z' : ('A' - 1 + n % 26)));
            n /= 26;
            if (yu == 0) {
                n--;
            }
        }

        builder.append('A' - 1 + n);

        return builder.reverse().toString();
    }

    public int majorityElement(int[] nums) {
        int times = 0;
        int majority = -1;

        for (int num : nums) {
            if (times == 0) {
                majority = num;
                times = 1;
            } else if (majority == num) {
                times++;
            } else {
                times--;
            }
        }
        return majority;
    }

    public void rotate(int[] nums, int k) {
        rotate(nums, 0, nums.length - 1 - k);
        rotate(nums, nums.length - k, nums.length - 1);
        rotate(nums, 0, nums.length - 1);
    }

    private void rotate(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
        }
    }

    public int hammingWeight(int n) {
        int result = 0;

        while (n != 0) {
            n = n & (n - 1);
            result++;
        }
        return result;
    }

    public int rob(int[] nums) {
        if (nums == null || nums.length <= 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }

        int prepre = nums[0];
        int pre = Math.max(prepre, nums[1]);
        int max = pre;

        for (int i = 2; i < nums.length; ++i) {
            int cur = Math.max(nums[i] + prepre, pre);
            max = Math.max(cur, max);

            prepre = pre;
            pre = cur;
        }

        return max;
    }

    public int rangeBitwiseAnd(int m, int n) {
        int result = 0;

        if (m == n) {
            return m;
        }

        int i = 0;
        while (n != 0) {
            if (!hasEvenNum(m, n)) {
                result |= 1 << i;
            }
            m >>= 1;
            n >>= 1;
            i++;
        }

        return result;
    }

    private boolean hasEvenNum(int start, int end) {
        if (start == end) {
            return (start & 1) == 0;
        } else {
            return true;
        }
    }

    public boolean isHappy(int n) {
        return isHappy(n, new HashSet<>());
    }

    private boolean isHappy(int n, Set<Integer> vistied) {
        if (n == 1) {
            return true;
        }

        if (!vistied.add(n)) {
            return false;
        }

        char[] split = String.valueOf(n).toCharArray();

        int m = 0;
        for (int i = 0; i < split.length; ++i) {
            m += (split[i] - '0') * (split[i] - '0');
        }

        return isHappy(m, vistied);
    }

    public ListNode removeElements(ListNode head, int val) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;

        ListNode pre = dummy;
        ListNode cur = head;

        while (cur != null) {
            if (cur.val == val) {
                pre.next = cur.next;
                cur = cur.next;
            } else {
                pre = cur;
                cur = cur.next;
            }
        }

        return dummy.next;
    }

    public boolean isIsomorphic(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }

        int[] map = new int[26];
        Arrays.fill(map, -1);
        for (int i = 0; i < s.length(); ++i) {
            final char cs = s.charAt(i);
            final char ct = t.charAt(i);

            if (cs == ct) {
                if (map[cs] != -1) return false;
                else map[cs] = ct;
            } else if (map[cs] == -1 ){
                if (map[ct] == -1) map[cs] = ct;
                else return false;
            } else if (map[cs] != ct){
                return false;
            }
        }

        return true;
    }

    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null)
            return head;

        ListNode newHead = reverseList(head.next);
        head.next.next = head;
        head.next = null;

        return newHead;
    }

    public boolean canFinish(int numCourses, int[][] prerequisites) {
        if (numCourses == 0) {
            return true;
        }
        if (prerequisites == null || prerequisites.length <= 0) {
            return true;
        }

        int[] degree = new int[numCourses];
        boolean[] visited = new boolean[numCourses];

        List<Integer>[] graph = new List[numCourses];

        for (int[] edge : prerequisites) {
            final int from = edge[1];
            final int to = edge[0];

            degree[to]++;

            if (graph[from] == null) {
                graph[from] = new ArrayList<>();
            }
            graph[from].add(to);
        }

        boolean[] backstrace = new boolean[numCourses];

        for (int i = 0; i < degree.length; ++i) {
            if (degree[i] == 0) {
                if (!courseDfs(visited, i, graph, backstrace)) {
                    return false;
                }
            }
        }

        for (boolean visit : visited) {
            if (!visit) {
                return false;
            }
        }

        return true;
    }

    private boolean courseDfs(boolean[] result, int index, List<Integer>[] graph, boolean[] visited) {
        result[index] = true;
        visited[index] = true;

        List<Integer> tos = graph[index];

        boolean res = true;
        if (tos != null) {
            for (int to : tos) {
                if (visited[to]) {
                    res = false;
                    break;
                }

                if (!result[to]) {
                    if (!courseDfs(result, to, graph, visited)) {
                        res = false;
                        break;
                    }
                }
            }
        }

        visited[index] = false;
        return res;
    }

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        int[] result = new int[numCourses];
        int curCnt = 0;

        int[] degree = new int[numCourses];
        List<Integer>[] graph = new List[numCourses];

        for (int[] edge : prerequisites) {
            final int from = edge[1];
            final int to = edge[0];

            degree[to]++;

            if (graph[from] == null) {
                graph[from] = new ArrayList<>();
            }
            graph[from].add(to);
        }

        Queue<Integer> queue = new LinkedList<>();

        for (int i = 0; i < numCourses; ++i) {
            if (degree[i] == 0) {
                queue.add(i);
            }
        }

        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i= 0; i < size; ++i) {
                final int from = queue.poll();
                result[curCnt++] = from;
                List<Integer> tos = graph[from];
                if (tos != null) {
                    for (int to : tos) {
                        if (--degree[to] == 0) {
                            queue.add(to);
                        }
                    }
                }
            }
        }

        return curCnt == numCourses ? result : new int[0];
    }

    private Trie mWordTrieRoot = null;
    private List<String> mFindWordResult;
    public List<String> findWords(char[][] board, String[] words) {
        mWordTrieRoot = new Trie();
        for (String word : words) {
            addWord(mWordTrieRoot, word);
        }

        mFindWordResult = new ArrayList<>();

        for (int i = 0; i < board.length; ++i) {
            for (int j = 0; j < board[0].length; ++j) {
                findWordDfs(board, i, j, mWordTrieRoot);
            }
        }

        return mFindWordResult;
    }

    private void findWordDfs(char[][] board, int x, int y, Trie root) {
        if (root == null) {
            return;
        }

        final int idx = board[x][y] - 'a';
        final Trie node = root.next[idx];
        if (node == null) {
            return;
        }

        board[x][y] = 0;
        if (node.word) {
            mFindWordResult.add(node.wordText);
            node.word = false;
        }

        if (x - 1 >= 0 && board[x-1][y] != 0)
            findWordDfs(board, x - 1, y, node);
        if (y - 1 >= 0 && board[x][y-1] != 0)
            findWordDfs(board, x, y-1, node);
        if (x + 1 < board.length && board[x + 1][y] != 0)
            findWordDfs(board, x + 1, y, node);
        if (y + 1 < board[0].length && board[x][y+1] != 0)
            findWordDfs(board, x, y + 1, node);

        board[x][y] = (char) ('a' + idx);
    }

    public int robII(int[] nums) {
        if (nums == null || nums.length <= 0) {
            return 0;
        }

        if (nums.length  == 1)
            return nums[0];
        // 假设第一个可以取，那只能取到倒数第二个
        int prepre = nums[0];
        int pre = Math.max(nums[1], prepre);
        int max = Math.max(prepre, pre);
        for (int i = 2; i < nums.length - 1; ++i) {
            int curMax = Math.max(nums[i] + prepre, pre);
            max = Math.max(curMax, max);

            prepre = pre;
            pre = curMax;
        }

        prepre = 0;
        pre = nums[1];
        for (int i = 2; i < nums.length; ++i) {
            int curMax = Math.max(nums[i] + prepre, pre);
            max = Math.max(curMax, max);

            prepre = pre;
            pre = curMax;
        }

        return max;
    }


    private static class Trie {
        private boolean word = false;
        private String wordText;
        private final Trie[] next = new Trie[26];
    }

    private void addWord(Trie root, String word) {
        if (root == null)
            return;

        Trie node = root;
        for (int i = 0; i < word.length(); ++i) {
            final int idx = word.charAt(i) - 'a';
            if (node.next[idx] == null) {
                node.next[idx] = new Trie();
            }
            node = node.next[idx];
        }
        node.word = true;
        node.wordText = word;
    }

    public String shortestPalindrome(String s) {
        return "";
    }

    private int getLongestPalindrome(String s) {
        return 0;
    }

    public int findKthLargest(int[] nums, int k) {
        return findKthLargest(nums, nums.length - k, 0, nums.length - 1);
    }

    private int findKthLargest(int[] nums, int k, int start, int end) {
        int partition = quickPartition(nums, start, end);
        if (partition == k) {
            return nums[partition];
        } else if (partition > k) {
            return findKthLargest(nums, k, start, partition - 1);
        } else {
            return findKthLargest(nums, k, partition + 1, end);
        }
    }

    private int quickPartition(int[] nums, int start, int end) {
        if (start >= end) {
            return start;
        }
        int pivotIndex = new Random().nextInt(end - start) + start;

        int base = nums[start];
        nums[start] = nums[pivotIndex];
        nums[pivotIndex] = base;

        int i = start;
        int j = end;

        while (i < j) {
            while (i < j && nums[j] >= base) j--;
            while (i < j && nums[i] <= base) i++;
            if (i < j) {
                int temp = nums[i]; nums[i] = nums[j]; nums[j] = temp;
            }
        }
        if (i != start) {
            int temp = nums[i]; nums[i] = base; nums[start] = temp;
        }

        return i;
    }


    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> result = new ArrayList<>();
        combinationSum3Dfs(k, 1, n, 0, new ArrayList<>(), result);
        return result;
    }

    private void combinationSum3Dfs(int k, int idx, int n, int sum, List<Integer> backstrace, List<List<Integer>> result) {
        if (backstrace.size() > k) {
            return;
        }

        if (backstrace.size() == k) {
            if (sum == n) {
                result.add(new ArrayList<>(backstrace));
            }
            return;
        }

        for (int i = idx; i <= 9; ++i) {
            backstrace.add(i);
            combinationSum3Dfs(k, i + 1, n, sum + i, backstrace, result);
            backstrace.remove(backstrace.size() - 1);
        }
    }

    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Map<Integer, Integer> pos = new HashMap<>();
        for (int i = 0; i < nums.length; ++ i) {
            int num = nums[i];
            Integer old = pos.get(num);
            if (old != null && i - old <= k) {
                return true;
            } else {
                pos.put(num, i);
            }
        }

        return false;
    }

    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        TreeMap<Long, Integer> map = new TreeMap<>();

        for (int i = 0; i < nums.length; ++i) {
            final long num = nums[i];
            final long min = num - t;
            final long max = num + t;

            Map.Entry<Long, Integer> floorEntry = map.floorEntry(max);
            if (floorEntry != null && floorEntry.getKey() >= min && i - floorEntry.getValue() <= k) {
                return true;
            }

            Map.Entry<Long, Integer> ceilEntry = map.ceilingEntry(min);
            if (ceilEntry != null && ceilEntry.getKey() <= max && i - ceilEntry.getValue() <= k) {
                return true;
            }

            map.put(num, i);
        }

        return false;
    }

    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }

        final TreeNode left = root.left;
        final TreeNode right = root.right;

        root.right = left;
        root.left = right;

        invertTree(left);
        invertTree(right);

        return root;
    }
}
