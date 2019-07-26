import javafx.util.Pair;

import java.util.*;
import java.util.function.Function;

public class SolutionHard {

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length;
        int n = nums2.length;

        final boolean isEven = ((m + n) & 1) == 0;
        if (m == 0) {
            return findMedianSortedArrays(nums2);
        } else if (n == 0) {
            return findMedianSortedArrays(nums1);
        }

        if (m > n) {
            int[] temp = nums1;
            nums1 = nums2;
            nums2 = temp;

            int tempCnt = m;
            m = n;
            n = tempCnt;
        }

        // 从最小的来看
        int iMin = 0;
        int iMax = m;

        while (iMin <= iMax) {
            int i = iMin + (iMax - iMin) / 2;

            int j = (m + n + 1) / 2 - i;

            final int iPre = i == 0 ? Integer.MIN_VALUE : nums1[i-1];
            final int jPre = j == 0 ? Integer.MIN_VALUE : nums2[j-1];
            final int iCur = i == m ? Integer.MAX_VALUE : nums1[i];
            final int jCur = j == n ? Integer.MAX_VALUE : nums2[j];
            int maxLeft = Math.max(iPre, jPre);
            int minRight = Math.min(iCur, jCur);

            if (maxLeft <= minRight) {
                return isEven ? (maxLeft + minRight) / 2.0 : maxLeft;
            } else if (iCur < jCur){
                iMin = i + 1;
            } else {
                iMax = i - 1;
            }
        }


        return 0;
    }

    private double findMedianSortedArrays(int[] nums) {
        final int length = nums.length;

        return (length & 1) == 0 ? (nums[(length >> 1) - 1] + nums[(length >> 1)]) / 2.0 :
                nums[length >> 1];
    }

    public ListNode reverseKGroup(ListNode head, int k) {
        if (k == 1 || head == null) {
            return head;
        }

        final int length = listNodeLength(head);

        if (length < k) {
            return head;
        }

        final int maxReverseLength = length / k * k;

        int i = 0;

        ListNode pre = null;
        ListNode cur = head;
        ListNode next;
        ListNode curTail = null;
        ListNode newHead = null;
        ListNode lastTail = null;

        while (cur != null && i < maxReverseLength) {
            if (curTail == null) {
                curTail = cur;
            }
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur= next;
            i++;

            if (i % k == 0) {
                if (newHead == null) {
                    newHead = pre;
                }

                if (lastTail != null) {
                    lastTail.next = pre;
                }
                lastTail = curTail;

                curTail = null;
                pre = null;
            }
        }

        if (maxReverseLength < length) {
            if (lastTail != null) {
                lastTail.next = cur;
            }
        }

        return newHead;
    }

    public static int listNodeLength(ListNode head) {
        int length=0;
        while (head != null) {
            length++;
            head = head.next;
        }
        return length;
    }

    public ListNode listNodeFromArray(int[] nums) {
        return fromArrayHelper(nums, 0);
    }

    private ListNode fromArrayHelper(int[] nums, int idx) {
        if (idx >= nums.length) {
            return null;
        } else {
            ListNode head = new ListNode(nums[idx]);
            head.next = fromArrayHelper(nums, idx + 1);
            return head;
        }
    }

    public void printListNode(ListNode head) {
        while (head != null) {
            System.out.print(head.val + ",");
            head = head.next;
        }
    }

    public int trap(int[] height) {
        if (height == null || height.length <= 1) {
            return 0;
        }

        int left = 0;
        int right = height.length - 1;

        int result = 0;
        while (left < right) {
            final int hl = height[left];
            final int hr = height[right];

            if (hl <= hr) {
                while (++left < right && height[left] <= hl) {
                    result += hl - height[left];
                }
            } else {
                while (left < --right && height[right] <= hr) {
                    result += hr - height[right];
                }
            }
        }

        return result;
    }

    private boolean isDigit(char c) {
        return c >= '0' && c <= '9';
    }
    public boolean isNumber(String s) {
        if (s == null || s.length() <= 0) {
            return false;
        }

        s = s.trim();

        if (s.length() <= 0) {
            return false;
        }

        int pointIndex = -1;
        int eIndex = -1;
        boolean eHasSign = false;
        boolean eHasNum = false;
        // point之前有没有数字
        // point之后有没有数字
        for (int i = 0; i < s.length(); ++i) {
            final char c = s.charAt(i);

            if (i == 0) {
                if (c == '.' || c == '+' || c == '-') {
                    if (s.length() == 1) {
                        return false;
                    } else if (c == '.') {
                        pointIndex = i;

                        if (!(pointIndex - 1 >= 0 && isDigit(s.charAt(pointIndex-1))) &&
                                !(pointIndex + 1 <= s.length() - 1 && isDigit(s.charAt(pointIndex+1)))) {
                            return false;
                        }
                    }
                } else if (!isDigit(c)) {
                    return false;
                }

                continue;
            }

            if (c == '.') {
                if (eIndex == -1) {
                    if (pointIndex != -1)
                        return false;
                    else
                        pointIndex = i;

                    if (!(pointIndex - 1 >= 0 && isDigit(s.charAt(pointIndex-1))) &&
                            !(pointIndex + 1 <= s.length() - 1 && isDigit(s.charAt(pointIndex+1)))) {
                        return false;
                    }
                } else {
                    return false;
                }
            } else if (isDigit(c)) {
                if (eIndex != -1) {
                    eHasNum = true;
                }
            } else if (c == 'e') {
                if (eIndex != -1 || i == s.length() - 1) {
                    return false;
                }
                eIndex = i;

                if (pointIndex == - 1 && (i - 1 < 0 || !isDigit(s.charAt(i-1)))){
                    return false;
                }
            } else if (c == '+' || c == '-') {
                if (eIndex == -1) {
                    return false;
                } else if (eHasSign) {
                    return false;
                } else if (i == eIndex + 1){
                    eHasSign = true;
                } else {
                    return false;
                }

            } else {
                return false;
            }
        }

        return eIndex == -1 || eHasNum;
    }

    public List<String> fullJustify(String[] words, int maxWidth) {
        List<String> result = new ArrayList<>();

        if (words == null || words.length <= 0) {
            return result;
        }

        int curLineStartIndex = 0;
        int curLineLength = 0;
        int curLineWordLength = 0;
        for (int i = 0; i < words.length; ++i) {
            final String word = words[i];
            if (curLineLength == 0) {
                curLineStartIndex = i;
                curLineLength = word.length();
                curLineWordLength = curLineLength;
            } else if (curLineLength + 1 + word.length() <= maxWidth) {
                curLineLength += 1 + word.length();
                curLineWordLength += word.length();
            } else {
                 // 需要换行了
                String line = words[curLineStartIndex];
                final int endIdx = i - 1;
                final int lineWordsCnt = endIdx - curLineStartIndex + 1;
                if (lineWordsCnt == 1) {
                    for (int j = 1; j <= maxWidth - curLineWordLength; ++j) {
                        line += ' ';
                    }
                } else {
                    final int wordInternals = lineWordsCnt - 1;
                    final int wordSpaceInterval = (maxWidth - curLineWordLength) / wordInternals;
                    final int wordSpaceIntervalLeft = (maxWidth - curLineWordLength) % wordInternals;

                    for (int j = 1; j <= wordInternals; j++) {
                        for (int k = 1; k <= wordSpaceInterval; ++k) {
                            line += ' ';
                        }
                        if (j <= wordSpaceIntervalLeft) {
                            line += ' ';
                        }

                        line += words[curLineStartIndex + j];
                    }
                }

                result.add(line);

                curLineLength = 0;
                --i;
            }
        }

        String line = words[curLineStartIndex];
        // 处理最后一行
        for (int i = curLineStartIndex + 1; i < words.length; ++i) {
            line += ' ';
            line += words[i];
        }

        while (line.length() < maxWidth) {
            line += ' ';
        }

        result.add(line);

        return result;
    }

    public int minDistance(String word1, String word2) {
        if (word1.length() == 0) {
            return word2.length();
        } else if (word2.length() == 0) {
            return word1.length();
        }

        int[][] dp = new int[word1.length() + 1][word2.length() + 1];

        dp[0][0] = 0;
        for (int i = 1; i <= word1.length(); ++i) {
            dp[i][0] = i;
        }
        for (int j = 1; j <= word2.length(); ++j) {
            dp[0][j] = j;
        }

        for (int i = 1; i <= word1.length(); ++i) {
            for (int j = 1; j <= word2.length(); ++j) {
                final char c1 = word1.charAt(i-1);
                final char c2 = word2.charAt(j-1);
                // 不相同的话，就替换
                dp[i][j] = dp[i-1][j-1] + (c1 == c2 ? 0 : 1);

                // delete掉任何一个
                dp[i][j] = Math.min(dp[i][j], 1 + Math.min(dp[i-1][j], dp[i][j-1]));

            }
        }

        return dp[word1.length()][word2.length()];
    }

    public int maximalRectangle(char[][] matrix) {
        if (matrix == null || matrix.length <= 0 || matrix[0] == null || matrix[0].length <= 0) {
            return 0;
        }

        int[] preOneCnts = new int[matrix[0].length];

        int result = 0;

        for (int i = 0; i < matrix.length; ++i) {
            // update preOneCnts
            for (int j = 0; j < matrix[0].length; ++j) {
                preOneCnts[j] = matrix[i][j] == '0' ? 0 : 1 + preOneCnts[j];
            }

            result = Math.max(result, getMaxRectangle(preOneCnts));
        }

        return result;
    }

    private int getMaxRectangle(int[] heights) {
        if (heights == null || heights.length <= 0) {
            return 0;
        }

        int result = 0;

        Stack<Integer> stack = new Stack<>();

        int startIdx = -1;
        for (int i = 0; i <= heights.length; ++i) {
            final int height = i < heights.length ? heights[i] : 0;

            while (!stack.isEmpty() && heights[stack.peek()] > height) {
                final int headIdx = stack.pop();
                final int preIdx = stack.isEmpty() ? startIdx : stack.peek();

                result = Math.max(result, (i - preIdx - 1) * heights[headIdx]);
            }

            if (height != 0) {
                stack.push(i);
            } else {
                startIdx = i;
            }
        }

        return result;
    }

    public int numDistinct(String s, String t) {
        return numDistinctDp(s, t);
    }

    private int numDistinctHelper(CharSequence s, CharSequence t, int i, int j) {
        if (j < 0) {
            return 1;
        }
        if (i < j) {
            return 0;
        } else if (i == j) {
            for (int k = 0; k <= i; ++k) {
                if (s.charAt(k) != t.charAt(k)) {
                    return 0;
                }
            }
            return 1;
        }

        int res;
        if (s.charAt(i) == t.charAt(j)) {
            res = numDistinctHelper(s, t, i - 1, j - 1) + numDistinctHelper(s, t, i-1, j);
        } else {
            res = numDistinctHelper(s, t, i-1, j);
        }

        return res;
    }

    private int numDistinctDp(String s, String t) {
        int[][] dp = new int[s.length() + 1][t.length() + 1];

        for (int i = 0; i <= s.length(); ++i) {
            dp[i][0] = 1;
        }

        for (int i = 1; i < dp.length; ++i) {
            for (int j = 1; j < dp[0].length; ++j) {
                if (s.charAt(i-1) == t.charAt(j-1)) {
                    dp[i][j] = dp[i-1][j-1] + dp[i-1][j];
                } else {
                    dp[i][j] = dp[i-1][j];
                }
            }
        }

        return dp[s.length()][t.length()];
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

    public int minCut(String s) {
        return minCutDp(s);
    }

    private int minCutDp(String s) {
        if (s == null || s.length() <= 0) {
            return 0;
        }

        int[] dp = new int[s.length()];
        dp[0] = 0;

        boolean[][] cache = new boolean[s.length()][s.length()];

        // 预先处理奇数的
        for (int i = 0; i < s.length(); ++i) {
            for (int j = 0; i - j >= 0 && i + j < s.length(); ++j) {
                if (s.charAt(i-j) == s.charAt(i+j)) {
                    cache[i-j][i+j] = true;
                } else {
                    break;
                }
            }
        }

        // 再处理偶数个的
        for (int i = 0; i < s.length() - 1; ++i) {
            for (int j = 0; i - j >= 0 && i + 1 + j < s.length(); ++j) {
                if (s.charAt(i - j) == s.charAt(i+1+j)) {
                    cache[i-j][i+1+j] = true;
                } else {
                    break;
                }
            }
        }

        for (int i = 1; i < s.length(); ++i) {
            dp[i] = Integer.MAX_VALUE;
            for (int j = 0; j <= i; ++j) {
                if (cache[j][i]) {
                    if (j == 0) {
                        dp[i] = 0;
                        break;
                    } else {
                        dp[i] = Math.min(dp[i], dp[j - 1] + 1);
                    }
                }
            }
        }

        return dp[s.length() - 1];

    }

    private int minCutHelper(CharSequence s, int end) {
        if (end <= 0) {
            return 0;
        }

        int result = Integer.MAX_VALUE;
        for (int j = end; j >= 0; --j) {
            if (isPalindrome(s, j, end)) {
                if (j == 0) {
                    return 0;
                } else {
                    result = Math.min(result, minCutHelper(s,j - 1) + 1);
                }
            }
        }

        return result;
    }

    private boolean isPalindrome(CharSequence s, int start, int end) {
        while (start < end) {
            if (s.charAt(start) != s.charAt(end)) {
                return false;
            } else {
                start++;
                end--;
            }
        }
        return true;
    }

    /**
     * Definition for a point.*/
     public static class Point {
         int x;
         int y;
         Point() { x = 0; y = 0; }
         Point(int a, int b) { x = a; y = b; }
     }

    public static class PointCompareble extends Point {
        PointCompareble() { super(); }
        PointCompareble(int a, int b) { super(a, b); }
        PointCompareble(Point point) {
            super(point.x, point.y);
        }

        @Override
        public int hashCode() {
            return Objects.hash(x, y);
        }

        @Override
        public boolean equals(Object obj) {
            if (obj == this) {
                return true;
            }

            if (!(obj instanceof PointCompareble)) {
                return false;
            }

            return x == ((PointCompareble) obj).x && y == ((PointCompareble) obj).y;
        }
    }

     public static class Line {
         final int mType; // 0 为普通 1 为水平直线 2 为数值直线

         final int m, n; // k = m / n
         final int p, q; // v = p / q

         private Line(int type, int m, int n, int p, int q) {
             this.mType = type;
             this.m = m;
             this.n = n;
             this.p = p;
             this.q = q;
         }


         @Override
         public int hashCode() {
             return Objects.hash(mType, m, n, p, q);
         }

         @Override
         public boolean equals(Object obj) {
             if (obj == this)
                 return true;

             if (!(obj instanceof Line)) {
                 return false;
             }

             if (mType != ((Line) obj).mType) {
                 return false;
             }

             return m == ((Line) obj).m &&
                     n == ((Line) obj).n &&
                     p == ((Line) obj).p &&
                     q == ((Line) obj).q;
         }

         public static Line sConstructLine(PointCompareble point1, PointCompareble point2) {
             if (point1 == point2 || (point1.x == point2.x && point1.y == point2.y)) {
                 return null;
             }

             final int deltaX = point2.x - point1.x;
             final int deltaY = point2.y - point1.y;

             if (deltaY == 0) {
                 // 水平的
                 return new Line(1, 0, 1, point1.y, 1);
             } else if (deltaX == 0) {
                 // 竖直的
                 return new Line(2, 0, 0, point1.x, 1);
             } else {
                 // 普通的，m / n = y2 - y1 / x2 - x1，求deltaY和deltaX的最大公约数
                 int[] mn = {0,0};
                 gcd(deltaY, deltaX, mn);

                 int[] pq = {0, 0};
                 gcd(mn[1] * point2.y - mn[0] * point2.x, mn[1], pq);

                 return new Line(0, mn[0], mn[1], pq[0], pq[1]);
             }
         }


     }

    public static void gcd(int a, int b, int[] result) {
        int large = Math.max(Math.abs(a), Math.abs(b));
        int small = Math.min(Math.abs(a), Math.abs(b));

        while (small != 0) {
            int temp = small;
            small = large % small;
            large = temp;
        }

        int m;
        int n;
        if (a * b < 0) {
            m = 0 - Math.abs(a) / large;
        } else {
            m = Math.abs(a) / large;
        }

        n = Math.abs(b) / large;

        result[0] = m;
        result[1] = n;
    }

    private static class LineNode {
         int sum = 0;
         Set<Integer> set = new HashSet<>();
    }

    public int maxPoints(Point[] points) {
        if (points == null || points.length <= 0) {
            return 0;
        }

        int result = 1;

        Map<PointCompareble, Integer> pointsMap = new HashMap<>();
        for (Point point : points) {
            PointCompareble pointCompareble = new PointCompareble(point);
            int newValue = pointsMap.getOrDefault(pointCompareble, 0) + 1;
            pointsMap.put(pointCompareble, newValue);
            result = Math.max(result, newValue);
        }

        PointCompareble[] pointComparebles = new PointCompareble[pointsMap.size()];
        int[] cnts = new int[pointsMap.size()];

        int k = 0;
        for (Map.Entry<PointCompareble, Integer> entry : pointsMap.entrySet()) {
            pointComparebles[k] = entry.getKey();
            cnts[k] = entry.getValue();
            ++k;
        }

        Map<Line, LineNode> lineSetMap = new HashMap<>();

        for (int i = 0; i < pointComparebles.length - 1; ++i) {
            for (int j = i + 1; j < pointComparebles.length; ++j) {
                final Line newLine = Line.sConstructLine(pointComparebles[i], pointComparebles[j]);
                final LineNode lineNode = lineSetMap.computeIfAbsent(newLine, new Function<Line, LineNode>() {
                    @Override
                    public LineNode apply(Line line) {
                        return new LineNode();
                    }
                });
                if (lineNode.set.add(i)) {
                    lineNode.sum += cnts[i];
                }
                if (lineNode.set.add(j)) {
                    lineNode.sum += cnts[j];
                }

                result = Math.max(result, lineNode.sum);
            }
        }

        return result;
    }

    public int maximumGap(int[] nums) {
        if (nums == null || nums.length < 2) {
            return 0;
        }

        return 0;
    }

    public int calculate(String s) {
        List<String> postExpresses = inExpressionToPost(s);

        return getPostExpressionValue(postExpresses);
    }

    private int getPostExpressionValue(List<String> postFixList) {
         Stack<Integer> stack = new Stack<>();

        for(int i=0;i<postFixList.size();i++){
            String word = postFixList.get(i);
            if(word.length()==1 && (word.charAt(0)=='+'||word.charAt(0)=='-'||word.charAt(0)=='*'||word.charAt(0)=='/')){
                int number2 = stack.pop();
                int number1 = stack.pop();
                if(word.charAt(0)=='+'){
                    int number = number1+number2;
                    stack.push(number);
                }else if(word.charAt(0)=='-'){
                    int number = number1-number2;
                    stack.push(number);
                }else if(word.charAt(0)=='*'){
                    int number = number1*number2;
                    stack.push(number);
                }else{
                    int number = number1/number2;
                    stack.push(number);
                }
            }else{
                int number = Integer.parseInt(word);
                stack.push(number);
            }
        }
        return stack.peek();
    }

    private List<String> inExpressionToPost(String s) {
         ArrayList<String> postExpress = new ArrayList<>();

         Stack<Character> stack = new Stack<>();

         for (int i = 0; i < s.length(); ++i) {
             final char c = s.charAt(i);

             if (Character.isDigit(c)) {
                 int result = c - '0';
                 while (i + 1 < s.length() && Character.isDigit(s.charAt(i + 1))) {
                     result = result * 10 + (s.charAt(i + 1) - '0');
                     i++;
                 }
                 postExpress.add(String.valueOf(result));
             } else if (c == ' ') {
                 continue;
             } else if (c == '(') {
                 stack.push(c);
             } else if (c == ')') {
                 while (!stack.isEmpty() && '(' != stack.peek()) {
                     postExpress.add(String.valueOf(stack.pop()));
                 }
                 if (!stack.isEmpty() && '(' == stack.peek()) {
                     stack.pop();
                 }
             } else {
                 while (!stack.isEmpty() && getOpreationPriority(c) <= getOpreationPriority(stack.peek())) {
                     postExpress.add(String.valueOf(stack.pop()));
                 }

                 stack.push(c);
             }
         }

         while (!stack.isEmpty()) {
             postExpress.add(String.valueOf(stack.pop()));
         }

         return postExpress;
    }

    private int getOpreationPriority(char c) {
         if (c == '*' || c == '/') {
             return 2;
         } else if (c == '-' || c == '+') {
             return 1;
         } else {
             return 0;
         }
    }

    public int countDigitOne(int n) {
        int left = 0;
        int multi = 1;

        int result = 0;
        while (n / multi != 0) {
            if (multi == 1) {
                result += n % 10 == 0 ? n / 10 : (n / 10 + 1);
                multi *= 10;
            } else {
                int yu = n / multi;
                left = n % multi;

                int last = yu % 10;
                if (last == 0) {
                    result += multi * (yu / 10);
                } else if (last >= 2) {
                    result += multi * (yu / 10 + 1);
                } else {
                    result += multi * (yu / 10) + left + 1;
                }
                if (Integer.MAX_VALUE / multi < 10) {
                    break;
                }
                multi *= 10;
            }
        }

        return result;
    }

    private final String[] LESS_THAN_20 = {"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
    private final String[] TENS = {"", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};
    private final String[] THOUSANDS = {"", "Thousand", "Million", "Billion"};

    public String numberToWords(int num) {
        if (num == 0) return "Zero";

        int i = 0;
        String words = "";

        while (num > 0) {
            if (num % 1000 != 0)
                words = helper(num % 1000) +THOUSANDS[i] + " " + words;
            num /= 1000;
            i++;
        }

        return words.trim();
    }

    private String helper(int num) {
        if (num == 0)
            return "";
        else if (num < 20)
            return LESS_THAN_20[num] + " ";
        else if (num < 100)
            return TENS[num / 10] + " " + helper(num % 10);
        else
            return LESS_THAN_20[num / 100] + " Hundred " + helper(num % 100);
    }

    public List<String> addOperators(String num, int target) {
        List<String> all = addOperatorsHelper(num, 0);

        List<String> result = new LinkedList<>();

        for (String one : all) {
            if (one.length() == num.length()) {
                if (one.compareTo(String.valueOf(target)) == 0) {
                    result.add(one);
                }
                continue;
            }

            int temp = getPostExpressionValue(inExpressionToPost(one));

            if (temp == target) {
                result.add(one);
            }
        }

        return result;
    }

    private static final String MAX_INTEGER = String.valueOf(Integer.MAX_VALUE);

    private List<String> addOperatorsHelper(CharSequence num, int start) {
        List<String> result = new LinkedList<>();
        if (start == num.length()) {
            result.add("");
            return result;
        }

        for (int i = start; i < num.length(); ++i) {
            final CharSequence head = num.subSequence(start, i + 1);
            if (head.length() >= 2 && head.charAt(0) == '0') {
                continue;
            } else if (head.length() > MAX_INTEGER.length()) {
                continue;
            } else if (head.length() == MAX_INTEGER.length() && MAX_INTEGER.compareTo(head.toString()) < 0) {
                continue;
            }

            List<String> listNext = addOperatorsHelper(num, i + 1);
            for (String next : listNext) {
                if (next.isEmpty()) {
                    result.add(head.toString());
                } else {
                    result.add(head + "*" + next);
                    result.add(head + "+" + next);
                    result.add(head + "-" + next);
                }
            }
        }

        return result;
    }

    public List<String> removeInvalidParentheses(String s) {
        List<String> result = new LinkedList<>();

        if (s == null || s.length() <= 0) {
            result.add("");
            return result;
        }

        if (isValidParentheses(s)) {
            result.add(s);
            return result;
        }

        Queue<String>  queue = new LinkedList<>();

        queue.add(s);

        Set<String> set = new HashSet<>();
        set.add(s);

        boolean findShortest = false;

        while (!queue.isEmpty()) {
            final String head = queue.poll();

            for (int i = 0; i < head.length(); ++i) {
                final char c = head.charAt(i);
                if (c != '(' && c != ')') {
                    continue;
                }
                final String cut = head.substring(0, i) + head.substring(i + 1);

                if (findShortest) {
                    if (set.add(cut) && isValidParentheses(cut)) {
                        result.add(cut);
                    }
                } else if (isValidParentheses(cut)) {
                    findShortest = true;
                    set.add(cut);
                    result.add(cut);
                } else if (set.add(cut)){
                    queue.add(cut);
                }
            }
        }

        return result;
    }

    private boolean isValidParentheses(String s) {
        if (s == null || s.length() <= 0) {
            return true;
        }

        int left = 0;
        for (int i = 0; i < s.length(); ++i) {
            final char c = s.charAt(i);
            if (c == '(') {
                left++;
            } else if (c == ')') {
                left--;
            }

            if (left < 0) {
                return false;
            }
        }

        return left == 0;
    }

    public int longestValidParentheses(String s) {
        if (s == null || s.length() <= 1) {
            return 0;
        }

        int leftCnt = 0;
        int result = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);

        int[] dp = new int[s.length()];

        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == '(') {
                map.put(++leftCnt, i);
            } else {
                leftCnt--;

                if (leftCnt < 0) {
                    leftCnt = 0;
                    map.clear();
                    map.put(0, i);
                } else {
                    // 上一个有值
                    Integer oldValue = map.put(leftCnt, i);
                    if (oldValue != null) {
                        dp[i] = (i - oldValue) + (oldValue == -1 ? 0 : dp[oldValue]);
                        result = Math.max(dp[i], result);
                    }
                }
            }
        }

        return result;
    }

    public int longestIncreasingPath(int[][] matrix) {
        if (matrix == null || matrix.length <= 0 || matrix[0] == null || matrix[0].length <= 0) {
            return 0;
        }

        final int rows = matrix.length;
        final int columns = matrix[0].length;
        final int size = rows * columns;

        int[][] cached = new int[rows][columns];

        int result = 0;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                result = Math.max(result, longestIncreasingPath(matrix, i, j, cached));
            }
        }

        return result;
    }

    int[][] matrixDirections = {
            {0,1},
            {0,-1},
            {1,0},
            {-1,0}
    };

    private int longestIncreasingPath(int[][] matrix, int x, int y, int[][] cached) {
        final int cache = cached[x][y];
        if (cache != 0) {
            return cache;
        }

        int result = 1;
        for (int[] dirction : matrixDirections) {
            final int newx = x + dirction[0];
            final int newy = y + dirction[1];

            if (newx >= 0 && newx < matrix.length && newy >= 0 && newy < matrix[0].length
                    && matrix[x][y] < matrix[newx][newy]) {

                result = Math.max(result, 1 + longestIncreasingPath(matrix, newx, newy, cached));
            }
        }

        cached[x][y] = result;

        return result;
    }

    public int maxCoins(int[] nums) {

        if (nums == null || nums.length <= 0) {
            return 0;
        }

        int[][] dp = new int[nums.length][nums.length];

        for (int k = 0; k < nums.length; ++k) {
            for (int i = 0; i < nums.length; ++i) {
                for (int t = 0; t <= k && i + k < nums.length; ++t) {
                    final int leftMax = t - 1 >= 0 ? dp[i][i + t-1] : 0;
                    final int rightMax = t < k ? dp[i + t + 1][i + k] : 0;
                    final int middle = nums[i + t] * (i >= 1 ? nums[i-1] : 1) * (i + k + 1 < nums.length ? nums[i+k+1] : 1);

                    dp[i][i + k] = Math.max(dp[i][i+k], leftMax + rightMax + middle);
                }
            }
        }

        return dp[0][nums.length - 1];
    }

    public List<Integer> countSmaller(int[] nums) {

        if (nums == null || nums.length <= 0) {
            return new ArrayList<>();
        }

        int[] result = new int[nums.length];
        int[] indexs = new int[nums.length];
        for (int i = 0; i < indexs.length; ++i) {
            indexs[i] = i;
        }

        countSmallerMergeSort(nums, indexs, 0, nums.length - 1, new int[indexs.length], result);

        List<Integer> intList = new ArrayList<Integer>();
        for (int i : result) {
            intList.add(i);
        }

        return intList;
    }

    private void countSmallerMergeSort(int[] nums, int[] indexs, int start, int end, int[] temp, int[] result) {
        if (start >= end) {
            return;
        }

        int middle = start + (end - start) / 2;

        countSmallerMergeSort(nums, indexs, start, middle, temp, result);
        countSmallerMergeSort(nums, indexs, middle + 1, end, temp, result);

        int left = start;
        int right = middle + 1;

        int tempIdx = start;

        int smallerCnt = 0;

        while (left <= middle) {
            while (right <= end && nums[indexs[right]] < nums[indexs[left]]) {
                temp[tempIdx++] = indexs[right];
                right++;
                smallerCnt++;
            }

            result[indexs[left]] += smallerCnt;
            temp[tempIdx++] = indexs[left];
            left++;
        }

        while (right <= end) {
            temp[tempIdx++] = indexs[right];
            right++;
        }

        for (int i = start; i <= end; ++i) {
            indexs[i] = temp[i];
        }
    }

    public int tallestBillboard(int[] rods) {
        if (rods == null || rods.length <= 0) {
            return 0;
        }

        int maxDiff = 0;
        for (int rod : rods) {
            maxDiff += rod;
        }

        maxDiff /= 2;

        int[] dp = new int[maxDiff + 1];
        int[] dpNext = new int[maxDiff + 1];

        for (int rod : rods) {
            if (rod == 0) {
                continue;
            }
//            for (int diff = 0; diff < dp.length; ++diff) {
//                if (diff == rod) {
//                    if (dp[rod] != 0) {
//                        dpNext[0] = Math.max(dpNext[0], dp[rod]);
//                    }
//                    if (diff + rod <= maxDiff) {
//                        dpNext[diff + rod]
//                    }
//                }
//            }
        }

        return 0;
    }

    private void tallestBillboardHelper(int[] rods, int i, int sum1, int sum2) {
    }

    private class RectPoint {
        final int x;
        final int y;

        RectPoint(int x, int y) {
            this.x = x;
            this.y = y;
        }

        @Override
        public int hashCode() {
            return Objects.hash(x, y);
        }

        @Override
        public boolean equals(Object obj) {
            return obj instanceof RectPoint && this.x == ((RectPoint) obj).x && this.y == ((RectPoint) obj).y;
        }
    }

    public boolean isRectangleCover(int[][] rectangles) {
        if (rectangles == null || rectangles.length <= 0) {
            return false;
        }

        Set<RectPoint> set = new HashSet<>();

        int minBottom = Integer.MAX_VALUE;
        int maxTop = Integer.MIN_VALUE;
        int minLeft = Integer.MAX_VALUE;
        int maxRight = Integer.MIN_VALUE;

        int sumArea = 0;

        for (int[] rectangle : rectangles) {

            final int left = rectangle[0];
            final int bottom = rectangle[1];
            final int right = rectangle[2];
            final int top = rectangle[3];

            sumArea += (top - bottom) * (right - left);

            minBottom = Math.min(minBottom, bottom);
            maxTop = Math.max(maxTop, top);
            minLeft = Math.min(minLeft, left);
            maxRight = Math.max(maxRight, right);

            checkPoint(new RectPoint(left, bottom), set);
            checkPoint(new RectPoint(left, top), set);
            checkPoint(new RectPoint(right, bottom), set);
            checkPoint(new RectPoint(right, top), set);
        }

        return set.size() == 4
                && set.contains(new RectPoint(minLeft, minBottom))
                && set.contains(new RectPoint(minLeft, maxTop))
                && set.contains(new RectPoint(maxRight, minBottom))
                && set.contains(new RectPoint(maxRight, maxTop))
                && sumArea == (maxRight - minLeft) * (maxTop - minBottom);

    }

    private void checkPoint(RectPoint rectPoint, Set<RectPoint> set) {
        if (set.contains(rectPoint)) {
            set.remove(rectPoint);
        } else {
            set.add(rectPoint);
        }
    }


    public boolean canCross(int[] stones) {
        return canCrossDp2(stones);
    }

    private boolean canCrossDp(int[] stones) {
        SortedSet<Integer>[] dp = new SortedSet[stones.length];

        final Comparator<Integer> comparator = new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2 - o1;
            }
        };

        dp[0] = new TreeSet<Integer>(comparator);

        dp[0].add(1);

        for (int i = 1; i < stones.length; ++i) {
            TreeSet<Integer> next = null;
            for (int j = i - 1; j >= 0; --j) {
                final int diff = stones[i] - stones[j];
                final SortedSet<Integer> pre = dp[j];

                if (pre == null) {
                    continue;
                } else if (diff > pre.iterator().next()){
                    break;
                } else if (pre.contains(diff)){
                    if (next == null) {
                        next = new TreeSet<>(comparator);
                    }
                    if (diff >= 2) {
                        next.add(diff - 1);
                    }

                    next.add(diff);
                    next.add(diff + 1);
                }
            }
            dp[i] = next;

        }

        return dp[stones.length - 1] != null;

    }

    private boolean canCrossDp2(int[] stones) {
        int[][] dp = new int[stones.length][];

        dp[0] = new int[2];

        dp[0][0] = dp[0][1] = 1;

        for (int i = 1; i < stones.length; ++i) {
            int[] next = null;
            for (int j = i - 1; j >= 0; --j) {
                final int diff = stones[i] - stones[j];
                final int[] pre = dp[j];

                if (pre == null) {
                    continue;
                } else if (diff > pre[1]){
                    break;
                } else if (diff >= pre[0] && diff <= pre[1]){
                    if (next == null) {
                        next = new int[2];
                        next[0] = Integer.MAX_VALUE;
                    }
                    next[1] = Math.max(next[1], diff + 1);

                    int curMin = Math.max(diff - 1, 1);

                    next[0] = Math.min(curMin, next[0]);
                }
            }
            dp[i] = next;
        }

        return dp[stones.length - 1] != null;

    }

    public boolean crossHelper(int[] stones, int k, int idx) {
        if (idx == stones.length - 1) {
            return true;
        } else if (idx >= stones.length) {
            return false;
        }

        for (int i = idx + 1; i < stones.length; ++i) {
            final int diff = stones[i] - stones[idx];
            if (diff > k + 1) {
                return false;
            } else if (diff < k - 1){
                continue;
            } else {
                int maxEnd = diff * (stones.length - 1 - i) + (stones.length - 1 - i) * (stones.length - i) / 2 + stones[i];
                if (stones[stones.length - 1] > maxEnd) {
                    continue;
                } else if (crossHelper(stones, diff, i)) {
                    return true;
                }
            }
        }

        return false;
    }

    public boolean crossHelperBfs(int[] stones) {

        Queue<Pair<Integer, Integer>> queue = new LinkedList<>();

        queue.add(new Pair<>(0, 0));

        while (!queue.isEmpty()) {
            final Pair<Integer, Integer> pair = queue.poll();
            final int k = pair.getKey();
            final int idx = pair.getValue();

            System.out.println(pair.toString());

            for (int i = idx + 1; i < stones.length; ++i) {
                final int diff = stones[i] - idx;
                if (diff > k + 1) {
                    break;
                } else if (diff >= k - 1){
                    if (i == stones.length - 1) {
                        return true;
                    } else {
                        queue.add(new Pair<>(diff, i));
                    }
                }
            }
        }

        return false;
    }

    public int splitArray(int[] nums, int m) {
        if (nums == null || nums.length <= 0) {
            return 0;
        }

        if (m > nums.length) {
            return 0;
        }

        int[] dp = new int[nums.length];
        int[] dpNext = new int[nums.length];
        int[] sums = new int[nums.length];

        dp[0] = nums[0];
        sums[0] = nums[0];
        for (int i = 1; i < nums.length; ++i) {
            dp[i] += dp[i-1] + nums[i];
            sums[i] = dp[i];
        }

        for (int n = 2; n <= m; ++n) {
            for (int i = n - 1; i < nums.length; ++i) {
                final int length = i - (n - 2);
                final int base = n - 2;
                final int end = i;
                final int[] dpTemp = dp;
                final DecreasingArray decreasingArray = new DecreasingArray() {
                    @Override
                    public int getAt(int idx) {
                        return sums[end] - sums[base + idx];
                    }
                };

                final IncreasingArray increasingArray = new IncreasingArray() {
                    @Override
                    public int getAt(int idx) {
                        return dpTemp[base + idx];
                    }
                };

                dpNext[i] = getMinMax(decreasingArray, increasingArray, length);
            }

            int[] temp = dp;
            dp = dpNext;
            dpNext = temp;
        }

        return dp[nums.length - 1];
    }

    private interface IncreasingArray {
        int getAt(int idx);
    }

    private interface DecreasingArray {
        int getAt(int idx);
    }

    private int getMinMax(DecreasingArray decreasingArray, IncreasingArray increasingArray, int length) {
        int left = 0;
        int right = length - 1;

        while (left <= right) {
            int middle = left + (right - left) / 2;
            final int decreaseMiddle = decreasingArray.getAt(middle);
            final int increaseMiddle = increasingArray.getAt(middle);
            if (decreaseMiddle == increaseMiddle) {
                return decreaseMiddle;
            } else if (decreaseMiddle > increaseMiddle) {
                left = middle + 1;
            } else {
                right = middle - 1;
            }
        }

        if (left == length) {
            return decreasingArray.getAt(length - 1);
        } else if (right == -1) {
            return increasingArray.getAt(0);
        }

        // 取left的值
        final int decreaseMiddle = decreasingArray.getAt(left);
        final int increaseMiddle = increasingArray.getAt(left);

        if (decreaseMiddle > increaseMiddle) {
            if (left < length - 2) {
                return Math.min(decreaseMiddle, increasingArray.getAt(left + 1));
            } else {
                return decreaseMiddle;
            }
        } else if (decreaseMiddle < increaseMiddle) {
            if (left > 0) {
                return Math.min(increaseMiddle, decreasingArray.getAt(left - 1));
            } else {
                return increaseMiddle;
            }
        } else {
            return decreaseMiddle;
        }
    }

    public int strongPasswordChecker(String s) {
        if (s == null || s.length() <= 0) {
            return 6;
        }

        int lower = 0;
        int uper = 0;
        int digit = 0;

        for (int i = 0; i < s.length(); ++i) {
            final char c = s.charAt(i);
            if (Character.isLowerCase(c)) {
                lower++;
            } else
                return 0;
        }

        return 0;
    }

    public int findKthNumber(int n, int k) {
        if (k == 1) {
            return 1;
        }

        final String s = String.valueOf(n);
        final int length = s.length();

        // 分3个阶段来获取，第一位为m，则看第一位为1到m-1，是不是在这个区间里面
        // 再看第一位m，是不是在这个区间里面
        // 再看剩下的，剩下的就是最大长度为length - 1位，且从m+1开始
        int oneMax = 0;
        int numPower = 1;
        int i = 1;
        while (i++ <= length) {
            oneMax += numPower;
            numPower *= 10;
        }

        final int first = s.charAt(0) - '0';

        final int sumPre = first > 1 ? oneMax * (first - 1) : 0;

        if (k <= sumPre) {
            return Integer.valueOf(findKStringFromStart('1', oneMax, k));
        }

        k -= sumPre;

        // 查看最多有多少个
        int sumCur = 1;

        int temp = oneMax / 10;

        for (i = 1; i < s.length(); ++i) {
            final int value = s.charAt(i) - '0';
            sumCur += value > 0 ? (value * temp) : 0;
            sumCur += 1;
            temp /= 10;
        }

        if (k <= sumCur) {
            String result = "";
            result += s.charAt(0);

            temp = oneMax / 10;
            int tempN = k - 1;

            while (temp >= 1 && tempN != 0) {

                int shang = (tempN - 1) / temp;

                final char next = (char) ('0' + shang);
                result += next;

                tempN -= 1 + shang * temp;

                temp /= 10;
            }


            return Integer.valueOf(result);
        }

        k -= sumCur;

        int multi = 1;
        int curPre = 0;
        for (i = s.length() - 2; i >= 1; --i) {

            curPre += multi;
            final int value = s.charAt(i) - '0';
            final int num = (9 - value) * curPre;

            if (k <= num && num > 0) {
                final char nextFirst = (char) (s.charAt(i) + 1);
                return Integer.valueOf(s.substring(0, i) + findKStringFromStart(nextFirst, curPre, k));
            }

            k -= num;

            multi = multi * 10;
        }

        // 从m+1到9的length - 1 位数

        final char nextFirst = (char) (s.charAt(0) + 1);
        return Integer.valueOf(findKStringFromStart(nextFirst, oneMax / 10, k));
    }

    private String findKStringFromStart(char first, int oneMax, int k) {
        boolean firstValue = true;

        String result = "";

        while (oneMax >= 1 && k != 0) {

            int shang = (k - 1) / oneMax;
            int yu = (k - 1) % oneMax;

            final char next = (char) ((shang + (firstValue ?  first : '0')));
            result += next;

            k = yu;

            oneMax /= 10;

            firstValue = false;
        }

        return result;
    }


    // 不一定是连续的
    public int numberOfArithmeticSlicesII(int[] A) {
        if (A == null || A.length <= 3) {
            return 0;
        }

        int sum = 0;

        Map<Integer, Integer>[] dp = new HashMap[A.length];

        dp[0] = new HashMap<>();

        dp[1] = new HashMap<>();
        dp[1].put(A[1] - A[0], 2);

        for (int i = 2; i < A.length; ++i) {
            Map<Integer, Integer> map = new HashMap<>();
            for (int j = i - 1; j >= 0; --j) {
                final int diff = A[i] - A[j];

                Map<Integer, Integer> pre = dp[j];

                final int old = pre.getOrDefault(diff, -1);
                if (old == -1) {
//                    sCombMap.put()
                }
            }
        }

        return 0;

    }

    // 必须是连续的
    public int numberOfArithmeticSlicesI(int[] A) {
        if (A == null || A.length < 3) {
            return 0;
        }

        int[] dp = new int[A.length];
        int sum = 0;

        for (int i = 2; i < A.length; ++i) {
            if (A[i] - A[i-1] == A[i-1] - A[i-2]) {
                dp[i] = dp[i-1] + 1;
                sum += dp[i];
            }
        }

        return sum;
    }

//    public int poorPigs(int buckets, int minutesToDie, int minutesToTest) {
//
//    }

    public int getMaxRepetitions(String s1, int n1, String s2, int n2) {
        if (s1.length() * n1 < s2.length() * n2) {
            return 0;
        }

        s1 = checkSubsequenceValid(s1, s2);

        if (s1 == null || s1.length() * n1 < s2.length() * n2) {
            return 0;
        }


        ArrayList<Integer> arrayLastRemainPos = new ArrayList<>();
        ArrayList<Integer> arrayLastRemainLength = new ArrayList<>();
        ArrayList<Integer> arrayCnt = new ArrayList<>();


        arrayLastRemainLength.add(0);
        arrayLastRemainPos.add(0);
        arrayCnt.add(0);

        // 查看loop
        int[] lastRemain = {0, 0};
        int k = -1;
        int cnt = 0;
        int i;
        for (i = 1; i <= n1; ++i) {
            cnt += getRemain(s1, s2, lastRemain[0], lastRemain[0] + lastRemain[1] + s1.length() - 1, lastRemain);

            boolean stop = false;
            for (int j = 0; j < arrayLastRemainPos.size(); ++j) {
                if(lastRemain[0] % s1.length() == arrayLastRemainPos.get(j) % s1.length()
                        && lastRemain[1] == arrayLastRemainLength.get(j)) {
                    k = j;
                    stop = true;
                    break;
                }
            }

            if (stop) {
                break;
            }
            arrayLastRemainPos.add(lastRemain[0]);
            arrayLastRemainLength.add(lastRemain[1]);
            arrayCnt.add(cnt);
        }

        if (k == -1) {
            return cnt / n2;
        }

        // loop 的位置是从哪到哪
        final int loopLength = i - k;

        final int oneLoopContainsCnt = cnt - arrayCnt.get(k);

        int total = 0;

        total += oneLoopContainsCnt * ((n1 - k) / loopLength) + arrayCnt.get(k);

        final int left = (n1 - k) % loopLength;

        total += left == 0 ? 0 : (arrayCnt.get(k + left) - arrayCnt.get(k));

        return total / n2;
    }

    private String checkSubsequenceValid(String s1, String s2) {
        boolean[] hash = new boolean[26];
        int kind = 0;
        for (int i = 0; i < s2.length(); ++i) {
            final int c = s2.charAt(i) - 'a';
            if (!hash[c]) {
                hash[c] = true;
                kind++;
            }
        }

        StringBuilder s1Builder = new StringBuilder();
        int s1Kind = 0;
        boolean[] s1hash = new boolean[26];
        for (int i = 0; i < s1.length(); ++i) {
            final int c = s1.charAt(i) - 'a';
            if (hash[c]) {
                s1Builder.append((char) (c + 'a'));
                if (!s1hash[c]) {
                    s1Kind++;
                    s1hash[c] = true;
                }
            }
        }

        if (s1Kind < kind) {
            return null;
        }

        return s1Builder.toString();
    }

    private int getRemain(String s1, String s2, int start, int end, int[] remain) {
        int j = 0;

        int lastEnd = start - 1;
        int cnt = 0;
        for (int i = start; i <= end; ++i) {
            final int c1 = s1.charAt(i % s1.length());
            final int c2 = s2.charAt(j);

            if (c1 == c2) {
                if (++j == s2.length()) {
                    lastEnd = i;
                    cnt++;
                    j = 0;
                }
            }
        }

        remain[0] = (lastEnd + 1);
        remain[1] = end - lastEnd;

        return cnt;
    }

    public String smallestGoodBase(String n) {
        long value = Long.valueOf(n);

        for (long k = 2; k <= value - 1; k++) {
            long temp = value;
            boolean find = true;
            while (temp != 0) {
                long left = temp % k;
                if (left != 1) {
                    find = false;
                    break;
                }
                temp = temp / k;
            }

            if (find) {
                return String.valueOf(k);
            }
        }

        return "";
    }

    public int findMaximizedCapital(int k, int W, int[] Profits, int[] Capital) {
        if (Profits == null || Profits.length <= 0 || Capital == null || Capital.length <= 0) {
            return 0;
        }

        Integer[] idxes = new Integer[Profits.length];
        for (int i = 0; i < idxes.length; ++i) {
            idxes[i] = i;
        }

        Arrays.sort(idxes, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return Capital[o1] - Capital[o2];
            }
        });

        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return Profits[o2] - Profits[o1];
            }
        });

        int sum = W;
        int startIdx = 0;
        while (k-- > 0) {
            while (startIdx < idxes.length && Capital[idxes[startIdx]] <= W) {
                maxHeap.add(idxes[startIdx]);
                startIdx++;
            }

            if (maxHeap.isEmpty()) {
                return sum;
            } else {
                int profit = Profits[maxHeap.poll()];
                W += profit;
                sum += profit;
            }
        }

        return sum;
    }

    public int findRotateSteps(String ring, String key) {
        List<Integer>[] hash = new ArrayList[26];

        for (int i = 0; i < ring.length(); ++i) {
            final int idx = ring.charAt(i) - 'a';
            List<Integer> list = hash[idx];
            if (list == null) {
                list = new ArrayList<>();
                hash[idx] = list;
            }
            list.add(i);
        }

        int[][] cache = new int[ring.length()][key.length()];

        return findMinRotateStep(ring, key, 0, 0, hash, cache);
    }

    private int findMinRotateStep(String ring, String key, int keyIdx, int ringIdx, List<Integer>[] hash, int[][] cache) {

        if (keyIdx == key.length()) {
            return 0;
        }

        if (cache[ringIdx][keyIdx]  != 0) {
            return cache[ringIdx][keyIdx];
        }

        final char keyChar = key.charAt(keyIdx);
        final char ringChar = ring.charAt(ringIdx);

        if (keyChar == ringChar) {
            int result =  1 + findMinRotateStep(ring, key, keyIdx + 1, ringIdx, hash, cache);
            cache[ringIdx][keyIdx] = result;
            return result;
        }

        final List<Integer> keyList = hash[keyChar - 'a'];

        // 找到比他小的
        int temp = Integer.MAX_VALUE;

        for (int target : keyList) {
            int minRotate = Math.min(Math.abs(target - ringIdx), ring.length() - Math.abs(target - ringIdx));
            temp = Math.min(temp, 1 + minRotate + findMinRotateStep(ring, key, keyIdx + 1, target, hash, cache));
        }

        cache[ringIdx][keyIdx] = temp;

        return temp;
    }

    public int findMinMoves(int[] machines) {
        int n = machines.length;
        int sum = 0;
        for (int num : machines) {
            sum += num;
        }
        if (sum % n != 0) {
            return -1;
        }
        int avg = sum / n;
        int[] leftSums = new int[n];
        int[] rightSums = new int[n];
        for (int i = 1; i < n; i ++) {
            leftSums[i] = leftSums[i-1] + machines[i-1];
        }
        for (int i = n - 2; i >= 0; i --) {
            rightSums[i] = rightSums[i+1] + machines[i+1];
        }
        int move = 0;
        for (int i = 0; i < n; i ++) {
            int expLeft = i * avg;
            int expRight = (n - i - 1) * avg;
            int left = 0;
            int right = 0;
            if (expLeft > leftSums[i]) {
                left = expLeft - leftSums[i];
            }
            if (expRight > rightSums[i]) {
                right = expRight - rightSums[i];
            }
            move = Math.max(move, left + right);
        }
        return move;
    }

    public int removeBoxes(int[] boxes) {
        return removeBoxesHelper(boxes, 0, boxes.length - 1, 0, new int[boxes.length][boxes.length][boxes.length]);
    }

    private int removeBoxesHelper(int[] boxes, int l, int r, int k, int[][][] dp) {
        if (r < l) {
            return 0;
        }

        if (dp[l][r][k] > 0) {
            return dp[l][r][k];
        }

        final int kOld = k;
        int i = l;
        for (; i + 1 <= r && boxes[i+1] == boxes[i]; ++i) {
            k++;
        }

        int res = (k + 1) * (k + 1) + removeBoxesHelper(boxes, i + 1, r, 0, dp);

        int m = i + 1;
        for (; m <= r; ++m) {
            if (boxes[m] == boxes[l]) {
                res = Math.max(res, removeBoxesHelper(boxes, i + 1, m - 1, 0, dp) + removeBoxesHelper(boxes, m, r, k + 1, dp));
            }
        }

       dp[l][r][kOld] = res;

        return res;
    }

    public int strangePrinter(String s) {
        if (s == null || s.length() <= 0) {
            return 0;
        }

        return strangePrinterHelper(s, 0, s.length() - 1, new int[s.length()][s.length()]);
    }

    private int strangePrinterHelper(String s, int l, int r, int[][] dp) {
        if (r < l) {
            return 0;
        }

        if (dp[l][r] > 0) {
            return dp[l][r];
        }

        int i = l;
        for (; i + 1 <= r && s.charAt(i+1) == s.charAt(i); i++);

        int res = 1 + strangePrinterHelper(s, i + 1, r, dp);

        for (int m = i + 1; m <= r; ++m) {
            if (s.charAt(m) == s.charAt(l)) {
                res = Math.min(res, strangePrinterHelper(s, i + 1, m - 1, dp) + strangePrinterHelper(s, m, r, dp));
            }
        }

        dp[l][r] = res;

        return res;
    }

    public int checkRecord(int n) {
        if (n == 1) {
            return 3;
        } else if (n == 2) {
            return 8;
        }

        final int mod = 1000000007;

        int L2 = 3;
        int L1 = 1;

        int P2 = 3;
        int P1 = 1;

        int A2 = 2;
        int A1 = 1;

        int pno2 = 2;
        int pno1 = 1;

        int lno2 = 2;
        int lno1 = 1;

        for (int i = 3; i <= n; ++i) {
            int L3 = ((P2 + P1) % mod + (A2 + A1) % mod) % mod;
            int P3 = ((P2 + L2) % mod + A2 % mod) % mod;
            int A3 = (pno2 + lno2) % mod;

            // 更新pno2和lno2


            int temp = L2;
            L2 = L3;
            L1 = temp;

            temp = P2;
            P2 = P3;
            P1 = temp;

            temp = A2;
            A2 = A3;
            A1 = temp;

            int pno3 = (pno2 + lno2) % mod;
            int lno3 = (pno2 + pno1) % mod;

            temp = pno2;
            pno2 = pno3;
            pno1 = temp;

            temp = lno2;
            lno2 = lno3;
            lno1 = temp;
        }

        return ((L2 + P2) % mod + A2 % mod) % mod;
    }

    public int findIntegers(int num) {
        if (num == 0) {
            return 1;
        } else if (num == 1) {
            return 2;
        }

        final String s = Integer.toString(num, 2);
        final int length = s.length();
        int[] zeros = new int[length];
        int[] ones = new int[length];

        zeros[0] = ones[0] = 1;

        for (int i = 1; i < length; ++i) {
            zeros[i] = zeros[i-1] + ones[i-1];
            ones[i] = zeros[i-1];
        }

        int result = zeros[length - 2] + ones[length - 2];

        char preC = '1';
        for (int i = 1; i < length; ++i) {
            final char c = s.charAt(i);
            if (c == '1') {
                final int leftLength = s.length() - i - 1;
                result += (leftLength == 0 ? 1 : zeros[leftLength - 1] + ones[leftLength - 1]);

                if (preC == c) {
                    return result;
                }
            }
            preC = c;
        }

        return result + 1;
    }

    public int scheduleCourse(int[][] courses) {
        if (courses == null || courses.length <= 0) {
            return 0;
        }

        Integer[] idxes = new Integer[courses.length];
        for (int i = 0; i < idxes.length; ++i) {
            idxes[i] = i;
        }
        Arrays.sort(idxes, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                if (courses[o1][1] < courses[o2][1]) {
                    return -1;
                } else if (courses[o1][1] > courses[o2][1]) {
                    return 1;
                } else {
                    return courses[o1][0] - courses[o2][0];
                }
            }
        });

        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2 - o1;
            }
        });

        int total = 0;
        for (int i = 0; i < idxes.length; ++i) {
            total += courses[idxes[i]][0];
            maxHeap.add(courses[idxes[i]][0]);
            if (total > courses[idxes[i]][1]) {
                total -= maxHeap.poll();
            }
        }

        return maxHeap.size();
    }

    public int kInversePairs(int n, int k) {
        int mod = 1000000007;
        if (k > n*(n-1)/2 || k < 0) return 0;
        if (k == 0 || k == n*(n-1)/2) return 1;
        long[][] dp = new long[n+1][k+1];
        dp[2][0] = 1;
        dp[2][1] = 1;
        for (int i = 3; i <= n; i++) {
            dp[i][0] = 1;
            for (int j = 1; j <= Math.min(k, i*(i-1)/2); j++) {
                dp[i][j] = dp[i][j-1] + dp[i-1][j];
                if (j >= i) dp[i][j] -= dp[i-1][j-i];
                dp[i][j] = (dp[i][j]+mod) % mod;
            }
        }
        return (int) dp[n][k];
    }

    public int kthSmallest(int[][] matrix, int k) {
        int low = matrix[0][0];

        int hi = matrix[matrix.length - 1][matrix[0].length - 1];

        while (low <= hi) {
            int middle = low + (hi - low) / 2;
            int count = getLessEqualCount(matrix, middle);
            if (count >= k) {
                hi = middle - 1;
            } else {
                low = middle + 1;
            }
        }

        return low;
    }

    private int getLessEqualCount(int[][] matrix, int val) {
        int count = 0;

        int row = 0;
        int column = matrix[0].length - 1;

        while (row < matrix.length && column >= 0) {
            if (matrix[row][column] > val) {
                column--;
            } else {
                count += column + 1;
                row++;
            }
        }

        return count;
    }

    public int findKthNumber(int m, int n, int k) {
        int low = 1;

        int hi = m * n;

        while (low <= hi) {
            int middle = low + (hi - low) / 2;
            int count = getLessEqualCount(m, n, middle);
            if (count >= k) {
                hi = middle - 1;
            } else {
                low = middle + 1;
            }
        }

        return low;
    }

    private int getLessEqualCount(int m, int n, int val) {
        int count = 0;

        int row = 0;
        int column = n - 1;

        while (row < m && column >= 0) {
            if ((row + 1) * (column + 1) > val) {
                column--;
            } else {
                count += column + 1;
                row++;
            }
        }

        return count;
    }

    public int smallestDistancePair(int[] nums, int k) {
        Arrays.sort(nums);

        PriorityQueue<int[]> minHeap = new PriorityQueue<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return nums[o1[1]] - nums[o1[0]] - (nums[o2[1]] - nums[o2[0]]);
            }
        });

        for (int i = 0; i < nums.length - 1; ++i) {
            minHeap.add(new int[]{i,i+1});
        }

        int temp = 0;
        int result = 0;
        while (++temp <= k) {
            final int[] min = minHeap.poll();

            result = nums[min[1]] - nums[min[0]];

            if (min[1] < nums.length - 1) {
                minHeap.add(new int[]{min[0], min[1] + 1});
            }
        }

        return result;
    }

    public int smallestDistancePair2(int[] nums, int k) {
        Arrays.sort(nums);

        int hi = nums[nums.length - 1] - nums[0];
        int lo = 0;

        while (lo <= hi) {
            int mi = lo + (hi - lo) / 2;
            if (distanceValid(nums, mi, k)) {
                hi = mi - 1;
            } else {
                lo = mi + 1;
            }
        }
        return lo;
    }

    private boolean distanceValid(int[] nums, int distance, int k) {
        int result = 0;

        int i = 0;
        int j = 0;
        while (i < nums.length) {
            while (j + 1 < nums.length && nums[j + 1] - nums[i] <= distance) {
                j++;
            }

            result += j - i;

            i++;

            if (i > j) {
                j = i;
            }
        }

        return result >= k;
    }

    public int cutOffTree(List<List<Integer>> forest) {
        int rows = forest.size();
        int columns = forest.get(0).size();

        // 找到所有的排序，
        ArrayList<Integer> treesPos = new ArrayList<>();
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                if (forest.get(i).get(j) > 1){
                    treesPos.add(i * rows + j);
                }
            }
        }

        Collections.sort(treesPos, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return forest.get(o1 / columns).get(o1 % columns) - forest.get(o2 / columns).get(o2 % columns);
            }
        });

        int result = 0;
        int prePos = 0;
        for (Integer pos : treesPos) {
            if (prePos == pos) {
                continue;
            }
            int steps = cutOffTree(forest, prePos / columns, prePos % columns, pos / columns, pos % columns);
            if (steps == -1) {
                return -1;
            } else {
                result += steps;
            }
            prePos = pos;
        }

        return result;
    }

    private static int[][] sTreeDir = {
            {1, 0},
            {-1, 0},
            {0, 1},
            {0, -1}
    };

    public int cutOffTree(List<List<Integer>> forest, int startRow, int startColumn, int endRow, int endColumn) {
        final int rows = forest.size();
        final int columns = forest.get(0).size();
        Queue<Integer> queue = new LinkedList<>();

        final int start = startRow * columns + startColumn;
        queue.add(start);
        Set<Integer> visited = new HashSet<>();
        visited.add(start);

        int level = 0;
        int cur = 1;
        int next = 0;

        while (!queue.isEmpty()) {
            final int head = queue.poll();
            cur--;

            final int headRow = head / columns;
            final int headColumn = head % columns;

            for (int[] dir : sTreeDir) {
                final int nextRow = dir[0] + headRow;
                final int nextColumn = dir[1] + headColumn;
                if (nextRow >= 0 && nextRow < rows && nextColumn >= 0 && nextColumn < columns) {
                    if (nextRow == endRow && nextColumn == endColumn) {

                        return level + 1;
                    } else if (forest.get(nextRow).get(nextColumn) == 0) {
                        continue;
                    }
                    final int nextPos = nextRow * columns + nextColumn;
                    if (!visited.add(nextPos)) {
                        continue;
                    } else {
                        queue.add(nextPos);
                        next++;
                    }
                }
            }

            if (cur == 0) {
                cur = next;
                next = 0;
                level += 1;
            }
        }

        return -1;
    }

    public long numPermsDISequence(String S) {
        return numPermsDISequence(S, 0, S.length() - 1, new HashMap<>());
    }

    static Map<String,Integer> sCombMap = new HashMap<String, Integer>();
    private static int comb(int m,int n){
        String key= m+","+n;
        if(n==0)
            return 1;
        if (n==1)
            return m;
        if(n>m/2)
            return comb(m,m-n);
        if(n>1){
            if(!sCombMap.containsKey(key))
                sCombMap.put(key, (comb(m-1,n-1)+comb(m-1,n)) % MOD);
            return sCombMap.get(key);
        }
        return -1;
    }

    private static int MOD = 1000000007;

    // 代表有k个数，组成从start到end的数量
    private int numPermsDISequence(String s, int start, int end, HashMap<String, Integer> cache) {
        if (start >= end) {
            return 1;
        }

        final String key = start + "," + end;
        final int cacheRes = cache.getOrDefault(key, -1);
        if (-1 != cacheRes) {
            return cacheRes;
        }

        // 思想就是将最后一个数一次放在不同的位置，然后统计左边的和右边的各有多少个
        int result = 0;
        for (int i = end; i >= start - 1; --i) {
            // 放到最后一个,最后一个必须为I
            if (i == end) {
                if (s.charAt(i) =='I')
                    result = (result + numPermsDISequence(s, start, end - 1, cache)) % MOD;
            } else if (i == start - 1) {
                // 第一个
                if (s.charAt(start) == 'D')
                    result = (result + numPermsDISequence(s, start + 1, end, cache)) % MOD;
            } else {
                // 放在中间
                if (s.charAt(i) == 'I' && s.charAt(i + 1) == 'D') {
                    final int total = end - start + 1;
                    final int pre = i - start + 1;
                    final int combCnt = comb(total, pre);
                    final int preRes = numPermsDISequence(s, start, i - 1, cache);
                    final int suffixRes = numPermsDISequence(s, i + 2, end, cache);

                    final int preMultiSuffix = (int) ((1L * preRes * suffixRes) % MOD);

                    int temp = (int) (1L * preMultiSuffix * combCnt % MOD);

                    result = (temp + result) % MOD;
                }
            }
        }

        cache.put(key, result);

        return result;
    }

    public int[] maxSumOfThreeSubarrays(int[] nums, int k) {
        return null;
    }


    public int maxProfit(int[] prices) {
        if (prices == null || prices.length <= 1) {
            return 0;
        }

        int min = prices[0];
        int result = 0;
        for (int i = 1; i < prices.length; ++i) {
            if (min > prices[i]) {
                min = prices[i];
            } else {
                result = Math.max(result, prices[i] - min);
            }
        }

        return result;
    }

    public int maxProfitII(int[] prices) {
        if (prices == null || prices.length <= 1) {
            return 0;
        }

        int start = 0;

        int result = 0;
        int i = 0;
        while (i < prices.length) {
            while (i + 1 < prices.length && prices[i+1] >= prices[i]) {
                i++;
            }

            if (i > start) {
                result += prices[i] - prices[start];
            }

            i++;
            start = i;
        }

        return result;
    }

    // 最多执行两次
    public int maxProfitIII(int[] prices) {

        if (prices == null || prices.length < 2) {
            return 0;
        }

        int[] maxRight = new int[prices.length];

        int resultRight = 0;
        int max = prices[prices.length - 1];
        maxRight[prices.length - 1] = resultRight;

        for (int i = prices.length - 2; i >= 0; --i) {
            if (max < prices[i]) {
                max = prices[i];
            } else {
                resultRight = Math.max(resultRight, max - prices[i]);
            }

            maxRight[i] = resultRight;
        }

        int min = prices[0];
        int result = 0;

        int resultTotal = 0;
        for (int i = 1; i < prices.length; ++i) {
            if (min > prices[i]) {
                min = prices[i];
            } else {
                result = Math.max(result, prices[i] - min);
            }

            resultTotal = Math.max(resultTotal, result + maxRight[i]);
        }

        return resultTotal;
    }

    // 最多执行k次
    public int maxProfitIV(int k, int[] prices) {
        int[][] dp = new int[prices.length][k + 1];
        for (int i = 0; i < dp.length; ++i) {
            for (int j = 0; j < dp[0].length; ++j) {
                dp[i][j] = -1;
            }
        }

        if (k >= prices.length - 1) {
            return maxProfitII(prices);
        }
        return maxProfitIV(prices, 0, k, dp);
    }

    private int maxProfitIV(int[] prices, int start, int k, int[][] dp) {

        if (k <= 0 || prices.length - start <= 1) {
            return 0;
        }

        if (dp[start][k] != -1) {
            return dp[start][k];
        }

        int result = 0;

        int total = 0;
        int min = prices[start];
        for (int i = start + 1; i < prices.length; ++i) {
            if (min > prices[i]) {
                min = prices[i];
            } else {
                result = Math.max(result, prices[i] - min);
            }

            total = Math.max(total, result + maxProfitIV(prices, i, k - 1, dp));
        }

        dp[start][k] = total;

        return total;
    }

    public int minStickers(String[] stickers, String target) {
        int m = stickers.length;
        int[][] map = new int[m][26];
        Map<String, Integer> dp = new HashMap<>();
        for (int i = 0; i < m; i++)
            for (char ch : stickers[i].toCharArray()) map[i][ch - 'a']++;
        dp.put("", 0);
        return helper(dp, map, target);
    }
    private int helper(Map<String, Integer> dp, int[][] map, String target) {
        if (dp.containsKey(target)) return dp.get(target);
        int ans = Integer.MAX_VALUE, n = map.length;
        int[] tar = new int[26];
        for (char ch : target.toCharArray()) tar[ch - 'a']++;
        // try every sticker
        for (int i = 0; i < n; i++) {
            //optimization
            if (map[i][target.charAt(0) - 'a'] == 0) continue;
            StringBuilder sb = new StringBuilder();
            // apply a sticker on every character a - z
            for (int j = 0; j < 26; j++) {
                if (tar[j] > 0) {
                    for (int k = 0; k < Math.max(0, tar[j] - map[i][j]); k++) {
                        sb.append((char)('a' + j));
                    }
                }
            }
            String s = sb.toString();
            int tmp = helper(dp, map, s);
            if (tmp != -1) ans = Math.min(ans, 1 + tmp);
        }
        dp.put(target, ans == Integer.MAX_VALUE ? -1 : ans);
        return dp.get(target);
    }

    public List<Integer> fallingSquares(int[][] positions) {
        Set<Integer> coords = new HashSet();
        for (int[] pos: positions) {
            coords.add(pos[0]);
            coords.add(pos[0] + pos[1] - 1);
        }
        List<Integer> sortedCoords = new ArrayList(coords);
        Collections.sort(sortedCoords);

        Map<Integer, Integer> index = new HashMap();
        int t = 0;
        for (int coord: sortedCoords) index.put(coord, t++);

        SegmentTree tree = new SegmentTree(sortedCoords.size());
        int best = 0;
        List<Integer> ans = new ArrayList();

        for (int[] pos: positions) {
            int L = index.get(pos[0]);
            int R = index.get(pos[0] + pos[1] - 1);
            int h = tree.query(L, R) + pos[1];
            tree.update(L, R, h);
            best = Math.max(best, h);
            ans.add(best);
        }
        return ans;
    }

//    class SegmentTree {
//        int N, H;
//        int[] tree, lazy;
//
//        SegmentTree(int N) {
//            this.N = N;
//            H = 1;
//            while ((1 << H) < N) H++;
//            tree = new int[2 * N];
//            lazy = new int[N];
//        }
//
//        private void apply(int x, int val) {
//            tree[x] = Math.max(tree[x], val);
//            if (x < N) lazy[x] = Math.max(lazy[x], val);
//        }
//
//        private void pull(int x) {
//            while (x > 1) {
//                x >>= 1;
//                tree[x] = Math.max(tree[x * 2], tree[x * 2 + 1]);
//                tree[x] = Math.max(tree[x], lazy[x]);
//            }
//        }
//
//        private void push(int x) {
//            for (int h = H; h > 0; h--) {
//                int y = x >> h;
//                if (lazy[y] > 0) {
//                    apply(y * 2, lazy[y]);
//                    apply(y * 2 + 1, lazy[y]);
//                    lazy[y] = 0;
//                }
//            }
//        }
//
//        public void update(int L, int R, int h) {
//            L += N;
//            R += N;
//            int L0 = L, R0 = R, ans = 0;
//            while (L <= R) {
//                if ((L & 1) == 1) apply(L++, h);
//                if ((R & 1) == 0) apply(R--, h);
//                L >>= 1;
//                R >>= 1;
//            }
//            pull(L0);
//            pull(R0);
//        }
//
//        public int query(int L, int R) {
//            L += N;
//            R += N;
//            int ans = 0;
//            push(L);
//            push(R);
//            while (L <= R) {
//                if ((L & 1) == 1) ans = Math.max(ans, tree[L++]);
//                if ((R & 1) == 0) ans = Math.max(ans, tree[R--]);
//                L >>= 1;
//                R >>= 1;
//            }
//            return ans;
//        }
//    }

    public String countOfAtoms(String formula) {
        Map<String, Integer> map = countOfAtomsHelper(formula);

        String result = "";

        for (Map.Entry<String, Integer> entry : map.entrySet()) {
            result += entry.getKey();
            if (entry.getValue() > 1) {
                result += entry.getValue();
            }
        }
        return result;
    }

    private int mAtomIndex = 0;
    private Map<String, Integer> countOfAtomsHelper(String formula) {
        Map<String, Integer> result = new TreeMap<>();
        if (mAtomIndex >= formula.length()) {
            return result;
        }

        String key = null;
        int keyTimes = 0;
        while (mAtomIndex < formula.length()) {
            final char c = formula.charAt(mAtomIndex);
            mAtomIndex++;

            if (c == '(') {
                atomsMerge(result, key, keyTimes);
                key = null;
                keyTimes = 0;
                atomsMerge(result, countOfAtomsHelper(formula));
            } else if (c == ')') {
                atomsMerge(result, key, keyTimes);

                int times = 0;
                while (mAtomIndex < formula.length() && Character.isDigit(formula.charAt(mAtomIndex))) {
                    times *= 10;
                    times += formula.charAt(mAtomIndex) - '0';
                    mAtomIndex++;
                }

                atomsMerge(result, times);

                return result;
            } else if (Character.isUpperCase(c)) {
                atomsMerge(result, key, keyTimes);
                keyTimes = 0;
                key = Character.toString(c);

            } else if(Character.isLowerCase(c)) {

                key += Character.toString(c);

            } else if (Character.isDigit(c)) {
                keyTimes *= 10;
                keyTimes += c - '0';
            }

            if (mAtomIndex >= formula.length()) {
                atomsMerge(result, key, keyTimes);
            }
        }
        return result;

    }
    private void atomsMerge(Map<String, Integer> to, int times) {
        if (to == null || times == 0) {
            return;
        }

        for (Map.Entry<String, Integer> entry : to.entrySet()) {
            to.put(entry.getKey(), entry.getValue() * times);
        }
    }

    private void atomsMerge(Map<String, Integer> to, Map<String, Integer> from) {
        if (from == null) {
            return;
        }

        for (Map.Entry<String, Integer> entry : from.entrySet()) {
            to.put(entry.getKey(), to.getOrDefault(entry.getKey(), 0) + entry.getValue());
        }
    }

    private void atomsMerge(Map<String, Integer> to, String from, int times) {
        if (from == null) {
            return;
        }

        if (times == 0) {
            times = 1;
        }
        to.put(from, to.getOrDefault(from, 0) + times);
    }

    public int countPalindromicSubsequences(String S) {
        if (S == null || S.length() <= 0) {
            return 0;
        }

        final int mod = 1000000000 + 7;

        final int length = S.length();

        int[][] dp = new int[length][length];

        for (int i = 0; i < length; ++i) {
            dp[i][i] = 1;
        }

        for (int k = 1; k <= length - 1; ++k) {
            for (int i = 0; i + k < length; ++i) {
                final int j = i + k;

                final int cleft = S.charAt(i);
                final int cright = S.charAt(j);
                if (cleft != cright) {
                    dp[i][j] = (dp[i + 1][j] + dp[i][j - 1]) % mod - dp[i+1][j-1] % mod;
                } else {
                    dp[i][j] = 2 * dp[i + 1][j - 1] % mod;
                    int left = i + 1;
                    int right = j - 1;

                    while (left <= right && S.charAt(left) != cleft) {
                        left++;
                    }

                    while (left <= right && S.charAt(right) != cleft) {
                        right--;
                    }

                    if (left == right) {
                        dp[i][j] += 1;
                    } else if (left > right) {
                        dp[i][j] += 2;
                    } else {
                        dp[i][j] -= dp[left + 1][right - 1];
                    }
                }

                dp[i][j] = (dp[i][j] + mod) % mod;
            }
        }

        return dp[0][length - 1];
    }

    public int evaluate(String expression) {
        return 0;
    }

    public int cherryPickup(int[][] grid) {
        int[][][] map = new int[grid.length][grid[0].length][2];

        for (int i = 0; i < grid.length; ++i) {
            for (int j = 0; j < map[0].length; ++j) {
                for (int k = 0; k < map[0][0].length; ++k) {
                    map[i][j][k] = -1;
                }
            }
        }
        int result = cherryPickupHelper(grid, 0, 0, true);
        return Math.max(0, result);
    }

    private int cherryPickupHelper(int[][] grid, int x, int y, boolean down) {
        if (x == 0 && y == 0 && !down) {
            return 0;
        }

        int result = Integer.MIN_VALUE;
        final int old = grid[x][y];
        grid[x][y] = 0;
        if (x == grid.length - 1 && y == grid[0].length - 1 && down) {
            int reverseTemp = cherryPickupHelper(grid, x, y, false);
            if (reverseTemp != Integer.MIN_VALUE) {
                result = old + reverseTemp;
            }
        } else {
            if (down) {
                // 向右走

                if (y + 1 < grid[0].length && grid[x][y+1] != -1) {
                    int rightTemp = cherryPickupHelper(grid, x, y + 1, down);
                    if (rightTemp != Integer.MIN_VALUE) {
                        result = Math.max(result, old + rightTemp);
                    }
                }

                // 往下走
                if (x + 1 < grid.length && grid[x + 1][y] != -1) {
                    int downTemp = cherryPickupHelper(grid, x + 1, y, down);
                    if (downTemp != Integer.MIN_VALUE) {
                        result = Math.max(result, old + downTemp);
                    }

                }
            } else {
                // 向右走

                if (y - 1 >= 0 && grid[x][y - 1] != -1) {
                    int rightTemp = cherryPickupHelper(grid, x, y - 1, down);
                    if (rightTemp != Integer.MIN_VALUE) {
                        result = Math.max(result, old + rightTemp);
                    }
                    grid[x][y] = old;
                }

                // 往下走
                if (x - 1 >= 0 && grid[x - 1][y] != -1) {
                    int downTemp = cherryPickupHelper(grid, x - 1, y, down);
                    if (downTemp != Integer.MIN_VALUE) {
                        result = Math.max(result, old + downTemp);
                    }

                }
            }
        }

        grid[x][y] = old;

        return result;
    }

    public String crackSafe(int n, int k) {
        /* backtracking: each node is a password
        https://www.youtube.com/watch?v=kRdlLahVZDc */

        int size = (int) Math.pow(k, n);  /* total num nodes inside graph */

        /* initialize password to be n digits of '0's */
        char[] password = new char[n];
        Arrays.fill(password, '0');
        StringBuilder res = new StringBuilder(String.valueOf(password));

        Set<String> visited = new HashSet<>();
        visited.add(res.toString());

        /* traverse each node exactly once to minimize the result  */
        if (dfs(res, visited, n, k, size)) return res.toString();
        return "";
    }

    /*  A node in the graph is one possible combination of n digits
        where each digit is chosen from [0,k)
        e.g.    n = 2, k = 2,
                total combination = 4 (00, 01, 10, 11)
                n = 3, k = 6
                total combination = 6^3
                (each of the 3 digits has 6 possibilities, [0-5])
    */
    public boolean dfs(StringBuilder res, Set<String> visited, int n, int k, int size) {
        /* base case: all nodes are visited  */
        if (visited.size() == size) return true;

        /* reuse (n-1) digits from last node to form new node */
        String prefix = res.substring(res.length()-n+1, res.length());

	   /* append one digit to prefix */
        for (char i = '0'; i<'0'+ k; i++) {
            String password = prefix+i;
            if (!visited.contains(password)) {
                res.append(i);
                visited.add(password);
                if (dfs(res, visited, n, k, size)) return true;
                visited.remove(password);  /* backtracking */
                res.deleteCharAt(res.length()-1);
            }
        }
        return false;
    }

    public int intersectionSizeTwo(int[][] intervals) {
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[1]));
        TreeSet<Integer> set = new TreeSet<>();

        for(int[] interval: intervals) {
            int start = interval[0];
            int end = interval[1];
            Integer higher = set.floor(end);                            // lower or equal to end or null
            Integer lower = higher != null ? set.lower(higher) : null;  // lower than higher or null

            if(higher == null || higher < start) {  // zero integers of that range in set
                set.add(end);
                set.add(end - 1);
            } else if(higher >= start && lower != null && lower < start) { // one element from that range in set
                if(set.contains(end)) {
                    set.add(end - 1);
                } else {
                    set.add(end);
                }
            }
        }
        return set.size();
    }

    public String makeLargestSpecial(String S) {
        return "";
    }

    public int maxChunksToSorted(int[] arr) {
        if (arr == null) {
            return 0;
        } else if (arr.length <= 1) {
            return arr.length;
        }

        int result = 0;

        int max = -1;
        for (int i = 0; i < arr.length; ++i) {
            if (max == -1) {
                max = arr[i];
            } else {
                max = Math.max(max, arr[i]);
            }

            if (i == max && arr[i] <= max) {
                result++;
                max = -1;
            }
        }

        return result;
    }

    public int maxChunksToSortedII(int[] arr) {
        if (arr == null) {
            return 0;
        } else if (arr.length <= 1) {
            return arr.length;
        }

        int[] sorted = new int[arr.length];
        for (int i = 0; i < sorted.length; ++i) {
            sorted[i] = arr[i];
        }

        Arrays.sort(sorted);

        int result = 0;

        int nonzero = 0;
        Map<Integer, Integer> map = new HashMap<>();

        for (int i = 0; i < sorted.length; ++i) {
            int sortval = sorted[i];
            final int oldSort = map.getOrDefault(sortval, 0) + 1;
            map.put(sortval, oldSort);

            if (oldSort == 0) {
                nonzero--;
            } else if (oldSort == 1) {
                nonzero++;
            }

            final int oldVa = map.getOrDefault(arr[i], 0) - 1;
            map.put(arr[i], oldVa);

            if (oldVa == 0) {
                nonzero--;
            } else if (oldVa == -1){
                nonzero++;
            }

            if (nonzero == 0) {
                result++;
            }
        }

        return result;
    }

    public int slidingPuzzle(int[][] board) {
        String target = "123450";
        String start = getStart(board);
        Queue<String> queue = new LinkedList();
        Set<String> visited = new HashSet();
        int moves = 0;
        queue.offer(start);

        while(!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                String current = queue.poll();
                if (current.equals(target)) {
                    return moves;
                }
                int zeroPosition = current.indexOf('0');
                int[] nextPositions = getNextPossiblePositions(zeroPosition);
                for(int nextPosition : nextPositions) {
                    String nextMove = swapPosition(current, zeroPosition, nextPosition);
                    if (visited.contains(nextMove)) {
                        continue;
                    }
                    queue.offer(nextMove);
                    visited.add(nextMove);
                }
            }
            moves++;
        }

        return -1;
    }

    private String swapPosition(String current, int zeroPosition, int nextPosition) {
        StringBuilder next = new StringBuilder();
        for(int i=0 ; i<current.length(); i++) {
            if(i == zeroPosition) {
                next.append(current.charAt(nextPosition));
            } else if(i == nextPosition) {
                next.append(current.charAt(zeroPosition));
            } else {
                next.append(current.charAt(i));
            }
        }

        return next.toString();
    }

    private int[] getNextPossiblePositions(int currentPosition) {
        if (currentPosition == 0) {
            return new int[]{1,3};
        } else if(currentPosition == 1) {
            return new int[]{0,2,4};
        } else if(currentPosition == 2) {
            return new int[]{1,5};
        } else if(currentPosition == 3) {
            return new int[]{0,4};
        } else if(currentPosition == 4) {
            return new int[]{1,3,5};
        } else if(currentPosition == 5) {
            return new int[]{2,4};
        }

        return null;
    }


    private String getStart(int[][] board) {
        StringBuilder start = new StringBuilder();
        for(int i=0 ; i<board.length; i++) {
            for(int j=0; j<board[i].length; j++) {
                start.append(board[i][j]);
            }
        }

        return start.toString();
    }

    public int swimInWater(int[][] grid) {
        return Dijkstra(grid);
    }

    private int Dijkstra(int[][] grid) {
        final int rows = grid.length;
        final int columns = grid[0].length;

        final int total = rows * columns;

        final boolean[] S = new boolean[total];
        final int[] distance = new int[total];

        final int[][] dijkstraDir = {
                {0, -1},
                {0, 1},
                {1, 0},
                {-1, 0}
        };

        for (int i = 0; i < total; ++i) {
            distance[i] = Integer.MAX_VALUE;
        }

        for (int[] dir : dijkstraDir) {
            final int row = dir[0];
            final int column = dir[1];
            if (row >= 0 && row < rows && column >= 0 && column < columns) {
                distance[row * columns + column] = Math.max(grid[row][column] - grid[0][0], 0);
            }
        }

        distance[0] = grid[0][0];
        S[0] = true;

        for (int k = 1; k <= total - 1; ++k) {

            int minDis = Integer.MAX_VALUE;
            int idx = -1;
            for (int i = 0; i < total; ++i) {
                if (!S[i] && distance[i] < minDis) {
                    minDis = distance[i];
                    idx = i;
                }
            }

            if (idx == total - 1) {
                return minDis + distance[0];
            }

            S[idx] = true;
            distance[idx] = minDis;

            final int oldRow = idx / columns;
            final int oldColumn = idx % columns;

            grid[oldRow][oldColumn] = minDis + distance[0];

            for (int[] dir : dijkstraDir) {

                final int row = dir[0] + oldRow;
                final int column = dir[1] + oldColumn;
                if (row >= 0 && row < rows && column >= 0 && column < columns) {
                    final int newIdx = row * columns + column;
                    if (!S[newIdx]) {
                        final int tempDis =  distance[idx] + Math.max(0, grid[row][column] - grid[oldRow][oldColumn]);
                        if (tempDis < distance[newIdx]) {
                            distance[newIdx] = tempDis;
                        }
                    }
                }
            }
        }

        return distance[total - 1] + distance[0];
    }

    public boolean reachingPoints(int sx, int sy, int tx, int ty) {
        if (sx == tx && sy == ty) {
            return true;
        } else if (sx > tx || sy > ty) {
            return false;
        }

        if (tx == ty) {
            return false;
        }

        if (tx == sx) {
            return (ty - sy) % sx == 0;
        } else if (ty == sy) {
            return (tx - sx) % sy == 0;
        }

        if (tx > ty) {
            return reachingPoints(sx, sy, tx - ty, ty);
        } else {
            return reachingPoints(sx, sy, tx, ty - tx);
        }
    }

    public int[] kthSmallestPrimeFraction(int[] A, int K) {

        PriorityQueue<int[]> minHeap = new PriorityQueue<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return A[o1[0]] * A[o2[1]] - A[o1[1]] * A[o2[0]];
            }
        });

        for (int i = 0; i < A.length - 1; ++i) {
            minHeap.add(new int[]{0, i + 1});
        }

        while (--K > 0) {
            int[] head = minHeap.poll();

            int start = head[0];
            int end = head[1];

            if (start == end - 1) {
                continue;
            }

            minHeap.add(new int[]{start + 1, end});
        }

        return minHeap.poll();
    }

    public int preimageSizeFZF(int K) {
        if (K == 0) {
            return 5;
        }

        // 走两次二分查找，查找等于K的最大值和最小值
        long max = preimageSizeFZFBinaryHelper(K, true);
        long min = preimageSizeFZFBinaryHelper(K, false);

        return (int) (max < min ? 0 : max - min + 1);
    }

    public long preimageSizeFZFBinaryHelper(int K, boolean max) {
        long start = 5;
        long right = 5L * (K + 1) - 1;

        while (start <= right) {
            long mid = start + (right - start) / 2;
            long zeros = trailZeros(mid);

            if (zeros < K) {
                start = mid + 1;
            } else if(zeros > K) {
                right = mid - 1;
            } else if (max) {
                start = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return max ? right : start;
    }

    public long trailZeros(long num) {
        long sum = 0;
        long left = num / 5;
        while (left > 0) {
            sum += left;
            left /= 5;
        }

        return sum;
    }

    public int bestRotation(int[] A) {
        return 0;
    }


    public int[] hitBricks(int[][] grid, int[][] hits) {
        int[] res = new int[hits.length];
        int m = grid.length;
        int n = grid[0].length;
        int idx = 0;
        for (int[] v: hits) {
            int r = v[0], c = v[1];
            int[] count = {0};
            grid[r][c] = 0;
            if(!reachTop(grid, r - 1, c, m, n)) {
                erase(grid, r - 1, c, m, n, count);
            }
            if(!reachTop(grid, r + 1, c, m, n)) {
                erase(grid, r + 1, c, m, n, count);
            }
            if(!reachTop(grid, r, c - 1, m, n)) {
                erase(grid, r, c - 1, m, n, count);
            }
            if(!reachTop(grid, r, c + 1, m, n)) {
                erase(grid, r, c + 1, m, n, count);
            }
            res[idx++] = count[0];
        }
        return res;
    }

    boolean reachTop(int[][] grid, int r, int c, int m, int n) {
        if (r < 0 || c < 0 || r == m || c == n || grid[r][c] == 0) {
            return false;
        }
        if (r == 0) {
            return true;
        }
        int tmp = grid[r][c];
        grid[r][c] = 0;
        boolean res = reachTop(grid, r + 1, c, m, n) || reachTop(grid, r, c + 1, m, n)
                || reachTop(grid, r - 1, c, m, n) || reachTop(grid, r, c - 1, m, n);
        grid[r][c] = tmp;
        return res;
    }

    void erase(int[][] grid, int r, int c, int m, int n, int[] count) {
        if (r < 0 || c < 0 || r == m || c == n || grid[r][c] == 0) {
            return;
        }
        ++count[0];
        grid[r][c] = 0;
        erase(grid, r + 1, c, m, n, count);
        erase(grid, r - 1, c, m, n, count);
        erase(grid, r, c + 1, m, n, count);
        erase(grid, r, c - 1, m, n, count);
    }

    public int numBusesToDestination(int[][] routes, int S, int T) {
        if(routes == null || routes.length == 0 || S == T) return 0;
        HashMap<Integer, HashSet<Integer>> map = new HashMap<Integer, HashSet<Integer>>();  //  <key: stop, value: set of routes>
        for(int i = 0; i < routes.length; i++){
            for(int stop : routes[i]){
                if(!map.containsKey(stop)) map.put(stop, new HashSet<Integer>());
                map.get(stop).add(i);
            }
        }

        HashSet<Integer> visited = new HashSet<Integer>(); // visited routes
        Queue<Integer> queue = new LinkedList<Integer>();

        for(int route : map.get(S)){
            queue.add(route);
        }

        int lastLevel = map.get(S).size();
        int curLevel = 0;
        int step = 1;

        while(!queue.isEmpty()){
            for(int i = 0; i < lastLevel && !queue.isEmpty(); i++){

                int curRouteId = queue.poll();

                if(visited.contains(curRouteId)) continue;
                visited.add(curRouteId);


                int[] curStops = routes[curRouteId];
                for(int stop: curStops){
                    if(stop == T) return step;
                    for(int route : map.get(stop)){
                        curLevel++;
                        queue.add(route);
                    }
                }
            }
            step++;
            lastLevel = curLevel;
            curLevel = 0;
        }
        return -1;
    }

    public int racecar(int target) {
        if (target == 0) {
            return 0;
        }

        int[] dp = new int[target + 1];

        dp[0] = 0;
        int boundary = 1;

        do {
            final int idx = (1 << boundary) - 1;

            if (idx == target) {
                return dp[idx];
            } else if (idx > target) {
                break;
            } else {
                dp[idx] = boundary;
                boundary++;
            }
        } while (true);


        for (int i = 1; i <= boundary - 1; ++i) {

            final int start = (1 << i) - 1;
            final int end = (1 << (i + 1)) - 1;

            for (int j = start + 1; j < end; ++j) {
                // 在超过的地方往回走
                dp[j] = (end > target ? boundary : dp[end]) + 1 + dp[end - j];

                // 在还没超过的地方继续往前走，或者往后走，再往前走
                for (int backStep = 0; backStep <= 1 << (i - 1); ++backStep) {
                    dp[j] = Math.min(dp[j], dp[start] + 1 + dp[backStep] + 1 + dp[backStep + j - start]);
                }

                if (j == target) {
                    return dp[j];
                }
            }
        }


        return 0;
    }

    public int largestIsland(int[][] grid) {
        int totalIsland = 0;

        Map<Integer, Integer> map = new HashMap<>();

        int mark = 2;
        for (int i = 0; i < grid.length; ++i) {
            for (int j = 0; j < grid[0].length; ++j) {
                int markCnt = markIsland(grid, i, j, mark);

                if (markCnt != 0) {
                    map.put(mark, markCnt);
                    mark++;
                    totalIsland += markCnt;
                }
            }
        }

        int total = grid.length * grid[0].length;

        if (totalIsland == total) {
            return total;
        } else if (totalIsland == 0) {
            return 1;
        }

        int result = 1;
        for (int x = 0; x < grid.length; ++x) {
            for (int y = 0; y < grid[0].length; ++y) {
                if (grid[x][y] == 0) {
                    Set<Integer> unioned = new HashSet<>();
                    if (x - 1 >= 0 && grid[x - 1][y] != 0) {
                        unioned.add(grid[x - 1][y]);
                    }

                    if (x + 1 < grid.length && grid[x+1][y] != 0) {
                        unioned.add(grid[x + 1][y]);
                    }

                    if (y - 1 >= 0 && grid[x][y-1] != 0) {
                        unioned.add(grid[x][y - 1]);
                    }

                    if (y + 1 < grid[0].length && grid[x][y+1] != 0) {
                        unioned.add(grid[x][y+1]);
                    }

                    int temp = 1;
                    for (int markVal : unioned) {
                        temp += map.get(markVal);
                    }

                    result = Math.max(result, temp);
                }
            }
        }

        return result;
    }

    private int markIsland(int[][] grid, int x, int y, int mark) {
        if (grid[x][y] != 1) {
            return 0;
        }

        int result = 1;

        grid[x][y] = mark;
        if (x - 1 >= 0 && grid[x - 1][y] == 1) {
            result += markIsland(grid, x-1, y, mark);
        }

        if (x + 1 < grid.length && grid[x+1][y] == 1) {
            result += markIsland(grid, x+1, y, mark);
        }

        if (y - 1 >= 0 && grid[x][y-1] == 1) {
            result += markIsland(grid, x, y-1, mark);
        }

        if (y + 1 < grid[0].length && grid[x][y+1] == 1) {
            result += markIsland(grid, x, y+1, mark);
        }

        return result;
    }

    public int uniqueLetterString(String S) {
        if (S == null || S.length() <= 0) {
            return 0;
        }

        // 便利的时候，记录A到Z最后两次出现的位置
        int[][] hash = new int[26][2];

        for (int[] one : hash) {
            one[0] = one[1] = -1;
        }

        final int mod = 1000000007;
        int result = 1;
        int pre = 1;
        hash[S.charAt(0) - 'A'][0] = 0;

        for (int i = 1; i < S.length(); ++i) {
            final int idx = S.charAt(i) - 'A';
            int next = 1 + pre;

            if (hash[idx][0] == -1 && hash[idx][1] == -1) {
                next += i;
                hash[idx][0] = i;
            } else if (hash[idx][0] != -1 && hash[idx][1] != -1) {
                // 两个都有
                next += (i - hash[idx][1] - 1);
                next -= hash[idx][1] - hash[idx][0];

                hash[idx][0] = hash[idx][1];
                hash[idx][1] = i;
            } else if (hash[idx][0] != -1 && hash[idx][1] == -1) {
                next += (i - hash[idx][0] - 1);
                next -= hash[idx][0] + 1;

                hash[idx][1] = i;
            }

            result = (next + result) % mod;
            pre = next;
        }

        return result;
    }

    public int consecutiveNumbersSum(int N) {
        if (N <= 1) {
            return N;
        }

        int start = 1;
        int end = 1;

        int result = 0;

        int sum = 0;
        while (end <= (N + 1) / 2) {
            sum += end;

            while (sum > N) {
                sum -= start;
                start++;
            }

            if (sum == N) {
                result++;
            }

            end++;
        }

        return result + 1;
    }

    public int consecutiveNumbersSumII(int N) {
        if (N <= 1) {
            return N;
        }

        // 从m开始，总共有n个数，组成的和是N
        // n * m + (n - 1) * n / 2 = N

        int maxn = (int) Math.round(Math.sqrt(2L * N));

        int result = 0;

        for (int n = 1; n <= maxn; ++n) {
            if ((N - (n - 1) * n / 2) % n == 0) {
                result++;
            }
        }

        return result;
    }

    public int[] sumOfDistancesInTree(int N, int[][] edges) {
        ArrayList<Set<Integer>> graph = new ArrayList<>();

        for (int i = 0; i < N; ++i) {
            graph.add(new HashSet<>());
        }

        for (int[] edge: edges) {
            graph.get(edge[0]).add(edge[1]);
            graph.get(edge[1]).add(edge[0]);
        }

        final int[] count = new int[N];
        Arrays.fill(count, 1);

        final int[] answer = new int[N];

        dfsDisInTree(graph, 0, -1, count, answer);
        dfsDisInTree2(graph, 0, -1, count, answer);

        return answer;

    }

    private void dfsDisInTree(List<Set<Integer>> graph, int node, int parent, int[] count, int[] answer) {
        Set<Integer> childs = graph.get(node);

        for (int child : childs) {
            if (child != parent) {
                dfsDisInTree(graph, child, node, count, answer);
                count[node] += count[child];
                answer[node] += answer[child] + count[child];
            }
        }
    }

    private void dfsDisInTree2(List<Set<Integer>> graph, int node, int parent, int[] count, int[] answer) {
        Set<Integer> childs = graph.get(node);

        for (int child : childs) {
            if (child != parent) {
                answer[child] = answer[node] + graph.size() - count[child] - count[child];
                dfsDisInTree2(graph, child, node, count, answer);
            }
        }
    }

    public int numSimilarGroups(String[] A) {
        int N = A.length;
        int W = A[0].length();
        DSU dsu = new DSU(N);

        if (N < W*W) { // If few words, then check for pairwise similarity: O(N^2 W)
            for (int i = 0; i < N; ++i)
                for (int j = i+1; j < N; ++j)
                    if (similar(A[i], A[j]))
                        dsu.union(i, j);

        } else { // If short words, check all neighbors: O(N W^3)
            Map<String, List<Integer>> buckets = new HashMap();
            for (int i = 0; i < N; ++i) {
                char[] L = A[i].toCharArray();
                for (int j0 = 0; j0 < L.length; ++j0)
                    for (int j1 = j0 + 1; j1 < L.length; ++j1) {
                        swap(L, j0, j1);
                        StringBuilder sb = new StringBuilder();
                        for (char c: L) sb.append(c);
                        buckets.computeIfAbsent(sb.toString(),
                                x-> new ArrayList<Integer>()).add(i);
                        swap(L, j0, j1);
                    }
            }

            for (int i1 = 0; i1 < A.length; ++i1)
                if (buckets.containsKey(A[i1]))
                    for (int i2: buckets.get(A[i1]))
                        dsu.union(i1, i2);
        }

        int ans = 0;
        for (int i = 0; i < N; ++i)
            if (dsu.parent[i] == i) ans++;

        return ans;
    }

    public boolean similar(String word1, String word2) {
        int diff = 0;
        for (int i = 0; i < word1.length(); ++i)
            if (word1.charAt(i) != word2.charAt(i))
                diff++;
        return diff <= 2;
    }

    public void swap(char[] A, int i, int j) {
        char tmp = A[i];
        A[i] = A[j];
        A[j] = tmp;
    }

    class DSU {
        int[] parent;
        public DSU(int N) {
            parent = new int[N];
            for (int i = 0; i < N; ++i)
                parent[i] = i;
        }
        public int find(int x) {
            if (parent[x] != x) parent[x] = find(parent[x]);
            return parent[x];
        }
        public void union(int x, int y) {
            parent[find(x)] = find(y);
        }
}

    public int shortestPathLength(int[][] graph) {
        return 0;
    }

    public int rectangleArea(int[][] rectangles) {
        return 0;
    }

    public List<List<Integer>> getSkyline(int[][] buildings) {
        int[][] sweeps = new int[buildings.length][3];

        for (int i = 0; i < buildings.length; ++i) {
            sweeps[i * 2][0] = buildings[i][0];
            sweeps[i * 2][1] = buildings[i][2];
            sweeps[i * 2][2] = 0;

            sweeps[i * 2 + 1][0] = buildings[i][1];
            sweeps[i * 2 + 1][1] = buildings[i][2];
            sweeps[i * 2 + 1][2] = 1;
        }

        Arrays.sort(sweeps, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] == o2[0]) {
                    return o1[2] - o2[2];
                } else {
                    return o1[0] - o2[0];
                }
            }
        });


        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2 - o1;
            }
        });

        int i = 0;

        int lastMax = -1;

        List<List<Integer>> result = new ArrayList<>();

        while (i < sweeps.length) {

            int end = i;
            while (end + 1 < sweeps.length && sweeps[end + 1][0] == sweeps[i][0]) {
                end++;
            }

            for (int j = i; j <= end; ++j) {
                if (sweeps[j][2] == 1) {
                    maxHeap.remove(sweeps[j][1]);
                } else {
                    maxHeap.add(sweeps[j][1]);
                }
            }

            int curMax = maxHeap.isEmpty() ? maxHeap.peek() : 0;

            if (curMax != lastMax) {
                ArrayList<Integer> one = new ArrayList<>();
                one.add(sweeps[i][0]);
                one.add(curMax);
                result.add(one);

                lastMax = curMax;
            }

            i = end + 1;
        }

        return result;
    }

    public int shortestSubarray(int[] A, int K) {
        int[] sum = new int[A.length + 1];
        for (int i = 0; i < A.length; ++i) {
            sum[i + 1] = sum[i] + A[i];
        }

        Deque<Integer> deque = new LinkedList<>();

        int ans = Integer.MAX_VALUE;
        for (int y = 0; y < sum.length; ++y) {
            while (!deque.isEmpty() && sum[deque.getLast()] >= sum[y]) {
                deque.removeLast();
            }

            while (!deque.isEmpty() && sum[y] - sum[deque.getFirst()] >= K) {
                ans = Math.min(ans, y - deque.removeFirst());
            }

            deque.addLast(y);
        }

        return ans == Integer.MAX_VALUE ? -1 : ans;
    }

    public int shortestPathAllKeys(String[] grid) {
        if (grid == null || grid.length <= 0) {
            return 0;
        }

        final int rows = grid.length;
        final int columns = grid[0].length();

        int allKeys = 0;
        int startX = -1;
        int startY = -1;

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                final char c = grid[i].charAt(j);
                if (c == '@') {
                    startX = i;
                    startY = j;
                } else if (c >= 'a' && c <= 'f') {
                    allKeys |= (1 << (c - 'a'));
                }
            }
        }

        if (startX == -1 || startY == -1) {
            return -1;
        } else if (allKeys == 0) {
            return 0;
        }

        Queue<Integer> queue = new LinkedList<>();
        int key = getShortestPath(startX, startY, 0);
        queue.add(key);

        int level = 0;
        int cur = 1;
        int next = 0;

        Set<Integer> visited = new HashSet<>();
        visited.add(key);

        while (!queue.isEmpty()) {
            int head = queue.poll();
            cur--;

            final int keys = head & 0x3F;
            int column = (head >> 6) & 0x1F;
            int row = (head >> 11) & 0x1F;

            for (int[] dir : ShortestPathDirs) {
                int newRow = row + dir[0];
                int newColumn = column + dir[1];
                if (newRow >= 0 && newRow < rows && newColumn >= 0 && newColumn < columns) {
                    final char c = grid[newRow].charAt(newColumn);
                    if (c >= 'a' && c <= 'f') {

                        int newKeys = keys;
                        newKeys |= (1 << (c - 'a'));

                        if (newKeys == allKeys) {
                            return level + 1;
                        } else {
                            int newKey = getShortestPath(newRow, newColumn, newKeys);
                            if (visited.add(newKey)) {
                                next++;
                                queue.add(newKey);
                            }

                        }
                    } else if (c >= 'A' && c <= 'F') {
                        if((keys & (1 << (c - 'A'))) == (1 << (c - 'A'))) {
                            int newKey = getShortestPath(newRow, newColumn, keys);
                            if (visited.add(newKey)) {
                                next++;
                                queue.add(newKey);
                            }
                        }
                    } else if (c == '.' || c == '@') {
                        int newKey = getShortestPath(newRow, newColumn, keys);
                        if (visited.add(newKey)) {
                            next++;
                            queue.add(newKey);
                        }
                    }
                }
            }

            if (cur== 0) {
                cur = next;
                next = 0;
                level += 1;
            }
        }

        return -1;

    }

    private int[][] ShortestPathDirs = {
            {1, 0},
            {-1, 0},
            {0, 1},
            {0, -1}
    };

    private int getShortestPath(int row, int column, int allKeys) {
        return allKeys | (column << 6) | (row << 11);
    }

    public int nthMagicalNumber(int N, int A, int B) {
        final int mod = 1000000007;
        if (A == B) {
            return (int)((1L * A * N) % mod);
        } else {
            return magicalNumberBinarySearch(A, B, N, mod);
        }
    }

    private int magicalNumberBinarySearch(int A, int B, int N, int mod) {

        int L = A / gcd(A, B) * B;

        long lo = Math.min(A, B);
        long hi = 1L * Math.max(A, B) * N;
        while (lo <= hi) {
            long mi = lo + (hi - lo) / 2;
            // If there are not enough magic numbers below mi...
            if (mi / A + mi / B - mi / L < N)
                lo = mi + 1;
            else
                hi = mi - 1;
        }

        return (int) (lo % mod);
    }

    int gcd(int x, int y) {
        if (x == 0) return y;
        return gcd(y % x, x);
    }

    public int profitableSchemes(int G, int P, int[] group, int[] profit) {
        return profitableSchemesHelper(G, P, group, profit, 0, 0, new int[G+1][10001][group.length]);
    }

    public int profitableSchemesHelper(int G, int P, int[] group, int[] profit, int start, int sum, int[][][] mem) {
        if (G < 0) {
            return 0;
        }

        if (start >= group.length) {
            return 0;
        }

        if (mem[G][sum][start] != 0) {
            return mem[G][sum][start];
        }

        int tem = 0;
        tem += profitableSchemesHelper(G, P, group, profit, start + 1, sum, mem);

        if (G >= group[start]) {
            if (sum + profit[start] >= P) {
                tem = (tem + 1) % 1000000007;
            }
            tem = (tem + profitableSchemesHelper(G - group[start], P, group, profit, start + 1, sum + profit[start], mem) ) % 1000000007;
        }

        mem[G][sum][start] = tem;

        return tem;
    }

    public int profitableSchemesDp(int G, int P, int[] group, int[] profit) {
        int K = group.length;
        int V = P;
        int MOD = 1_000_000_007;

        //  given g person,  create more the v profit
        int[][][] d = new int[K + 1][G + 1][V + 1];
        for (int k = 1; k <= K; ++k) {
            for (int g = 1; g <= G; ++g) {
                int need_person = group[k - 1];
                int get_value = profit[k - 1];
                for (int v = 0; v <= V; ++v) {
                    d[k][g][v] = 0;
                    // case 0, only use plan[k]
                    if (v <= get_value && g >= need_person) {
                        d[k][g][v] += 1;
                    }

                    // case 1: not use plan[k]
                    d[k][g][v] += (k < 1 ? 0 : d[k - 1][g][v]) % MOD;

                    // case 2: use plan[k] and use plan before
                    if (g > need_person) {
                        d[k][g][v] += (k < 1 ? 0 : d[k - 1][g - need_person][Math.max(0, v - get_value)]) % MOD;
                    }
                    d[k][g][v] %= MOD;
                }
            }
        }
        int sum = d[K][G][P];
        return sum;
    }

    public int reachableNodes(int[][] edges, int M, int N) {
        return 0;
    }

    public int superEggDrop(int K, int N) {
        return dp(K, N);
    }

    Map<Integer, Integer> memo = new HashMap();
    public int dp(int K, int N) {
        if (!memo.containsKey(N * 100 + K)) {
            int ans;
            if (N == 0)
                ans = 0;
            else if (K == 1)
                ans = N;
            else {
                int lo = 1, hi = N;
                while (lo + 1 < hi) {
                    int x = (lo + hi) / 2;
                    int t1 = dp(K-1, x-1);
                    int t2 = dp(K, N-x);

                    if (t1 < t2)
                        lo = x;
                    else if (t1 > t2)
                        hi = x;
                    else
                        lo = hi = x;
                }

                ans = 1 + Math.min(Math.max(dp(K-1, lo-1), dp(K, N-lo)),
                        Math.max(dp(K-1, hi-1), dp(K, N-hi)));
            }

            memo.put(N * 100 + K, ans);
        }

        return memo.get(N * 100 + K);
    }

    public String orderlyQueue(String S, int K) {
        if (K == 1) {
            String min = S;
            for (int i = 1; i < S.length() - 1; ++i) {
                String newString = S.substring(i) + S.substring(0, i);
                if (min.compareTo(newString) > 0) {
                    min = newString;
                }
            }
            return min;
        } else {
            int[] hash = new int[26];
            for (int i = 0; i < S.length(); ++i) {
                hash[S.charAt(i) - 'a']++;
            }

            StringBuilder builder = new StringBuilder();

            for (int i = 0; i < hash.length; ++i) {
                for (int j = 0; j < hash[i]; ++j) {
                    builder.append((char) ('a' + i));
                }
            }

            return builder.toString();
        }
    }

    public int atMostNGivenDigitSet(String[] D, int N) {
        int res = countCaseEqLen(D, N);

        int len = len(N);
        for(int i = 1; i < len; i++){
            res += Math.pow(D.length, i);
        }
        return res;
    }

    private int countCaseEqLen(String [] D, int N){
        String val = Integer.toString(N);
        int res = 0;
        for(int i = 0; i < val.length(); i++){
            int indexD = 0;

            while(indexD < D.length && val.charAt(i) > D[indexD].charAt(0)){
                indexD++;
            }

            res += indexD * Math.pow(D.length, val.length() - 1 - i);

            if(indexD == D.length){
                break;
            }

            if(i == val.length() - 1 && val.charAt(i) == D[indexD].charAt(0)){
                res++;
                break;
            }

            if(val.charAt(i) != D[indexD].charAt(0)){
                break;
            }

        }

        return res;
    }

    private int len(int num){
        int res = 0;
        while(num != 0){
            res++;
            num /= 10;
        }

        return res;
    }

    public int minMalwareSpread(int[][] graph, int[] initial) {
        int N = graph.length;
        DSUMalware dsu = new DSUMalware(N);
        for (int i = 0; i < N; ++i)
            for (int j = i+1; j < N; ++j)
                if (graph[i][j] == 1)
                    dsu.union(i, j);

        int[] count = new int[N];
        for (int node: initial)
            count[dsu.find(node)]++;

        int ans = -1, ansSize = -1;
        for (int node: initial) {
            int root = dsu.find(node);
            if (count[root] == 1) {  // unique color
                int rootSize = dsu.size(root);
                if (rootSize > ansSize) {
                    ansSize = rootSize;
                    ans = node;
                } else if (rootSize == ansSize && node < ans) {
                    ansSize = rootSize;
                    ans = node;
                }
            }
        }

        if (ans == -1) {
            ans = Integer.MAX_VALUE;
            for (int node: initial)
                ans = Math.min(ans, node);
        }
        return ans;
    }

    class DSUMalware {
        int[] p, sz;

        DSUMalware(int N) {
            p = new int[N];
            for (int x = 0; x < N; ++x)
                p[x] = x;

            sz = new int[N];
            Arrays.fill(sz, 1);
        }

        public int find(int x) {
            if (p[x] != x)
                p[x] = find(p[x]);
            return p[x];
        }

        public void union(int x, int y) {
            int xr = find(x);
            int yr = find(y);
            if (xr != yr) {
                p[xr] = yr;
                sz[yr] += sz[xr];
            }

        }

        public int size(int x) {
            return sz[find(x)];
        }
    }

    public int[] threeEqualParts(int[] A) {
        if (A == null || A.length <= 3) {
            return new int[]{-1, -1};
        }

        int onesCnt = 0;
        int zerosThird = 0;

        boolean recordZeros = true;
        for (int i = A.length - 1; i >= 0; --i) {
            if (A[i] == 0) {
                if (recordZeros) {
                    zerosThird++;
                }
            } else {
                recordZeros = false;
                onesCnt++;
            }
        }

        if (onesCnt == 0) {
            return new int[]{0, A.length - 1};
        } else if (onesCnt % 3 != 0) {
            return new int[]{-1, -1};
        }

        final int onesOneCnt = onesCnt / 3;

        int firstStart = 0;
        while (firstStart < A.length && A[firstStart] == 0) {
            firstStart++;
        }

        int firstOneCnt = 0;

        int firstOneStart = firstStart;
        int firstEnd = firstStart;

        for (; firstEnd < A.length; ++firstEnd) {
            if (A[firstEnd] == 1) {
                firstOneCnt++;
                if (firstOneCnt >= onesOneCnt) {
                    break;
                }
            }
        }

        while (zerosThird-- > 0) {
            if (++firstEnd >= A.length || A[firstEnd] != 0) {
                return new int[]{-1, -1};
            }
        }

        int secondStart = firstEnd + 1;
        while (secondStart < A.length && A[secondStart] == 0) {
            secondStart++;
        }

        for (int i = 0; i < firstEnd - firstOneStart; ++i) {
            if (i + secondStart >= A.length || A[i + secondStart] != A[firstOneStart + i]) {
                return new int[]{-1, -1};
            }
        }

        int thirdStart = secondStart + firstEnd - firstOneStart + 1;
        if (thirdStart >= A.length) {
            return new int[]{-1, -1};
        }

        final int resultThirdStart = thirdStart;

        while (thirdStart < A.length && A[thirdStart] == 0) {
            thirdStart++;
        }


        for (int i = 0; i < firstEnd - firstOneStart; ++i) {
            if (i + thirdStart >= A.length || A[i + thirdStart] != A[firstOneStart + i]) {
                return new int[]{-1, -1};
            }
        }

        return new int[]{firstEnd, resultThirdStart};
    }

    public int[] movesToStamp(String stamp, String target) {
        return null;
    }

    public int distinctSubseqII(String S) {
        if (S == null || S.length() <= 0) {
            return 0;
        }

        final int mod = 1000000007;

        int[] dp = new int[S.length()];
        dp[0] = 1;
        int[] hash = new int[26];
        Arrays.fill(hash, -1);
        hash[S.charAt(0) - 'a'] = 0;


        for (int i = 1; i < dp.length; ++i) {
            dp[i] = dp[i - 1] + dp[i - 1] + 1;
            final int idx = S.charAt(i) - 'a';
            if (hash[idx] != -1) {
                final int pre = hash[idx] - 1 >= 0 ? dp[hash[idx] - 1] : 0;
                dp[i] -= pre + 1;
            }
            hash[idx] = i;

            if (dp[i] < 0) {
                dp[i] += mod;
            }
            dp[i] %= mod;
        }

        return dp[S.length() - 1];
    }

    public String shortestSuperstring(String[] A) {
        return null;
    }

    public int minDeletionSize(String[] A) {
        if (A == null || A.length <= 0 || A[0] == null || A[0].length() <= 0) {
            return 0;
        }

        final int length = A[0].length();

        int[] dp = new int[length + 1];

        int result = 0;

        dp[1] = 0;

        result = dp[1] + length - 1;
        for (int i = 2; i <= length; ++i) {
            dp[i] = i - 1;
            for (int j = i - 1; j >= 0; --j) {
                boolean big = true;
                for (int k = 0; k < A.length; ++k) {
                    final char cur = A[k].charAt(i-1);
                    final char pre = j == 0 ? 0 : A[k].charAt(j - 1);
                    if (pre >= cur) {
                        big = false;
                    }
                }

                if (big) {
                    dp[i] = Math.min(dp[i], dp[j] + i - j - 1);
                }
            }

            result = Math.min(result, dp[i] + length - i);
        }

        return result;
    }

    public int leastOpsExpressTarget(int x, int target) {
        return opsExpressTargetHelper(x, target, new HashMap<>());
    }

    private int opsExpressTargetHelper(int x, int target, Map<Integer, Integer> dp) {
        if (x == target) {
            return 0;
        } else if (target == 1) {
            return 1;
        }

        int cache = dp.getOrDefault(target, -1);
        if (cache != -1) {
            return cache;
        }

        long temp = x;
        int count = 0;
        while (temp < target) {
            temp *= x;
            count++;
        }

        int ret1 = Integer.MAX_VALUE; // x * x* x - (temp - target)
        if (temp == target) {
            return count;
        } else if (temp - target < target) {
            ret1 = count + 1 + opsExpressTargetHelper(x, (int) (temp - target), dp);
        }

        int ret2 = Integer.MAX_VALUE; // x * x + (target - temp)
        temp /= x;

        ret2 = (count == 0 ? 2 : count) + opsExpressTargetHelper(x, (int) (target - temp), dp);


        int res = Math.min(ret1, ret2);

        dp.put(target, res);

        return res;
    }

    public int minCameraCover(TreeNode root) {
        int[] res = minCameraCoverHelper(root);

        return Math.min(res[1], res[2]);
    }

    // 0 below covered, not this 1 all covered, this not camera 2 all covered, this is camera
    public int[] minCameraCoverHelper(TreeNode root) {
        int[] res = {0,0, 9999};

        if (root == null) {
            return res;
        }

        int[] left = minCameraCoverHelper(root.left);
        int[] right = minCameraCoverHelper(root.right);

        res[0] = left[1] + right[1];

        res[1] = Math.min(left[2] + Math.min(right[1], right[2]),
                            right[2] + Math.min(left[1] , left[2]));

        res[2] = 1 + Math.min(left[0], Math.min(left[1] , left[2])) +
                Math.min(right[0], Math.min(right[1], right[2]));

        return res;
    }

    public int oddEvenJumps(int[] A) {
        if (A == null || A.length <= 0) {
            return 0;
        }

        final int length = A.length;

        TreeSet<Integer> set = new TreeSet<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                if (A[o1] != A[o2]) {
                    return A[o1] - A[o2];
                } else {
                    return o1 - o2;
                }
            }
        });

        TreeSet<Integer> set2 = new TreeSet<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                if (A[o1] != A[o2]) {
                    return A[o1] - A[o2];
                } else {
                    return o2 - o1;
                }
            }
        });

        boolean[][] dp = new boolean[length][2];
        dp[length - 1][0] = true;
        dp[length - 1][1] = true;

        int result = 1;

        set.add(length - 1);
        set2.add(length - 1);

        for (int i = length - 2; i >= 0; --i) {
            // 当i是基数跳跃的时候，要找出大于等于他的最小值

            final Integer smallestBiggerIdx = set.ceiling(i);
            if (smallestBiggerIdx == null || !dp[smallestBiggerIdx][1]) {
                dp[i][0] = false;
            } else {
                dp[i][0] = true;
                result++;
            }

            final Integer smallestSmallerIdx = set2.floor(i);
            if (smallestSmallerIdx == null || !dp[smallestSmallerIdx][0]) {
                dp[i][1] = false;
            } else {
                dp[i][1] = true;
            }
            set.add(i);
            set2.add(i);
        }

        return result;
    }

    public int subarraysWithKDistinct(int[] A, int K) {
        if (A == null || A.length <= 0) {
            return 0;
        }

        int[] hash = new int[A.length + 1];

        int left = 0;
        int end = 0;

        int result = 0;

        int kinds = 0;
        while (end < A.length) {
            final int cur = A[end];
            hash[cur]++;
            if (hash[cur] == 1) {
                kinds++;
            }

            while (left <= end && kinds == K) {
                result++;
                hash[A[left]]--;
                if (hash[A[left]] == 0) {
                    kinds--;
                }
                left++;
            }
            end++;
        }

        return result;
    }

    public int minKBitFlips(int[] A, int K) {
        if (A == null || A.length <= 0) {
            return 0;
        }

        int i = 0;

        int result = 0;
        int next = -1;
        while(i + K <= A.length) {
            if (A[i] == 0) {
                result++;
                next = -1;
                for (int j = i; j < i + K; ++j) {
                    A[j] = A[j] ^ 1;

                    if (A[j] == 0 && next == -1) {
                        next = j;
                    }
                }

                if (next == -1) {
                    next = i + K;
                }

                i = next;
            } else {
                ++i;
            }
        }

        boolean allOne = true;
        for (int j = 0; j < K; ++j) {
            if (A[A.length - 1 - j] == 0) {
                return -1;
            }
        }

        return result;

    }

    public int mergeStones(int[] stones, int K) {
        if (stones == null || stones.length < K) {
            return -1;
        }

        // f(n) = f(n - k + 1)
        if ((stones.length - K) % (K - 1) != 0) {
            return -1;
        }

        return 0;
    }

    public TreeNode recoverFromPreorder(String S) {
        return recoverFromPreorder(S, 0, S.length() - 1, 0);
    }

    private TreeNode recoverFromPreorder(String s, int i, int j, int depth) {
        if (i < j) {
            return null;
        } else if (i == j) {
            return new TreeNode(Integer.valueOf(s.substring(i, i+1)));
        }

        int numEnd = i;
        while (numEnd < s.length() && Character.isDigit(s.charAt(numEnd))) {
            numEnd++;
        }

        TreeNode root = new TreeNode(Integer.valueOf(s.substring(i, numEnd)));

        // 从i查找连续depth + 1个的位置
        int devider = 0;
        int leftIdx = -1;
        int rightIdx = -1;
        for (int k = numEnd; k <= j; ++k) {
            final char c = s.charAt(k);
            if (c == '-') {
                devider++;
            } else {
                if (devider == depth + 1) {
                    if (leftIdx == -1) {
                        leftIdx = k;
                    } else if (rightIdx == -1) {
                        rightIdx = k;
                    }
                }
                devider = 0;
            }
        }

        TreeNode leftNode = null;
        if (leftIdx != -1) {
            leftNode = recoverFromPreorder(s, leftIdx, rightIdx == -1 ? j : rightIdx - devider - 1 - 1, depth + 1);
        }

        TreeNode rightNode = null;
        if (rightIdx != -1) {
            rightNode = recoverFromPreorder(s, rightIdx, j, depth + 1);
        }

        root.left = leftNode;
        root.right = rightNode;

        return root;
    }

    public int numSubmatrixSumTarget(int[][] matrix, int target) {
        if (matrix == null || matrix.length <= 0 || matrix[0] == null || matrix[0].length <= 0) {
            return 0;
        }

        final int rows = matrix.length;
        final int columns = matrix[0].length;

        final int[] preColumnSum = new int[columns];

        int result = 0;
        for (int i = 0; i < rows; ++i) {
            Arrays.fill(preColumnSum, 0);
            for (int j = i; j < rows; ++j) {
                final Map<Integer, Integer> map = new HashMap<>();
                map.put(0, 1);
                int preSum = 0;
                for (int k = 0; k < columns; k++) {
                    preColumnSum[k] = preColumnSum[k] + matrix[j][k];
                    preSum += preColumnSum[k];

                    final int toTind = preSum - target;
                    final int oldCnt = map.getOrDefault(toTind, -1);
                    if (oldCnt != -1) {
                        result += oldCnt;
                    }
                    map.put(preSum, map.getOrDefault(preSum, 0) + 1);
                }
            }
        }

        return result;
    }

    public String shortestCommonSupersequence(String str1, String str2) {
        int[][] dp = new int[str1.length() + 1][str2.length() + 1];

        getLcs(str1, str2, dp);

        StringBuilder builder = new StringBuilder();

        int i = str1.length();
        int j = str2.length();

        while (i > 0 && j > 0) {
            if (dp[i][j] == dp[i-1][j]) {
                builder.append(str1.charAt(i-1));
                i--;
            } else if (dp[i][j] == dp[i][j-1]) {
                builder.append(str2.charAt(j-1));
                j--;
            } else {
                builder.append(str1.charAt(i-1));
                i--;
                j--;
            }
        }

        while (i > 0) {
            builder.append(str1.charAt(i-1));
            i--;
        }
        while (j > 0) {
            builder.append(str2.charAt(j-1));
            j--;
        }

        return builder.reverse().toString();
    }

    private void getLcs(String str1, String str2, int[][] dp) {
        for (int i = 1; i <= str1.length(); ++i) {
            for (int j = 1; j <= str2.length(); ++j) {
                if (str1.charAt(i - 1) == str2.charAt(j-1)) {
                    dp[i][j] = 1 + dp[i-1][j -1];
                } else {
                    dp[i][j] = Math.max(dp[i][j-1], dp[i-1][j]);
                }
            }
        }
    }

    public int findInMountainArray(int target, MountainArray mountainArr) {
        int highestIdx = findMountainHighestIdx(mountainArr);

        if (highestIdx == -1) {
            return -1;
        }

        int leftIdx = findMoutainBinarySearch(mountainArr, 0, highestIdx, target, true);
        if (leftIdx != -1) {
            return leftIdx;
        } else {
            return findMoutainBinarySearch(mountainArr, highestIdx, mountainArr.length() - 1,  target, false);
        }
    }

    private int findMoutainBinarySearch(MountainArray mountainArr, int start, int end, int target, boolean increase) {
        int left = start;
        int right = end;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            int midValue = mountainArr.get(mid);
            if (increase) {
                if (midValue == target) {
                    return mid;
                } else if (midValue > target) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (midValue == target) {
                    return mid;
                } else if (midValue < target) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
        }

        return -1;
    }

    private int findMountainHighestIdx(MountainArray mountainArr) {
        if (mountainArr == null || mountainArr.length() < 3) {
            return -1;
        }

        final int length = mountainArr.length();

        int left = 0;
        int right = length - 1;

        while (left <= right) {
            final int mid = left + (right - left) / 2;
            final int midValue = mountainArr.get(mid);

            final int pre = mid >= 1 ? mountainArr.get(mid - 1) : -1;

            final int next = mid + 1 < length ? mountainArr.get(mid + 1) : -1;

            if (pre < midValue && midValue > next) {
                return mid;
            } else if (pre < midValue && midValue < next){
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return -1;
    }

    public boolean parseBoolExpr(String expression) {
        if ("t".equals(expression)) {
            return true;
        } else if ("f".equals(expression)) {
            return false;
        }

        char op = expression.charAt(0);

        int count = 0;
        int pre = 2;

        boolean result = op == '&';
        for (int i = 1; i < expression.length(); ++i) {
            final char c = expression.charAt(i);
            if (c == '(') count++;
            if (c == ')') count--;

            if ((c == ',' && count == 1) || (c == ')' && count == 0)) {
                boolean next = parseBoolExpr(expression.substring(pre, i));
                pre = i + 1;

                if (op == '|') {
                    result |= next;
                } else if (op == '&') {
                    result &= next;
                } else if (op == '!') {
                    result = !next;
                }
            }
        }

        return result;
    }

}
