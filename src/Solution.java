import com.sun.tools.hat.internal.model.Root;

import java.util.*;
import java.util.function.Function;

public class Solution {

	// https://leetcode.com/problems/4sum/description/
	public List<List<Integer>> fourSum(int[] nums, int target) {
		ArrayList<List<Integer>> result = new ArrayList<>();

		if (nums == null || nums.length <= 0) {
			return result;
		}

		Arrays.sort(nums);

		List<Integer> one = new ArrayList<>();

		fourSum(nums, 0, 0, target, 0, one, result);

		return result;
	}

	public void fourSum(int[] nums, int index, int size, int target, int sum, List<Integer> one,
			List<List<Integer>> result) {

		for (int i = index; i < nums.length; i++) {
			sum += nums[i];
			one.add(nums[i]);

			if (size + 1 == 4) {
				if (sum == target)
					result.add(new ArrayList<>(one));
			} else if (size + 1 < 4) {

				// 计算出最大值和最小值，然后看看是不是需要舍弃还要继续进行
				// 还剩下多少元素，如果都不够了，则不需要了
				if (i + 1 < nums.length && size + nums.length - i >= 4) {
					int tempMin = sum;
					int tempMax = sum;
					for (int l = 1; l <= 4 - size - 1; l++) {
						tempMin += nums[i + 1 + l - 1];
						tempMax += nums[nums.length - l];
					}
					if (target >= tempMin && target <= tempMax) {
						fourSum(nums, i + 1, size + 1, target, sum, one, result);
					}
				}

			}

			sum -= nums[i];
			one.remove(one.size() - 1);

			while (i < nums.length - 1 && nums[i] == nums[i + 1]) {
				i++;
			}
		}
	}

	public List<List<Integer>> threeSum(int[] nums) {
		List<List<Integer>> result = new ArrayList<>();

		if (nums == null || nums.length < 3) {
			return result;
		}

		Arrays.sort(nums);

		for (int i = 0; i < nums.length - 2; i++) {
			List<List<Integer>> temp = twoSum(nums, -nums[i], i + 1);
			if (temp != null && temp.size() > 0) {
				for (List<Integer> list : temp) {
					list.add(0, nums[i]);
					result.add(new ArrayList<>(list));
				}
			}

			while (i < nums.length - 3 && nums[i] == nums[i + 1]) {
				i++;
			}
		}

		return result;
	}

	public List<List<Integer>> twoSum(int[] nums, int target, int start) {
		int s = start;
		int e = nums.length - 1;

		List<List<Integer>> result = new ArrayList<>();

		while (s < e) {
			if (nums[s] + nums[e] == target) {
				// 记录数据
				List<Integer> one = new ArrayList<>();
				one.add(nums[s]);
				one.add(nums[e]);
				result.add(one);
				while (s < e && nums[s] == nums[s + 1]) {
					s++;
				}

				while (s < e && nums[e] == nums[e - 1]) {
					e--;
				}
				s++;
				e--;

			} else if (nums[s] + nums[e] < target) {
				s++;
			} else {
				e--;
			}
		}

		return result;
	}

	public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
		int length = A.length;
		if (length <= 0) {
			return 0;
		}

		Map<Integer, Integer> result = new HashMap<>();

		int cnt = 0;
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < length; j++) {
				int sum = A[i] + B[j];
				if (result.containsKey(sum)) {
					result.put(sum, result.get(sum) + 1);
				} else {
					result.put(sum, 1);
				}
			}
		}

		for (int k = 0; k < length; k++) {
			for (int l = 0; l < length; l++) {
				int sum = C[k] + D[l];
				if (result.containsKey(-sum)) {
					cnt += result.get(-sum);
				}
			}
		}

		return cnt;
	}

	// https://leetcode.com/problems/3sum-closest/description/

	// 正常的是n*n*n的复杂度，应该能做到n*n的
	public int threeSumClosest(int[] nums, int target) {
		if (nums == null || nums.length < 3) {
			return -1;
		}

		Arrays.sort(nums);

		int closest = Integer.MAX_VALUE;
		int delta = Integer.MAX_VALUE;
		for (int i = 0; i < nums.length - 2; i++) {
			int newTarget = target - nums[i];
			int newClosest = twoSumClosest(nums, newTarget, i + 1);
			int newDelta = Math.abs(newClosest + nums[i] - target);
			if (newDelta < delta) {
				delta = newDelta;
				closest = newClosest + nums[i];
			}
		}

		return closest;
	}

	public int twoSumClosest(int[] nums, int target, int start) {
		int s = start;
		int e = nums.length - 1;

		int closest = Integer.MAX_VALUE;
		int delta = Integer.MAX_VALUE;

		while (s < e) {
			int temp = nums[s] + nums[e];
			if (temp == target) {
				return target;
			} else if (temp > target) {
				e--;
			} else {
				s++;
			}

			if (Math.abs(temp - target) < delta) {
				delta = Math.abs(temp - target);
				closest = temp;
			}
		}

		return closest;
	}

	// https://leetcode.com/problems/search-a-2d-matrix-ii/description/
	public boolean searchMatrix(int[][] matrix, int target) {
		if (matrix == null || matrix.length <= 0 || matrix[0] == null || matrix[0].length <= 0) {
			return false;
		}

		int row = 0;
		int column = matrix[0].length - 1;

		while (row < matrix.length && column >= 0) {
			if (matrix[row][column] == target) {
				return true;
			} else if (matrix[row][column] > target) {
				column--;
			} else {
				row++;
			}
		}

		return false;
	}

	// https://leetcode.com/problems/search-a-2d-matrix/description/
	public boolean searchMatrix1(int[][] matrix, int target) {
		if (matrix == null || matrix.length <= 0 || matrix[0] == null || matrix[0].length <= 0) {
			return false;
		}

		int row_num = matrix.length;
		int col_num = matrix[0].length;

		int begin = 0, end = row_num * col_num - 1;

		while (begin <= end) {
			int mid = (begin + end) / 2;
			int mid_value = matrix[mid / col_num][mid % col_num];

			if (mid_value == target) {
				return true;

			} else if (mid_value < target) {
				// Should move a bit further, otherwise dead loop.
				begin = mid + 1;
			} else {
				end = mid - 1;
			}
		}

		return false;
	}

	// https://leetcode.com/problems/longest-harmonious-subsequence/description/
	public int findLHS(int[] nums) {
		if (nums == null || nums.length <= 1) {
			return 0;
		}

		int result = 0;
		for (int i = 0; i < nums.length; i++) {
			int below = nums[i] - 1;
			int upper = nums[i] + 1;
			int belowCnt = 0;
			int upperCnt = 0;
			boolean hasBelow = false;
			boolean hasUpper = false;
			for (int j = 0; j < nums.length; j++) {
				if (nums[j] == nums[i]) {
					belowCnt++;
					upperCnt++;
				} else if (nums[j] == below) {
					hasBelow = true;
					belowCnt++;
				} else if (nums[j] == upper) {
					hasUpper = true;
					upperCnt++;
				}
			}

			if (hasBelow) {
				result = Math.max(result, belowCnt);
			} else if (hasUpper) {
				result = Math.max(result, upperCnt);
			}
		}

		return result;
	}

	public int findLHS1(int[] nums) {
		if (nums == null || nums.length <= 1) {
			return 0;
		}

		Map<Integer, Integer> map = new TreeMap<>(new Comparator<Integer>() {
			public int compare(Integer obj1, Integer obj2) {
				// 降序排序
				return obj1 - obj2;
			}
		});

		for (int i = 0; i < nums.length; i++) {
			if (!map.containsKey(nums[i]))
				map.put(nums[i], 1);
			else
				map.put(nums[i], 1 + map.get(nums[i]));
		}

		Iterator<Map.Entry<Integer, Integer>> iterator = map.entrySet().iterator();
		if (!iterator.hasNext()) {
			return 0;
		}
		Map.Entry<Integer, Integer> entry = iterator.next();
		int lastKey = entry.getKey();
		int lastCnt = entry.getValue();
		int result = 0;
		while (iterator.hasNext()) {
			entry = iterator.next();
			int nextKey = entry.getKey();
			int nextCnt = entry.getValue();

			if (nextKey - lastKey == 1) {
				result = Math.max(result, lastCnt + nextCnt);
			}

			lastCnt = nextCnt;
			lastKey = nextKey;
		}

		return result;
	}

	// https://leetcode.com/problems/palindrome-pairs/description/
	// 这个是最简单的方法，用前缀
	public List<List<Integer>> palindromePairs(String[] words) {
		List<List<Integer>> result = new ArrayList<>();
		if (words == null || words.length < 2) {
			return result;
		}

		for (int i = 0; i < words.length; i++) {
			for (int j = 0; j < words.length; j++) {
				if (i != j) {
					String newWord = words[i] + words[j];
					if (isPalindrome(newWord)) {
						List<Integer> one = new ArrayList<>();
						one.add(i);
						one.add(j);
						result.add(one);
					}
				}
			}
		}

		return result;
	}

	public boolean isPalindrome(String word) {
		if (word == null || word.length() <= 0)
			return false;
		if (word.length() == 1) {
			return true;
		}

		int start = 0;
		int end = word.length() - 1;

		while (start <= end) {
			if (word.charAt(start) == word.charAt(end)) {
				start++;
				end--;
			} else {
				return false;
			}
		}

		return true;
	}

	// https://leetcode.com/problems/diagonal-traverse/description/
	public int[] findDiagonalOrder(int[][] matrix) {
		if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0) {
			return new int[0];
		}

		final int row = matrix.length;
		final int column = matrix[0].length;
		int[] result = new int[row * column];
		int index = 0;

		boolean up = true;
		for (int sum = 0; sum <= row + column - 1; ++sum) {
			if (up) {
				// 往上走,row是不断变小，最大值是row-1和sum的最小值
				int rstart = Math.min(row - 1, sum);
				int rend = Math.max(0, sum - (column - 1));
				for (int r = rstart; r >= rend; --r) {
					result[index++] = matrix[r][sum - r];
				}
			} else {
				// 往下走,row是不断变大，最大值是row-1和sum的最小值
				int rstart = Math.max(0, sum - (column - 1));
				int rend = Math.min(sum, row - 1);

				for (int r = rstart; r <= rend; ++r) {
					result[index++] = matrix[r][sum - r];
				}
			}

			up = !up;
		}

		return result;
	}

	// https://leetcode.com/problems/candy/description/
	public int candy(int[] ratings) {
		if (ratings == null || ratings.length == 0) {
			return 0;
		}

		int[] candy = new int[ratings.length];
		candy[0] = 1;
		for (int i = 1; i < ratings.length; i++) {
			if (ratings[i] > ratings[i - 1]) {
				candy[i] = candy[i - 1] + 1;
			} else {
				candy[i] = 1;
			}
		}

		for (int i = ratings.length - 2; i >= 0; --i) {
			if (ratings[i] > ratings[i + 1] && candy[i] <= candy[i + 1]) {
				candy[i] = candy[i + 1] + 1;
			}
		}

		int result = 0;
		for (int i = 0; i < ratings.length; i++) {
			result += candy[i];
		}

		return result;
	}

	// https://leetcode.com/problems/wildcard-matching/description/
	public boolean isMatch(String s, String p) {
		if (p == null) {
			return false;
		}

		if (p.length() == 0) {
			if (s.length() == 0)
				return true;
			else
				return false;
		}

		boolean[][] dp = new boolean[p.length()][s.length() + 1];
		dp[0][0] = p.charAt(0) == '*';

		for (int i = 1; i < p.length(); ++i) {
			dp[i][0] = dp[i - 1][0] && p.charAt(i) == '*';
		}

		for (int j = 1; j <= s.length(); j++) {
			char pc = p.charAt(0);
			if (pc == '*')
				dp[0][j] = true;
			else if (j == 1) {
				dp[0][j] = isEqual(pc, s.charAt(j - 1));
			} else {
				dp[0][j] = false;
			}
		}

		for (int i = 1; i < p.length(); i++) {
			for (int j = 1; j <= s.length(); j++) {
				char pc = p.charAt(i);
				char sc = s.charAt(j - 1);

				dp[i][j] = dp[i - 1][j - 1] && isEqual(pc, sc);
				if (!dp[i][j]) {
					if (pc == '*') {
						dp[i][j] |= dp[i][j - 1] | dp[i - 1][j];
					}
				}
			}
		}

		return dp[p.length() - 1][s.length()];
	}

	private boolean isEqual(char p, char s) {
		return p == '?' || p == s;
	}

	// https://leetcode.com/problems/longest-substring-without-repeating-characters/description/
	public int lengthOfLongestSubstring(String s) {
		if (s == null || s.length() <= 0) {
			return 0;
		}

		int start = 0;
		int end = 0;

		int result = 0;
		int[] hash = new int[256];

		while (end < s.length()) {
			int idx = s.charAt(end) - 0;
			if (hash[idx] == 0) {
				hash[idx]++;
				result = Math.max(result, end - start + 1);
				end++;
			} else {
				while (start <= end && hash[idx] >= 1) {
					if (hash[s.charAt(start)] > 0)
						hash[s.charAt(start)] = 0;
					start++;
				}
			}
		}

		return result;

	}

	// https://leetcode.com/problems/word-search-ii/description/
	public List<String> findWords(char[][] board, String[] words) {
	        
			List<String> result = new ArrayList<>();
			if (board == null || board.length <= 0 ||
					board[0] == null || board[0].length <= 0) {
				return result;
			}
			
			if (words == null || words.length <= 0) {
				return result;
			}
			
			TrieNode root = new TrieNode();
			
			for (String word : words) {
				addWord(root, word);
			}
			
			boolean[][] walk = new boolean[board.length][board[0].length];
			Set<String> setResult = new HashSet<>();
			for (int i = 0; i < board.length; ++i) {
				for (int j = 0; j < board[0].length; ++j) {
					findWords(board, walk, i, j, root, "", setResult);
				}
			}
			
			result.addAll(setResult);
			
			return result;
	    }

	static int[][] sDirection = { { -1, 0 }, // up
			{ 1, 0 }, // down
			{ 0, -1 }, // left
			{ 0, 1 } // right
	};

	// https://leetcode.com/problems/word-search/description/
	public boolean exist(char[][] board, String word) {
		if (board == null || board.length <= 0 || board[0] == null || board[0].length <= 0) {
			return false;
		}
		
		if (word == null || word.length() <= 0) {
			return false;
		}
		
		for (int i = 0; i < board.length; ++i) {
			for (int j = 0; j < board[0].length; ++j) {
				if (exist(board, word, i, j, 0)) {
					return true;
				}
			}
		}
		
		return false;
    }
	
	private boolean exist(char[][] board, CharSequence word, int row, int column, int start) {
		if (start >= word.length()) {
			return false;
		}
		char c = word.charAt(start);
		if (c != board[row][column]) {
			return false;
		}
		
		if (start == word.length() - 1) {
			return true;
		}
		
		board[row][column] = '#';
		for (int i = 0; i < sDirection.length; ++i) {
			int newRow = row + sDirection[i][0];
			int newColumn = column + sDirection[i][1];
			
			if (newRow >= 0 && newRow < board.length 
					&& newColumn >= 0 && newColumn < board[0].length
					&& board[newRow][newColumn] != '#') {
				if (exist(board, word, newRow, newColumn, start + 1)) {
					board[row][column] = c;
					return true;
				}
			}
		}
		board[row][column] = c;
		return false;
	}
	
	private boolean checkWalk(boolean[][] walk, int row, int column) {
		if (row < 0 || row >= walk.length) {
			return false;
		}

		if (column < 0 || column >= walk[0].length) {
			return false;
		}

		if (walk[row][column]) {
			return false;
		}

		return true;
	}

	private void findWords(char[][] board, boolean[][] walk, int row, int column, TrieNode root, String last, Set<String> result) {
		if (!checkWalk(walk, row, column)) {
			return;
		}
		
		String cur = last + board[row][column];
		TrieNode node = findPrefix(root, cur);
		if (node == null) {
			return;
		}
		
		walk[row][column] = true;
		
		if (node.words > 0) {
			result.add(cur);
		}
		
		for (int i = 0; i < sDirection.length; i++) {
			int newRow = row + sDirection[i][0];
			int newColumn = column + sDirection[i][1];
			
			findWords(board, walk, newRow, newColumn, root, cur, result);
		}
		
		walk[row][column] = false;
	}
	
	static class TrieNode {
		char c;
		int words; // 代表以这个点结尾的单词数
		TrieNode[] next = new TrieNode[26];
	}
	
	private void addWord(TrieNode root, String word) {
		addWord(root, word, 0);
	}
	
	private void addWord(TrieNode root, CharSequence word, int start) {
		if (start >= word.length()) {
			return;
		}
		
		char c = word.charAt(start);
		int index = c - 'a';
		TrieNode node = root.next[index];
		if (node == null) {
			node = new TrieNode();
			node.c = c;
			root.next[index] = node;
		}
		
		if (start == word.length() - 1) {
			node.words++;
		} else {
			addWord(node, word, start + 1);
		}
	}
	
	@SuppressWarnings("unused")
	private boolean findWord(TrieNode root, String word) {
		TrieNode node = findPrefix(root, word);
		return node != null && node.words > 0;
	}
	/*
	 * 查找以prefix开头的单词是否存在
	 */
	private TrieNode findPrefix(TrieNode root, String prefix) {
		return findPrefix(root, prefix, 0);
	}
	
	private TrieNode findPrefix(TrieNode root, CharSequence prefix, int start) {
		if (root == null || start >= prefix.length()) {
			return null;
		}
		
		char c = prefix.charAt(start);
		int index = c - 'a';
		
		if (root.next[index] == null) {
			return null;
		}
		
		// 找到了这个点，判断是否到头了
		if (start == prefix.length() - 1) {
			return root.next[index];
		} else {
			return findPrefix(root.next[index], prefix, start + 1);
		}
	}
	
	static int[][] ssDirection = { 
			{ -1, 0 }, // up
			{ 1, 0 }, // down
			{ 0, -1 }, // left
			{ 0, 1 } // right
	};

	private boolean outofBound(int rowEnd, int columnEnd, int row, int column) {
		if (row < 0 || row >= rowEnd) {
			return true;
		}

		if (column < 0 || column >= columnEnd) {
			return true;
		}
		
		return false;
	}
	
	// https://leetcode.com/problems/out-of-boundary-paths/description/
	public int findPaths(int m, int n, int N, int i, int j) {
		long[][][] dp = new long[m][n][N+1];
		
		for (int o = 0; o < m; ++o) 
			for (int p = 0; p < n; ++p)
				for (int q = 0; q <= N; ++q)
					dp[o][p][q] = -1;
		
        return (int)findPathsRecurse(m, n, N, i, j, dp) % (1000000000 + 7);
    }

	public long findPathsRecurse(int rowCount, int columnCout, int leftStep, int row, int column, long[][][] dp) {
		if (leftStep <= 0 || outofBound(rowCount, columnCout, row, column)) {
			return 0;
		}
		
		if (dp[row][column][leftStep] != -1) {
			return dp[row][column][leftStep];
		}
		long result = 0;
		for (int i = 0; i < ssDirection.length; ++i) {
			int newRow = row + ssDirection[i][0];
			int newColumn = column + ssDirection[i][1];
			if (outofBound(rowCount, columnCout, newRow, newColumn)) {
				result++;
			} else {
				result += findPathsRecurse(rowCount, columnCout, leftStep - 1, newRow, newColumn, dp) ;
			}
		}
		
		result %= (1000000000 + 7);
		dp[row][column][leftStep] = result;
		return result;
	}
	
	// https://leetcode.com/problems/degree-of-an-array/description/
	public int findShortestSubArray(int[] nums) {
        if (nums == null || nums.length <= 0) {
        		return 0;
        }
        
        Map<Integer, Integer> mapCnt = new HashMap<>();
        Map<Integer, Integer> mapFirstPos = new HashMap<>();
        
        int maxCnt = 0;
        int result = Integer.MAX_VALUE;
        for (int i = 0; i < nums.length; ++i) {
        		int cnt;
        		if (mapCnt.containsKey(nums[i])) {
        			cnt = mapCnt.get(nums[i]) + 1;
        			mapCnt.put(nums[i], cnt);
        		} else {
        			cnt = 1;
        			mapCnt.put(nums[i], cnt);
        		}
        		
        		if (!mapFirstPos.containsKey(nums[i])) {
        			mapFirstPos.put(nums[i], i);
        		}
        		if (cnt == maxCnt) {
        			result = Math.min(result, i - mapFirstPos.get(nums[i]) + 1);
        		} else if (cnt > maxCnt) {
        			maxCnt = cnt;
        			result = i - mapFirstPos.get(nums[i]) + 1;
        		}
        }
        
        return result;
    }
	
	// https://leetcode.com/problems/count-of-range-sum/description/
	public int countRangeSum(int[] nums, int lower, int upper) {
        if (nums == null || nums.length <= 0) {
        		return 0;
        }
        
        long[] sums = new long[nums.length];
        long[] sort = new long[nums.length];
        sums[0] = nums[0];
        
        for (int i = 1; i < nums.length; i++) {
        		sums[i] = sums[i - 1] + nums[i];
        }
        
        return countRange(sums, sort, 0, nums.length - 1, lower, upper);
    }
	
	// 二分查找，找到第一个大于等于给定数据的index
	private int findFirstUpper(long[] sums, int start, int end, long target) {
		if (start > end) {
			return -1;
		}
		
		int oldEnd = end;
		while (start <= end) {
			int mid = start + (end - start) / 2;
			if (sums[mid] < target) {
				start = mid + 1;
			} else {
				end = mid - 1;
			}
		}
		
		if (end == oldEnd) {
			return -1;
		}
		
		return end + 1;
	}
	
	// 二分查找，找到最后一个小于等于给定数据
	private int findLastLower(long[] sums, int start, int end, long target) {
		if (start > end) {
			return -1;
		}
		
		int oldStart = start;
		while (start <= end) {
			int mid = start + (end - start) / 2;
			if (sums[mid] > target) {
				end = mid - 1;
			} else {
				start = mid + 1;
			}
		}
		
		if (start == oldStart) {
			return -1;
		}
		
		return start - 1;
	}
	
	private int countRange(long[] sums, long[] sort, int start, int end, int lower, int upper) {
		if (start > end){
			return 0;
		} else if (start == end) {
			if (sums[end] >= lower && sums[end] <= upper) {
				return 1;
			} else {
				return 0;
			}
		}
		
		int mid = (start + end) / 2;
		
		int leftCnt = countRange(sums, sort, start, mid, lower, upper);
		int rightCnt = countRange(sums, sort, mid + 1, end, lower, upper);
		
		
		int result = leftCnt + rightCnt;
		
		// 此时，sums的的left和right已经全部排好顺序
		for (int i = mid + 1; i <= end; i++) {
			long max = sums[i] - lower;
			long min = sums[i] - upper;
			
			int firstIndex = findFirstUpper(sums, start, mid, min);
			int lastIndex = findLastLower(sums, start, mid, max);
			
			if (firstIndex != -1 && lastIndex != -1) {
				result += (lastIndex - firstIndex + 1);
			}
		}
		
		int leftStart = start;
		int rightStart = mid + 1;
		int k = start;
		while (leftStart <= mid && rightStart <= end) {
			if (sums[leftStart] <= sums[rightStart]) {
				sort[k++] = sums[leftStart++];
			} else {
				sort[k++] = sums[rightStart++];
			}
		}
		
		while (leftStart <= mid) {
			sort[k++] = sums[leftStart++];
		}
		
		while (rightStart <= end) {
			sort[k++] = sums[rightStart++];
		}
		
		while (start <= end) {
			sums[start] = sort[start];
			start++;
		}
		return result;
	}
	
	// https://leetcode.com/problems/stickers-to-spell-word/description/
	// 我擦，理解错了
	public int minStickers2(String[] stickers, String target) {
        if (target == null || target.length() <= 0) {
        		return 0;
        }
        
        if (stickers == null || stickers.length <= 0) {
        		return -1;
        }
        
        int[][] dpMin = new int[target.length()][target.length()];
        int[][] dpFind = new int[target.length()][target.length()];
        
        for (int i = 0; i < target.length(); i++) {
        		for (int j = i; j < target.length(); j++) {
        			dpMin[i][j] = -2;
        			dpFind[i][j] = 2;
        		}
        }
        
        
        return minStickers(stickers, target, 0, target.length() - 1, dpMin, dpFind);
        
    }
	
	private int minStickers(String[] stickers, CharSequence target, int start, int end, int[][] dpMin, int[][] dpFind) {
		if (dpMin[start][end] != -2) {
			return dpMin[start][end];
		}
		
		int result = Integer.MAX_VALUE;
		if (findTarget(stickers, target, start, end, dpFind)) {
			dpMin[start][end] = 1;
			return 1;
		}
		
		for (int i = start; i <= end - 1; i++) {
			boolean findLeft = findTarget(stickers, target, start, i, dpFind);
			boolean findRight = findTarget(stickers, target, i + 1, end, dpFind);
			
			int leftMin = -1;
			if (findLeft) {
				leftMin = 1;
			} else {
				leftMin = minStickers(stickers, target, start, i, dpMin, dpFind);
			}
			
			int rightMin = -1;
			if (result != -1) {
				if (findRight) {
					rightMin = 1;
				} else {
					rightMin = minStickers(stickers, target, i + 1, end, dpMin, dpFind);
				}
			}
			
			if (leftMin != -1 && rightMin != -1) {
				result = Math.min(result, leftMin + rightMin);
			}
		}
		
		result = (result == Integer.MAX_VALUE ? -1 : result);
		dpMin[start][end] = result;
		
		return result;
		
	}
	
	private boolean findTarget(String[] stickers, CharSequence target, int start, int end, int[][] dpFind) {
		if (dpFind[start][end] != 2) {
			return dpFind[start][end] == 1;
		}
		
		for (String sticker : stickers) {
			if (sticker.contains(target.subSequence(start, end + 1))) {
				dpFind[start][end] = 1;
				return true;
			}
		}
		
		dpFind[start][end] = 0;
		return false;
	}
	
	// 总是超时啊
	public int minStickers1(String[] stickers, String target) {
        if (target == null || target.length() <= 0) {
        		return 0;
        }
        
        if (stickers == null || stickers.length <= 0) {
        		return -1;
        }
        
        int[] targetHash = new int[26];
        int[] before = new int[26];
        int[] after = new int[26];
        for (int i = 0; i < target.length(); i++) {
        		targetHash[target.charAt(i) - 'a']++;
        }
        
        int[][] stickersHash = new int[stickers.length][26];
        for (int i = 0; i < stickers.length; ++i) {
        		for (int j = 0; j < stickers[i].length(); ++j) {
        			stickersHash[i][stickers[i].charAt(j) - 'a']++;
        		}
        }
        Map<StikerNode, Integer> cache = new HashMap<>();
        return minStickers(targetHash, stickersHash, 0, before, after, cache);
    }
	
	static class StikerNode {
		public int i;
		public String left;
		
		@Override
		public boolean equals(Object obj) {
			// TODO Auto-generated method stub
			if (!(obj instanceof StikerNode))
				return false;
			
			return i == ((StikerNode) obj).i && left.equals(((StikerNode) obj).left);
		}
	}
	
	private String hashToString(int[] target) {
		StringBuilder builder = new StringBuilder();
		for (int idx = 0; idx < target.length; ++idx) {
			for (int i = 0; i < target[idx]; ++i) {
				builder.append( (char) ('a' + idx));
			}
		}
		
		return builder.toString();
	}
	
	private void putMinStikers(int[] target, int i, Map<StikerNode, Integer> map, int min) {
		String str = hashToString(target);
		StikerNode node = new StikerNode();
		node.i = i;
		node.left = str;
		
		map.put(node, min);
		
		System.out.println(node.left + "_" + i + ":" + min);
	}
	
	public int getMinStikers(int[] target, int i, Map<StikerNode, Integer> map) {
		String str = hashToString(target);
		StikerNode node = new StikerNode();
		node.i = i;
		node.left = str;
		
		return map.getOrDefault(node, -2);
	}
	
	public int minStickers(int[] target, int[][] stickers, int i, int[] before, int[] after, Map<StikerNode, Integer> cache) {
		if (isTargetEmpty(target)) {
			return 0;
		}
		if (i >= stickers.length) {
			return -1;
		}
		
		int minCache = getMinStikers(target, i, cache);
		if (minCache != -2) {
			return minCache;
		}
		
		int maxConsume = maxCunsumeSticker(target, stickers[i]);
		if (maxConsume == 0) {
			
			int temp = minStickers(target, stickers, i + 1, before, after, cache);
			putMinStikers(target, i, cache, temp);
			return temp;
		}
		
		// 不用这个
		int result = minStickers(target, stickers, i + 1, before, after, cache);
		if (result == -1) {
			result = Integer.MAX_VALUE;
		}
		
		for (int j = 0; j < target.length; ++j)
			before[j] = target[j];

		
		int consume;
		for (consume = 1; consume <= maxConsume; ++consume) {
			after = new int[target.length];
			boolean consumed = consumeSticker(before, stickers[i], after);
			if (consumed) {
				int nextMin = minStickers(after, stickers, i + 1, before, after, cache);
				if (nextMin != -1) {
					result = Math.min(result, nextMin + consume);
				}
				
				for (int j = 0; j < target.length; ++j) {
					before[j] = after[j];
				}
			} else {
				if (consume != 1) {
					int nextMin = minStickers(before, stickers, i + 1, before, after, cache);
					if (nextMin != -1) {
						result = Math.min(result, nextMin + consume - 1);
					}
				}
				
				break;
			}
		}
		
		result = result == Integer.MAX_VALUE ? -1 : result;
		putMinStikers(target, i, cache, result);
		
		return result;
	}
	
	public boolean isTargetEmpty(int[] target) {
		for (int i = 0; i < 26; ++i) {
			if (target[i] > 0) {
				return false;
			}
		}
		return true;
	}
	
	// 返回表示是否还可以消费
	private boolean consumeSticker(int[] target, int[] sticker, int[] result) {
		boolean consumed = false;
		for (int i = 0; i < target.length; ++i) {
			result[i] = target[i];
			if (target[i] <= 0) continue;
			if (sticker[i] <= 0) continue;
			
			result[i] = target[i] - sticker[i] > 0 ? target[i] - sticker[i] : 0;
			consumed = true;
		}
		
		return consumed;
	}
	
	// 计算最多消耗多少个这样的sticker，消耗多了就没啥用了
	public int maxCunsumeSticker(int[] target, int[] sticker) {
		int result = 0;
		for (int i = 0; i < 26; ++i) {
			if (target[i] == 0) {
				continue;
			}
			if (sticker[i] <= 0) {
				continue;
			}
			
			result = Math.max(result, (target[i] + sticker[i] - 1) / sticker[i]);
		}
		
		return result;
	}
	
	public int minStickers(String[] stickers, String target) { 
		if (stickers == null || stickers.length <= 0) {
			return -1;
		}
		
		if (target == null || target.isEmpty()) {
			return 0;
		}
		
		String hashTarget = toHashStr(target);
		for (int i = 0; i < stickers.length; ++i) {
			stickers[i] = toHashStr(stickers[i]);
		}
		
		return minStickers(hashTarget, stickers);
		
	}
	
	static class MinStickersComparator implements Comparator<String> {
		public final String mTarget;
		
		public MinStickersComparator(String target) {
			// TODO Auto-generated constructor stub
			mTarget = target;
		}
		
		@Override
		public int compare(String o1, String o2) {
			// TODO Auto-generated method stub
			return containsLetter(mTarget, o2) - containsLetter(mTarget, o1);
		}
		
		private int containsLetter(String target, String src) {
			int[] t = new int[26];
			int result = 0;
			
			for (int i = 0; i < target.length(); ++i)
				t[ target.charAt(i) - 'a']++;
			
			for (int i = 0; i < src.length(); ++i) {
				if (t[src.charAt(i) - 'a'] > 0) {
					t[src.charAt(i) - 'a']--;
					result++;
				}
			}
			
			return result;
		}
	}
	
	private String toHashStr(String target) {
		StringBuilder builder = new StringBuilder();
		
		int[] hash = new int[26];
		for (int i = 0; i < target.length(); ++i) {
			hash[target.charAt(i) - 'a']++;
		}
		for (int idx = 0; idx < hash.length; ++idx) {
			for (int i = 0; i < hash[idx]; ++i) {
				builder.append( (char) ('a' + idx));
			}
		}
		
		return builder.toString();
	}
	
	private String spellSticker(String target, String sticker) {
		int[] hashTarget = new int[26];
		for (int i = 0; i < target.length(); ++i) {
			hashTarget[target.charAt(i) - 'a']++;
		}
		
		int[] hashSticker = new int[26];
		for (int i = 0; i < sticker.length(); ++i) {
			hashSticker[sticker.charAt(i) - 'a']++;
		}
		
		StringBuilder builder = new StringBuilder();
		for (int i = 0; i < 26; ++i) {
			if (hashSticker[i] > 0 && hashTarget[i] > 0) {
				hashTarget[i] = hashTarget[i] >= hashSticker[i] ? hashTarget[i] - hashSticker[i] : 0;
			}
			for (int j = 0; j < hashTarget[i]; ++j) {
				builder.append( (char) ('a' + i));
			}
		}
		
		return builder.toString();
	}
	
	private int minStickers(String target, String[] stickers) {
		
		Map<String, Integer> cache = new HashMap<>();
		
		Queue<String> queue = new LinkedList<>();
		queue.add(target);
		
		int cur = 1;
		int next = 0;
		int level = 1;
		while (!queue.isEmpty()) {
			String head = queue.poll();
			cur--;
			
			// 给stickers排序，从最大的开始
			Arrays.sort(stickers, new MinStickersComparator(head));
			
			// 然后
			for (int  i = 0; i < stickers.length; ++i) {
				// 生成新的target
				String newTarget = spellSticker(head, stickers[i]);
				if (newTarget.equals(head)) {
					break;
				}
				
				if (newTarget.isEmpty()) {
					return level;
				}
				if (!cache.containsKey(newTarget)) {
					cache.put(newTarget, level);
					next++;
					queue.offer(newTarget);
				}
			}
			
			if (cur == 0) {
				cur = next;
				next = 0;
				level++;
			}
		}
		
		return -1;
	}
	
	
	// https://leetcode.com/problems/perfect-squares/description/
//	public int numSquares(int n) {
//        return 0;
//    }
	
	// https://leetcode.com/problems/matchsticks-to-square/description/
	public boolean makesquare(int[] nums) {
		if (nums == null || nums.length < 4) {
			return false;
		}
		
		// 怎样分成四份，每份的和是一样的
		
		int total = 0;
		for (int i = 0; i < nums.length; ++i) {
			total += nums[i];
		}
		
		if ((total % 4) != 0) {
			return false;
		}
		
		int arverage = total / 4;
		if (arverage <= 0) {
			return false;
		}
		
		for (int i = 0; i < nums.length; i++) {
			if (nums[i] > arverage) {
				return false;
			}
		}
		
		Arrays.sort(nums);
		
		return makesquare(nums, 0, arverage, 0, 0, 0, 0);
    }
	
	private boolean makesquare(int[] nums, int idx, int average, int sum1, int sum2, int sum3, int sum4) {
		if (idx == nums.length) {
			if (sum1 == average && sum2 == average && sum3 == average && sum4 == average) {
				return true;
			} else {
				return false;
			}
		}
		
		if (sum1 > average || sum2 > average || sum3 > average || sum4 > average) {
			return false;
		}
		
		int value = nums[idx];
		if (sum1 < average && sum1 + value <= average) {
			if (makesquare(nums, idx + 1, average, sum1 + value, sum2, sum3, sum4)) {
				return true;
			}
		}
		
		if (sum1 != sum2 && sum2 < average && sum2 + value <= average) {
			if (makesquare(nums, idx + 1, average, sum1, sum2 + value, sum3, sum4)) {
				return true;
			}
		}
		
		if (sum2 != sum3 && sum3 != sum1 && sum3 < average && sum3 + value <= average) {
			if (makesquare(nums, idx + 1, average, sum1, sum2, sum3 + value, sum4)) {
				return true;
			}
		}
		
		if (sum3 != sum4 && sum4 != sum2 && sum4 != sum1 && sum4 < average && sum4 + value <= average) {
			if (makesquare(nums, idx + 1, average, sum1, sum2, sum3, sum4 + value)) {
				return true;
			}
		}
		return false;
	}
	
	private void printArrays(int[] nums) {
		for (int num : nums) {
			System.out.print(num + ",");
		}
		System.out.println();
	}
	
	// https://leetcode.com/problems/24-game/description/
	public boolean judgePoint24(int[] nums) {
        boolean[] result = new boolean[1];
        
        permute(nums, 0, result);
        
        return result[0];
    }
	
	public void permute(int[] nums, int start, boolean[] result) {
		if (result[0]) {
			return;
		}
		if (start == nums.length) {
			result[0] = judgePoint24(nums, 0, nums.length - 1, 24);
			return;
		}
		
		for (int i = start; i < nums.length; i++) { 
			swap(nums, start, i);
			permute(nums, start + 1, result);
			swap(nums, start, i);
		}
	}

	
	private boolean isEqual(double src, double dest) {
		return Math.abs(src - dest) <= 0.000001;
	}
	
	public boolean judgePoint24(int[] nums, int start, int end, double target) {
		if (start > end) 
			return false;
		// 一个数
		if (end == start)
			return isEqual(nums[start], target);
		
		// 二个数
		if (end - start == 1) {
			return isEqual(target, (nums[start] + nums[end])) ||
					isEqual(target, (nums[start] - nums[end])) ||
					isEqual(target, (nums[start] * nums[end])) ||
					isEqual(target, (nums[start] * 1.0f / nums[end]));
		}
		// 三个数
		else if (end - start == 2) {
			// 左边一个，右边两个
			// 先处理+ - * /的情况
			if (judgePoint24(nums, start + 1, end, target - nums[start])) {
				return true;
			} else if (judgePoint24(nums, start + 1, end, nums[start] - target)) {
				return true;
			} else if (judgePoint24(nums, start + 1, end, target * 1.0f / nums[start])) {
				return true;
			} else if (judgePoint24(nums, start + 1, end, nums[start] * 1.0f / target)) {
				return true;
			}
		
			// 左边两个，右边一个
			if (judgePoint24(nums, start, end - 1, target - nums[end])) {
				return true;
			} else if (judgePoint24(nums, start, end - 1, nums[end] + target)) {
				return true;
			} else if (judgePoint24(nums, start, end - 1, target * 1.0f / nums[end])) {
				return true;
			} else if (judgePoint24(nums, start, end - 1, target * 1.0f * nums[end])) {
				return true;
			}
			
			return false;
		} 
		// 四个数
		else if (end - start == 3) {
			// 左边一个，右边三个
			// 先处理+ - * /的情况
			if (judgePoint24(nums, start + 1, end, target - nums[start])) {
				return true;
			} else if (judgePoint24(nums, start + 1, end, nums[start] - target)) {
				return true;
			} else if (judgePoint24(nums, start + 1, end, target * 1.0f / nums[start])) {
				return true;
			} else if (judgePoint24(nums, start + 1, end, nums[start] * 1.0f / target)) {
				return true;
			}
		
			// 左边三个，右边一个
			if (judgePoint24(nums, start, end - 1, target - nums[end])) {
				return true;
			} else if (judgePoint24(nums, start, end - 1, nums[end] + target)) {
				return true;
			} else if (judgePoint24(nums, start, end - 1, target * 1.0f / nums[end])) {
				return true;
			} else if (judgePoint24(nums, start, end - 1, target * 1.0f * nums[end])) {
				return true;
			}
			
			double[] allValues = new double[6];
			allValues[0] = nums[start] + nums[start + 1];
			allValues[1] = nums[start] * nums[start + 1];
			allValues[2] = nums[start] - nums[start + 1];
			allValues[3] = nums[start + 1] - nums[start];
			allValues[4] = nums[start] * 1.0f / nums[start + 1];
			allValues[5] = nums[start + 1] * 1.0f / nums[start];
			
			for (int i = 0; i < allValues.length; ++i) {
				// 左边两个，右边两个
				if (judgePoint24(nums, end - 1, end, target - allValues[i])) {
					return true;
				} else if (judgePoint24(nums, end - 1, end, allValues[i] - target)) {
					return true;
				} else if (judgePoint24(nums, end - 1, end, target * 1.0f / allValues[i])) {
					return true;
				} else if (allValues[i] != 0 && judgePoint24(nums, end - 1, end, allValues[i] * 1.0f / target)) {
					return true;
				}
			}
			
			return false;
		}
		
		return false;
		
	}
	
	// https://leetcode.com/problems/remove-duplicate-letters/description/
	static class LetterNode {
		char c;
		LetterNode next;
		LetterNode pre;
	}
	
	private void deleteNode(LetterNode head, LetterNode tail, LetterNode node) {
		if (head.next == tail) {
			return;
		}
		
		LetterNode pre = node.pre;
		LetterNode next = node.next;
		
		pre.next = next;
		next.pre = pre;
	}
	
	private void addTail(LetterNode head, LetterNode tail, LetterNode node) {
		LetterNode tailPre = tail.pre;
		
		tailPre.next = node;
		node.pre = tailPre;
		node.next = tail;
		tail.pre = node;
	}
	
	public String removeDuplicateLetters(String s) {
		if (s == null || s.length() <= 0) {
			return "";
		}
		
		LetterNode head = new LetterNode();
		LetterNode tail = new LetterNode();
		
		head.next = tail;
		tail.pre = head;
		
		int[] hash = new int[26];
		for (int i = 0; i < s.length(); ++i) {
			hash[s.charAt(i) - 'a']++;
		}
		
		Map<Character, LetterNode> cache = new HashMap<>();
		
		for (int idx = 0; idx < s.length(); ++idx) {
			char c = s.charAt(idx);
			if (!cache.containsKey(c)) {
				LetterNode node = new LetterNode();
				node.c = c;
				addTail(head, tail, node);
				cache.put(c, node);
				hash[c - 'a']--;
				continue;
			}
			
			LetterNode node = cache.get(c);
			
			// 从node开始遍历
			LetterNode temp = node.next;
			LetterNode min = node;
			while (temp != tail) {
				if (min.c > temp.c) {
					min = temp;
				}
				
				if (hash[temp.c - 'a'] <= 0) {	
					break;
				}
				
				temp = temp.next;
			}
			
			// 找到一个比他小
			if (min.c < c) {
				// 删除从node到min之前的所有节点，并且将新的加入到末尾
				temp = node;
				while (temp != min) {
					LetterNode next = temp.next;
					cache.remove(temp.c);
					deleteNode(head, tail, temp);
					temp = next;					
				}
				
				node = new LetterNode();
				node.c = c;
				addTail(head, tail, node);
				cache.put(c, node);
				hash[c - 'a']--;
				
			} else {
				// 不用新加入的，用老的
				hash[c - 'a']--;
			}
		}
		
		StringBuilder builder = new StringBuilder();
		LetterNode node = head;
		while (node != tail) {
			builder.append(node.c);
			node = node.next;
		}
		
		return builder.toString();
	}
	
	// https://leetcode.com/problems/next-greater-element-i/description/
	public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        final int cnt = nums2.length;
       
        Map<Integer, Integer> map = new HashMap<>();
        Stack<Integer> stack = new Stack<>();
        
        for (int i = 0; i < cnt; ++i) {
        		if (stack.isEmpty()) {
        			stack.push(i);
        			continue;
        		}
        		
        		int top = stack.peek();
        		if (nums2[top] > nums2[i]) {
        			stack.push(i);
        			continue;
        		}
        		
        		while (!stack.isEmpty() && nums2[stack.peek()] < nums2[i]) {
        			top = stack.pop();
        			map.put(nums2[top], nums2[i]);
        		}
        		
        		stack.push(i);
        }
        
        int[] res = new int[nums1.length];
        for (int i = 0; i < nums1.length; i++) {
        		res[i] = map.getOrDefault(nums1[i], -1);
        }
        
        return res;
    }
	
	public int[] nextGreaterElements(int[] nums) {
        Stack<Integer> stack = new Stack<>();
        int[] result = new int[nums.length];
        for (int i = 0; i < nums.length; ++i) {
        		result[i] = -1;
        }
        for (int i = 0; i < nums.length * 2; ++i) {
        		while (!stack.isEmpty() && nums[stack.peek() % nums.length] < nums[i % nums.length]) {
        			result[stack.pop() % nums.length] = nums[i % nums.length];
        		}
        		stack.push(i);
        }
        
        return result;
    }
	
	// https://leetcode.com/problems/binary-tree-level-order-traversal/description/

	
	public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        
        if (root == null) {
        		return result;
        }
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        
        int cur = 1;
        int next = 0;
        List<Integer> level = new ArrayList<>();
        
        while (!queue.isEmpty()) {
        		TreeNode node = (TreeNode) queue.poll();
        		level.add(node.val);
        		
        		if (node.left != null) {
        			queue.add(node.left);
        			next++;
        		}
        		
        		if (node.right != null) {
        			queue.add(node.right);
        			next++;
        		}
        		
        		if (--cur == 0) {
        			cur = next;
        			next = 0;
        			result.add(new ArrayList<>(level));
        			level.clear();
        		}
        }
        
        return result;
    }
	
	public List<List<Integer>> levelOrderBottom(TreeNode root) {
		List<List<Integer>> result = new ArrayList<>();
        
        if (root == null) {
        		return result;
        }
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        
        int cur = 1;
        int next = 0;
        List<Integer> level = new ArrayList<>();
        
        while (!queue.isEmpty()) {
        		TreeNode node = (TreeNode) queue.poll();
        		level.add(node.val);
        		
        		if (node.left != null) {
        			queue.add(node.left);
        			next++;
        		}
        		
        		if (node.right != null) {
        			queue.add(node.right);
        			next++;
        		}
        		
        		if (--cur == 0) {
        			cur = next;
        			next = 0;
        			result.add(0, new ArrayList<>(level));
        			level.clear();
        		}
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
		
		int cur = 1;
		int next = 0;
		int level = 0;
		
		LinkedList<Integer> line = new LinkedList<>();
		while (!queue.isEmpty()) {
			TreeNode node = (TreeNode) queue.poll();
			line.add(node.val);
			
			if (node.left != null) {
				queue.add(node.left);
				next++;
			}
			
			if (node.right != null) {
				queue.add(node.right);
				next++;
			}
			
			if (--cur == 0) {
				cur = next;
				next = 0;
				
				if ((level & 1) == 0) {
					result.add(new LinkedList<>(line));
				} else {
					LinkedList<Integer> reverse = new LinkedList<>();
					while (!line.isEmpty()) {
						reverse.add(line.pollLast());
					}
					result.add(reverse);
				}
				
				line.clear();
				
				level++;
			}
		}
		
		return result;
    }
	
	public boolean isSameTree(TreeNode p, TreeNode q) {
		if (p == null && q == null)
			return true;
		if (p == null && q != null)
			return false;
		if (p != null && q == null)
			return false;
		
		if (p.val != q.val) {
			return false;
		}
		
		return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }
	
	public int minDepth(TreeNode root) {
        if (root == null)
        		return 0;
        
        if (root.left == null && root.right == null)
        		return 1;
        else if (root.left == null && root.right != null) 
        		return 1 + minDepth(root.right);
        else if (root.left != null && root.right == null)
        		return 1 + minDepth(root.left);
        else
        		return Math.min(1 + minDepth(root.left), 1 + minDepth(root.right));
    }
	
	// https://leetcode.com/problems/redundant-connection/description/
	public int[] findRedundantConnection(int[][] edges) {
		int[] result = new int[2];
		
		int N = edges.length;
		
		int[] union = new int[N];
		for (int i = 0; i < N; i++) {
			union[i] = i;
		}
		
		for (int i = 0; i < N; ++i) {
			int p1 = edges[i][0] - 1;
			int p2 = edges[i][1] - 1;
			
			int union1 = union[p1];
			int union2 = union[p2];
			
			if (union1 == union2) {
				result[0] = p1 + 1;
				result[1] = p2 + 1;
			} else {
				// 将id统一
				for (int j = 0; j < N; ++j) {
					if (union[j] == union2) {
						union[j] = union1;
					}
				}
			}
		}
		
		return result;
        
    }
	
	private int findUnion(int[] union, int idx) {
		while(idx != union[idx]) {
			idx = union[idx];
		}
		
		return idx;
	}
	
	public int[] findRedundantConnectionI(int[][] edges) {
		int[] result = new int[2];
		
		int N = edges.length;
		
		int[] union = new int[N];
		for (int i = 0; i < N; i++) {
			union[i] = i;
		}
		
		for (int i = 0; i < N; ++i) {
			int p1 = edges[i][0] - 1;
			int p2 = edges[i][1] - 1;
			
			int union1 = findUnion(union, p1);
			int union2 = findUnion(union, p2);
			
			if (union1 == union2) {
				result[0] = p1 + 1;
				result[1] = p2 + 1;
			} else {
				// 将id统一
				union[union2] = union1;
			}
		}
		
		return result;
        
    }
	
	public int[] findRedundantDirectedConnection(int[][] edges) {
		int[] result = new int[2];
		final int N = edges.length;
		
		int[] parents = new int[N];
		int dupIdx1 = -1;
		int dupIdx2 = -1;
		
		int[][] childParents = new int[N][2];
		// childParents,index为child结点,0为child的parent，1为child出现的索引
		for (int i = 0; i < N; ++i) {
			childParents[i][0] = -1;
			childParents[i][1] = -1;
			parents[i] = i;
		}
		
		for (int i = 0; i < N; ++i) {
			int parentNode = edges[i][0] - 1;
			int childNode = edges[i][1] - 1;
			
			if (childParents[childNode][0] == -1) {
				childParents[childNode][0] = parentNode;
				childParents[childNode][1] = i;
			} else {
				dupIdx1 = childParents[childNode][1];
				dupIdx2 = i;
				break;
			}
		}
		
		if (dupIdx1 == -1) {
			// 没有一个child有两个父结点
			for (int i = 0; i < N; ++i) {
				// 判断是否有环
				int parentNode = edges[i][0] - 1;
				int childNode = edges[i][1] - 1;
				
				// 判断childNode连接上parent，会不会有环，有环，则记录
				if (isCircle(parents, parentNode, childNode)) {
					result[0] = parentNode + 1;
					result[1] = childNode + 1;
					
				} else {
					parents[childNode] = parentNode;
				}
			}
			
			return result;
		} else {
			for (int i = 0; i < N; ++i) {
				if (i == dupIdx2) {
					continue;
				}
				// 判断是否有环
				int parentNode = edges[i][0] - 1;
				int childNode = edges[i][1] - 1;
				
				// 判断childNode连接上parent，会不会有环，有环，则记录
				if (isCircle(parents, parentNode, childNode)) {
					result[0] = edges[dupIdx1][0];
					result[1] = edges[dupIdx1][1];
					return result;
				} else {
					parents[childNode] = parentNode;
				}
			}
			
			result[0] = edges[dupIdx2][0];
			result[1] = edges[dupIdx2][1];
			
			return result;
		}
	}
	
	private boolean isCircle(int[] parents, int parentNode, int childNode) {
		while (parentNode != parents[parentNode]) {
			parentNode = parents[parentNode];
			if (parentNode == childNode) {
				return true;
			}
		}
		return false;
	}
	
	// https://leetcode.com/problems/word-break/description/
	public boolean wordBreak(String s, List<String> wordDict) {
        if (wordDict == null || s == null || s.length() <= 0) {
        		return false;
        }
        
        Set<String> set = new HashSet<>();
        set.addAll(wordDict);
        
        boolean[][] dp = new boolean[s.length()][s.length()];
        for (int i = 0; i < s.length(); ++i) {
        		dp[i][0] = set.contains(String.valueOf(s.charAt(i)));
        }
        for (int j = 1; j < s.length(); ++j) {
	    		for (int i = 0; i + j < s.length(); i++) {
	    			for (int k = 0; k < j; ++k) {
	    				System.out.print(String.format("[%d, %d] [%d, %d], ", i, i + k, i + k + 1, j - k - 1));
	    				if (dp[i][k] && dp[i + k + 1][j - k - 1]) {
	    					
	    					
	    					dp[i][j] = true;
	    					break;
	    				}
	    			}
	    			
	    			if (!dp[i][j]) {
	    				dp[i][j] = set.contains(s.substring(i, i + j+1));
	    			}
	    		}
	    		System.out.println();
        }
        
        return dp[0][s.length() - 1];
        
    }
	
	public List<String> wordBreakII(String s, List<String> wordDict) {
        Set<String> set = new HashSet<>();
        set.addAll(wordDict);
        
        return wordBreak(s, s.length() - 1, set);
    }
	
	private List<String> wordBreak(CharSequence s, int end, Set<String> wordDict) {
		List<String> result = new ArrayList<>();
		if (end < 0) {
			result.add("");
			return result;
		}
		
		for (int i = end; i >= 0; --i) {
			CharSequence suffix = s.subSequence(i, end + 1);
			if (wordDict.contains(suffix)) {
				List<String> prefixs = wordBreak(s, i - 1, wordDict);
				for (String prefix : prefixs) {
					if (prefix.isEmpty()) {
						result.add(suffix.toString());
					} else {
						result.add(prefix + " " + suffix.toString());
					}
				}
			}
		}
		
		return result;
	}
	
	private static class TrieNodeII {
		public char c;
		public boolean isWord;
		public TrieNodeII[] next = new TrieNodeII[26];
		public TrieNodeII(char c) {
			// TODO Auto-generated constructor stub
			this.c = c;
		}
	}
	
	private static void addTrieWord(TrieNodeII root, String word) {
		addTrieWord(root, word, 0);
	}
	
	private static void addTrieWord(TrieNodeII root, CharSequence word, int start) {
		if (start >= word.length()) {
			return;
		}
		char c = word.charAt(start);
		TrieNodeII node = root.next[c - 'a'];
		if (node == null) {
			node = new TrieNodeII(c);
			root.next[c - 'a'] = node;
		}
		
		if (start == word.length() - 1) {
			node.isWord = true;
			return;
		}
		
		addTrieWord(node, word, start + 1);
	}
	
	private static boolean findTrieWord(TrieNodeII root, CharSequence word, int start) {
		
		TrieNodeII node = root;
		for (int idx = start; idx < word.length(); ++idx) {
			char c = word.charAt(idx);
			node = node.next[c - 'a'];
			if (node == null)
				return false;
			if (idx == word.length() - 1)
				return node.isWord;
			
		}
		return false;
	}
	
	private static boolean findTrieWord(TrieNodeII root, CharSequence word, int start, int end) {
		
		TrieNodeII node = root;
		for (int idx = start; idx <= end; ++idx) {
			char c = word.charAt(idx);
			node = node.next[c - 'a'];
			if (node == null)
				return false;
			if (idx == end)
				return node.isWord;
			
		}
		return false;
	}
	
	public List<String> wordBreakDp(String s, List<String> wordDict) {
		if (!wordBreakDpBoolean(s, wordDict)) {
			return new ArrayList<>();
		}
		
        TrieNodeII root = new TrieNodeII('#');
        for (String word : wordDict) {
        		addTrieWord(root, word);
        }
        
        List<String>[] dp = new ArrayList[s.length() + 1];
        
        List<String> empty = new ArrayList<>();
        empty.add("");
        
        dp[0] = empty;
        
        for (int i = 1; i <= s.length(); ++i) {
        		List<String> next = new ArrayList<>();
        		for (int j = 1; j <= i; ++j) {
        			List<String> prefixs = dp[j-1];
        			if (prefixs.size() <= 0) continue;
        			String sub = s.substring(j - 1, i);
        			
        			if (findTrieWord(root, sub, 0)) {
        				for (String prefix : prefixs) {
        					if (prefix.isEmpty()) {
        						next.add(sub);
        					} else {
        						next.add(prefix + " " + sub);
        					}
        				}
        			}
        		}
        		dp[i] = next;
        }
        
        return dp[s.length()];
    }
	
	public boolean wordBreakDpBoolean(String s, List<String> wordDict) {
        TrieNodeII root = new TrieNodeII('#');
        for (String word : wordDict) {
        		addTrieWord(root, word);
        }
        
       boolean[] dp = new boolean[s.length() + 1];
        
        dp[0] = true;
        
        for (int i = 1; i <= s.length(); ++i) {
        		for (int j = 1; j <= i; ++j) {
        			if (dp[j - 1] && findTrieWord(root, s.substring(j - 1, i), 0)) {
        				dp[i] = true;
            			break;
        			}
        		}
        }
        
        return dp[s.length()];
    }
	
	public boolean wordBreak(CharSequence s, TrieNodeII root) {
       boolean[] dp = new boolean[s.length() + 1];
        
        dp[0] = true;
        
        for (int i = 1; i <= s.length(); ++i) {
        		for (int j = 1; j <= i; ++j) {
        			if (dp[j - 1] && findTrieWord(root, s, j - 1, i - 1)) {
        				dp[i] = true;
            			break;
        			}
        		}
        }
        
        return dp[s.length()];
    }
	
	public boolean wordBreakRecurse(CharSequence s, TrieNodeII root, int start) {
		if (start >= s.length()) {
			return true;
		}
		TrieNodeII node = root;
		for (int idx = start; idx <= s.length() - 1; ++idx) {
			char c = s.charAt(idx);
			node = node.next[c - 'a'];
			if (node == null) {
				return false;
			} 
			
			if (node.isWord && wordBreakRecurse(s, root, idx + 1)) {
				return true;
			}
		}
		
		return false;
    }
	
	public List<String> findAllConcatenatedWordsInADict(String[] words) {
		Map<Integer, List<String>> map = new TreeMap<>();
		
		for (String word : words) {
			int length = word.length();
			List<String> list = map.get(length);
			if (list == null) {
				list = new ArrayList<>();
				map.put(length, list);
			}
			list.add(word);
		}
		
		TrieNodeII root = new TrieNodeII('#');
		List<String> result = new ArrayList<>();
		
		boolean firstLength = true;
		for (Map.Entry<Integer, List<String>> entry: map.entrySet()) {
			List<String> list = entry.getValue();
			if (firstLength) {
				firstLength = false;
				for (String word : list) {
					addTrieWord(root, word);
				}
			} else {
				for (String word : list) {
					boolean res = wordBreakRecurse(word, root, 0);
					if (res) {
						result.add(word);
					}
					
				}
				for (String word : list) {
					addTrieWord(root, word);
				}
			}
		}
		
		return result;
    }
	
	// https://leetcode.com/problems/zuma-game/description/
	public int findMinStep(String board, String hand) {
        int[] handLeft = new int[5];
        for (int i = 0; i < hand.length(); ++i) {
        		handLeft[ballToIndext(hand.charAt(i))]++;
        }
        
        return findMinStepRecurse(board, handLeft);
    }
	
	private int findMinStepRecurse(String board, int[] handLeft) {
		if (board.isEmpty()) {
			return 0;
		}
		
		int start = 0;
		int end = 0;
		
		int idx = 0;
		
		int min = Integer.MAX_VALUE;
		
		while (idx <= board.length()) {
			if (idx == board.length() || board.charAt(end) != board.charAt(idx)) {
				
				int use = useHandBall(board.charAt(start), end - start + 1, handLeft);
				if (use != 0) {
					
					// 需要把这个球去掉，并且再走一遍
					String newBoard = board.substring(0, start) + board.substring(end + 1);
					newBoard = deleteOver3Times(newBoard);
					
					int nextMin = findMinStepRecurse(newBoard, handLeft);
					if (nextMin != -1) {
						min = Math.min(min, nextMin + use);
					}
					
					recycleHandBall(board.charAt(start), end - start + 1, handLeft);
				}
				
				start = idx;
				end = idx;
			} else {
				end = idx;
			}
			idx++;
		}
		
		return min == Integer.MAX_VALUE ? -1 : min;
	}
	
	private int ballToIndext(char c) {
		int idx = 0;
		switch (c) {
		case 'R': 
			idx = 0;
			break;
		case 'Y':
			idx = 1;
			break;
		case 'B':
			idx = 2;
			break;
		case 'G':
			idx = 3;
			break;
		case 'W':
			idx = 4;
			break;
		default:
			idx = -1;
			break;
		}
		return idx;
	}
	
	private int useHandBall(char c, int cnt, int[] handleft) {
		int idx = ballToIndext(c);
	
		int use = 0;
		if (handleft[idx] > 0 && handleft[idx] >= 3 - cnt) {
			use = 3 - cnt;
			handleft[idx] -= use;
		}
		
		return use;
	}
	
	private void recycleHandBall(char c, int cnt, int[] handLeft) {
		int idx = ballToIndext(c);
		handLeft[idx] += 3 - cnt;
	}
	
	private String deleteOver3Times(String board) {
		if (board.isEmpty())
			return board;
		
		StringBuilder builder = new StringBuilder();
		
		int index = 0;
		
		int start = 0;
		int end = 0;
		
		while (index <= board.length()) {
			if (index == board.length() || board.charAt(index) != board.charAt(start)) {
				if (end - start + 1 < 3) {
					builder.append(board.substring(start, end + 1));
				}
				
				start = index;
				end = index;
			} else {
				end = index;
			}
			index++;
		}
		
		String newBoard = builder.toString();
		if (newBoard.equals(board)) {
			return newBoard;
		} else {
			return deleteOver3Times(newBoard);
		}
	}
	
	// https://leetcode.com/problems/unique-substrings-in-wraparound-string/description/
	public int findSubstringInWraproundString(String p) {
        if (p == null || p.length() <= 0) {
        		return 0;
        }
        
        int[] dp = new int[26];
        
        int maxLength = 1;
        dp[p.charAt(0) - 'a'] = 1;
        for (int i = 1; i < p.length(); ++i) {
        		char cur = p.charAt(i);
        		char pre = p.charAt(i - 1);
        		
        		if ( (pre == 'z' && cur == 'a') || (cur == pre + 1) ) {
        			maxLength++;
        		} else {
        			maxLength = 1;
        		}
        		
        		dp[cur - 'a'] = Math.max(dp[cur - 'a'], maxLength);
        }
        
        int result = 0;
        for (int i = 0; i < 26; ++i) {
        		result += dp[i];
        }
        
        return result;
    }
	
	// https://leetcode.com/problems/the-skyline-problem/description/
	public List<int[]> getSkyline(int[][] buildings) {
        SkyNode[] nodes = new SkyNode[buildings.length * 2];
        
        for (int i = 0; i < buildings.length; i++) {
        		nodes[2 * i] = new SkyNode(buildings[i][0], true, buildings[i][2]);
        		nodes[2 * i + 1] = new SkyNode(buildings[i][1], false, buildings[i][2]);
        }
        
        Arrays.sort(nodes, new Comparator<SkyNode>() {
        		@Override
        		public int compare(SkyNode o1, SkyNode o2) {
        			// TODO Auto-generated method stub
        			if (o1.pos != o2.pos) {
        				return o1.pos - o2.pos;
        			}
        			
        			if (o1.isLeft && o1.isLeft) {
        				return o1.height - o2.height;
        			}
        			
        			if (o1.isLeft) {
        				return -1;
        			} else {
        				return 1;
        			}
        		}
		});
        
        List<int[]> result = new LinkedList<>();
        
        PriorityQueue<Integer> heap = new PriorityQueue<>(new Comparator<Integer>() {
	        	@Override
	        	public int compare(Integer o1, Integer o2) {
	        		// TODO Auto-generated method stub
	        		return o2 - o1;
	        	}
        	
		});
        
        int i = 0;
        int lastMax = -1;
   
        while (i < nodes.length) {
        		int start = i;
        		int end = i;
        		while (end < nodes.length - 1 && nodes[end].pos == nodes[end + 1].pos) {
        			end++;
        		}
        		
        		for (int j = start; j <= end; j++) {
        			SkyNode node = nodes[j];
            		if (node.isLeft) {
            			heap.add(node.height);
            		} else {
            			heap.remove(node.height);
            		}
        		}
        		
        		int newMax = 0;
        		if (!heap.isEmpty()) {
        			newMax = heap.peek();
        		}
        		
        		if (newMax != lastMax) {
        			int[] one = new int[2];
        			one[0] = nodes[start].pos;
        			one[1] = newMax;
        			lastMax = newMax;
        			result.add(one);
        		}
        		
        		i = end + 1;
        		
        }
        
        return result;
        
    }
	
	private static class SkyNode {
		public final int pos;
		public final boolean isLeft;
		public final int height;
		
		public SkyNode(int pos, boolean isLeft, int height) {
			this.pos = pos;
			this.isLeft = isLeft;
			this.height = height;
		}
	}
	
	// https://leetcode.com/problems/falling-squares/description/

	// https://leetcode.com/problems/generate-parentheses/description/
	public List<String> generateParenthesis(int n) {
        return generateParenthesisRecurse(n, new HashMap<Integer, List<String>>());
    }
	
	private List<String> generateParenthesisRecurse(int n, Map<Integer, List<String>> cache) {
		List<String> result = new LinkedList<>();
		if (n == 0) {
			result.add("");
			return result;
		}
		
		if (n == 1) {
			result.add("()");
			return result;
		}
		
		if (cache.containsKey(n)) {
			return cache.get(n);
		}
		
		for (int i = 0; i < n; ++i) {
			List<String> middle = generateParenthesisRecurse(i, cache);
			List<String> right = generateParenthesisRecurse(n - 1 - i, cache);
			
			for (String mid : middle) {
				for (String rig : right) {
					result.add("(" + mid + ")" + rig);
				}
			}
		}
		
		cache.put(n, result);
		return result;
	}
	
	public List<String> generateParenthesisII(int n) {
        List<String>[] dp = new List[n + 1];
        dp[0] = new LinkedList<>(Arrays.asList(""));
        
        for (int i = 1; i <= n; i++) {
        		List<String> result = new LinkedList<>();
        		for (int j = 0; j < i; j++) {
        			List<String> middle = dp[j];
        			List<String> right = dp[i - 1 - j];
        			
        			for (String m : middle) {
        				for (String r : right) {
        					result.add("(" + m + ")" + r);
        				}
        			}
        		}
        		dp[i] = result;
        }
        
        return dp[n];
    }
	
	// https://leetcode.com/problems/regular-expression-matching/description/
	public boolean isMatchII(String s, String p) {
        if (!validPatern(p)) {
        		return false;
        }
        
        boolean[][] dp = new boolean[p.length() + 1][s.length() + 1];
        dp[0][0] = true;
        
        for (int i = 1; i <= p.length(); ++i) {
        		dp[i][0] = (i & 1) == 0 && p.charAt(i - 1) == '*';
        		if ((i & 1) == 0 && !dp[i][0]) {
        			break;
        		}
        }
        
        for (int i = 1; i <= p.length(); ++i) {
        		for (int j = 1; j <= s.length(); ++j) {
        			char pc = p.charAt(i - 1);
        			char sc = s.charAt(j - 1);
        			dp[i][j] = dp[i - 1][j - 1] && isEqualII(sc, pc);
        			if (!dp[i][j]) {
        				 if (pc == '*') {
        					 // 匹配0个或者1个
        					 dp[i][j] = dp[i - 2][j] || dp[i-1][j];
        					 // 匹配多个
        					 if (!dp[i][j]) {
        						 dp[i][j] = dp[i][j-1] && isEqualII(sc, p.charAt(i-2));
        					 }
        				 } 
        			}
        			
     
        		}
        }
        
        return dp[p.length()][s.length()];
    }
	
	private boolean isEqualII(char s, char p) {
		return p == '.' || p == s;
	}
	
	private boolean validPatern(String pattern) {
		if (pattern == null || pattern.length() <= 0) {
			return true;
		}
		
		for (int i = 0; i < pattern.length(); ++i) {
			if (pattern.charAt(i) == '*') {
				if (i == 0)
					return false;
				else if (pattern.charAt(i - 1) == '*')
					return false;
			}
		}
		
		return true;
	}
	
	// https://leetcode.com/problems/combination-sum/description/
	public List<List<Integer>> combinationSum(int[] candidates, int target) {
        // candidates是从小到大排序的
		List<List<Integer>> result = new LinkedList<>();
		List<Integer> arrays = new LinkedList<>();
		
		Arrays.sort(candidates);
		
		combinationSumDfs(candidates, 0, target, 0, arrays, result);
		
		return result;
    }
	
	private void combinationSumDfs(int[] candidates, int start, int target, int sum, List<Integer> arrays, List<List<Integer>> result) {
		if (sum > target) {
			return;
		}
		
		if (sum == target) {
			result.add(new LinkedList<>(arrays));
			return;
		}
		
		if (start >= candidates.length) {
			return;
		}
		
		int maxIndexCnt = (target - sum) / candidates[start];
		if (maxIndexCnt <= 0) {
			return;
		}
		
		for (int i = 0; i <= maxIndexCnt; ++i) {
			sum += i * candidates[start];
			for (int j = 1; j <= i; ++j)
				arrays.add(candidates[start]);
			combinationSumDfs(candidates, start + 1, target, sum, arrays, result);
			
			sum -= i * candidates[start];
			
			for (int j = 1; j <= i; ++j)
				arrays.remove(arrays.size() - 1);
		}
	}
	
	public List<List<Integer>> combinationSum2(int[] candidates, int target) {
		List<List<Integer>> result = new LinkedList<>();
		List<Integer> arrays = new LinkedList<>();
		
		Arrays.sort(candidates);
		
		combinationSumDfs2(candidates, 0, target, 0, arrays, result);
		
		return result;
    }
	
	private void combinationSumDfs2(int[] candidates, int start, int target, int sum, List<Integer> arrays, List<List<Integer>> result) {
		if (sum > target) {
			return;
		}
		
		if (sum == target) {
			result.add(new LinkedList<>(arrays));
			return;
		}
		
		if (start >= candidates.length) {
			return;
		}
		
		if (target - sum < candidates[start]) {
			return;
		}
		
		for (int i = start; i <= candidates.length - 1; ++i) {
			
			arrays.add(candidates[i]);
			combinationSumDfs2(candidates, i + 1, target, sum + candidates[i], arrays, result);
			arrays.remove(arrays.size() - 1);
			
			while (i <= candidates.length - 2 && candidates[i] == candidates[i+1]) {
				i++;
			}
		}
	}
	
	public List<List<Integer>> combine(int n, int k) {
		List<List<Integer>> result = new LinkedList<>();
		List<Integer> arrays = new LinkedList<>();
		
		combine(n, 0, k, 0, arrays, result);
		
		return result;
    }
	
	private void combine(int n, int start, int k, int cnt, List<Integer> arrays, List<List<Integer>> result) {
		if (cnt >= k) {
			result.add(new LinkedList<>(arrays));
			return;
		}
		
		for (int i = start; i <= n; i++) {
			arrays.add(i);
			combine(n, i + 1, k, cnt + 1, arrays, result);
			arrays.remove(arrays.size() - 1);
		}
	}
	
	// https://leetcode.com/problems/jump-game/description/
	public boolean canJump(int[] nums) {
        if (nums == null || nums.length <= 0) {
        		return false;
        }
        
        int maxIndex = 0;
        for (int i = 0; i < nums.length; ++i) {
        		if (maxIndex < i) {
        			return false;
        		}
        		
        		maxIndex = Math.max(maxIndex, nums[i] + i);
        		if (maxIndex >= nums.length - 1) {
        			return true;
        		}
        }
        
        return false;
    }
	
	// https://leetcode.com/problems/course-schedule/description/
	public boolean canFinish(int numCourses, int[][] prerequisites) {
        GraphDirected graphDirected = new GraphDirected(numCourses);
        
        for (int i = 0; i < prerequisites.length; ++i) {
        		graphDirected.addEdge(prerequisites[i][1], prerequisites[i][0]);
        }
        
        return graphDirected.hasCircle();
    }
	
	public int[] findOrder(int numCourses, int[][] prerequisites) {
		GraphDirected graphDirected = new GraphDirected(numCourses);
        
        for (int i = 0; i < prerequisites.length; ++i) {
        		graphDirected.addEdge(prerequisites[i][1], prerequisites[i][0]);
        }
        
        return graphDirected.findOrder();
    }
	
	private class GraphDirected {
		private final int mCapacity;
		private final List<Integer>[] mAdjacents;
		private final int[] mInDegrees;
		private final Set<Integer> mZeroInDegrees;
		
		public GraphDirected(int capacity) {
			mCapacity = capacity;
			mAdjacents = new List[mCapacity];
			mInDegrees = new int[mCapacity];
			mZeroInDegrees = new HashSet<>();
			for (int i = 0; i < capacity; ++i) {
				mZeroInDegrees.add(i);
			}
		}
		
		public void addEdge(int from, int to) {
			if (mAdjacents[from] == null) {
				mAdjacents[from] = new LinkedList<Integer>();
			}
			
			mAdjacents[from].add(to);
			
			mInDegrees[to]++;
			
			mZeroInDegrees.remove(to);
		}
		
		public boolean hasCircle() {
			Queue<Integer> queue = new LinkedList<>();
			
			queue.addAll(mZeroInDegrees);
			
			while (!queue.isEmpty()) {
				int top = queue.poll();
				
				// 减少所有的入度
				if (mAdjacents[top] != null) {
					for (int adjacent : mAdjacents[top]) {
						mInDegrees[adjacent]--;
						if (mInDegrees[adjacent] == 0) {
							mZeroInDegrees.add(adjacent);
							queue.offer(adjacent);
						}
					}
				}
			}
			
			return mZeroInDegrees.size() >= mCapacity;
		}
		
		public int[] findOrder() {
			Queue<Integer> queue = new LinkedList<>();
			
			int[] result = new int[mCapacity];
			
			queue.addAll(mZeroInDegrees);
			
			int idx = 0;
			while (!queue.isEmpty()) {
				int top = queue.poll();
				
				result[idx++] = top;
				
				// 减少所有的入度
				if (mAdjacents[top] != null) {
					for (int adjacent : mAdjacents[top]) {
						mInDegrees[adjacent]--;
						if (mInDegrees[adjacent] == 0) {
							mZeroInDegrees.add(adjacent);
							queue.offer(adjacent);
						}
					}
				}
			}
			
			if (idx == mCapacity) {
				return result;
			} else {
				return new int[0];
			}
	    }
	}
	
	
	
	// https://leetcode.com/problems/minimum-height-trees/description/
	public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        GraphUndirected graphUndirected = new GraphUndirected(n);
        
        for (int i = 0; i < edges.length; i++) {
        		graphUndirected.addEdge(edges[i][0], edges[i][1]);
        }
        
        return graphUndirected.getHeight();
    }
	
	private class GraphUndirected {
		private final int mCapacity;
		private final Set<Integer>[] mAdjacents;
		private final int[] mAdjacentsCnt;
		
		public GraphUndirected(int capacity) {
			mCapacity = capacity;
			mAdjacents = new Set[mCapacity];
			mAdjacentsCnt = new int[mCapacity];
		}
		
		public void addEdge(int from, int to) {
			if (mAdjacents[from] == null)
				mAdjacents[from] = new HashSet<>();
			mAdjacents[from].add(to);
			
			if (mAdjacents[to] == null)
				mAdjacents[to] = new HashSet<>();
			mAdjacents[to].add(from);
			
			mAdjacentsCnt[from]++;
			mAdjacentsCnt[to]++;
		}
		
		private List<Integer> getHeight() {
			List<Integer> result = new LinkedList<>();
			Queue<Integer> queue = new LinkedList<>();
			
			for (int i = 0; i < mCapacity; ++i) {
				if (mAdjacentsCnt[i] <= 1) {
					queue.add(i);
				}
			}
			
			int cur = queue.size();
			int next = 0;
			
			int left = mCapacity - queue.size();
			
			while (left >= 1) {
				int leaf = queue.poll();
				cur--;
				
				
				for (int adjacent : mAdjacents[leaf]) {
					mAdjacentsCnt[adjacent]--;
					mAdjacents[adjacent].remove(leaf);
					if (mAdjacentsCnt[adjacent] == 1) {
						queue.add(adjacent);
						next++;
					}
				}
				
				if (cur == 0) {
					cur = next;
					left -= next;
					next = 0;
				}
			}
			
			result.addAll(queue);
			
			return result;
		}
	}
	
	// https://leetcode.com/problems/find-the-duplicate-number/description/
//	public int findDuplicate(int[] nums) {
//        return 0;
//    }
	
	// https://leetcode.com/problems/linked-list-cycle/description/
	
	// Definition for singly-linked list.
	public static class ListNode {
	     int val;
	     ListNode next;
	     ListNode(int x) {
	         val = x;
	         next = null;
	     }
	}
	
	public boolean hasCycle(ListNode head) {
        if (head == null)
        		return false;
        
        ListNode slow = head;
        ListNode fast = head;
        
        while (slow != null && slow.next != null 
        		&& fast != null && fast.next != null && fast.next.next != null) {
        		slow = slow.next;
        		fast = fast.next.next;
        		
        		if (slow == fast) {
        			return true;
        		}
        }
        
        return false;
    }
	
	// 寻找起点
	public ListNode detectCycle(ListNode head) {
		if (head == null)
    			return null;
    
	    ListNode slow = head;
	    ListNode fast = head;
	    
	    boolean hasCycle = false;
	    while (slow != null && slow.next != null 
	    		&& fast != null && fast.next != null && fast.next.next != null) {
	    		slow = slow.next;
	    		fast = fast.next.next;
	    		
	    		if (slow == fast) {
	    			hasCycle = true;
	    			break;
	    		}
	    }
	    
	    if (!hasCycle)
	    		return null;
	    
	    slow = head;
	    
	    while (slow != fast) {
	    		fast = fast.next;
	    		slow = slow.next;
	    }
	    
	    return slow;
    }
	
	// https://leetcode.com/problems/first-missing-positive/description/
	public int firstMissingPositive(int[] nums) {
        if (nums == null || nums.length <= 0) {
        		return 1;
        }
        
        for (int i = 0; i < nums.length; ++i) {
        		while (nums[i] > 0 && nums[i] != i + 1 && nums[i] <= nums.length
        				&& nums[nums[i] - 1] != nums[i]) {
        			int index = nums[i] - 1;
        			// 交换index和i相对应的值
        			int temp = nums[index];
        			nums[index] = nums[i];
        			nums[i] = temp;
        		}
        }
        
        for (int i = 0; i < nums.length; ++i) {
        		if (nums[i] != i + 1) {
        			return i + 1;
        		}
        }
        
        return nums.length + 1;
    }
	
	// https://leetcode.com/problems/merge-intervals/description/
	// Definition for an interval.
	public static class Interval {
	    int start;
	    int end;
	    Interval() { start = 0; end = 0; }
	    Interval(int s, int e) { start = s; end = e; }
	    
	    @Override
	    public String toString() {
	    	// TODO Auto-generated method stub
	    		return String.format("[%d, %d]", start, end);
	    }
	}
	
	public List<Interval> merge(List<Interval> intervals) {
        List<Interval> result = new LinkedList<>();
        if (intervals == null || intervals.isEmpty()) {
        		return result;
        }
        
        Collections.sort(intervals, new Comparator<Interval>() {
        		@Override
        		public int compare(Interval o1, Interval o2) {
        			// TODO Auto-generated method stub
        			return o1.start - o2.start;
        		}
		});
        
        int i = 0;
        while (i < intervals.size()) {
        		Interval first = intervals.get(i);
        		int firstEnd = first.end;
        		while (i + 1 < intervals.size()) {
        			Interval next = intervals.get(i + 1);
        			if (next.start <= firstEnd) {
        				firstEnd = Math.max(firstEnd, next.end);
        				i++;
        			} else {
        				break;
        			}
        		}
        		i++;
        		result.add(new Interval(first.start, firstEnd));
        }
        
        return result;
    }
	
	public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
        List<Interval> result = new LinkedList<>();
        
        int i = 0;
        boolean added = false;
        while (i < intervals.size()) {
        		// 最简单的，都大于
        		Interval cur = intervals.get(i);
        		if (added) {
        			result.add(cur);
        			i++;
        			continue;
        		}
        		
        		if (newInterval.start > cur.end) {
        			result.add(cur);
        			i++;
        		} else if (newInterval.end < cur.start){
        			result.add(newInterval);
        			result.add(cur);
        			added = true;
        			i++;
        		} 
        		else {
        			int start = Math.min(newInterval.start, cur.start);
        			int end;
        			// 找到他都包含什么
        			int j = i;
        			while (j < intervals.size() && intervals.get(j).end < newInterval.end) {
        				j++;
        			}
        			
        			if (j == i) {
        				end = Math.max(cur.end, newInterval.end);
        				i++;
        			} else if (j == intervals.size()) {
        				end = newInterval.end;
        				i = j;
        			} else {
        				if (intervals.get(j).start > newInterval.end) {
        					end = Math.max(intervals.get(j-1).end, newInterval.end);
        					i = j;
        				} else {
        					end = Math.max(intervals.get(j).end, newInterval.end);
        					i = j + 1;
        				}
        			}
        			
        			result.add(new Interval(start, end));
        			added = true;
        		}
        }
        
        if (intervals.size() <= 0 || newInterval.start > intervals.get(intervals.size() - 1).end) {
        		result.add(newInterval);
        }
        
        return result;
    }

	// https://leetcode.com/problems/teemo-attacking/description/
	public int findPoisonedDuration(int[] timeSeries, int duration) {
        int sum = 0;
        if (timeSeries == null || timeSeries.length <= 0 || duration <= 0) {
        		return sum;
        }
        
        int i = 0;
        while (i < timeSeries.length) {
        		int start = timeSeries[i];
        		int end = timeSeries[i] + duration - 1;
        		
        		while (i + 1 < timeSeries.length) {
        			if (timeSeries[i + 1] <= end) {
        				end = Math.max(end, timeSeries[i+1] + duration - 1);
        				i++;
        			} else {
        				break;
        			}
        		}
        		
        		i++;
        		sum += end - start + 1;
        }
        
        return sum;
    }
	
	// https://leetcode.com/problems/dota2-senate/discuss/
	public String predictPartyVictory(String senate) {
		char[] arr = senate.toCharArray();
		int r = 0;
		int d = 0;
		for (int i = 0; i < arr.length; ++i) {
			if (arr[i] == 'R') r++;
			else d++;
		}
		
		Stack<Character> stack = new Stack<>();
		int start = 0;
		while (r > 0 && d > 0) {
			char c = arr[start];
			if (c != ' ') {
				if (stack.isEmpty() || stack.peek() == c) {
					stack.push(c);
				} else {
					stack.pop();
					if (c == 'R') r--;
					else d--;
					arr[start] = ' ';
				}
			}
			start = (start + 1) % arr.length;
		}
		if (r > 0)
			return "Radiant";
		else
			return "Dire";
	}
	
	public String predictPartyVictoryII(String senate) {
		char[] arr = senate.toCharArray();
		int r = 0;
		int d = 0;
		for (int i = 0; i < arr.length; ++i) {
			if (arr[i] == 'R') r++;
			else d++;
		}
		
		int sr = 0;
		int sd = 0;
		int start = 0;
		while (r > 0 && d > 0) {
			char c = arr[start];
			if (c != ' ') {
				if (sr == 0 && sd == 0) {
					if (c == 'R') 
						sr++;
					else
						sd++;
				} else if (sr != 0) {
					if (c == 'R') 
						sr++;
					else {
						sr--;
						d--;
						arr[start] = ' ';
					}
				} else {
					if (c == 'D') 
						sd++;
					else {
						sd--;
						r--;
						arr[start] = ' ';
					}
				}
			}
			start = (start + 1) % arr.length;
		}
		if (r > 0)
			return "Radiant";
		else
			return "Dire";
	}
	
	// https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/
	public int maxProfit(int[] prices) {
        if (prices == null || prices.length <= 1) {
        		return 0;
        }
        
        int result = 0;
        int min = prices[0];
        for (int i = 1; i < prices.length; ++i) {
        		min = min < prices[i] ? min : prices[i];
        		result = result > prices[i] - min ? result : prices[i] - min;
        }
        
        return result;
    }
	
	public int maxProfitII(int[] prices) {
        if (prices == null || prices.length <= 1) {
        		return 0;
        }
        
        int result = 0;
        int min = prices[0];
        int max = prices[0];
        
        for (int i = 1; i < prices.length; ++i) {
        		if (prices[i] >= prices[i-1]) {
        			// 往上走，更新最大值
        			max = prices[i];
        		} else {
        			// 往下走，更新最小值
        			if (max > min) {
        				result += max - min;
        			}
        			min = prices[i];
        			max = min;
        		}
        }
        if (max > min)
        		result += max - min;
        
        return result;
    }
	
	// Design an algorithm to find the maximum profit. You may complete at most two transactions.
	public int maxProfitIII(int[] prices) {
        if (prices == null || prices.length <= 1) {
        		return 0;
        }
        
        int[] left = new int[prices.length];
        left[0] = 0;
        int leftResult = 0;
        int min = prices[0];
        for (int i = 1; i < prices.length; ++i) {
        		leftResult = Math.max(leftResult, prices[i] - min);
        		min = Math.min(prices[i], min);
        		left[i] = leftResult;
        }
        
        int max = prices[prices.length - 1];
        int rightResult = 0;
        
        int result = left[prices.length - 1];
        for (int i = prices.length - 2; i >= 0; --i) {
        		rightResult = Math.max(rightResult, max - prices[i]);
        		result = Math.max(result, rightResult + left[i]);
        		max = Math.max(max, prices[i]);
        }
        
        return result;
    }
	
	// 上面那种算法还是慢啊，
	public int maxProfitIII2(int[] prices) {
		if (prices == null || prices.length <= 1) {
	    		return 0;
	    }
	    
	    int min = prices[0];
	    int max = prices[0];
	    int[] twoMax = {0, 0};
	    for (int i = 1; i < prices.length; ++i) {
	    		if (prices[i] >= prices[i-1]) {
	    			// 往上走，更新最大值
	    			max = prices[i];
	    		} else {
	    			// 往下走，更新最小值
	    			if (max > min) {
	    				updateTwoMax(twoMax, max - min);
	    			}
	    			min = prices[i];
	    			max = min;
	    		}
	    }
	    if (max > min) {
	    		updateTwoMax(twoMax, max - min);
	    }
	    
	    return twoMax[0] + twoMax[1];
    }
	
	private void updateTwoMax(int[] maxs, int update) {
		if (update > maxs[1]) {
			maxs[0] = maxs[1];
			maxs[1] = update;
		} else if (update > maxs[0] && update < maxs[1]) {
			maxs[0] = update;
		}
	}
	
	// Design an algorithm to find the maximum profit. You may complete at most k transactions
	public int maxProfitIV(int k, int[] prices) {
        if (k <= 0 || prices == null || prices.length <= 1) {
        		return 0;
        }
        
        return maxProfitIVDp(k, prices);
    }
	
	// timeout
	private int maxProfitIVDfs(int left, int[] prices, int end) {
		if (left <= 0) {
			return 0;
		}
		
		if (end <= 0) {
			return 0;
		}
		
		int max = prices[end];
		int oneResult = 0;
		int result = 0;
		for (int i = end - 1; i >= 0; --i) {
			oneResult = Math.max(oneResult, max - prices[i]);
			max = Math.max(max, prices[i]);
			result = Math.max(result, maxProfitIVDfs(left - 1, prices, i) + oneResult);
			
		}
		
		return result;
	}
	
	private int maxProfitIVDp(int k, int[] prices) {
		if (k <= 0 || prices == null || prices.length <= 1) {
			return 0;
		}
		
		if (k > prices.length) {
			k = prices.length;
		}
		
		int result = 0;
		int[][] dp = new int[k + 1][prices.length];
		for (int i = 1; i <= k; ++i) {
			int tempMax = dp[i-1][0] - prices[0];
			for (int j = 1; j < prices.length; ++j) {
				dp[i][j] = Math.max(dp[i][j-1], prices[j] + tempMax);
				tempMax = Math.max(tempMax, dp[i-1][j] - prices[j]);
			}
		}

		return dp[k][prices.length - 1];
	}
	
	// https://leetcode.com/problems/maximum-sum-of-3-non-overlapping-subarrays/description/
	public int[] maxSumOfThreeSubarrays(int[] nums, int k) {
        if (nums == null || nums.length < 3 * k) {
        		return new int[0];
        }
        int[] result = new int[3];
        int[] sums = new int[nums.length];
        sums[0] = nums[0];
        for (int i = 1; i < nums.length; ++i) {
        		sums[i] = sums[i-1] + nums[i];
        }
        int[] traverse = new int[3];
        SumOfThreeSubRes res = new SumOfThreeSubRes();
        
        maxSumOfThreeSubarrays(nums, sums, 0, 3, 0, k, traverse, res);
        
        return res.mResult;
    }
	
	private class SumOfThreeSubRes {
		int[] mResult = new int[3];
		int mSum = Integer.MIN_VALUE;
		
		public void update(int sum, int[] result) {
			if (sum > mSum) {
				mResult[0] = result[0];
				mResult[1] = result[1];
				mResult[2] = result[2];
				mSum = sum;
			} else if (sum == mSum){
				boolean big = isBig(result, 0);
				if (big) {
					mResult[0] = result[0];
					mResult[1] = result[1];
					mResult[2] = result[2];
				}
			}
		}
		
		private boolean isBig(int[] result, int start) {
			if (start >= 3) {
				return false;
			} 
			if (mResult[start] > result[start])
				return true;
			if (mResult[start] < result[start])
				return false;
			
			return isBig(result, start + 1);
		}
	}
	
	// timeout
	private void maxSumOfThreeSubarrays(int[] nums, int[] sums, int start, int left, int sum, int k, int[] traverse, SumOfThreeSubRes res) {
		if (left <= 0) {
			res.update(sum, traverse);
			return;
		}
		
		// 剩下的已经不够分了
		if (start >= nums.length || nums.length - start < left * k) {
			return;
		}
		
		if (nums.length - start - 1 >= left * k) {
			maxSumOfThreeSubarrays(nums, sums, start + 1, left, sum, k, traverse, res);
		}
		
		int intervalSum = sums[start + k - 1] - (start == 0 ? 0 : sums[start - 1]);
		traverse[3 - left] = start;
		maxSumOfThreeSubarrays(nums, sums, start + k, left - 1, sum + intervalSum, k, traverse, res);
	}
	
	// 还不是最优解啊
	public int[] maxSumOfThreeSubarraysDp(int[] nums, int k) {
		if (nums == null || nums.length < 3 * k) {
			return new int[0];
		}
		
		int[] sums = new int[nums.length];
		sums[0] = nums[0];
		for (int i = 1; i < nums.length; ++i) {
			sums[i] = sums[i-1] + nums[i];
		}
	
		int[][] dp = new int[4][nums.length + 1];
		
		int thirdFirstMaxIdx = 0;
		int thirdMax = 0;
		for (int i = 1; i <= 3; ++i) {
			for (int j = k * i; j <= nums.length; ++j) {
				dp[i][j] = dp[i][j-1];
				if (j - k >= 0) {
					int interval = sums[j-1] - ((j - k) == 0 ? 0 : sums[j-k-1]);
					dp[i][j] = Math.max(dp[i][j], dp[i-1][j-k] + interval);
					
				}
				
				if (i == 3) {
					if (thirdMax < dp[i][j]) {
						thirdMax = dp[i][j];
						thirdFirstMaxIdx = j;
					}
				}
			}
		}
		
		int find = thirdMax - (sums[thirdFirstMaxIdx - 1] - sums[thirdFirstMaxIdx -1-k]); 
		int secondFirstMaxIdx = findFirstTarget(dp[2], find);
		
		
		find = dp[2][secondFirstMaxIdx] - (sums[secondFirstMaxIdx - 1] - sums[secondFirstMaxIdx - 1 - k]);
		int firstFirstMaxIdx = findFirstTarget(dp[1], find);
		
		int[] result = new int[3];
		result[0] = firstFirstMaxIdx - k;
		result[1] = secondFirstMaxIdx - k;
		result[2] = thirdFirstMaxIdx - k;
		
		return result;
	}
	
	// 前提是，target肯定会出现
	private int findFirstTarget(int[] nums, int target) {
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
		
		return left;
	}
	
	// https://leetcode.com/problems/course-schedule-iii/description/
	public int scheduleCourse(int[][] courses) {
        // TODO
		return 0;
    }
	
	// https://leetcode.com/problems/gas-station/description/
	public int canCompleteCircuit(int[] gas, int[] cost) {
        int min = Integer.MAX_VALUE;
        int sum = 0;
        int idx = -1;
        for (int i = 0; i < gas.length; ++i) {
        		sum += gas[i] - cost[i];
        		if (sum < min) {
        			min = sum;
        			idx = i;
        		}
        }
        
        if (sum < 0 || idx == -1)
        		return -1;
        else
        		return (idx + 1) % gas.length;
    }
	
	// https://leetcode.com/problems/n-queens/description/
    public List<List<String>> solveNQueens(int n) {
        char[][] traverse = new char[n][n];
        for (int i = 0; i < n; ++i)
        		for (int j = 0; j < n; ++j)
        			traverse[i][j] = '.';
        List<List<String>> result = new ArrayList<>(n);
        	solveNQueensDfs(n, 0, traverse, result);
        	
        	return result;
    }
    
    // 在第row行，能不能放在第column列
    private boolean isValid(char[][] traverse, int row, int column) {
    		if (row == 0)
    			return true;
    		
    		// 判断在同一列有没有
    		for (int i = 0; i < row; ++i) 
    			if (traverse[i][column] == 'Q')
    				return false;
    		
    		// 判断在同一个斜线上有没有
    		for (int i = row - 1, j = column - 1; i >= 0 && j >= 0; --i, --j) {
    			if (traverse[i][j] == 'Q')
    				return false;
    		}
    		
    		for (int i = row - 1, j = column + 1; i >= 0 && j <= traverse.length - 1; --i, ++j) 
    			if (traverse[i][j] == 'Q')
    				return false;
    		
    		return true;
    }
    
    private void solveNQueensDfs(int n, int start, char[][] traverse, List<List<String>> result) {
    		if (start >= n) {
    			List<String> one = new ArrayList<>(n);
    			for (int i = 0; i < n; ++i) {
    				one.add(String.valueOf(traverse[i], 0, n));
    			}
    			result.add(one);
    			return;
    		}
    		
    		for (int i = 0; i < n; ++i) {
    			if (isValid(traverse, start, i)) {
    				traverse[start][i] = 'Q';
    				solveNQueensDfs(n, start + 1, traverse, result);
    				traverse[start][i] = '.';
    			}
    		}
    }
    
    public int totalNQueens(int n) {
    		if (n == 0)
    			return 0;
    		
    		int[] traverse = new int[n];
    		int[] result = {0};
    		totalNQueens(n, 0, traverse, result);
    		
    		return result[0];
    }
    
    private boolean isValid(int n, int row, int column, int[] traverse) {
    		if (row == 0) return true;
    		for (int i = 0; i < row; ++i)
    			if (traverse[i] - 1 == column)
    				return false;
    		
    		for (int i = 0; i < row; ++i) {
    			if (i - row == traverse[i] - 1 - column || 
    					i - row == column - traverse[i] + 1) {
    				return false;
    			}
    		}
    		
    		return true;
    }
    
    private void totalNQueens(int n, int row, int[] traverse, int[] result) {
    		if (row >= n) {
    			result[0]++;
    			return;
    		}
    		
    		for (int i= 0; i < n; ++i) {
    			if (isValid(n, row, i, traverse)) {
    				traverse[row] = i + 1;
    				totalNQueens(n, row + 1, traverse, result);
    				traverse[row] = 0;
    			}
    		}
    }
    
    // https://leetcode.com/problems/reconstruct-original-digits-from-english/description/
    public String originalDigits(String s) {
    		
    		int[] count = new int[10];
    		
    		for (int i = 0; i < s.length(); ++i) {
    			char c = s.charAt(i);
    			switch (c) {
				case 'z':
					count[0]++;
					break;
				case 'w':
					count[2]++;
					break;
				case 'u':
					count[4]++;
					break;
				case 'x':
					count[6]++;
					break;
				case 'g':
					count[8]++;
					break;
				case 'o':
					count[1]++;
					break;
				case 't':
					count[3]++;
					break;
				case 'f':
					count[5]++;
					break;
				case 's':
					count[7]++;
					break;
				case 'i':
					count[9]++;
					break;
				default:
					break;
				}
    		}
    		
    		count[1] = count[1] - count[0] - count[2] - count[4];
    		count[3] = count[3] - count[2] - count[8];
    		count[5] = count[5] - count[4];
    		count[7] = count[7] - count[6];
    		count[9] = count[9] - count[5] - count[6] - count[8];
    		
    		StringBuilder builder = new StringBuilder();
    		for (int i = 0; i <= 9; i++) {
    			for (int j = 1; j <= count[i]; ++j)
    				builder.append(i);
    		}
    		
    		return builder.toString();
    }
    
    // https://leetcode.com/problems/power-of-four/description/
    public boolean isPowerOfFour(int num) {
    		if (num == 0) return false;
    		if (num == 1) return true;
    			
    		return num > 0 && (num & (num - 1)) == 0 && (num - 1) % 3 == 0;
    }

    // https://leetcode.com/problems/sort-colors/description/
    public void sortColors(int[] nums) {
        int[] hash = new int[3];
        for (int i = 0; i < nums.length; ++i) {
        		hash[nums[i]]++;
        }
        int i = 0;
        while (i < hash[0]) nums[i++] = 0;
        while (i - hash[0] < hash[1]) nums[i++] = 1;
        while (i < nums.length)  nums[i++] = 2;
    }
    
    // https://leetcode.com/problems/sort-list/description/
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null)
        		return head;
        
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null && fast.next.next != null) {
        		slow = slow.next;
        		fast = fast.next.next;
        }
        
        ListNode right = slow.next;
        slow.next = null;
        ListNode leftHead = sortList(head);
        ListNode rightHead = sortList(right);
        
        ListNode newHead = null;
        ListNode preNode = newHead;
        while (leftHead != null && rightHead != null) {
        		if (leftHead.val <= rightHead.val) {
        			if (newHead == null) {
        				newHead = leftHead;
        				preNode = newHead;
        			} else {
        				preNode.next = leftHead;
        				preNode = leftHead;
        			}
        			leftHead = leftHead.next;
        		} else {
        			if (newHead == null) {
        				newHead = rightHead;
        				preNode = newHead;
        			} else {
        				preNode.next = rightHead;
        				preNode = rightHead;
        			}
        			rightHead = rightHead.next;
        		}
        }
        if (leftHead != null) {
        		preNode.next = leftHead;
        } else if (rightHead != null) {
        		preNode.next = rightHead;
        }
        
        return newHead;
    }
    
    // https://leetcode.com/problems/insertion-sort-list/description/
    public ListNode insertionSortList(ListNode head) {
        if (head == null || head.next == null) {
        		return head;
        }
        
        ListNode end = head.next;
        ListNode endPre = head;
        while (end != null) {
        		ListNode left = head;
        		ListNode pre = null;
        		while (left.val <= end.val && left != end) {
        			pre = left;
        			left = left.next;
        		}
        		
        		// 需要把end取出来
        		if (left != end) {
        			// 直接插入到最前面
        			if (pre == null) {
        				// 移出end
        				endPre.next = end.next;
        				// 插入到最前面
        				end.next = left;
        				// 重新
        				head = end;
        				// 重新给end和endpre复制
        				end = endPre.next;
        			} else {
        				// 需要插入到中间
        				endPre.next = end.next;
        				pre.next = end;
        				end.next = left;
        				
        				end = endPre.next;
        			}
        		} else {
        			endPre = end;
        			end = end.next;
        		}
        }
        
        return head;
    }
    
    public ListNode arrayToListNodes(int[] nums, int start) {
    		if (start == nums.length)
    			return null;
    		
    		ListNode head = new ListNode(nums[start]);
    		head.next = arrayToListNodes(nums, start + 1);
    		
    		return head;
    }
    
    public ListNode mergeKLists(ListNode[] lists) {
    		if (lists == null || lists.length == 0)
    			return null;
    		if (lists.length == 1) 
    			return lists[0];
    		
    		PriorityQueue<ListNode> minHeap = new PriorityQueue<>(new Comparator<ListNode>() {
    			@Override
    			public int compare(ListNode o1, ListNode o2) {
    				// TODO Auto-generated method stub
    				return o1.val - o2.val;
    			}
    		});
    		
    		ListNode head = null;
    		ListNode tail = null;
    		for (ListNode node : lists) {
    			if (node != null)
    				minHeap.add(node);
    		}
    		
    		while (!minHeap.isEmpty()) {
    			ListNode min = minHeap.poll();
    			if (head == null) {
    				head = tail = min;
    			} else {
    				tail.next = min; tail = min;
    			}
    			if (min.next != null)
    				minHeap.add(min.next);
    		}
    		
    		if (tail != null)
    			tail.next = null;
    		
    		return head;
    }
    
    // https://leetcode.com/problems/odd-even-linked-list/description/
    public ListNode oddEvenList(ListNode head) {
        if (head == null || head.next == null || head.next.next == null) {
        		return head;
        }
        
        ListNode oddEnd = head;
        ListNode evenEnd = head.next;
        ListNode evenStart = head.next;
        
        while (evenEnd != null && evenEnd.next != null) {
        		ListNode nextOdd = evenEnd.next;
        		
        		evenEnd.next = nextOdd.next;
        		evenEnd = nextOdd.next;
        		
        		oddEnd.next = nextOdd;
        		nextOdd.next = evenStart;
        		oddEnd = nextOdd;
        }
        
        return head;
    }
    
    // https://leetcode.com/problems/sliding-window-median/description/
    public double[] medianSlidingWindow(int[] nums, int k) {
        double[] result = new double[nums.length - k + 1];
        if (k == 1) {
        		for (int i = 0; i < nums.length; ++i)
        			result[i] = nums[i];
        		return result;
        }
        
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(new Comparator<Integer>() {
        		@Override
        		public int compare(Integer o1, Integer o2) {
        			// TODO Auto-generated method stub
        			if (o2 > o1)
        				return 1;
        			else if (o2 == o1)
        				return 0;
        			else
        				return -1;
        		}
        });
        
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        
        addWindow(nums, 0, k, maxHeap, minHeap);
        result[0] = getMedianWindow(k, maxHeap, minHeap);
        
        for (int i = 1; i <= nums.length - k; ++i) {
        		updateWindow(maxHeap, minHeap, nums[i-1], nums[i + k - 1], k);
        		result[i] = getMedianWindow(k, maxHeap, minHeap);
        }
        
        return result;
    }
    
    private double getMedianWindow( int k,
    		PriorityQueue<Integer> maxHeap, 
    		PriorityQueue<Integer> minHeap) {
    		if ((k & 1) == 0) {
    			int minTop = minHeap.peek();
    			int maxTop = maxHeap.peek();
    			if (minTop == maxTop)
    				return minTop;
    			else
    				return (long)minHeap.peek() + ((double)((long)maxHeap.peek() - (long)minHeap.peek())) / 2;
    		}
    		else {
    			return (long)maxHeap.peek();
    		}
    }
    
    private void addWindow(int[] nums, int start, int k, 
    		PriorityQueue<Integer> maxHeap, 
    		PriorityQueue<Integer> minHeap) {
    		
    		int cnt = 0;
    		for (int i = start; i <= start + k - 1; i++) {
    			cnt++;
    			if (cnt == 1) {
    				maxHeap.add(nums[i]);
    				continue;
    			}
    			int leftCnt = (cnt + 1) / 2;
    			if (nums[i] > maxHeap.peek()) minHeap.add(nums[i]);
    			else maxHeap.add(nums[i]);
    			
    			while (maxHeap.size() < leftCnt) maxHeap.add(minHeap.poll());
    			while (maxHeap.size() > leftCnt) minHeap.add(maxHeap.poll());
    		}
    		
    		System.out.println(maxHeap);
    		System.out.println(minHeap);
    }
    
    private void updateWindow(
    		PriorityQueue<Integer> maxHeap, 
    		PriorityQueue<Integer> minHeap,
    		int delete,
    		int add,
    		int k) {
    		
    		if (delete == add) return;
    		
    		if (delete > maxHeap.peek())
    			minHeap.remove(delete);
    		else
    			maxHeap.remove(delete);
    		
    		if (maxHeap.isEmpty()) {
    			if (add >= minHeap.peek())
    				minHeap.add(add);
    			else 
    				maxHeap.add(add);
    		} else {
    			if (add <= maxHeap.peek())
    				maxHeap.add(add);
    			else
    				minHeap.add(add);
    		}
    		
    		int leftCnt = (k + 1) / 2;
    		while (maxHeap.size() < leftCnt) maxHeap.add(minHeap.poll());
		while (maxHeap.size() > leftCnt) minHeap.add(maxHeap.poll());
		
		System.out.println(maxHeap);
		System.out.println(minHeap);
    }
    
    // https://leetcode.com/problems/add-two-numbers/description/
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    		ListNode tail1 = l1;
    		ListNode tail2 = l2;
    		ListNode head = null;
    		ListNode tail = head;
    		int add = 0;
    		while (tail1 != null || tail2 != null) {
    			int temp;
    			if (tail1 == null) {
    				temp = tail2.val + add;
    			} else if (tail2 == null) {
    				temp = tail1.val + add;
    			} else {
    				temp = tail1.val + tail2.val + add;
    			}
    			
    			if (temp >= 10) {
    				temp = temp - 10;
    				add = 1;
    			} else {
    				add = 0;
    			}
    			
    			if (head == null) {
    				head = new ListNode(temp);
    				tail = head;
    			} else {
    				tail.next = new ListNode(temp);
    				tail = tail.next;
    			}
    		}
    		
    		if (add >= 1) {
    			tail.next = new ListNode(1);
    		}
    		
    		return head;
    }
    
    // https://leetcode.com/problems/longest-palindromic-substring/description/
    public String longestPalindrome(String s) {
        if (s == null || s.length() <= 0)
        		return "";
        if (s.length() == 1)
        		return s;
        
        int max = 0;
        int maxIdx = -1;
        int[][] dp = new int[s.length()][s.length()];
        
        for (int i = 0; i < s.length(); ++i) {
        		dp[0][i] = s.charAt(s.length() - 1) == s.charAt(i) ? 1 : 0;
        		if (dp[0][i] > max) {
        			max = dp[0][i];
        			maxIdx = i;
        		}
        }
        
        for (int i = 0; i < s.length(); ++i) {
        		dp[i][0] = s.charAt(0) == s.charAt(s.length() - 1 - i) ? 1 : 0;
        		if (dp[i][0] > max) {
        			max = dp[i][0];
        			maxIdx = 0;
        		}
        }
        
        for (int i = 1; i < s.length(); ++i) {
        		for (int j = 1; j < s.length(); ++j) {
        			if (s.charAt(s.length() - 1 - i) == s.charAt(j)) {
        				dp[i][j] = dp[i-1][j-1] + 1;
        				if (dp[i][j] > max) {
        					max = dp[i][j];
        					maxIdx = j;
        				}
        			}
        		}
        }
        
        return s.substring(maxIdx - max + 1, maxIdx + 1);
    }
    
    // https://leetcode.com/problems/shortest-palindrome/description/
    public String shortestPalindrome(String s) {
        if (s == null || s.length() <= 1)
        		return s;
        
        int longest = getLongestPalindromeFromStart(s);
        if (longest == s.length())
        		return s;
        
        StringBuilder sb = new StringBuilder();
        for (int i = s.length() - 1; i >= longest; i--) {
        		sb.append(s.charAt(i));
        }
        
        sb.append(s);
        
        return sb.toString();
    }
    
    public int getLongestPalindromeFromStart(String src) {
    		if (src == null || src.length() <= 0)
    			return 0;
    		if (src.length() == 1)
    			return 1;
    		
    		final int length = src.length();
    		
    		int[] next = new int[2*length + 2];
    		next[0] = -1;
    		
    		int i = 0;
    		int k = -1;
    		while (i <= 2 * length) {
    			char ic;
    			if (i <= length - 1) ic = src.charAt(i);
    			else if (i == length) ic = '#';
    			else ic = src.charAt(2 * length - i);
    			
    			char kc = '#';
    			if (k == -1) {}
    			else if (k <= length - 1) kc = src.charAt(k);
    			else if (k == length) kc = '#';
    			else kc = src.charAt(2 * length - k);
    		
    			if (k == -1 || ic == kc) {
    				i++;
    				k++;
    				next[i] = k;
    			} else {
    				k = next[k];
    			}
    		}
    		
    		return next[2*length + 1];
    }
    
    // https://leetcode.com/problems/different-ways-to-add-parentheses/description/
    public List<Integer> diffWaysToCompute(String input) {
    		List<Integer> nums = new LinkedList<>();
    		List<Character> ops = new LinkedList<>();
    		
    		int temp =0;
    		for (int i = 0; i < input.length(); ++i) {
    			char c = input.charAt(i);
    			if (c >= '0' && c <= '9') {
    				temp = temp * 10 + (c - '0');
    			} else {
    				nums.add(temp);
    				temp = 0;
    				ops.add(c);
    			}
    		}
    		nums.add(temp);
    		
    		Integer[] numArray = new Integer[nums.size()];
    		numArray = nums.toArray(numArray);
    		
    		Character[] opArray = new Character[ops.size()];
    		opArray = ops.toArray(opArray);
    		
    		return diffWaysToCompute(numArray, opArray, 0, numArray.length - 1);
    		
    }
    
    private List<Integer> diffWaysToCompute(Integer[] nums, Character[] ops, int start, int end) {
    		List<Integer> result = new LinkedList<>();
    		if (start == end) {
    			result.add(nums[start]);
    			return result;
    		} else if (start > end) {
    			return result;
    		}
    		
    		for (int i = start; i <= end-1; ++i) {
    			List<Integer> pres = diffWaysToCompute(nums, ops, start, i);
    			List<Integer> posts = diffWaysToCompute(nums, ops, i + 1, end);
    			
    			for (int pre : pres) {
    				for (int post : posts) {
    					if (ops[i] == '+')
    						result.add( pre + post);
    					else if (ops[i] == '-')
    						result.add(pre - post);
    					else
    						result.add(pre * post);
    				}
    			}
    		}
    		
    		return result;
    }
    
    // https://leetcode.com/problems/longest-univalue-path/description/
    public int longestUnivaluePath(TreeNode root) {
        int[] result = {0};
        longestUnivaluePath(root, result);
        return result[0];
    }
    
    // 返回值，代表从root往左或者右的最大距离
    private int longestUnivaluePath(TreeNode root, int[] result) {
    		if (root == null) {
    			return 0;
    		}
    		
    		int leftMax = longestUnivaluePath(root.left, result);
    		int rightMax = longestUnivaluePath(root.right, result);
    		
    		int curMax = 0;
    		int curResult = 0;
    		if (root.left != null && root.left.val == root.val) {
    			curMax = Math.max(curMax, leftMax + 1);
    			curResult = leftMax + 1;
    		}
    		if (root.right != null && root.right.val == root.val) {
    			curMax = Math.max(curMax, rightMax + 1);
    			curResult += rightMax + 1;
    		}
    		
    		result[0] = Math.max(result[0], curResult);
    		
    		return curMax;
    }
    
    // https://leetcode.com/problems/path-sum/description/
    public boolean hasPathSum(TreeNode root, int sum) {
        return hasPathSum(root, sum, 0);
    }
    
    private boolean hasPathSum(TreeNode root, int sum, int temp) {
    		if (root == null) {
    			return false;
    		}
    		
    		if (root.left == null && root.right == null) {
    			return temp + root.val == sum;
    		}
    		
    		boolean result = false;
    		if (root.left != null) {
    			result = hasPathSum(root.left, sum, temp + root.val);
    		}
    		if (!result && root.right != null) {
    			result = hasPathSum(root.right, sum, temp + root.val);
    		}
    		
    		return result;
    }
    
    // https://leetcode.com/problems/path-sum-ii/description/
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> result = new LinkedList<>();
        List<Integer> one = new LinkedList<>();
        
        pathSum(root, sum, 0, one, result);
        
        return result;
    }
    
    private void pathSum(TreeNode root, int sum, int temp, List<Integer> traverse, List<List<Integer>> result) {
    		if (root == null) {
    			return;
    		}
    		if (root.left == null && root.right == null) {
    			if (temp + root.val == sum) {
    				List<Integer> one = new LinkedList<>();
    				one.addAll(traverse);
    				one.add(root.val);
    				result.add(one);
    			}
    			
    			return;
    		}
    		
    		if (root.left != null) {
    			traverse.add(root.val);
    			pathSum(root.left, sum, temp + root.val, traverse, result);
    			traverse.remove(traverse.size() - 1);
    		}
    		
    		if (root.right != null) {
    			traverse.add(root.val);
    			pathSum(root.right, sum, temp + root.val, traverse, result);
    			traverse.remove(traverse.size() - 1);
    		}
    		
    }
    
    // https://leetcode.com/problems/path-sum-iii/description/
    public int pathSumII(TreeNode root, int sum) {
        int[] result = {0};
        Map<Integer, Integer> cache = new HashMap<>();
        cache.put(0, 1);
        pathSumRecurse(root, sum, 0, cache, result);
        
        return result[0];
    }
    
    private void pathSumRecurse(TreeNode root, int sum, int temp, Map<Integer, Integer> cache, int[] result) {
    		if (root == null) {
    			return;
    		}
    		
    		int newSum = temp + root.val;
    		result[0] += cache.getOrDefault(newSum - sum, 0);
    		
    		cache.put(newSum, cache.getOrDefault(newSum, 0) + 1);
		pathSumRecurse(root.left, sum, temp + root.val, cache, result);
		if (cache.get(newSum) > 1) 
			cache.put(newSum, cache.get(newSum) - 1);
		else
			cache.remove(newSum);
		
		cache.put(newSum, cache.getOrDefault(newSum, 0) + 1);
		pathSumRecurse(root.right, sum, temp, cache, result);
		if (cache.get(newSum) > 1) 
			cache.put(newSum, cache.get(newSum) - 1);
		else
			cache.remove(newSum); 
    }
    
    // https://leetcode.com/problems/sum-root-to-leaf-numbers/description/
    public int sumNumbers(TreeNode root) {
        int[] sum = {0};
        sumNumbers(root, 0, sum);
        return sum[0];
    }
    
    private void sumNumbers(TreeNode root, int traverse, int[] sum) {
    		if (root == null)
    			return;
    		
    		int newTraverse = traverse * 10 + root.val;
    		
    		if (root.left == null && root.right == null) {
    			sum[0] += newTraverse;
    			return;
    		}
    		
    		if (root.left != null) {
    			sumNumbers(root.left, newTraverse, sum);
    		}
    		
    		if (root.right != null) {
    			sumNumbers(root.right, newTraverse, sum);
    		}
    }
    
    // https://leetcode.com/problems/second-minimum-node-in-a-binary-tree/description/
    public int findSecondMinimumValue(TreeNode root) {
    		if (root == null)
    			return -1;
    		
    		int second = findSecondMinimumValue(root, root.val);
    		
    		if (second == root.val) return -1;
    		else return second;
    }
    
    private int findSecondMinimumValue(TreeNode root, int small) {
    		if (root == null) {
    			return small;
    		}
    		
    		if (root.val > small)
    			return root.val;
    		
    		int left = findSecondMinimumValue(root.left, small);
    		int right = findSecondMinimumValue(root.right, small);
    		
    		if (left <= right) {
    			return left > small ? left : right;
    		} else {
    			return right > small ? right : left;
    		}
    }
    
    // https://leetcode.com/problems/maximum-width-of-binary-tree/description/
    private class TreeNumNode {
    		TreeNode node;
    		int idx;
    		public TreeNumNode(TreeNode node, int idx) {
    			this.node = node;
    			this.idx = idx;
    		}
    }
    
    public int widthOfBinaryTree(TreeNode root) {
        if (root == null)
        		return 0;
        
        Queue<TreeNumNode> queue = new LinkedList<>();
        queue.add(new TreeNumNode(root, 0));
        
        int cur = 1;
        int next =0;
        int curLeftIdx = 0;
        int nextLeftIdx = 0;
        int maxWidth = 1;
        while (!queue.isEmpty()) {
        		TreeNumNode node = queue.poll();
        		cur--;
        		
        		if (node.node.left != null) {
        			queue.add(new TreeNumNode(node.node.left, 2 * node.idx + 1));
        			next++;
        			if (next == 1) {
        				nextLeftIdx = 2 * node.idx + 1;
        			}
        		}
        		if (node.node.right != null) {
        			queue.add(new TreeNumNode(node.node.right, 2 * node.idx + 2));
        			next++;
        			if (next == 1) {
        				nextLeftIdx = 2 * node.idx + 2;
        			}
        		}
        		
        		if (cur == 0) {
        			cur = next;
        			next = 0;
        			maxWidth = Math.max(maxWidth, node.idx - curLeftIdx + 1);
        			curLeftIdx = nextLeftIdx;
        			nextLeftIdx = 0;
        		}
        }
        
        return maxWidth;
    }
    	
    // https://leetcode.com/problems/maximum-binary-tree/description/
    public TreeNode constructMaximumBinaryTree(int[] nums) {
    		if (nums == null || nums.length <= 0)
    			return null;
    		
    		return constructMaximumBinaryTree(nums, 0, nums.length - 1);
    }
    
    private TreeNode constructMaximumBinaryTree(int[] nums, int start, int end) {
    		if (start > end) return null;
    		if (start == end) 
    			return new TreeNode(nums[start]);
    		
    		int max = nums[start];
    		int maxIdx = start;
    		for (int i = start + 1; i <= end; i++) {
    			if (max < nums[i]) {
    				max = nums[i];
    				maxIdx = i;
    			}
    		}
    		
    		TreeNode root = new TreeNode(max);
    		
    		root.left = constructMaximumBinaryTree(nums, start, maxIdx - 1);
    		root.right = constructMaximumBinaryTree(nums, maxIdx + 1, end);
    		
    		return root;
    }
    
    // https://leetcode.com/problems/binary-tree-maximum-path-sum/description/
    public int maxPathSum(TreeNode root) {
        int[] result = {Integer.MIN_VALUE};
        maxPathSum(root, result);
        return result[0];
    }
    
    private int maxPathSum(TreeNode root, int[] result) {
    		if (root.left == null && root.right == null) {
    			result[0] = Math.max(result[0], root.val);
    			return root.val;
    		}
    		
    		int leftMax = Integer.MIN_VALUE;
    		if (root.left != null) {
    			leftMax = maxPathSum(root.left, result);
    		}
    		
    		int rightMax = Integer.MIN_VALUE;
    		if (root.right != null) {
    			rightMax = maxPathSum(root.right, result);
    		}
    		
    		
    		int sideMax = Math.max(leftMax, rightMax);
    		int sideMin = Math.min(leftMax, rightMax);
    		
    		int tempMax;
    		// 最大值小于0
    		if (sideMax <= 0) {
    			tempMax = Math.max(sideMax, root.val);
    		} else if (sideMin >= 0){
    			tempMax = Math.max(leftMax + rightMax + root.val, sideMax);
    		} else {
    			if (root.val >= 0)
    				tempMax = root.val + sideMax;
    			else
    				tempMax = sideMax;
    		}
    		
    		result[0] = Math.max(result[0], tempMax);
    		
    		return sideMax > 0 ? root.val + sideMax : root.val;
    }
    
    // https://leetcode.com/problems/binary-tree-postorder-traversal/description/
    public List<Integer> postorderTraversal(TreeNode root) {
    		List<Integer> result = new LinkedList<>();
		if (root == null)
			return result;
		
		Stack<TreeNode> stack = new Stack<>();
		TreeNode right = root;
		Set<TreeNode> backNodes = new HashSet<>();
		
		while (!stack.isEmpty() || right != null) {
			
			while (right != null) {
				stack.push(right);
				right = right.left;
			}
			
			if (!stack.isEmpty()) {
				TreeNode top = stack.peek();
				if (backNodes.contains(top)) {
					result.add(top.val);
					stack.pop();
				} else {
					right = top.right;
					if (right == null) {
						result.add(top.val);
						stack.pop();
					} else {
						backNodes.add(top);
					}
				}
			}
		}
		
		return result;
    }
    
    public List<Integer> preorderTraversal(TreeNode root) {
    		List<Integer> result = new LinkedList<>();
    		if (root == null)
    			return result;
    		
    		Stack<TreeNode> stack = new Stack<>();
    		TreeNode right = root;
    		
    		while (!stack.isEmpty() || right != null) {
    			
    			while (right != null) {
    				stack.push(right);
    				result.add(right.val);
    				right = right.left;
    			}
    			
    			if (!stack.isEmpty()) {
    				right = stack.pop().right;
    			}
    		}
    		
    		return result;
    }
    
    public List<Integer> inorderTraversal(TreeNode root) {
		List<Integer> result = new LinkedList<>();
		if (root == null)
			return result;
		
		Stack<TreeNode> stack = new Stack<>();
		TreeNode right = root;
		
		while (!stack.isEmpty() || right != null) {
			
			while (right != null) {
				stack.push(right);
				right = right.left;
			}
			
			if (!stack.isEmpty()) {
				TreeNode top = stack.pop();
				result.add(top.val);
				right = top.right;
			}
		}
		
		return result;
    }
    
    // https://leetcode.com/problems/binary-tree-right-side-view/description/
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> result = new LinkedList<>();
        if (root == null)
        		return result;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        
        int cur = 1;
        int next = 0;
        while (!queue.isEmpty()) {
        		TreeNode head = queue.poll();
        		cur--;
        		
        		if (head.left != null) {
        			queue.add(head.left);
        			next++;
        		}
        		
        		if (head.right != null) {
        			queue.add(head.right);
        			next++;
        		}
        		
        		if (cur == 0) {
        			result.add(head.val);
        			cur = next;
        			next = 0;
        		}
        }
        
        return result;
    }
    
    public List<Integer> rightSideViewII(TreeNode root) {
    		List<Integer> result = new LinkedList<>();
    		
    		rightSideViewDfs(root, 0, result);
    		
    		return result;
    }
    
    private void rightSideViewDfs(TreeNode root, int h, List<Integer> result) {
    		if (root == null)
    			return;
    		
    		if (h + 1 > result.size()) {
    			result.add(root.val);
    		}
    		
    		if (root.right != null) {
    			rightSideViewDfs(root.right, h + 1, result);
    		}
    		if (root.left != null) {
    			rightSideViewDfs(root.left, h + 1, result);
    		}
    }
    
    /**
     * Definition for binary tree with next pointer.*/
     public class TreeLinkNode {
         int val;
         TreeLinkNode left, right, next;
         TreeLinkNode(int x) { val = x; }
     }
     
    // https://leetcode.com/problems/populating-next-right-pointers-in-each-node/description/
    public void connect(TreeLinkNode root) {
        if (root == null)
        		return;
        
        TreeLinkNode preLevelStart = root;
        while (preLevelStart != null) {
            TreeLinkNode end = null;
            TreeLinkNode levelStart = preLevelStart;
            while (levelStart != null && levelStart.left != null) {
                if (end != null) {
                    end.next = levelStart.left;
                }
                levelStart.left.next = levelStart.right;
                end = levelStart.right;
                levelStart = levelStart.next;
            }
            preLevelStart = preLevelStart.left;
        }
    }
    
    public void connectII(TreeLinkNode root) {
        if (root == null)
        		return;
        
        TreeLinkNode preLevelStart = root;
        while (preLevelStart != null) {
        		TreeLinkNode nextLevelStart = null;
        		TreeLinkNode end = null;
        		TreeLinkNode levelStart = preLevelStart;
        		while (levelStart != null) {
        			if (levelStart.left != null && levelStart.right != null) {
        				if (end != null) end.next = levelStart.left;
        				if (nextLevelStart == null) nextLevelStart = levelStart.left;
        				levelStart.left.next = levelStart.right;
        				end = levelStart.right;
        			} else if (levelStart.left != null && levelStart.right == null) {
        				if (end != null) end.next = levelStart.left;
        				if (nextLevelStart == null) nextLevelStart = levelStart.left;
        				end = levelStart.left;
        			} else if (levelStart.right != null && levelStart.left == null) {
        				if (end != null) end.next = levelStart.right;
        				if (nextLevelStart == null) nextLevelStart = levelStart.right;
        				end = levelStart.right;
        			}
        			levelStart = levelStart.next;
        		}
        		preLevelStart = nextLevelStart;
        }
        
    }
    
    // https://leetcode.com/problems/add-one-row-to-tree/description/
    public TreeNode addOneRow(TreeNode root, int v, int d) {
        if (d == 1) {
        		TreeNode head = new TreeNode(v);
        		head.left = root;
        		return head;
        }
        
        addOneRow(root, 1, v, d);
        
        return root;
    }
    
    public void addOneRow(TreeNode root, int depth, int v, int d) {
    		if (root == null)
    			return;
    		if (depth >= d - 1) {
    			TreeNode left = root.left;
    			TreeNode right = root.right;
    			
    			root.left = new TreeNode(v);
    			root.left.left = left;
    			
    			root.right = new TreeNode(v);
    			root.right.right = right;
    			
    			return;
    		}
    		
    		addOneRow(root.left, depth + 1, v, d);
    		addOneRow(root.right, depth + 1, v, d);
    }
    
    // https://leetcode.com/problems/cut-off-trees-for-golf-event/description/
    public int cutOffTree(List<List<Integer>> forest) {
        if (forest == null || forest.size() <= 0) return -1;
        final int rows = forest.size();
        final int columns = forest.get(0).size();
        if (columns <= 0) return -1;
        
        PriorityQueue<CutTreeNode> minHead = new PriorityQueue<>(new Comparator<CutTreeNode>() {
	    		@Override
	    		public int compare(CutTreeNode o1, CutTreeNode o2) {
	    			// TODO Auto-generated method stub
	    			if (o1.trees > o2.trees)
	    				return 1;
	    			else if (o1.trees < o2.trees)
	    				return -1;
	    			else
	    				return 0;
	    		}
        });
        
        int[][] trees = new int[rows][columns];
        for (int i = 0; i < rows; ++i)
        		for (int j = 0; j < columns; ++j) {
        			trees[i][j] = forest.get(i).get(j);
        			if (trees[i][j] > 0 || (i == 0 && j == 0))
        				minHead.add(new CutTreeNode(i, j, trees[i][j]));
        		}
        
        // 需要排序
        int sum = 0;
        int srow = 0;
        int scolumn = 0;
        
        CutTreeNode min = minHead.peek();
        if (min.row == 0 && min.column == 0) {
        		srow = 0;
        		scolumn = 0;
        } else {
        		// 需要先走到最小的
        		boolean[][] visited = new boolean[rows][columns];
        		visited[0][0] = true;
        		int minStep = getMinStep(trees, 0, 0, min.row, min.column, visited);
        		if (minStep == -1)
        			return -1;
        		sum += minStep;
        		
        		srow = min.row;
        		scolumn = min.column;
        }
        
        minHead.poll();
        
        while (!minHead.isEmpty()) {
        		CutTreeNode node = minHead.poll();
        		boolean[][] visited = new boolean[rows][columns];
        		visited[srow][scolumn] = true;
        		int minStep = getMinStep(trees, srow, scolumn, node.row, node.column, visited);
        		if (minStep == -1)
        			return -1;
        		else
        			sum += minStep;
        		srow = node.row;
        		scolumn = node.column;
        }
        
        return sum;
    }
    
    private class CutTreeNode {
    		int row;
    		int column;
    		int trees;
    		
    		public CutTreeNode(int row, int column, int trees) {
    			this.row = row;
    			this.column = column;
    			this.trees = trees;
    		}
    }
    
    private static int[][] sCutDirection = {
    		{-1, 0},
    		{1, 0},
    		{0, -1},
    		{0, 1},
    };
    private int getMinStep(int[][] trees, int srow, int scolumn, int erow, int ecolumn, boolean[][] visited) {
    	
    		if (srow == erow && ecolumn == scolumn)
    			return 0;
    		
    		Queue<Integer> queue = new LinkedList<>();
    		
    		int cur = 1;
    		int next = 0;
    		int step = 0;
    		
    		queue.add(srow);
    		queue.add(scolumn);
    		visited[srow][scolumn] = true;
    		
    		while (!queue.isEmpty()) {
    			int toprow = queue.poll();
    			int topcolumn = queue.poll();
    			cur--;
    			
    			for (int i = 0; i < sCutDirection.length; ++i) {
    				int newrow = toprow + sCutDirection[i][0];
    				int newcolumn = topcolumn + sCutDirection[i][1];
    				if (newrow == erow && newcolumn == ecolumn) {
    					return step + 1;
    				}
    				
    				if (newrow >= 0 && newrow < trees.length 
    						&& newcolumn >= 0 && newcolumn < trees[0].length
    						&& !visited[newrow][newcolumn]
    						&& trees[newrow][newcolumn] >= 1) {
    					queue.add(newrow);
    					queue.add(newcolumn);
    					visited[newrow][newcolumn] = true;
    					next++;
    				}
    			}
    			
    			if (cur == 0) {
    				cur = next;
    				next = 0;
    				step++;
    			}
    		}
    		
    		return -1;
    }
    
    // https://leetcode.com/problems/recover-binary-search-tree/description/
//    TreeNode[] pre = new TreeNode[1];
//    TreeNode[] first = new TreeNode[2];
//    TreeNode[] second = new TreeNode[2];
//    public void recoverTree(TreeNode root) {
//    		pre[0] = null;
//    		first[0] = first[1] = null;
//    		second[0] = second[1] = null;
//
//    		recoverTreeInOrder(root);
//
//        if (first[1] == null) {
//        		return;
//        }
//
//        if (second[1] == null) {
//        		int temp = first[0].val;
//        		first[0].val = first[1].val;
//        		first[1].val = temp;
//        } else {
//        		int temp = first[0].val;
//        		first[0].val = second[1].val;
//        		second[1].val = temp;
//        }
//    }
    
//    private void recoverTreeInOrder(TreeNode root) {
//    		if (root == null)
//    			return;
//
//    		if (first[1] != null && second[1] != null)
//    			return;
//
//    		recoverTreeInOrder(root.left);
//    		if (pre[0] != null) {
//    			if (root.val < pre[0].val) {
//    				if (first[1] == null) {
//    					first[0] = pre[0];
//    					first[1] = root;
//    				} else if (second[1] == null) {
//    					second[0] = pre[0];
//    					second[1] = root;
//    				}
//    			}
//    		}
//    		pre[0] = root;
//
//    		recoverTreeInOrder(root.right);
//    }
    
    // https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description/
    private LinkedList<TreeNode> pTraverse = new LinkedList<>();
    private LinkedList<TreeNode> qTraverse = new LinkedList<>();
    private List<TreeNode> temTraverse = new LinkedList<>();
    private TreeNode pFind;
    private TreeNode qFind;
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (p == q)
        		return p;
        
        pTraverse.clear();
        qTraverse.clear();
        temTraverse.clear();
        pFind = p;
        qFind = q;
        
        lowestCommonAncestor(root);
        
        Iterator<TreeNode> pIterator = pTraverse.iterator();
        Iterator<TreeNode> qIterator = qTraverse.iterator();
        
        TreeNode result = null;
        while (pIterator.hasNext() && qIterator.hasNext()) {
        		TreeNode pNode = pIterator.next();
        		TreeNode qNode = qIterator.next();
        		if (pNode == qNode) {
        			result = pNode;
        		} else {
        			break;
        		}
        }
        
        return result;
    }
    
    
    private void lowestCommonAncestor(TreeNode root) {
    		if (root == null)
    			return;
    		if (!pTraverse.isEmpty() && !qTraverse.isEmpty())
    			return;
    		
    		if (root == pFind) {
    			pTraverse.addAll(temTraverse);
    			pTraverse.add(root);
    		} else if (root == qFind) {
    			qTraverse.addAll(temTraverse);
    			qTraverse.add(root);
    		}
    		
    		temTraverse.add(root);
    		lowestCommonAncestor(root.left);
    		lowestCommonAncestor(root.right);
    		temTraverse.remove(temTraverse.size() - 1);
    }
    
    // https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/description/
    // BST
    public TreeNode lowestCommonAncestorII(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return root;
        TreeNode max = p.val >= q.val ? p : q;
        TreeNode min = p.val < q.val ? p : q;
        if (root.val >= min.val && root.val <= max.val)
        		return root;
        else if (root.val > max.val) {
        		return lowestCommonAncestorII(root.left, p, q);
        } else {
        		return lowestCommonAncestorII(root.right, p, q);
        } 
    }
    
    
    // https://leetcode.com/problems/closest-leaf-in-a-binary-tree/description/
    public int findClosestLeaf(TreeNode root, int k) {
        List<TreeNode> traverse = new LinkedList<>();
        List<TreeNode> result = new LinkedList<>();
        boolean[] find = {false};
        findLeaf(root, k, traverse, find, result);
        if (!find[0])
        		return -1;
        
        TreeNode findNode = result.get(result.size() - 1);
        if (findNode.left == null && findNode.right == null)
        		return k;
       
        TreeNode[] path = new TreeNode[result.size()];
        path = result.toArray(path);
        int[] min = {Integer.MAX_VALUE, -1};
        findClosestLeaf(path, 0, root, 0, min);
        
        return min[1];
    }
    
    private void findClosestLeaf(TreeNode[] path, int start, TreeNode root, int level, int[] min) {
    		if (root == null)
    			return;
    		
    		if (root.left == null && root.right == null) {
    			// 算出距离
    			int tempMin = (level - start + 1) + path.length - start;
    			if (min[0] > tempMin) {
    				min[0] = tempMin;
    				min[1] = root.val;
    			}
    			
    			return;
    		}
    		
    		// 更新start
    		if (start < path.length) {
    			if (root == path[start]) {
    				start++;
    			}
    		}
    		
    		if (root.left != null) {
    			findClosestLeaf(path, start, root.left, level + 1, min);
    		}
    		if (root.right != null) {
    			findClosestLeaf(path, start, root.right, level + 1, min);
    		}
    		
    }
    
    private void findLeaf(TreeNode root, int k, List<TreeNode> traverse, boolean[] find, List<TreeNode> result) {
    		if (root == null || find[0])
    			return;
    		
    		if (k == root.val) {
    			result.addAll(traverse);
    			result.add(root);
    			find[0] = true;
    			return;
    		}
    		
    		traverse.add(root);
    		findLeaf(root.left, k, traverse, find, result);
    		findLeaf(root.right, k, traverse, find, result);
    		traverse.remove(traverse.size() - 1);
    }
    
    // https://leetcode.com/problems/unique-binary-search-trees/description/
    public int numTrees(int n) {
        int[] dp = new int[1 + n];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; ++i) {
        		
        		for (int j = 0; j <= i - 1; ++j) {
        			dp[i] += dp[j] * dp[i-1-j];
        		}
        }
        
        return dp[n];
    }
    
    // https://leetcode.com/problems/unique-binary-search-trees-ii/description/
    public List<TreeNode> generateTrees(int n) {
    		return generateTrees(1, n);
    }
    
    private List<TreeNode> generateTrees(int start, int end) {
    		List<TreeNode> result = new LinkedList<>();
    		if (start > end) {
    			return result;
    		}else if (start == end) {
    			TreeNode node = new TreeNode(start);
    			result.add(node);
    			return result;
    		}
    		
    		for (int i = start; i <= end; i++) {
    			List<TreeNode> leftHeads = generateTrees(start, i - 1);
    			if (leftHeads.isEmpty()) leftHeads.add(null);
    			List<TreeNode> rightHeads = generateTrees(i + 1, end);
    			if (rightHeads.isEmpty()) rightHeads.add(null);
    			for (TreeNode left : leftHeads) {
    				for (TreeNode right : rightHeads) {
    					TreeNode head = new TreeNode(i);
    					head.left = left;
    					head.right = right;
    					result.add(head);
    				}
    			}
    		}
    		
    		return result;
    }
    
    // https://leetcode.com/problems/max-sum-of-rectangle-no-larger-than-k/description/
    public int maxSumSubmatrix(int[][] matrix, int k) {
        final int rows = matrix.length;
        final int columns = matrix[0].length;
        
        int max = Integer.MIN_VALUE;
        
        int[] sumsColumn = new int[columns + 1];
        int[] sort = new int[columns + 1];
        for (int i = 0; i < rows; ++i) {
        		int[] sums = new int[columns]; 
        		for (int j = i; j < rows; ++j) {
        			for (int l = 0; l < columns; ++l) {
        				sums[l] += matrix[j][l];
        				sumsColumn[l + 1] = sumsColumn[l] + sums[l];
        			}
        			
        			max = Math.max(max, getSubMaxSum(sumsColumn, sort, k, 0, sort.length - 1));
        			if (max == k)
        				return k;
        		}
        }
        
        return max;
    }
    
    private int getSubMaxSum(int[] sums, int[] sort, int k, int start, int end) {
    		if (start >= end)
    			return Integer.MIN_VALUE;
    		
    		int mid = start + (end - start) / 2;
    		
    		int leftAns = getSubMaxSum(sums, sort, k, start, mid);
    		if (leftAns == k) return k;
    		int rightAns = getSubMaxSum(sums, sort, k, mid + 1, end);
    		if (rightAns == k) return k;
    		
    		int ans = Math.max(leftAns, rightAns);
    		
    		// 处理最大值
    		for (int i = start, j = mid + 1; i <= mid && j <= end; ++i) {
    			while (j <= end && sums[j] - sums[i] <= k)
    				j++;
    			if (j - 1 >= mid + 1) {
    				ans = Math.max(ans, sums[j-1] - sums[i]);
    			}
    			if (ans == k)
    				return k;
    		}
    		
    		int leftStart = start;
    		int rightStart = mid + 1;
    		int sortStart = start;
    		while (leftStart <= mid && rightStart <= end) {
    			if (sums[leftStart] <= sums[rightStart]) {
    				sort[sortStart++] = sums[leftStart++];
    			} else 
    				sort[sortStart++] = sums[rightStart++];
    		}
    		while (leftStart <= mid)
    			sort[sortStart++] = sums[leftStart++];
    		while (rightStart <= end)
    			sort[sortStart++] = sums[rightStart++];
    		
    		System.arraycopy(sort, start, sums, start, end - start + 1);
    		
    		return ans;
    }
    
    // https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/description/
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        return buildTree(inorder, postorder, 0, inorder.length - 1, 0, postorder.length - 1);
    }
    
    public TreeNode buildTree(int[] inorder, int[] postorder, int instart, int inend, int poststart, int postend) {
    		if (instart > inend) return null;
    		if (instart == inend) 
    			return new TreeNode(inorder[instart]);
    		
    		TreeNode head = new TreeNode(postorder[postend]);
    		
    		// 找到后序在中序的位置
    		int find = -1;
    		for (int i = instart; i <= inend; ++i) {
    			if (inorder[i] == postorder[postend]) {
    				find = i;
    				break;
    			}
    		}
    		
    		head.left = buildTree(inorder, postorder, instart, find - 1, poststart, poststart + find - instart - 1);
    		head.right = buildTree(inorder, postorder, find + 1, inend, poststart + find - instart, postend - 1);
    		
    		return head;
    }
    
    // https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/
    public TreeNode buildTreeII(int[] preorder, int[] inorder) {
        return buildTree(inorder, preorder, 0, inorder.length - 1, 0, preorder.length - 1);
    }
    
    public TreeNode buildTreeII(int[] inorder, int[] preorder, int instart, int inend, int prestart, int prend) {
		if (instart > inend) return null;
		if (instart == inend) 
			return new TreeNode(inorder[instart]);
		
		TreeNode head = new TreeNode(preorder[prestart]);
		
		// 找到后序在中序的位置
		int find = -1;
		for (int i = instart; i <= inend; ++i) {
			if (inorder[i] == preorder[prestart]) {
				find = i;
				break;
			}
		}
		
		head.left = buildTree(inorder, preorder, instart, find - 1, prestart + 1, prestart + find - instart);
		head.right = buildTree(inorder, preorder, find + 1, inend, prestart + find - instart + 1, prend);
		
		return head;
    }
    
    // https://leetcode.com/problems/house-robber-iii/description/
    public int rob(TreeNode root) {
    		int[] maxes = {0, 0};
    		int[] result = {0};
    		rob(root, maxes, result);
    		return result[0];
    }
    
    // 0 代表包含头结点，1 代表不包含头结点
    private void rob(TreeNode root, int[] maxes, int[] result) {
        if (root == null) {
        		maxes[0] = 0;
        		maxes[1] = 0;
        		result[0] = Math.max(result[0], 0);
        		return;
        } 
        
        int[] leftMaxes = new int[2];
        rob(root.left, leftMaxes, result);
        int[] rightMaxes = new int[2];
        rob(root.right, rightMaxes, result);
        
        maxes[0] = root.val + leftMaxes[1] + rightMaxes[1];
        maxes[1] = Math.max(leftMaxes[0], leftMaxes[1]) + Math.max(rightMaxes[0], rightMaxes[1]);
        
        result[0] = Math.max(result[0], Math.max(maxes[0], maxes[1]));
    }
    
    // https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/description/
    public TreeNode sortedListToBST(ListNode head) {
	    	if (head == null)
	    		return null;
	    if (head.next == null)
	    		return new TreeNode(head.val);
	    
	    ListNode slow = head;
	    ListNode fast = head;
	    ListNode pre = null;
	    while (fast != null && fast.next != null) {
	            pre = slow;
	    		slow = slow.next;
	    		fast = fast.next.next;
	    }
	    
	    ListNode right = slow.next;
	    slow.next = null;
	    pre.next = null;
	    
	    TreeNode root = new TreeNode(slow.val);
	    root.left = sortedListToBST(head);
	    root.right = sortedListToBST(right);
	    
	    return root;
    }
    
    // https://leetcode.com/problems/merge-two-binary-trees/description/
    public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null)
        		return null;
        
        TreeNode root;
        if (t1 != null && t2 != null) {
        		root = new TreeNode(t1.val + t2.val);
        		root.left = mergeTrees(t1.left, t2.left);
        		root.right = mergeTrees(t1.right, t2.right);
        }
        else if (t1 != null) {
        		root = new TreeNode(t1.val);
        		root.left = mergeTrees(t1.left, null);
        		root.right = mergeTrees(t1.right, null);
        } else {
	        	root = new TreeNode(t2.val);
	    		root.left = mergeTrees(null, t2.left);
	    		root.right = mergeTrees(null, t2.right);
        }
       
        return root;
    }
    
    // https://leetcode.com/problems/delete-node-in-a-bst/description/
    public TreeNode deleteNode(TreeNode root, int key) {
        if (root == null)
        		return root;
        
        TreeNode[] find = {null, null};
        findNode(root, key, find);
        if (find[0] == null)
        		return root;
        
        if (find[1] == null) {
        		// 需要删除的是顶点
        		if (root.left == null && root.right == null)
        			return null;
        		else if (root.left == null)
        			return root.right;
        		else {
        			TreeNode pre = root.left;
        			TreeNode right = root.left;
        			while (right.right != null) {
        				pre = right;
        				right = right.right;
        			}
        			
        			if (right == root.left) {
        				root.left.right = root.right;
        				return root.left;
        			} 
        			
        			pre.right = right.left;
        			
        			right.left = root.left;
        			right.right = root.right;
        			
        			return right;
        			
        		}
        } 
        
        // 删除的不是顶点
        boolean leftChild = find[1].left == find[0];
        
        // 叶子结点 
        if (find[0].left == null && find[0].right == null) {
        		if (leftChild) find[1].left = null;
        		else find[1].right = null;
        		return root;
        } else if (find[0].left == null) {
        		if (leftChild) find[1].left = find[0].right;
        		else find[1].right = find[0].right;
        		return root;
        } else if (find[0].right == null) {
	        	if (leftChild) find[1].left = find[0].left;
	    		else find[1].right = find[0].left;
	    		return root;
        } else {
        		// 左右都不为空
        		TreeNode pre = find[0].left;
			TreeNode right = find[0].left;
			while (right.right != null) {
				pre = right;
				right = right.right;
			}
			
			if (right == find[0].left) {
				find[0].left.right = find[0].right;
				if (leftChild) find[1].left = find[0].left;
        			else find[1].right = find[0].left;
				return root;
			} 
			
			pre.right = right.left;
			
			right.left = find[0].left;
			right.right = find[0].right;
			
			if (leftChild) find[1].left = right;
			else find[1].right = right;
			
			return root;
        }
        
    }
    
    public void findNode(TreeNode root, int key, TreeNode[] find) {
        if (root == null)
        		return;
        
        if (root.val == key) {
        		find[0] = root;
        		return;
        }
        
        find[1] = root;
        if (key > root.val) {
        		findNode(root.right, key, find);
        } else {
        		findNode(root.left, key, find);
        }
    }
    
    // 这个是可以改变node中的值
    public TreeNode deleteNodeDfs(TreeNode root, int key) {
    		if (root == null)
    			return null;
    		if (key < root.val) {
    			root.left = deleteNodeDfs(root.left, key);
    			return root;
    		} else if (key > root.val) {
    			root.right = deleteNodeDfs(root.right, key);
    			return root;
    		}
    		
    		// root的值
    		if (root.left == null)
    			root = root.right;
    		else if (root.right == null)
    			root = root.left;
    		else {
    			int max = findMax(root.left);
    			root.val = max;
    			root.left = deleteNodeDfs(root.left, max);
    		}
    		
    		return root;
    }
    
    private int findMax(TreeNode root) {
        while (root.right != null) {
            root = root.right;
        }
        return root.val;
    }
    
    // https://leetcode.com/problems/trim-a-binary-search-tree/description/
    public TreeNode trimBST(TreeNode root, int L, int R) {
        if (root == null)
        		return null;
        if (L <= root.val && R >= root.val) {
        		root.left = trimBST(root.left, L, R);
        		root.right = trimBST(root.right, L, R);
        } else if (root.val < L) {
        		root = trimBST(root.right, L, R);
        } else {
        		root = trimBST(root.left, L, R);
        }
        
        return root;
    }
    
    // https://leetcode.com/problems/most-frequent-subtree-sum/description/
    private Map<Integer, Integer> mapFrequentSum ;
    private List<Integer> listResult;
    private int mapFrequent;
    public int[] findFrequentTreeSum(TreeNode root) {
        mapFrequent = Integer.MIN_VALUE;
        listResult = new LinkedList<>();
        mapFrequentSum = new HashMap<>();
        
        findFrequentTreeSumDfs(root);
        
        if (listResult.size() <= 0)
        		return new int[0];
        else {
        		int[] result = new int[listResult.size()];
        		int i = 0;
        		for (int sum : listResult)
        			result[i++] = sum;
        		return result;
        }
    }
    
    private int findFrequentTreeSumDfs(TreeNode root) {
    		if (root == null) {
    			return 0;
    		}
    		
    		int leftSum = findFrequentTreeSumDfs(root.left);
    		int rightSum = findFrequentTreeSumDfs(root.right);
    		
    		int curSum = root.val + leftSum + rightSum;
    		
    		mapFrequentSum.put(curSum, mapFrequentSum.getOrDefault(curSum, 0) + 1);
    		int curFrequent = mapFrequentSum.get(curSum);
    		if (curFrequent > mapFrequent) {
    			mapFrequent = curFrequent;
    			listResult.clear();
    			listResult.add(curSum);
    		} else if (curFrequent == mapFrequent) {
    			listResult.add(curSum);
    		}
    		
    		return curSum;
    }
    
    // https://leetcode.com/problems/find-bottom-left-tree-value/description/
    int maxDepth;
    int bottomLeftValue;
    public int findBottomLeftValue(TreeNode root) {
        maxDepth = -1;
        findBottomLeftValue(root, 0);
        return bottomLeftValue;
    }
    
    private void findBottomLeftValue(TreeNode root, int depth) {
    		if (root == null)
    			return;
    		
    		if (depth > maxDepth) {
    			maxDepth = depth;
    			bottomLeftValue = root.val;
    		}
    		
    		findBottomLeftValue(root.left, depth + 1);
    		findBottomLeftValue(root.right, depth + 1);
    }
    
    // https://leetcode.com/problems/print-binary-tree/description/
    public List<List<String>> printTree(TreeNode root) {
        int depth = getMaxDepth(root);
        return printTree(root, 1, depth);
    }
    
    private List<List<String>> printTree(TreeNode root, int depth, int maxDepth) {
    		List<List<String>> linkedList = new LinkedList<>();
    		
    		if (depth == maxDepth) {
    			List<String> line = new LinkedList<>();
    			line.add(root == null ? "" : String.valueOf(root.val));
    			linkedList.add(line);
    			
    			return linkedList;
    		} else if (depth > maxDepth) {
    			return linkedList;
    		}
    		
    		List<List<String>> leftResult = printTree(root == null ? null : root.left, depth + 1, maxDepth);
    		List<List<String>> rightResult = printTree(root == null ? null : root.right, depth + 1, maxDepth);
    		
    		List<String> line = new LinkedList<>();
    		int cnt = leftResult.get(0).size();
    		for (int i = 0; i < cnt; ++i)
    			line.add("");
    		line.add(root == null ? "" : String.valueOf(root.val));
    		for (int i = 0; i < cnt; ++i)
    			line.add("");
    		linkedList.add(line);
    		
    		Iterator<List<String>> itLeft = leftResult.iterator();
    		Iterator<List<String>> itRight = rightResult.iterator();
    		while (itLeft.hasNext()) {
    			line = new LinkedList<>();
    			line.addAll(itLeft.next());
    			line.add("");
    			line.addAll(itRight.next());
    			linkedList.add(line);
    		}
    		
    		return linkedList;
    }
    
    private int getMaxDepth(TreeNode root) {
    		if (root == null)
    			return 0;
    		return 1 + Math.max(getMaxDepth(root.left), getMaxDepth(root.right));
    }
    
    // https://leetcode.com/problems/verify-preorder-serialization-of-a-binary-tree/description/
    public boolean isValidSerializationII(String preorder) {
		if (preorder == null) return false;
	
		String[] nodes = preorder.split(",");
		int idx = 0;
		String right = nodes[0];
		Stack<String> stack = new Stack<>();
		
		while ((!stack.isEmpty() || !"#".equals(right)) && idx < nodes.length) {
			String left = right;
			while (!"#".equals(left)) {
				stack.push(left);
				if (idx + 1 < nodes.length) {
					left = nodes[++idx];
				} else {
					return false;
				}
			}
			
			stack.pop();
			if (idx + 1 < nodes.length) 
				right = nodes[++idx];
			else 
				return false;
		}
		
		return stack.isEmpty() && idx == nodes.length - 1;
    }
    
    // https://leetcode.com/problems/network-delay-time/description/
    public int networkDelayTime(int[][] times, int N, int K) {
        if (times == null || times[0].length == 0 || K > N) {
        		return -1;
        }
        
        Map<Integer, Integer> minTime = new HashMap<>();
        Map<Integer, Map<Integer, Integer>> graph = new HashMap<>();
        
        for (int i = 0; i < times.length; ++i) {
        		Map<Integer, Integer> adjecents = graph.get(times[i][0]);
        		if (adjecents == null) {
        			adjecents = new HashMap<>();
        			graph.put(times[i][0], adjecents);
        		}
        		
        		adjecents.put(times[i][1], Math.min(adjecents.getOrDefault(times[i][1], Integer.MAX_VALUE), times[i][2]));
        }
        
        minTime.put(K, 0);
        Queue<Integer> queue = new LinkedList<>();
        queue.add(K);
        
        while (!queue.isEmpty()) {
        		int top = queue.poll();
        		
        		int delay = minTime.get(top);
        		
        		Map<Integer, Integer> adjecents = graph.get(top);
        		if (adjecents == null)
        			continue;
        		
        		for (Map.Entry<Integer, Integer> entry : adjecents.entrySet()) {
        			int adjecent = entry.getKey();
        			int time = entry.getValue();
        			
        			if (minTime.containsKey(adjecent)) {
        				int oldDelay = minTime.get(adjecent);
        				if (oldDelay <= time + delay) {
        					continue;
        				}
        			} 
        			
        			minTime.put(adjecent, time + delay);
				queue.add(adjecent);
        		}
        }
        
        if (minTime.size() < N) {
        		return -1;
        }
        
        int result = Integer.MIN_VALUE;
        for (Integer time : minTime.values()) {
        		result = Math.max(result, time);
        }
        
        return result;
    }
    
    // https://leetcode.com/problems/daily-temperatures/description/
    public int[] dailyTemperatures(int[] temperatures) {
        if (temperatures == null || temperatures.length <= 0)
        		return null;
        
        int[] result = new int[temperatures.length];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < temperatures.length; ++i) {
        		while (!stack.isEmpty() && temperatures[stack.peek()] < temperatures[i]) {
        			int idx = stack.pop();
        			result[idx] = i - idx;
        		}
        		stack.push(i);
        }
        
        return result;
    }
    
    // https://leetcode.com/problems/monotone-increasing-digits/description/
    public int monotoneIncreasingDigits(int N) {
        if (N < 10) {
        		return N;
        }
        
        String s = Integer.toString(N);
        int lowerIdx = -1;
        for (int i = 1; i < s.length(); ++i) {
        		if (s.charAt(i) >= s.charAt(i-1)) {
        			continue;
        		} else {
        			lowerIdx = i - 1;
        			break;
        		}
        }
        
        if (lowerIdx == -1)
        		return N;
        
        while (lowerIdx > 0 && s.charAt(lowerIdx) == s.charAt(lowerIdx - 1)) {
        		lowerIdx--;
        }
        
        StringBuilder builder = new StringBuilder();
        builder.append(s.substring(0, lowerIdx));
    		builder.append(s.charAt(lowerIdx) - '0' - 1);
    		for (int i = lowerIdx + 1; i < s.length(); ++i)
    			builder.append(9);
    		
    		return Integer.valueOf(builder.toString());
        
    }
    
    // https://leetcode.com/problems/asteroid-collision/description/
    public int[] asteroidCollision(int[] asteroids) {
        if (asteroids == null)
        		return new int[0];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < asteroids.length; ++i) {
        		if (stack.isEmpty()) {
        			stack.push(asteroids[i]);
        			continue;
        		}
        		
        		int top = stack.peek();
        		if (top < 0) {
        			stack.push(asteroids[i]);
        			continue;
        		}
        		
        		if (asteroids[i] > 0) {
        			stack.push(asteroids[i]);
        			continue;
        		}
        		
        		if (top == -asteroids[i]) {
        			stack.pop();
        		} else if (top > -asteroids[i]) {
        			
        		} else {
        			while (!stack.isEmpty() && stack.peek() > 0 && stack.peek() < -asteroids[i]) {
        				stack.pop();
        			}
        			if (stack.isEmpty()) {
        				stack.push(asteroids[i]);
        			} else {
        				if (stack.peek() == -asteroids[i]) {
        					stack.pop();
        				}
        			}
        		}
        }
        
        if (stack.isEmpty())
        		return new int[0];
        
        int[] result = new int[stack.size()];
        for (int i = result.length - 1; i >= 0; --i) {
        		result[i] = stack.pop();
        }
        
        return result;
    }
    
    // https://leetcode.com/problems/maximum-length-of-repeated-subarray/description/
    public int findLength(int[] A, int[] B) {
        if (A == null || A.length == 0 || B == null || B.length == 0)
        		return 0;
        
        int result = 0;
        int[][] dp = new int[A.length + 1][B.length + 1];
        for (int i = 1; i <= A.length; ++i) {
        		for (int j = 1; j <= B.length; ++j) {
        			int ac = A[i-1];
        			int bc = B[j-1];
        			if (ac == bc) {
        				dp[i][j] = 1 + dp[i-1][j-1];
        				result = Math.max(result, dp[i][j]);
        			}
        		}
        }
        
        return result;
    }
    
    // https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/description/
    public int maxProfit(int[] prices, int fee) {
        int result = 0;
        if (prices == null || prices.length <= 1) {
        		return result;
        }
        
        int lastNone = 0;
        int lastHas = -prices[0];
        
        for (int i = 1; i < prices.length; ++i) {
        		int newNone = Math.max(0, Math.max(lastNone, lastHas + prices[i] - fee));
        		int newHas = Math.max(lastHas, lastNone - prices[i]);
        		result = Math.max(result, Math.max(newNone, newHas));
        		lastNone = newNone;
        		lastHas = newHas;
        }
        
        return result;
    }
    
    // https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/description/
    public int maxProfitIV(int[] prices) {
        int result = 0;
        if (prices == null || prices.length <= 1) {
        		return result;
        }
        
        int lastNoneNoSell = 0;
        int lastNoneSell = 0;
        int lastHas = -prices[0];
        
        for (int i = 1; i < prices.length; ++i) {
        		int newNoneNoSell = Math.max(lastNoneNoSell, lastNoneSell);
        		int newNoneSell = lastHas + prices[i];
        		int newHas = Math.max(lastHas, lastNoneNoSell - prices[i]);
        		
        		result = Math.max(result, Math.max(newNoneNoSell, newNoneSell));
        		
        		lastNoneNoSell = newNoneNoSell;
        		lastNoneSell = newNoneSell;
        		lastHas = newHas;
        }
        
        return result;
    }
    
    // https://leetcode.com/problems/maximum-product-subarray/description/
    public int maxProduct(int[] nums) {
        if (nums == null || nums.length <= 0)
        		return -1;
        
        int result = nums[0];
        int max = nums[0];
        int min = nums[0];
        
        for (int i = 1; i < nums.length; ++i) {
        		int newmax = max * nums[i];
        		int newmin = min * nums[i];
        		
        		max = Math.max(nums[i], Math.max(newmax, newmin));
        		min = Math.min(nums[i], Math.min(newmax, newmin));
        		
        		result = Math.max(result, max);
        }
        
        return result;
    }
    
    // https://leetcode.com/problems/subarray-product-less-than-k/description/
    public int numSubarrayProductLessThanK(int[] nums, int k) {
    		if (nums == null || nums.length <= 0)
    			return 0;
    		int product = 1;
    		int left = 0;
    		int right = 0;
    		int result = 0;
    		while (right < nums.length) {
    			while (right < nums.length && nums[right] * product < k) {
    				product *= nums[right];
    				right++;
    			}
    			
    			if (right == nums.length) {
    				result += (long)(right - left) * (long)(right - left + 1)  / 2;
    				return result;
    			}
    			
    			if (right == left) {
    				left++;
    				right++;
    				product = 1;
    			} else {
    				result += right - left;
    				product /= nums[left];
    				left++;
    			}
    		}
    		
    		return result;
    }
    
    // https://leetcode.com/problems/subarray-sum-equals-k/description/
    public int subarraySum(int[] nums, int k) {
        if (nums == null || nums.length <= 0)
        		return 0;
        
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        
        int result = 0;
        int sum = 0;
        for (int i = 0; i < nums.length; ++i) {
        		sum += nums[i];
        		result += map.getOrDefault(sum - k, 0);
        		map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        
        return result;
    }
    
    // https://leetcode.com/problems/minimum-ascii-delete-sum-for-two-strings/description/
    public int minimumDeleteSum(String s1, String s2) {
        int[][] dp = new int[s1.length()+1][s2.length()];
        for (int i = 1; i <= s1.length(); ++i) {
            dp[i][0] = dp[i-1][0] + s1.charAt(i-1);
        }
        
        for (int j = 1; j <= s2.length(); ++j) {
        		dp[0][j] = dp[0][j-1] + s2.charAt(j-1);
        }
        
        for (int i = 1; i <= s1.length(); ++i) {
        		for (int j = 1; j <= s2.length(); ++j) {
        			char c1 = s1.charAt(i-1);
        			char c2 = s2.charAt(j-1);
        			if (c1 == c2) {
        				dp[i][j] = Math.min(dp[i-1][j-1], Math.min(dp[i-1][j], dp[i][j-1]) + c1);
        			} else {
        				dp[i][j] = Math.min(dp[i-1][j-1] + c1 + c2, Math.min(dp[i-1][j] + c1, dp[i][j-1] + c2));
        			}
        		}
        }
        
        return dp[s1.length()][s2.length()];
    }
    
    // https://leetcode.com/problems/valid-parenthesis-string/description/
    Stack<Character> stringsttack;
    public boolean checkValidString(String s) {
        if (s == null)
        		return true;
        stringsttack = new Stack<>();
        return checkValidString(s, 0);
    }
    
    private boolean checkValidString(CharSequence s, int start) {
    		if (start >= s.length())
    			return stringsttack.isEmpty();
    		
    		final char c = s.charAt(start);
    		boolean empty = stringsttack.isEmpty();
    		boolean result = false;
    		if (c == '(') {
			stringsttack.push(c);
			if (checkValidString(s, start + 1))
				return true;
			else {
				stringsttack.pop();
				return false;
			}
    		} else if (c == ')') {
    			if (!empty && stringsttack.peek() == '(') {
    				stringsttack.pop();
    				if (checkValidString(s, start + 1))
    					return true;
    				else {
    					stringsttack.push('(');
    					return false;
    				}
    			} else {
    				return false;
    			}
    		} else if (c == '*') {
    			if (empty) {
    				result = checkValidString(s, start + 1);
    				if (!result) {
    					stringsttack.push('(');
    					if (checkValidString(s, start + 1))
    						return true;
    					else {
    						stringsttack.pop();
    						return false;
    					}
    				}
    			} else if (stringsttack.peek() == '('){
    				result = checkValidString(s, start + 1);
    				if (!result) {
    					stringsttack.pop();
        				if (checkValidString(s, start + 1))
        					return true;
        				else {
        					stringsttack.push('(');
        				}
    				}
    				if (!result) {
    					stringsttack.push('(');
    					if (checkValidString(s, start + 1))
    						return true;
    					else {
    						stringsttack.pop();
    					}
    				}
    			}
    		}
    		
    		return result;
    }
    
    // https://leetcode.com/problems/number-of-longest-increasing-subsequence/description/
    public int findNumberOfLIS(int[] nums) {
        if (nums == null || nums.length <= 0)
        		return 0;
        
        int[][] dp = new int[nums.length][2];
        dp[0][0] = 1;
        dp[0][1] = 1;
        int max = 1;
        int maxCnt = 1;
        for (int i = 1; i < nums.length; ++i) {
        		dp[i][0] = 1;
        		dp[i][1] = 1;
        		for (int j = i - 1; j >= 0; j--) {
        			if (nums[i] > nums[j]) {
        				if (dp[i][0] < dp[j][0] + 1) {
        					dp[i][0] = dp[j][0] + 1;
        					dp[i][1] = 1;
        				} else if (dp[i][0] == dp[j][0] + 1){
        					dp[i][1]++;
        				}
        			}
        		}
        		
        		if (dp[i][0] > max) {
        			max = dp[i][0];
        			maxCnt = dp[i][1];
        		} else if (dp[i][0] == max) {
        			maxCnt += dp[i][1];
        		}
        }
        
        return maxCnt;
    }
    
    // https://leetcode.com/problems/partition-equal-subset-sum/description/
    boolean[] partitionResult = {false};
    public boolean canPartition(int[] nums) {
    		if (nums == null || nums.length <= 0)
    			return false;
    		
    		int sums = 0;
    		for (int num : nums) {
    			sums += num;
    		}
    		
    		if ((sums & 1) != 0) {
    			return false;
    		}
    		
    		int average = sums >> 1;
    		partitionResult[0] = false;
    		
    		for (int num : nums) {
    			if (num > average) {
    				return false;
    			}
    		}
    		Arrays.sort(nums);
    		
    		canPartition(nums, nums.length - 1, 0, average);
    		
    		return partitionResult[0];
    }
    
    private void canPartition(int[] nums, int start, int sum, int average) {
    		if (start < 0 || partitionResult[0]) {
    			return;
    		}
    		
    		if (sum + nums[start] == average) {
    			partitionResult[0] = true;
    			return;
    		}
    		
    		if (sum + nums[start] > average) {
    			canPartition(nums, start - 1, sum, average);
    		} else {
    			canPartition(nums, start - 1, sum + nums[start], average);
        		if (partitionResult[0])
        			return;
        		
        		canPartition(nums, start - 1, sum, average);
    		}    	
    }
    
    // https://leetcode.com/problems/partition-to-k-equal-sum-subsets/description/
    public boolean canPartitionKSubsets(int[] nums, int k) {
        if (k <= 1)
        		return true;
        
        if (k > nums.length)
        		return false;
        
        int sum = 0;
        for (int num : nums) {
        		sum += num;
        }
        
        if (sum % k != 0)
        		return false;
        
        int average = sum / k;
        
        Arrays.sort(nums);
        
        resKSubsets[0] = false;
        
        canPartitionKSubsets(nums, average, new int[k], nums.length - 1);
        
        return resKSubsets[0];
    }
    
    private boolean[] resKSubsets = {false};
    private void canPartitionKSubsets(int[] nums, int average, int[] sums, int start) {
    		if (resKSubsets[0])
    			return;
    		
    		if (start < 0) {
    			resKSubsets[0] = true;
    			for (int i = 0; i < sums.length; ++i) {
    				if (sums[i] != average) {
    					resKSubsets[0] = false;
    					break;
    				}
    			}
    			
    			return;
    		}
    		
    		for (int i = 0; i < sums.length; ++i) {
    			if (sums[i] + nums[start] <= average) {
    				sums[i] += nums[start];
    				canPartitionKSubsets(nums, average, sums, start - 1);
    				sums[i] -= nums[start];
    			}
    		}
    		
    }
    
    // https://leetcode.com/problems/maximum-length-of-pair-chain/description/
    public int findLongestChain(int[][] pairs) {
    		if (pairs == null || pairs.length <= 0)
    			return 0;
    		
    		Arrays.sort(pairs, new Comparator<int[]>() {
    			@Override
    			public int compare(int[] o1, int[] o2) {
    				// TODO Auto-generated method stub
    				if (o1[0] < o2[0])
    					return -1;
    				else if (o1[0] > o2[0])
    					return 1;
    				else
    					return 0;
    			}
		});
    		
    		int[] dp = new int[pairs.length];
    		dp[0] = 1;
    		int max = 1;
    		for (int i = 1; i < pairs.length; ++i) {
    			dp[i] = 1;
    			for (int j = i - 1; j >= 0; --j) {
    				if (pairs[j][1] < pairs[i][0])
    					dp[i] = Math.max(dp[j] + 1, dp[i]);
    			}
    			
    			max = Math.max(max, dp[i]);
    		}
    		
    		return max;
    }

    
    // https://leetcode.com/problems/shopping-offers/description/
    public int shoppingOffers(List<Integer> price, List<List<Integer>> special, List<Integer> needs) {
        if (price == null || price.size() <= 0 || needs == null || needs.size() <= 0)
        		return 0;
        
        int[] prices = new int[price.size()];
        for (int i = 0; i < price.size(); ++i)
        		prices[i] = price.get(i);
        
        int[] needss = new int[needs.size()];
        for (int i = 0; i < price.size(); ++i) {
        		needss[i] = needs.get(i);
        }
        
        if (special != null) {
        		Iterator<List<Integer>> it = special.iterator();
        		while (it.hasNext()) {
        			List<Integer> offers = it.next();
        			int result = 0;
        			for (int i = 0; i < needss.length; ++i) {
        				result += offers.get(i) * prices[i];
        			}
        			
        			if (result <= offers.get(needss.length)) {
        				it.remove();
        			}
        		}
        }
        return shoppingOffers(prices, special, needss);
    }
    
    private int shoppingOffers(int[] price, List<List<Integer>> special, int[] needs) {
    		if (isEmpty(needs))
    			return 0;
    		
    		boolean canConsume = false;
    		int min = Integer.MAX_VALUE;
    		for (int i = 0; special != null && i < special.size(); ++i) {
    			List<Integer> offers = special.get(i);
    			if (canConsume(needs, offers)) {
    				canConsume = true;
    				consumeOffer(needs, offers);
    				min = Math.min(min, offers.get(price.length) + shoppingOffers(price, special, needs));
    				unconsumeOffer(needs, offers);
    			}
    		}
    		
    		if (canConsume) {
    			return min;
    		}
    		
		int result = 0;
		for (int i = 0; i < needs.length; ++i) {
			if (needs[i] > 0) {
				result += needs[i] * price[i];
			}
		}
    		
    		return result;
    }
    
    private boolean canConsume(int[] needs, List<Integer> special) {
    		for (int i = 0; i < needs.length; ++i) {
    			int offer = special.get(i);
    			if (offer != 0) {
    				if (offer > needs[i])
    					return false;
    			}
    		}
    		
    		return true;
    }
    
    private void consumeOffer(int[] needs, List<Integer> special) {
    		for (int i = 0; i < needs.length; ++i) {
    			int offer = special.get(i);
    			if (offer != 0) {
    				needs[i] -= offer;
    			}
		}
    }
    
    private void unconsumeOffer(int[] needs, List<Integer> special) {
    		for (int i = 0; i < needs.length; ++i) {
    			int offer = special.get(i);
			if (offer != 0) {
				needs[i] += offer;
			}
		}
    }
    
    private boolean isEmpty(int[] needs) {
    		for (int need : needs) {
    			if (need != 0)
    				return false;
    		}
    		return true;
    }
    
    // https://leetcode.com/problems/01-matrix/description/
    public int[][] updateMatrix(int[][] matrix) {
    		if (matrix == null || matrix.length <= 0 || matrix[0] == null || matrix[0].length <= 0)
    			return matrix;
    		
    		for (int i = 0; i < matrix.length; ++i) {
    			for (int j = 0; j < matrix[0].length; ++j) {
    				if (matrix[i][j] != 0) {
    					matrix[i][j] = getMatrixMinDistance(matrix, i, j);
    				}
    			}
    		}
    		
    		return matrix;
    }
    
    private class MatrixNode {
    		int row;
    		int column;
    		
    		public MatrixNode(int row, int column) {
			// TODO Auto-generated constructor stub
    			this.row = row;
    			this.column = column;
		}
    		
    		@Override
    		public boolean equals(Object obj) {
    		// TODO Auto-generated method stub
    			return row == ((MatrixNode) obj).row && column == ((MatrixNode) obj).column;
    		}
    }
    
    private int getMatrixMinDistance(int[][] matrix, int row, int column) {
    		final int rows = matrix.length;
    		final int columns = matrix[0].length;
    		
    		if (row < 0 || row >= rows || column < 0 || column >= columns) {
    			return -1;
    		}
    		
    		if (matrix[row][column] == 0)
    			return 0;
    		
    		MatrixNode start = new MatrixNode(row, column);
    		
    		Set<MatrixNode> visited = new HashSet<>();
    		visited.add(start);
    		
    		Queue<MatrixNode> queue = new LinkedList<>();
    		queue.add(start);
    		
    		int level = 0;
    		int cur = 1;
    		int next = 0;
    		
    		int[][] direction = {
    				{-1, 0},
    				{1, 0},
    				{0, -1},
    				{0, 1}
    		}; 
    		
    		while (!queue.isEmpty()) {
    			MatrixNode node = queue.poll();
    			cur--;
    			
    			for (int i = 0; i < direction.length; ++i) {
    				int nrow = node.row + direction[i][0];
    				int ncolumn = node.column + direction[i][1];
    				if (nrow >= 0 && nrow < rows && ncolumn >= 0 && ncolumn < columns) {
    					if (matrix[nrow][ncolumn] == 0)
    						return level;
    					else {
    						MatrixNode nextNode = new MatrixNode(nrow, ncolumn);
    						if (!visited.contains(nextNode)) {
    							visited.add(nextNode);
    							queue.add(nextNode);
    							next++;
    						}
    					}
    				}
    			}
    			
    			if (cur == 0) {
        			cur = next;
        			next = 0;
        			level++;
        		}
    		}
    		
    		return -1;
    }
    
    // https://leetcode.com/problems/longest-word-in-dictionary-through-deleting/description/
    public String findLongestWord(String s, List<String> d) {
    		int min = Integer.MAX_VALUE;
    		String minWord = "";
    		for (String word : d) {
    			if (containWord(s, word)) {
    				if (min > s.length() - word.length()) {
    					min = s.length() - word.length();
    					minWord = word;
    				} else if (min == s.length() - word.length()) {
    					if ("".equals(minWord)) minWord = word;
    					else minWord = minWord.compareTo(word) <= 0 ? minWord : word; 
    				}
    			}
    		}
    		
    		return minWord;
    }
    
    private boolean containWord(CharSequence s, CharSequence word) {
    		if (s.length() < word.length())
    			return false;
    		
    		int widx = 0;
    		int sidx = 0;
    		while (sidx < s.length() && widx < word.length()) {
    			if (s.charAt(sidx) == word.charAt(widx)) {
    				widx++;
    			}
    			sidx++;
    		}
    		
    		return widx == word.length();
    }
    
    // https://leetcode.com/problems/coin-change-2/description/
    public int change(int amount, int[] coins) {
        return changeDp(amount, coins);
    }
    
    private int change(int amount, int sum, int[] coins, int start) {
    		if (sum > amount) {
			return 0;
		} else if (sum == amount) {
			return 1;
		}
    		
    		if (start >= coins.length) {
    			return 0;
    		}
    	
    		int result = 0;
    		for (int i = 0; i <= (amount - sum) / coins[start]; ++i) {
    			result += change(amount, sum + i * coins[start], coins, start + 1);
    		}
    		
    		return result;
    }
    
    public int changeDp(int amount, int[] coins) {
    		if (coins == null || coins.length <= 0)
    			return amount == 0 ? 1 : 0;
    		
    		int[][] dp = new int[coins.length + 1][amount + 1];
    		for (int i = 0; i <= coins.length; ++i) {
    			dp[i][0] = 1;
    		}
    		
    		for (int i = 1; i <= coins.length; ++i) {
    			for (int j = 1; j <= amount; ++j) {
    				dp[i][j] = dp[i-1][j];
    				if (j >= coins[i-1]) {
    					dp[i][j] += dp[i][j - coins[i-1]];
    				}
    			}
    		}
    		
    		return dp[coins.length][amount];
    }
    
    // https://leetcode.com/problems/minimum-genetic-mutation/description/
    public int minMutation(String start, String end, String[] bank) {
    		if (bank == null || bank.length <= 0)
    			return -1;
    		Set<String> set = new HashSet<>();
    		for (String word : bank) {
    			set.add(word);
    		}
    		
    		if (!set.contains(end)) {
    			return -1;
    		}
    		
    		char[] change = {'A', 'C', 'G', 'T'};
    		Queue<String> queue = new LinkedList<>();
    		queue.add(start);
    		
    		int level = 0;
    		while (!queue.isEmpty()) {
    			int size = queue.size();
    			for (int i = 0; i < size; ++i) {
    				String head = queue.poll();
    				
    				for (int j = 0; j < head.length(); ++j) {
    					for (int k = 0; k < change.length; ++k) {
    						if (change[k] == head.charAt(j))
    							continue;
						String gene = head.substring(0, j) + String.valueOf(change[k]) + head.substring(j + 1);
						if (!set.contains(gene))
							continue;
						if (gene.equals(end))
							return level + 1;
						else {
							queue.add(gene);
							set.remove(gene);
						}
    					}
    				}
    			}
    			level++;
    		}
    		
    		return -1;
    }
    
    // https://leetcode.com/problems/target-sum/description/
    public int findTargetSumWays(int[] nums, int S) {
        return findTargetSumWays(nums, S, nums.length - 1);
    }
    
    private int findTargetSumWays(int[] nums, int S, int end) {
    		int result = 0;
    		if (end == 0) {
    			if (nums[0] == S) 
    				result++;
    			if (nums[0] == -S)
    				result++;
    			
    			return result;
    		}
    		
    		
    		result += findTargetSumWays(nums, S + nums[end], end - 1);
    		result += findTargetSumWays(nums, S - nums[end], end - 1);
    		
    		return result;
    }
    
    // https://leetcode.com/problems/increasing-subsequences/description/
//    public List<List<Integer>> findSubsequences(int[] nums) {
//    		return null;
//    }
    
    // https://leetcode.com/problems/predict-the-winner/description/
    public boolean PredictTheWinner(int[] nums) {
    		if (nums == null || nums.length <= 1)
    			return true;
    		
    		int sums = 0;
    		for (int num : nums) sums += num;
    		
    		int[][] dp = new int[nums.length][nums.length];
    		for (int j = 0; j < nums.length; ++j) {
    			int curSum = 0;
    			for (int i = j; i >= 0; --i) {
    				curSum += nums[i];
    				if (i == j) {
    					dp[i][j] = nums[i];
    				} else {
    					dp[i][j] = Math.max(curSum - dp[i+1][j], 
    							curSum - dp[i][j-1]);
    				}
    			}
    		}
    		
    		return dp[0][nums.length-1] >= sums - dp[0][nums.length - 1]; 
    }
    
    // https://leetcode.com/problems/can-i-win/description/
    public boolean canIWin(int maxChoosableInteger, int desiredTotal) {
        int[] nums = new int[maxChoosableInteger];
        for (int i = 0; i < maxChoosableInteger; ++i) {
        	nums[i] = i + 1;
		}

		WinResult[0] = true;

		canIWin(nums, 0, 0, desiredTotal, true);

        return WinResult[0];
    }

	private void swap2(int[] nums, int one, int two) {
		int temp = nums[one];
		nums[one] = nums[two];
		nums[two] = temp;
	}

	private boolean WinResult[] = {true};
    private void canIWin(int[] nums, int idx, int sum, int desiredTotal, boolean first) {
    	if (!WinResult[0]) {
    		return;
		}

    	for (int i = idx; i < nums.length; ++i) {
    		swap(nums, i, idx);
    		if (sum + nums[i] >= desiredTotal) {
    			if (!first)
    				WinResult[0] = false;
			} else {
				canIWin(nums, idx + 1, sum + nums[i], desiredTotal, !first);
			}

			swap(nums, i, idx);
		}
	}
    
    // https://leetcode.com/problems/remove-k-digits/description/
    public String removeKdigits(String num, int k) {
    		if (k >= num.length())
    			return "";
    		if (k <= 0)
    			return num;
    		
    		Stack<Character> stack = new Stack<>();
    		int cnt = 0;
    		int i = 0;
    		for (i = 0; i < num.length() && cnt < k; ++i) {
    			final char c = num.charAt(i);
    			while (cnt < k && !stack.isEmpty() && stack.peek() > c) {
    				cnt++;
    				stack.pop();
    			}
    			stack.push(c);
    		}
    		
    		while (cnt++ < k) {
    			stack.pop();
    		}
    		
    		
    		StringBuilder builder = new StringBuilder();
    		for (int j = 0; j < stack.size(); ++j) {
    			builder.append(stack.get(j));
    		}
    		if (i < num.length()) {
    			builder.append(num.substring(i));
    		}
    		
    		String result = builder.toString();
    		int j = 0;
    		for (j = 0; j < result.length(); ++j) {
    			if (result.charAt(j) != '0')
    				break;
    		}
    		
    		if (j >= result.length()) {
    			return "0";
    		} else {
    			return result.substring(j);
    		}
    }
    
    // https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/description/
    public int longestSubstring(String s, int k) {
    		return 0;
    }

	// https://leetcode.com/problems/valid-parenthesis-string/description/
	public boolean checkValidStringII(String s) {
    	if (s == null || s.length() <= 0)
    		return true;

    	int cnt = 0;
    	int cp = 0;

    	for (int i = 0; i < s.length(); ++i) {
    		final char c = s.charAt(i);
    		if (c == '(') {
    			cnt++;
    			cp++;
			} else if (c == ')') {
    			if (cnt > 0) {
    				cnt--;
				}
				cp--;
			} else {
    			if (cnt > 0) {
    				cnt--;
				}
    			cp++;
			}

			if (cp < 0)
				return false;
		}

		return cnt == 0;
	}

	// https://leetcode.com/problems/coin-change/description/
	public int coinChange(int[] coins, int amount) {
		return coinChange(coins, 0, amount);
	}

	private int coinChange(int[] coins, int start, int amount) {
    	if (amount == 0) {
    		return 0;
		}

		if (start >= coins.length) {
    		return -1;
		}

		int min = Integer.MAX_VALUE;

    	int maxCnt = amount / coins[start];

    	for (int i = 0; i <= maxCnt; ++i) {
    		int tempMin = coinChange(coins, start + 1, amount - coins[start] * i);
    		if (tempMin != -1) {
    			min = Math.min(min, tempMin + i);
			}
		}

		return min == Integer.MAX_VALUE ? -1 : min;
	}

	public int coinChangeII(int[] coins, int amount) {
    	if (amount < 0) return -1;
    	else if (amount == 0) return 0;

    	int[][] dp = new int[coins.length + 1][amount + 1];
		dp[0][0] = 0;

		for (int j = 1; j <= amount; ++j) {
			dp[0][j] = -1;
		}

		for (int i = 1; i <= coins.length; i++) {
			dp[i][0] = 0;
		}

		for (int i = 1; i <= coins.length; ++i) {
			for (int j = 1; j <= amount; ++j) {
				int min = dp[i-1][j];
				if (j >= coins[i-1]) {
					int tempMin = dp[i][j - coins[i-1]];
					if (tempMin != -1) {
						tempMin += 1;
					}

					if (tempMin != -1) {
						if (min == -1) {
							min = tempMin;
						} else {
							min = Math.min(min, tempMin);
						}
					}
				}

				dp[i][j] = min;
			}
		}

		return dp[coins.length][amount];

	}

	// https://leetcode.com/problems/word-ladder-ii/description/
	public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {

		Queue<String> queue = new LinkedList<>();

		queue.add(beginWord);

		final Map<String, Integer> matchedWords = new HashMap<>();

		final Map<String, List<String>> prefixs = new HashMap<>();

		final Set<String> words = new HashSet<>(wordList);

		if (!words.contains(endWord)) {
			return new LinkedList<>();
		}

		int cur = 1;
		int next = 0;
		int level = 0;

		while (!queue.isEmpty()) {
			String head = queue.poll();



			for (int i = 0; i < head.length(); ++i) {
				final char c = head.charAt(i);

				for (char j = 'a'; j <= 'z'; ++j) {
					if (j == c) {
						continue;
					}

					final String newStr = head.substring(0, i) + j + head.substring(i+1);

					if (newStr.equals(endWord)) {
						List<String> lists = prefixs.get(endWord);
						if (lists == null) {
							lists = new LinkedList<>();
							prefixs.put(endWord, lists);
						}
						lists.add(head);
					} else {
						int matchedLevel = matchedWords.getOrDefault(newStr, -1);
						if ((-1 != matchedLevel) && (level > matchedLevel)) {
							// 不做任何处理
						} else if (words.contains(newStr)) {
							matchedWords.put(newStr, level);

							List<String> lists = prefixs.get(newStr);
							if (lists == null) {
								lists = new LinkedList<>();
								prefixs.put(newStr, lists);
							}
							lists.add(head);

							if (-1 == matchedLevel) {
								queue.add(newStr);
								next++;
							}
						}
					}



				}
			}

			cur--;
			if (cur == 0) {
				cur = next;
				next = 0;
				level++;

				if (prefixs.get(endWord) != null) {
					break;
				}
			}
		}

		// 根据保存的结果回溯出结果

		final List<List<String>> result = findLadders(prefixs, endWord, beginWord);

		print(result);

		return result;
	}

	public List<List<String>> findLadders(Map<String, List<String>> prefixs, String start, String end) {
    	List<List<String>> result = new LinkedList<>();

    	if (start.equals(end)) {
    		result.add(new LinkedList<String>(Arrays.asList(start)));
    		return result;
		}

		if (prefixs.get(start) == null) {
    		return result;
		}

    	for (String prefix : prefixs.get(start)) {
			List<List<String>> temp = findLadders(prefixs, prefix, end);

			for (List<String> one : temp) {
				one.add(start);
				result.add(one);
			}
		}

		return result;
	}

	private void print(List<List<String>> result) {
    	for (List<String> list : result) {
    		for (String one : list) {
    			System.out.print(one + " ");
			}

			System.out.println();
		}
	}

	// https://leetcode.com/problems/strong-password-checker/description/
	public int strongPasswordChecker(String s) {
		return 0;
	}

	// https://leetcode.com/problems/reverse-pairs/description/
	public int reversePairs(int[] nums) {
    	if (nums == null || nums.length <= 1) {
    		return 0;
		}
		return reversePairs(nums, 0, nums.length - 1, new int[nums.length]);
	}

	private int reversePairs(int[] nums, int start, int end, int[] temp) {
    	if (start >= end) {
    		return 0;
		}

		int middle = start + (end - start) / 2;

    	int leftCnt = reversePairs(nums, start, middle, temp);
    	int rightCnt = reversePairs(nums, middle + 1, end, temp);

    	// 能否在排序的同时，将个数也算出来呢
		int leftStart = start;
		int leftEnd = middle;
		int rightStart = middle + 1;
		int rightEnd = end;
		int idx = start;
		int k = rightStart;
		int cnt = 0;
		int res = 0;
		while (leftStart <= leftEnd) {

			while (k <= rightEnd && (long)nums[k] * 2 < (long)nums[leftStart]) {
				cnt++;
				k++;
			}

			res += cnt;

			while (rightStart <= rightEnd && nums[rightStart] < nums[leftStart]) {
				temp[idx++] = nums[rightStart++];
			}

			temp[idx++] = nums[leftStart++];


		}

		System.arraycopy(temp, start, nums, start, idx - start);

		return res + leftCnt + rightCnt;
	}

	// https://leetcode.com/problems/count-of-smaller-numbers-after-self/description/
	public List<Integer> countSmaller(int[] nums) {
		List<Integer> result = new LinkedList<>();
		if (nums == null || nums.length <= 0) {
			return result;
		}

		int[] idxs = new int[nums.length];
		for (int i = 0; i < idxs.length; ++i) {
			idxs[i] = i;
		}
		int[] resultArray = new int[nums.length];

		countSmaller(nums, idxs, 0, nums.length - 1, resultArray);

		for (int one : resultArray) {
			result.add(one);
		}

		return result;
	}

	public void countSmaller(int[] nums, int [] idxs, int start, int end, int[] result) {
    	if (start >= end) {
    		return;
		}

		int middle = start + (end - start) / 2;

    	countSmaller(nums, idxs, start, middle, result);
    	countSmaller(nums, idxs, middle + 1, end, result);

    	int leftStart = start;
    	int rightStart = middle + 1;
    	int k = rightStart;
    	int[] tempIdxs = new int[end - start + 1];
    	int tempIdx = 0;

    	int cnt = 0;
    	while (leftStart <= middle) {

    		while (k <= end && nums[idxs[k]] < nums[idxs[leftStart]]) {
    			k++;
    			cnt++;
			}

			result[idxs[leftStart]] += cnt;

    		while (rightStart <= end && nums[idxs[rightStart]] < nums[idxs[leftStart]]) {
    			tempIdxs[tempIdx++] = idxs[rightStart++];
			}

			tempIdxs[tempIdx++] = idxs[leftStart++];
		}

		System.arraycopy(tempIdxs, 0, idxs, start, tempIdx);
	}

	// https://leetcode.com/problems/queue-reconstruction-by-height/description/
	public int[][] reconstructQueue(int[][] people) {
		if (people == null || people.length <= 1) {
			return people;
		}

		int[] idxs = new int[people.length];
		for (int i = 0; i < idxs.length; ++i) {
			idxs[i] = i;
		}

		int[] decrements = new int[people.length];

		for (int i = 0; i < people.length - 1; ++i) {

			// 找出最小的，先根据k排序，再根据h排序
			int minIdx = i;
			for (int j = i + 1; j < people.length; ++j) {
				final int minV = people[idxs[minIdx]][1] - decrements[idxs[minIdx]];
				final int tempV = people[idxs[j]][1] - decrements[idxs[j]];
				if (minV < tempV) {
					continue;
				} else if (minV > tempV) {
					minIdx = j;
				} else if (people[idxs[minIdx]][0] > people[idxs[j]][0]) {
					minIdx = j;
				}
			}

			if (minIdx != i) {
				int tempIdx = idxs[minIdx];
				idxs[minIdx] = idxs[i];
				idxs[i] = tempIdx;
			}

			// 依次更新i后面的decrement的值

			for (int j = i + 1; j < people.length; ++j) {
				if (people[idxs[j]][0] <= people[idxs[i]][0]) {
					decrements[idxs[j]] += 1;
				}
			}
		}

		int[][] result = new int[people.length][2];

		for (int i = 0; i < people.length; i++) {
			result[i][0] = people[idxs[i]][0];
			result[i][1] = people[idxs[i]][1];
		}

		return result;
	}

	// https://leetcode.com/problems/create-maximum-number/description/
	public int[] maxNumber(int[] nums1, int[] nums2, int k) {
    	int[] temp = new int[k];
    	int[] result = new int[k];
    	int[] res1 = new int[Math.min(nums1.length, k)];
    	int[] res2 = new int[Math.min(nums2.length, k)];
		for (int i = Math.max(0, k - nums2.length); i <= Math.min(nums1.length, k); ++i) {
			// 第一个去i个，第二个取k - i个
			getMaxNumber(nums1, res1, i);
			getMaxNumber(nums2, res2, k - i);
			getMaxNumber(res1, i, res2, k - i, temp);
			if (compareArray(temp, 0, result, 0) > 0) {
				System.arraycopy(temp, 0, result, 0, k);
			}
		}
		return result;
	}

	private int compareArray(int[] one, int startOne, int[] another, int startAnother) {
    	int i = startOne;
    	int j = startAnother;
    	while (i < one.length && j < another.length) {
    		if (one[i] < another[j]) return -1;
    		else if (one[i] > another[j]) return 1;
    		i++;
    		j++;
		}

		if (i >= one.length && j >= another.length) return 0;
		if (i < one.length) {
			return compareArray(one, i, another, startAnother);
		} else {
			return compareArray(one, startOne, another, j);
		}
//			int k = 0;
//			int l = another.length - startAnother;
//			while (i < one.length) {
//				if (one[i] > another[startAnother + k % l]) return 1;
//				else if (one[i] < another[startAnother + k % l]) return -1;
//				k++;
//				i++;
//			}
//		} else {
//			int k = 0;
//			int l = one.length - startOne;
//			while (i < one.length) {
//				if (another[i] > one[startOne + k % l]) return 1;
//				else if (another[i] < one[startOne + k % l]) return -1;
//				k++;
//				i++;
//			}
//		}
	}

	// 从nums里面，取出k个值，组成最大的数字
	public void getMaxNumber(int[] nums, int[] result, int k) {
    	if (k <= 0) {
    		return;
		} else if (k >= nums.length) {
			System.arraycopy(nums, 0, result, 0, nums.length);
			return;
		}

		// 模拟一个简单的栈
		int resIdx = 0; // 下一次需要写入的index
		for (int i = 0; i < nums.length; ++i) {
			if (resIdx == 0) {
				result[resIdx++] = nums[i];
			} else if (nums[i] <= result[resIdx - 1] && resIdx < k) {
				result[resIdx++] = nums[i];
			} else if (nums[i] > result[resIdx - 1]){
				while (resIdx >= 1 && nums[i] > result[resIdx - 1] && (nums.length - i) >= (k - resIdx + 1)) {
					resIdx--;
				}
				result[resIdx++] = nums[i];
			}
		}
	}

	public void getMaxNumber(int[] nums1, int cnt1, int[] nums2, int cnt2, int[] result) {
    	int i = 0;
    	int j = 0;
    	int k = 0;
    	while (i < cnt1 && j < cnt2) {
    		if (compareArray(nums1, i, nums2, j) <= 0) result[k++] = nums2[j++];
    		else result[k++] = nums1[i++];
		}
		while (i < cnt1) result[k++] = nums1[i++];
    	while (j < cnt2) result[k++] = nums2[j++];
	}

	public int scompare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
		if (o1.getValue() > o2.getValue()) {
			return 1;
		} else if (o1.getValue() < o2.getValue()) {
			return -1;
		} else {
			return o2.getKey().compareTo(o1.getKey());
		}
	}

	// https://leetcode.com/problems/top-k-frequent-words/description/
	public List<String> topKFrequent(String[] words, int k) {
		List<String> result = new LinkedList<>();
		if (k <= 0 || words == null || words.length <= 0 || k > words.length) {
			return result;
		}

		Map<String, Integer> map = new HashMap<>();
		for (String word : words) {
			map.put(word, map.getOrDefault(word, 0) + 1);
		}

		// 最小堆
		PriorityQueue<Map.Entry<String, Integer>> queue = new PriorityQueue<>(new Comparator<Map.Entry<String, Integer>>() {
			@Override
			public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
				return scompare(o1, o2);
			}
		});

		for (Map.Entry<String, Integer> entry : map.entrySet()) {
			if (queue.size() < k) {
				queue.add(entry);
			} else {
				Map.Entry head = queue.peek();
				if (scompare(entry, head) > 0) {
					queue.poll();
					queue.add(entry);
				}
			}
		}

		while(!queue.isEmpty()) {
			result.add(0, queue.poll().getKey());
		}

		return result;
	}


	// https://leetcode.com/problems/sliding-window-maximum/description/
	public int[] maxSlidingWindow(int[] nums, int k) {
		if (nums == null || nums.length <= 0) {
			return new int[0];
		}

		int[] result = new int[nums.length - k + 1];

		LinkedList<Integer> queue = new LinkedList<>();

		for (int i = 0; i < nums.length; ++i) {
			while (!queue.isEmpty() && nums[queue.getLast()] < nums[i]) {
				queue.removeLast();
			}

			queue.addLast(i);

			if (i >= k-1) {
				result[i-k+1] = nums[queue.getFirst()];
			}

			while (i < nums.length - 1 && !queue.isEmpty() && queue.getFirst() <= i + 1 - k) {
				queue.removeFirst();
			}
		}

		return result;
	}

	// https://leetcode.com/problems/minimum-window-substring/description/
	public String minWindow(String s, String t) {
		int[] ht = new int[256];
		int[] st = new int[256];
		int cntt = 0;
		for (int i = 0; i < t.length(); ++i) {
			ht[t.charAt(i)]++;
			cntt++;
		}

		int tempcnt = 0;
		int start = 0;
		int end = 0;

		int minLength = Integer.MAX_VALUE;
		int minStart = 0;
		int minEnd = 0;
		while (end < s.length()) {
			char c = s.charAt(end);
			if (st[c] < ht[c]) {
				tempcnt++;
			}
			st[c]++;

			while (tempcnt >= cntt && start <= end) {
				if (end - start + 1 < minLength) {
					minLength = end - start + 1;
					minStart = start;
					minEnd = end;
				}

				c = s.charAt(start);
				if (ht[c] > 0) {
					if (st[c] <= ht[c]) {
						tempcnt--;
					}
					st[c]--;
				}

				start++;

			}

			end++;

		}

		if (minLength == Integer.MAX_VALUE) {
			return "";
		} else
			return s.substring(minStart, minEnd + 1);
	}


	//https://leetcode.com/problems/permutation-in-string/description/
	public boolean checkInclusion(String s1, String s2) {
    	// s1是要查找的关键字
		if (s1.length() > s2.length()) {
			return false;
		}

		int[] hash2 = new int[26];
		int cnt2 = 0;

		int[] hash1 = new int[26];
		int cnt1 = s1.length();
		for (int i = 0; i < s1.length(); ++i) {
			hash1[s1.charAt(i) - 'a']++;
		}

		int start = 0;
		int end = 0;

		while (end < s2.length()) {
			final int idx = s2.charAt(end) - 'a';

			if (hash1[idx] == 0) {
				end++;
				start = end;
				cnt2 = 0;
				Arrays.fill(hash2, 0);
				continue;
			}

			while (start <= end && hash2[idx] + 1 > hash1[idx]) {
				hash2[s2.charAt(start) - 'a']--;
				cnt2--;
				start++;
			}
			hash2[idx]++;
			cnt2++;

			if (cnt1 == cnt2 && Arrays.equals(hash1, hash2)) {
				return true;
			}
			end++;
		}

		return false;

	}

	// https://leetcode.com/problems/substring-with-concatenation-of-all-words/description/
	public List<Integer> findSubstring(String s, String[] words) {
		List<Integer> result = new LinkedList<>();
		if (s == null || s.isEmpty() || words == null || words.length <= 0) {
			return result;
		}

		int[] hashWords = new int[26];
		int cntWords = 0;

		int[] hashSrc = new int[26];
		int cntSrc = 0;

		for (String word : words) {
			for (int i = 0; i < word.length(); ++i) {
				hashWords[word.charAt(i) - 'a']++;
				cntWords++;
			}
		}

		int start = 0;
		int end = 0;
		Map<String, Boolean> cacheResult = new HashMap<>();

		while (end < s.length()) {
			final int idx = s.charAt(end) - 'a';

			if (hashWords[idx] == 0) {
				end++;
				start = end;
				cntSrc = 0;
				Arrays.fill(hashSrc, 0);
				continue;
			}

			while (start <= end && hashSrc[idx] + 1 > hashWords[idx]) {
				hashSrc[s.charAt(start) - 'a']--;
				cntSrc--;
				start++;
			}
			hashSrc[idx]++;
			cntSrc++;

			if (cntSrc == cntWords && Arrays.equals(hashWords, hashSrc)) {
				// hash的总和是相等的
				final String temp = s.substring(start, end + 1);
				final Boolean cacheRes = cacheResult.get(temp);
				boolean res;
				if (cacheRes == null) {
					res = consistOfWords(s, start, words, 0);
					cacheResult.put(temp, res);
				} else {
					res = cacheRes;
				}

				if (res) {
					result.add(start);
				}
			}

			end++;
		}

		return result;

	}

	public boolean consistOfWords(CharSequence s, int start, String[] words, int j) {
    	if (j >= words.length) {
    		return true;
		}
		System.out.println("start: " + start + ", j: " + j);
    	for (int i = j; i < words.length; ++i) {
    		final String word = words[i];

    		boolean res = false;
    		if (word.equals(s.subSequence(start, start + word.length()))) {
				swapWord(words, j, i);
				res = consistOfWords(s, start + word.length(), words, j+1);
				swapWord(words, j, i);

				if (res) {
					return true;
				}
			}
		}

		return false;
	}

	private void swapWord(String[] words, int i, int j) {
    	final String temp = words[i];
    	words[i] = words[j];
    	words[j] = temp;
	}


	// https://leetcode.com/problems/unique-paths-ii/description/
	public int uniquePathsWithObstacles(int[][] obstacleGrid) {
		if (obstacleGrid == null || obstacleGrid.length <= 0 ||
				obstacleGrid[0] == null || obstacleGrid[0].length <= 0) {
			return 0;
		}

		final int rows = obstacleGrid.length;
		final int columns = obstacleGrid[0].length;
		int[][] dp = new int[rows][columns];

		dp[0][0] = obstacleGrid[0][0] == 0 ? 1 : 0;
		if (dp[0][0] == 0) {
			return 0;
		}

		for (int i = 1; i < columns; ++i) {
			if (obstacleGrid[0][i] == 1) break;
			else dp[0][i] = 1;
		}

		for (int i = 1; i < rows; ++i) {
			if (obstacleGrid[i][0] == 1) break;
			else dp[i][0] = 1;
		}

		for (int i = 1; i < rows; ++i) {
			for (int j = 1; j < columns; ++j) {
				if (obstacleGrid[i][j] == 0) {
					dp[i][j] = dp[i][j - 1] + dp[i-1][j];
				}
			}
		}

		return dp[rows-1][columns-1];
	}

	// https://leetcode.com/problems/dungeon-game/description/
	public int calculateMinimumHP(int[][] dungeon) {
		if (dungeon == null || dungeon.length <= 0 ||
				dungeon[0] == null || dungeon[0].length <= 0) {
			return 0;
		}

		final int rows = dungeon.length - 1;
		final int columns = dungeon[0].length - 1;

		final int[][] dp = new int[rows+1][columns+1];

		if (dungeon[rows][columns] >= 0) {
			dp[rows][columns] = 1;
		} else {
			dp[rows][columns] = 1 - dungeon[rows][columns];
		}

		for (int i = columns - 1; i >= 0; --i) {
			dp[rows][i] = dungeon[rows][i] >= dp[rows][i+1] ? 1 : dp[rows][i+1] - dungeon[rows][i];
		}

		for (int i = rows - 1; i >= 0; --i) {
			dp[i][columns] = dungeon[i][columns] >= dp[i+1][columns] ? 1 : dp[i+1][columns] - dungeon[i][columns];
		}

		for (int i = rows - 1; i >= 0; --i)
			for (int j = columns - 1; j >= 0; --j) {
				// 先取右边的
				dp[i][j] = dungeon[i][j] >= dp[i][j+1] ? 1 : dp[i][j+1] - dungeon[i][j];
				dp[i][j] = Math.min(dp[i][j], dungeon[i][j] >= dp[i+1][j] ? 1 : dp[i+1][j] - dungeon[i][j]);
			}

		return dp[0][0];
	}

	public int calculateMinimumHP2(int[][] dungeon) {
    	dfsMinHp(dungeon, 0, 0, 0, 0);
    	return minHpResult;
	}

	int minHpResult = Integer.MAX_VALUE;
	private void dfsMinHp(int[][] dungeon, int row, int column, int minHp, int left) {
		if (minHp > minHpResult) {
			return;
		}

		if (left + dungeon[row][column] >= 1) {
			left = left + dungeon[row][column];
		} else {
			minHp += 1 - left - dungeon[row][column];
			left = 1;
		}
    	if (row >= dungeon.length - 1 && column >= dungeon[0].length - 1) {
			minHpResult = Math.min(minHp, minHpResult);
		} else if (row < dungeon.length - 1 && column < dungeon[0].length - 1) {
    		dfsMinHp(dungeon, row + 1, column, minHp, left);
    		dfsMinHp(dungeon, row, column + 1, minHp, left);
		} else if (row < dungeon.length - 1) {
			dfsMinHp(dungeon, row + 1, column, minHp, left);
		} else {
			dfsMinHp(dungeon, row, column + 1, minHp, left);
		}
	}

	// https://leetcode.com/problems/cherry-pickup/description/
	public int cherryPickup(int[][] grid) {

		// 先从0，0到n-1， n-1
		final int n = grid.length;

		// 0代表从左边来， 1代表从上边来
		int[][][] dp = new int[n][n][2];

		int result = 0;
		int loop = 0;
		while (loop++ <= 1) {
			dp[0][0][0] = grid[0][0];

			for (int i = 1; i < n; ++i) {
				if (dp[0][i-1][0] == -1 || grid[0][i] == -1) {
					dp[0][i][0] = -1;
				} else {
					dp[0][i][0] = grid[0][i] + dp[0][i-1][0];
					dp[0][i][1] = 0;
				}
			}
			for (int i = 1; i < n; ++i) {
				if (dp[i-1][0][0] == -1 || grid[i][0] == -1) {
					dp[i][0][0] = -1;
				} else {
					dp[i][0][0] = grid[i][0] + dp[i-1][0][0];
					dp[i][0][1] = 1;
				}
			}

			for (int i = 1; i < n; ++i) {
				for (int j = 1; j < n; ++j) {
					if (grid[i][j] == -1 || (dp[i-1][j][0] == -1 && dp[i][j-1][0] == -1) ) {
						dp[i][j][0] = -1;
					} else if (dp[i-1][j][0] != -1 && dp[i][j-1][0] != -1){
						if (dp[i-1][j][0] >= dp[i][j-1][0]) {
							dp[i][j][0] = dp[i-1][j][0] + grid[i][j];
							dp[i][j][1] = 1;
						} else {
							dp[i][j][0] = dp[i][j-1][0] + grid[i][j];
							dp[i][j][1] = 0;
						}
					} else if (dp[i-1][j][0] != -1 && dp[i][j-1][0] == -1) {
						dp[i][j][0] = dp[i-1][j][0] + grid[i][j];
						dp[i][j][1] = 1;
					} else if (dp[i-1][j][0] == -1 && dp[i][j-1][0] != -1) {
						dp[i][j][0] = dp[i][j-1][0] + grid[i][j];
						dp[i][j][1] = 0;
					}
				}
			}


			if (dp[n-1][n-1][0] == -1 || dp[n-1][n-1][0] == 0) {
				return result;
			}

			result += dp[n-1][n-1][0];

			if (loop ==1) {
				// 回溯回去
				int rowStart = n-1;
				int columnStart = n - 1;
				while (rowStart >= 0 && columnStart >= 0) {
					grid[rowStart][columnStart] = 0;
					if (dp[rowStart][columnStart][1] == 0) {
						// 从左边来
						columnStart--;
					} else {
						rowStart--;
					}
				}
			}

		}

		return result;
	}

	// https://leetcode.com/problems/smallest-range/description/
	public int[] smallestRange(List<List<Integer>> nums) {
		int[] result = new int[2];
		if (nums == null || nums.size() <= 0) {
			return result;
		}

		final int kinds = nums.size();

		int[] hash = new int[nums.size()];
		int kindCnt = 0;

		int start = 0;
		int end = 0;
		int minGap = Integer.MAX_VALUE;

		int resultStart = 0;
		int resultEnd = 0;

		PriorityQueue<ValueIndex> minHeap = new PriorityQueue<>(new Comparator<ValueIndex>() {
			@Override
			public int compare(ValueIndex o1, ValueIndex o2) {
				return o1.value - o2.value;
			}
		});

		Iterator<Integer>[] iterators = new Iterator[nums.size()];
		int i = 0;
		int totalNum = 0;
		for (List<Integer> list : nums) {
			Iterator<Integer> it = list.iterator();
			iterators[i] = it;
			if (it.hasNext()) {
				minHeap.add(new ValueIndex(i, it.next()));
				totalNum += list.size();
			}
			i++;

		}

		ValueIndex[] valueIndices = new ValueIndex[totalNum];
		i = 0;
		while (!minHeap.isEmpty()) {
			ValueIndex valueIndex = minHeap.poll();
			valueIndices[i++] = valueIndex;
			if (iterators[valueIndex.index].hasNext()) {
				minHeap.add(new ValueIndex(valueIndex.index, iterators[valueIndex.index].next()));
			}
		}

		while (end < valueIndices.length) {
			ValueIndex valueIndex = valueIndices[end];

			if (hash[valueIndex.index] == 0) {
				kindCnt++;
			}
			hash[valueIndex.index]++;

			while (kindCnt >= kinds && start <= end) {
				if (valueIndex.value - valueIndices[start].value < minGap) {
					minGap = valueIndex.value - valueIndices[start].value;
					resultStart = start;
					resultEnd = end;
				}

				final int startIndex = valueIndices[start].index;
				hash[startIndex]--;

				if (hash[startIndex] == 0) {
					kindCnt--;
				}

				start++;
			}
			end++;
		}

		result[0] = valueIndices[resultStart].value;
		result[1] = valueIndices[resultEnd].value;

		return result;
	}

	private static class ValueIndex {
		public int index;
		public int value;

		public ValueIndex(int index, int value) {
			this.index = index;
			this.value = value;
		}
	}

	// https://leetcode.com/problems/interleaving-string/description/
	public boolean isInterleave(String s1, String s2, String s3) {
		if (s3.length() != (s1.length() + s2.length())) {
			return false;
		}

		boolean[][] dp = new boolean[s1.length() + 1][s2.length() + 1];

		dp[0][0] = true;

		for (int i = 1; i <= s2.length(); ++i) {
			if (s3.charAt(i-1) != s2.charAt(i-1)) {
				break;
			} else {
				dp[0][i] = true;
			}
		}

		for (int i = 1; i <= s1.length(); ++i) {
			if (s3.charAt(i-1) == s1.charAt(i-1)) {
				dp[i][0] = true;
			} else {
				break;
			}
		}

		for (int i = 1; i <= s1.length(); ++i) {
			for (int j = 1; j <= s2.length(); ++j) {
				final char c3 = s3.charAt(i+j-1);
				final char c2 = s2.charAt(j-1);
				final char c1 = s1.charAt(i-1);

				if (c3 != c2 && c3 != c1) {
					dp[i][j] = false;
				} else if (c3 == c2 && c3 != c1) {
					dp[i][j] = dp[i][j-1];
				} else if (c3 != c2 && c3 == c1) {
					dp[i][j] = dp[i-1][j];
				} else {
					dp[i][j] = dp[i-1][j] || dp[i][j-1];
				}
			}
		}

		return dp[s1.length()][s2.length()];
	}

	//https://leetcode.com/problems/decode-ways/description/
//	public int numDecodings(String s) {
//		if (s.isEmpty()) {
//			return 0;
//		}
//
//		int[] dp = new int[s.length() + 1];
//		dp[0] = 1;
//
//		char c = s.charAt(0);
//		if (c >= '1' && c <= '9') {
//			dp[1] = 1;
//		} else {
//			return 0;
//		}
//
//		for (int i = 2; i <= s.length(); ++i) {
//			c = s.charAt(i-1);
//			if (c >= '1' && c <= '9')
//				dp[i] = dp[i-1];
//
//			int temp = Integer.valueOf(s.substring(i-2, i));
//			if (temp >= 10 && temp <= 26) {
//				dp[i] += dp[i-2];
//			}
//
//			if (dp[i] == 0) {
//				return 0;
//			}
//		}
//
//		return dp[s.length()];
//	}

	// https://leetcode.com/problems/decode-ways-ii/description/
	public int numDecodings(String s) {
		if (s.isEmpty()) {
			return 0;
		}

		final int more = 1000000000 + 7;

		int[] dp = new int[s.length() + 1];
		dp[0] = 1;

		char c = s.charAt(0);
		if (c >= '1' && c <= '9') {
			dp[1] = 1;
		} else if (c == '*') {
			dp[1] = 9;
		} else {
			return 0;
		}

		for (int i = 2; i <= s.length(); ++i) {
			char c1 = s.charAt(i - 1);
			char c0 = s.charAt(i - 2);
			if (c1 >= '1' && c1 <= '9') {
				dp[i] = dp[i - 1];
			} else if (c1 == '*') {
				dp[i] = dp[i-1] % more * 9 ;
			}

			if (c1 >= '0' && c1 <= '9' && c0 >= '0' && c0 <= '9') {
				int temp = Integer.valueOf(s.substring(i-2, i));
				if (temp >= 10 && temp <= 26) {
					dp[i] = dp[i] % more + dp[i-2] % more;
				}
			} else if (c1 >= '0' && c1 <= '9' && c0 == '*') {
				if (c1 <= '6') {
					dp[i] = dp[i] % more + dp[i-2] % more * 2;
				} else {
					dp[i] = dp[i] % more + dp[i-2] % more;
				}
			} else if (c1 == '*' && c0 >= '1' && c0 <= '2') {
				if (c0 == '1') {
					dp[i] = dp[i] % more + dp[i-2] % more * 9;
				} else {
					dp[i] = dp[i] % more + dp[i-2] % more * 6;
				}
			} else if (c1 == '*' && c0 == '*') {
				dp[i] = dp[i] % more + dp[i-2] % more * 15;
			}

			dp[i] %= more;

		}

		return dp[s.length()];
	}

	// https://leetcode.com/problems/minimum-number-of-refueling-stops/description/
	public int minRefuelStops(int target, int startFuel, int[][] stations) {
		int range = startFuel;
		int index = 0;
		int res = 0;

		PriorityQueue<Integer> queue = new PriorityQueue<>(new Comparator<Integer>() {
			@Override
			public int compare(Integer o1, Integer o2) {
				return o2 - o1;
			}
		});

		while (range < target) {
			while (index < stations.length && stations[index][0] <= range) {
				queue.add(stations[index][1]);
				index++;
			}

			if (queue.isEmpty()) {
				return -1;
			}

			range += queue.poll();
			res++;
		}

		return res;
	}

	// https://leetcode.com/problems/house-robber/description/
	public int rob(int[] nums) {
		if (nums == null || nums.length <= 0) {
			return 0;
		}

		if (nums.length == 1) {
			return nums[0];
		}

		int[] dp = new int[nums.length];
		dp[0] = nums[0];
		dp[1] = Math.max(nums[0], nums[1]);
		int result = Math.max(dp[0], dp[1]);
		for (int i = 2; i < nums.length; ++i) {
			dp[i] = Math.max(dp[i-1], dp[i-2] + nums[i]);
			result = Math.max(result, dp[i]);
		}

		return result;
	}

	// https://leetcode.com/problems/soup-servings/description/
	public double soupServings(int N) {
		Map<SoupKey, Double> dp = new HashMap<>();
		return soupServingsHelper(N, N, dp);
	}

	private class SoupKey {
		int n;
		int m;
		public SoupKey(int n, int m) {
			this.n = n;
			this.m = m;
		}

		@Override
		public boolean equals(Object obj) {
			if (obj == this) {
				return true;
			} else if (!(obj instanceof SoupKey)) {
				return false;
			} else {
				return this.n == ((SoupKey) obj).n && this.m == ((SoupKey) obj).m;
			}
		}

		@Override
		public int hashCode() {
			int result = 17;
			result = result * 31 + n;
			result = result * 31 + m;
			return result;

		}
	}

	public double soupServingsHelper(int n, int m, Map<SoupKey, Double> dp) {
		if (n <= 0 && m <= 0) {
			return 0.5f;
		} else if (n <= 0 && m > 0) {
			return 1.0f;
		} else if (n > 0 && m <= 0) {
			return 0;
		}

		SoupKey key = new SoupKey(n, m);
		if (dp.containsKey(key)) {
			return dp.get(key);
		}

		double res = 0.25 * (soupServingsHelper(n-100, m, dp)
			+ soupServingsHelper(n - 75, m - 25, dp)
			+ soupServingsHelper(n - 50, m - 50, dp)
			+ soupServingsHelper(n - 25, m - 75, dp));

		dp.put(key, res);

		return res;
	}

	// https://leetcode.com/problems/delete-and-earn/description/
	public int deleteAndEarn(int[] nums) {
		if (nums == null || nums.length <= 0) {
			return 0;
		}

		int[] hashTemp = new int[10001];
		int diffCnt = 0;
		for (int num : nums) {
			if (hashTemp[num] == 0) {
				diffCnt++;
			}
			hashTemp[num]++;
		}

		int[] arrays = new int[diffCnt];
		int j = 0;
		for (int i = 0; i < hashTemp.length; ++i) {
			if (hashTemp[i] != 0) {
				arrays[j++] = i;
			}
		}

		int[] dp = new int[diffCnt+1];
		if (diffCnt == 0) {
			return 0;
		} else if (diffCnt == 1) {
			return arrays[0] * hashTemp[arrays[0]];
		}

		dp[1] = arrays[0] * hashTemp[arrays[0]];
		int result = dp[1];

		for (int i = 2; i < dp.length; ++i) {
			if (arrays[i-1] == arrays[i-2] + 1) {
				dp[i] = Math.max(dp[i-2] + arrays[i-1] * hashTemp[arrays[i-1]], dp[i-1]);
			} else {
				dp[i] = dp[i-1] + arrays[i-1] * hashTemp[arrays[i-1]];
			}

			result = Math.max(result, dp[i]);
		}

		return result;
	}

	// https://leetcode.com/problems/burst-balloons/description/
	public int maxCoins(int[] nums) {
		if (nums == null || nums.length <= 0) {
			return 0;
		}

		int[][] dp = new int[nums.length][nums.length];

		for (int diff = 0; diff <= nums.length - 1; ++diff) {
			for (int i = 0; i + diff < nums.length; ++i) {
				int max = Integer.MIN_VALUE;
				for (int j = 0; j <= diff; ++j) {
					final int leftMax = (i + j - 1 >= 0 ? dp[i][i+j-1] : 0);
					final int rightMax = (i +j + 1 < nums.length ? dp[i+j+1][i+diff] : 0);
					final int center = nums[i+j] * (i - 1 >= 0 ? nums[i-1] : 1) * (i + diff + 1 < nums.length ? nums[i+ diff + 1] : 1);
					max = Math.max(leftMax + rightMax + center, max);
				}
				dp[i][i+diff] = max;
			}
		}

		return dp[0][nums.length-1];
	}

	// https://leetcode.com/problems/2-keys-keyboard/description/
	public int minSteps(int n) {
		if (n <= 1) {
			return 0;
		}

		if ((n & 1) == 0) {
			return 2 + minSteps(n >> 1);
		} else {
			for (int i = n / 2; i >= 3; --i) {
				if (n % i == 0) {
					return n / i + minSteps(i);
				}
			}

			return n;
		}
	}

	// https://leetcode.com/problems/shortest-path-visiting-all-nodes/description/
	public int shortestPathLength(int[][] graph) {

		int N = graph.length;

		Queue<Tuple> queue = new LinkedList<>();
		Set<Tuple> set = new HashSet<>();

		for(int i = 0; i < N; i++){
			int tmp = (1 << i);
			set.add(new Tuple(tmp, i, 0));
			queue.add(new Tuple(tmp, i, 1));
		}

		while(!queue.isEmpty()){
			Tuple curr = queue.remove();

			if(curr.bitMask == (1 << N) - 1){
				return curr.cost - 1;
			} else {
				int[] neighbors = graph[curr.curr];

				for(int v : neighbors){
					int bitMask = curr.bitMask;
					bitMask = bitMask | (1 << v);

					Tuple t = new Tuple(bitMask, v, 0);
					if(!set.contains(t)){
						queue.add(new Tuple(bitMask, v, curr.cost + 1));
						set.add(t);
					}
				}
			}
		}
		return -1;
	}

	class Tuple {
		int bitMask;
		int curr;
		int cost;

		public Tuple(int bit, int n, int c) {
			bitMask = bit;
			curr = n;
			cost = c;
		}

		public boolean equals(Object o) {
			Tuple p = (Tuple) o;
			return bitMask == p.bitMask && curr == p.curr && cost == p.cost;
		}

		public int hashCode() {
			return 1331 * bitMask + 7193 * curr + 727 * cost;
		}
	}

	// https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/
	public List<String> letterCombinations(String digits) {
		final Map<Character, Set<Character>> map = new HashMap<>();

		map.put('2', new HashSet<>(Arrays.asList('a', 'b', 'c')));
		map.put('3', new HashSet<>(Arrays.asList('d', 'e', 'f')));
		map.put('4', new HashSet<>(Arrays.asList('g', 'h', 'i')));
		map.put('5', new HashSet<>(Arrays.asList('j', 'k', 'l')));
		map.put('6', new HashSet<>(Arrays.asList('m', 'n', 'o')));
		map.put('7', new HashSet<>(Arrays.asList('p', 'q', 'r', 's')));
		map.put('8', new HashSet<>(Arrays.asList('t', 'u', 'v')));
		map.put('9', new HashSet<>(Arrays.asList('w', 'x', 'y', 'z')));

		return letterCombinations(digits, 0, map);
	}

	private List<String> letterCombinations(CharSequence digits, int index, Map<Character, Set<Character>> map) {


		if (index >= digits.length()) {

			return new ArrayList<>(Arrays.asList(""));
		}

		final ArrayList<String> result = new ArrayList<>();
		List<String> next = letterCombinations(digits, index + 1, map);
		for (Character c : map.get(digits.charAt(index))) {
			for (String nextStr : next) {
				result.add(c + nextStr);
			}
		}

		return result;
	}

	// https://leetcode.com/problems/zigzag-conversion/description/
	public String convert(String s, int numRows) {
		if (s == null || s.length() <= 0) {
			return s;
		}

		if (numRows <= 1 || numRows >= s.length()) {
			return s;
		}

		final int length = s.length();
		int row = 0;
		StringBuilder result = new StringBuilder(length);
		while (row < numRows) {
			int start = row;
			int delta = 2 * numRows - 2;
			int delta2 = delta - 2 * row;

			for (; start < length; start += delta) {
				result.append(s.charAt(start));
				if (start + delta2 < length && delta2 != 0 && delta2 != delta) {
					result.append(s.charAt(start + delta2));
				}
			}

			row++;
		}

		return result.toString();
	}

	// https://leetcode.com/problems/next-permutation/description/
	public void nextPermutation(int[] nums) {
		if (nums == null || nums.length <= 1) {
			return;
		}

		// 找到第一个大于她的
		int start = nums.length - 1;
		while (start > 0) {
			if (nums[start] <= nums[start - 1]) {
				start--;
			} else {
				break;
			}
		}

		if (start <= 0) {
			swapArray(nums, 0, nums.length - 1);
			return;
		}

		// 二分查找，找到最小的比他大的数
		final int find = nums[start - 1];
		int left = start;
		int right = nums.length - 1;
		while (left <= right) {
			int middle = left + (right - left) / 2;
			if (nums[middle] > find) {
				left = middle + 1;
			} else {
				right = middle - 1;
			}
		}

		// 交换right与start -1的值
		swap(nums, right, start - 1);

		swapArray(nums, start, nums.length - 1);
	}

	private void swapArray(int[] nums, int start, int end) {
		while (start < end) {
			swap2(nums, start, end);
			start++;
			end--;
		}
	}

	// https://leetcode.com/problems/search-in-rotated-sorted-array/description/
	public int search(int[] nums, int target) {
		if (nums == null)
			return -1;

		int left = 0;
		int right = nums.length - 1;

		while (left <= right) {
			int middle = left + (right - left) / 2;

			if (nums[middle] == target) {
				return middle;
			} else if (nums[middle] > target) {
				if (nums[middle] > nums[right] && nums[right] >= target) {
					left = middle + 1;
				} else {
					right = middle - 1;
				}
			} else {
				if (nums[left] > nums[middle] && nums[right] < target) {
					right = middle - 1;
				} else {
					left = middle + 1;
				}
			}
		}

		return -1;
	}

	// https://leetcode.com/problems/permutations/description/
	public List<List<Integer>> permute(int[] nums) {
		return permute(nums, 0, nums.length - 1);
	}

	private List<List<Integer>> permute(int[] nums, int start, int end) {
		List<List<Integer>> result = new LinkedList<>();
		if (nums == null || start > end) {
			result.add(new LinkedList<>());
			return result;
		}

		for (int i = start; i <= end; ++i) {
			swapPermute(nums, start, i);
			List<List<Integer>> next = permute(nums, start + 1, end);
			for (List<Integer> one : next) {
				LinkedList<Integer> temp = new LinkedList<>();
				temp.add(nums[start]);
				temp.addAll(one);
				result.add(temp);
			}
			swapPermute(nums, start, i);
		}

		return result;
	}



	// https://leetcode.com/problems/permutations-ii/description/
	public List<List<Integer>> permuteUnique(int[] nums) {
		Arrays.sort(nums);
		return permuteUnique(nums, 0, nums == null ? -1 : nums.length - 1);
	}

	public List<List<Integer>> permuteUnique(int[] nums, int start, int end) {
		List<List<Integer>> result = new LinkedList<>();
		if (nums == null || start > end) {
			result.add(new LinkedList<>());
			return result;
		}

		HashSet<Integer> visited = new HashSet<>();
		for (int i = start; i <= end; ++i) {
			if (visited.add(nums[i])) {
				swapPermute(nums, i, start);
				List<List<Integer>> next = permuteUnique(nums, start + 1, end);
				for (List<Integer> one : next) {
					LinkedList<Integer> temp = new LinkedList<>();
					temp.add(nums[start]);
					temp.addAll(one);
					result.add(temp);
				}
				swapPermute(nums, i, start);
			}

		}

		return result;
	}

	// https://leetcode.com/problems/rotate-image/description/
	public void rotate(int[][] matrix) {
		if (matrix == null || matrix.length <= 0
				|| matrix[0] == null || matrix[0].length <= 0) {
			return;
		}

		for (int i = 0; i <= matrix.length / 2; ++i) {
			rotate(matrix, i, matrix.length - 1 - i);
		}
	}

	// 每次旋转一层
	public void rotate(int[][] matrix, int top, int bottom) {
		if (bottom <= top) {
			return;
		}

		for (int column = top; column < bottom; column++) {
			// 交换4次
			// 上右交换
			int temp1 = matrix[column][bottom];
			matrix[column][bottom] = matrix[top][column];

			// 右下交换
			int temp2 = matrix[bottom][bottom - (column - top)];
			matrix[bottom][bottom - (column - top)] = temp1;

			// 下走交换
			int temp3 = matrix[bottom - (column - top)][top];
			matrix[bottom - (column - top)][top] = temp2;


			matrix[top][column] = temp3;
		}
	}

	// https://leetcode.com/problems/group-anagrams/description/
	public List<List<String>> groupAnagrams(String[] strs) {
		Map<Integer,List<String>> map = new HashMap<>();
		List<List<String>> res = new ArrayList<>();
		for(String word : strs)
		{
			int[] count = new int[26];
			for(char ch : word.toCharArray())
			{
				count[ch - 'a']++;
			}
			Integer temp = Arrays.hashCode(count);
			if(map.containsKey(temp))
			{
				List<String> list = map.get(temp);
				list.add(word);
				map.put(temp,list);
			}
			else
			{
				List<String> list = new ArrayList<>();
				list.add(word);
				map.put(temp,list);
			}
		}
		for(Integer k : map.keySet())
		{
			res.add(map.get(k));
		}
		return res;

	}

	// https://leetcode.com/problems/permutation-sequence/description/
	public String getPermutation(int n, int k) {

		int[] nums = new int[n];
		for (int i = 0; i < n; ++i) {
			nums[i] = i + 1;
		}


		getPermutationHelper(nums, 0, k);
		return getPermutationResult;
	}

	private String getPermutationResult = null;
	private int kth = 0;

	public void getPermutationHelper(int[] nums, int index, int k) {
		if (kth >= k) {
			return;
		}
		if (index >= nums.length) {
			kth++;
			if (kth == k) {
				String result = "";
				for (int num : nums) {
					result += String.valueOf(num);
				}
				getPermutationResult = result;
			}
			return;
		}

		int[] temp = new int[nums.length];
		for (int i = index; i < nums.length; ++i) {
			swapPermute(nums, i, index);
			System.arraycopy(nums, index + 1, temp, index + 1, nums.length - index - 1);
			Arrays.sort(nums, index + 1, nums.length);
			getPermutationHelper(nums, index + 1, k);
			System.arraycopy(temp, index + 1, nums, index + 1, nums.length - index - 1);
			swapPermute(nums, i, index);
		}
	}

	private void swapPermute(int[] nums, int from, int to) {
		int temp = nums[from];
		nums[from] = nums[to];
		nums[to] = temp;
	}

	// https://leetcode.com/problems/rotate-list/description/
	public ListNode rotateRight(ListNode head, int k) {
		if (head == null || head.next == null) {
			return head;
		}

		int length = 0;
		ListNode temp = head;
		ListNode tail = null;
		while (temp != null) {
			length++;
			tail = temp;
			temp = temp.next;
		}

		int left = k % length;
		if (left == 0) {
			return head;
		}

		ListNode pre = null;
		ListNode next = head;
		int cnt = 1;
		while (cnt <= length - left) {
			cnt++;
			pre = next;
			next = next.next;
		}

		// 把next
		tail.next = head;
		pre.next = null;

		return next;
	}

	// https://leetcode.com/problems/minimum-path-sum/description/
	public int minPathSum(int[][] grid) {
		if (grid == null || grid.length <= 0 || grid[0] == null || grid[0].length <= 0) {
			return 0;
		}

		final int m = grid.length;
		final int n = grid[0].length;

		int[][] dp = new int[m][n];

		dp[0][0] = grid[0][0];
		for (int i = 1; i < n; ++i) {
			dp[0][i] = dp[0][i-1] + grid[0][i];
		}

		for (int j = 1; j < m; ++j) {
			dp[j][0] = dp[j-1][0] + grid[j][0];
		}

		for (int i = 1; i < n; ++i) {
			for (int j = 1; j < m; ++j) {
				dp[j][i] = grid[j][i] + Math.min(dp[j-1][i], dp[j][i - 1]);
			}
		}

		return dp[m-1][n-1];
	}

	// https://leetcode.com/problems/simplify-path/description/
	public String simplifyPath(String path) {
		if (path == null || path.length() <= 0) {
			return path;
		}

		String[] paths = path.split("/");
		if (paths == null || paths.length <= 0) {
			return "/";
		}

		Stack<String> stack = new Stack<>();

		for (String one : paths) {
			if (one == null || one.isEmpty() || ".".equals(one)) {
				continue;
			}

			if ("..".equals(one)) {
				if (!stack.isEmpty())
					stack.pop();
			} else {
				stack.push(one);
			}
		}

		String result = "";
		while (!stack.isEmpty()) {
			String head = stack.pop();

			result = "/" + head + result;
		}

		return result.isEmpty() ? "/" : result;
	}

	// https://leetcode.com/problems/set-matrix-zeroes/description/
	public void setZeroes(int[][] matrix) {
		List<Integer> r = new ArrayList<>();
		List<Integer> c = new ArrayList<>();
		for(int i =0; i < matrix.length; i ++)
		{
			for(int j=0; j<matrix[0].length; j ++){
				if(matrix[i][j]==0) {
					r.add(i); c.add(j);
				}
			}
		}
		for(int x: r) for(int c1=0; c1<matrix[0].length;c1++) matrix[x][c1]=0;
		for(int x: c) for(int r1=0; r1<matrix.length;r1++) matrix[r1][x]=0;
	}

	// https://leetcode.com/problems/subsets/description/
	public List<List<Integer>> subsets(int[] nums) {
		List<List<Integer>> result = new LinkedList<>();
		subsetsHelper(result, new LinkedList<>(), nums, 0);
		return result;
	}

	public void subsetsHelper(List<List<Integer>> result, List<Integer> backstrack, int[] nums, int index) {
		if (index >= nums.length) {
			LinkedList<Integer> one = new LinkedList<>();
			one.addAll(backstrack);
			result.add(one);
			return;
		}

		backstrack.add(nums[index]);
		subsetsHelper(result, backstrack, nums, index + 1);
		backstrack.remove(backstrack.size() - 1);
		subsetsHelper(result, backstrack, nums, index + 1);
	}

	// https://leetcode.com/problems/surrounded-regions/description/
	public void solve(char[][] board) {
		if (board == null || board.length <= 0
				|| board[0] == null || board[0].length <= 0) {
			return;
		}

		for (int i = 0; i < board.length; ++i) {
			for (int j = 0; j < board[0].length; ++j) {
				if (board[i][j] == 'O') {
					if (!findExit(board, i, j)) {
						board[i][j] = 'X';
					}
				}
			}
		}
	}

	public void solve2(char[][] board) {
		if (board == null || board.length <= 0
				|| board[0] == null || board[0].length <= 0) {
			return;
		}

		// 四个边
		for (int i = 0; i < board[0].length; ++i) {
			if (board[0][i] == 'O') {
				unionAdjacent(board, 0, i);
			}
		}

		for (int i = 1; i < board.length; ++i) {
			if (board[i][board[0].length-1] == 'O') {
				unionAdjacent(board, i, board[0].length - 1);
			}
		}

		for (int i = board[0].length - 2; i >= 0; --i) {
			if (board[board.length - 1][i] == 'O') {
				unionAdjacent(board, board.length - 1, i);
			}
		}

		for (int i = 1; i < board.length - 1; ++i) {
			if (board[i][0] == 'O') {
				unionAdjacent(board, i, 0);
			}
		}

		for (int i = 0; i < board.length; ++i) {
			for (int j = 0; j < board[0].length; ++j) {
				if (board[i][j] == '#') {
					board[i][j] = 'O';
				} else if (board[i][j] == 'O') {
					board[i][j] = 'X';
				}
			}
		}
	}



	private void unionAdjacent(char[][] board, int sx, int sy) {
		board[sx][sy] = '#';

		for (int i = 0; i < sBoardDirection.length; ++i) {
			final int nx = sBoardDirection[i][0] + sx;
			final int ny = sBoardDirection[i][1] + sy;

			if (nx >= 0 && nx <= board.length - 1 &&
					ny >= 0 && ny <= board[0].length - 1 &&
					board[nx][ny] == 'O') {
				unionAdjacent(board, nx, ny);
			}
		}
	}

	private boolean findExit(char[][] board, int sx, int sy) {
		if (sx <= 0 || sx >= board.length - 1 ||
				sy <= 0 || sy >= board[0].length - 1) {
			return true;
		}

		boolean result = false;
		board[sx][sy] = 'X';
		for (int i = 0; i < sBoardDirection.length; ++i) {
			final int nx = sBoardDirection[i][0] + sx;
			final int ny = sBoardDirection[i][1] + sy;

			if (nx >= 0 && nx <= board.length - 1 &&
					ny >= 0 && ny <= board[0].length - 1 &&
					board[nx][ny] == 'O') {
				result = findExit(board, nx, ny);
				if (result) {
					break;
				}
			}
		}
		board[sx][sy] = 'O';
		return result;
	}


	//Definition for undirected graph.
	class UndirectedGraphNode {
	    int label;
	    List<UndirectedGraphNode> neighbors;
	    UndirectedGraphNode(int x) {
	    	label = x;
	    	neighbors = new ArrayList<UndirectedGraphNode>();
	    }
	};

	public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
		return cloneGraphHelper(node, new HashMap<>());
	}

	private UndirectedGraphNode cloneGraphHelper(UndirectedGraphNode node, Map<Integer, UndirectedGraphNode> map) {
		if (node == null) {
			return null;
		}

		if (map.containsKey(node.label)) {
			return map.get(node.label);
		}

		UndirectedGraphNode newNode = new UndirectedGraphNode(node.label);
		map.put(newNode.label, newNode);

		if (node.neighbors == null || node.neighbors.size() <= 0) {
			return newNode;
		}

		newNode.neighbors = new ArrayList<>();
		for (UndirectedGraphNode neighbor : node.neighbors) {
			newNode.neighbors.add(cloneGraph(neighbor));
		}

		return newNode;
	}

	// https://leetcode.com/problems/single-number-ii/description/
	public int singleNumber(int[] nums) {
		if (nums == null || nums.length <= 0) {
			return -1;
		} else if (nums.length % 3 != 1) {
			return -1;
		}

		return 0;
	}


	//Definition for singly-linked list with a random pointer.
	public static class RandomListNode {
	    int label;
	    RandomListNode next, random;
	    RandomListNode(int x) { this.label = x; }
	};

	public RandomListNode copyRandomList(RandomListNode head) {
		if (head == null) {
			return null;
		}

		Map<RandomListNode, RandomListNode> map = new HashMap<>();

		RandomListNode newHead = new RandomListNode(head.label);

		map.put(head, newHead);

		RandomListNode temp = head;

		RandomListNode preNew = newHead;
		while (temp.next != null) {
			temp = temp.next;
			RandomListNode newNext = new RandomListNode(temp.label);
			map.put(temp, newNext);
			preNew.next = newNext;
			preNew = newNext;
		}

		temp = head;

		while (temp != null) {
			if (temp.random != null) {
				map.get(temp).random = map.get(temp.random);
			}
			temp = temp.next;
		}

		return newHead;
	}

	// https://leetcode.com/problems/reorder-list/description/
	public void reorderList(ListNode head) {
		if (head == null) {
			return;
		}

		ListNode slow = head;
		ListNode fast = head;

		while (fast.next != null && fast.next.next != null) {
			slow = slow.next;
			fast = fast.next.next;
		}

		if (fast == slow) {
			return;
		}

		while (fast.next != null) {
			slow = slow.next;
			fast = fast.next;
		}

		// reverse slow之后的点
		ListNode leftHead = slow.next;
		slow.next = null;

		ListNode leftPre = null;
		ListNode newHead = null;
		while (leftHead != null) {
			newHead = leftHead;
			ListNode next = leftHead.next;
			leftHead.next = leftPre;
			leftPre = leftHead;
			leftHead = next;
		}

		// 将newHead插入到从head到slow之间的结点中
		ListNode insertedAfter = head;
		ListNode toInsert = newHead;
		while (insertedAfter != null && toInsert != null) {
			ListNode insertedNext = insertedAfter.next;
			ListNode toInsertNext = toInsert.next;
			insertedAfter.next = toInsert;
			toInsert.next = insertedNext;

			insertedAfter = insertedNext;
			toInsert = toInsertNext;
		}
	}

	// https://leetcode.com/problems/evaluate-reverse-polish-notation/description/
	public int evalRPN(String[] tokens) {
		Stack<Integer> stack = new Stack<>();

		for (String token : tokens) {
			if (!isOp(token)) {
				stack.push(Integer.valueOf(token));
			} else {
				int two = stack.pop();
				int one = stack.pop();

				final int result;
				if (token.equals("+")) {
					result = one + two;
				} else if (token.equals("-")) {
					result = one - two;
				} else if (token.equals("*")) {
					result = one * two;
				} else {
					result = one / two;
				}
				stack.push(result);
			}
		}

		return stack.pop();
	}

	private boolean isOp(String token) {
		return token.equals("+") || token.equals("-") || token.equals("*") || token.equals("/");
	}

	// https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/description/
	public int findMin(int[] nums) {
		if (nums == null || nums.length <= 0) {
			return -1;
		}

		int left = 0;
		int right = nums.length - 1;
		while (left <= right) {
			int middle = left + (right - left) / 2;
			if (nums[left] < nums[right]) {
				return nums[left];
			} else if (nums[middle] > nums[right]) {
				left = middle + 1;
			} else {
				right = middle - 1;
			}
		}

		return -1;
	}

	// https://leetcode.com/problems/find-peak-element/description/
	public int findPeakElement(int[] nums) {
		return 0;
	}

	// https://leetcode.com/problems/repeated-dna-sequences/description/
	public List<String> findRepeatedDnaSequences(String s) {
		List<String> result = new LinkedList<>();
		if (s == null || s.length() <= 10) {
			return result;
		}

		Set<Long> set = new HashSet<>();
		Set<Long> resultSet = new HashSet<>();

		long hash = 0L;

		long power = 1L;
		for (int i = 0; i < 9; ++i) {
			power <<= 2;
		}

		// 4进制
		for (int i = 0; i < 10; ++i) {
			final char c = s.charAt(i);
			hash = hash * 4 + charToInt(c);
		}

		set.add(hash);

		for (int i = 10; i < s.length(); ++i) {
			hash = (hash - power * charToInt(s.charAt(i-10))) * 4 + charToInt(s.charAt(i));

			if (!set.add(hash)) {
				if (resultSet.add(hash)) {
					result.add(longToStr(hash));
				}
			}
		}

		return result;
	}

	private long charToInt(char c) {
		if (c == 'A') {
			return 0L;
		} else if (c == 'C') {
			return 1L;
		} else if (c == 'G') {
			return 2L;
		} else {
			return 3L;
		}
	}

	private char intToChar(long c) {
		if (c == 0L) {
			return 'A';
		} else if (c == 1L) {
			return 'C';
		} else if (c == 2L) {
			return 'G';
		} else {
			return 'T';
		}
	}

	private String longToStr(long n) {
		String result = "";
		int i = 0;
		while (i++ < 10) {
			result = intToChar(n & 3) + result;
			n >>= 2;
		}

		return result;
	}

	// https://leetcode.com/problems/number-of-islands/description/
	public int numIslands(char[][] grid) {
		if (grid == null || grid.length <= 0 ||
				grid[0] == null || grid[0].length <= 0) {
			return 0;
		}

		int result = 0;
		for (int i = 0; i < grid.length; ++i) {
			for (int j = 0; j < grid[0].length; ++j) {
				if (grid[i][j] == '1') {
					result++;
					numIslandsHelper(grid, i, j);
				}
			}
		}

		return result;
	}

	private void numIslandsHelper(char[][] grid, int x, int y) {
		grid[x][y] = '2';

		for (int[] direct : sBoardDirection) {
			final int newx = x + direct[0];
			final int newy = y + direct[1];

			if (newx >= 0 && newx < grid.length
					&& newy >= 0 && newy < grid[0].length
					&& grid[newx][newy] == '1') {
				numIslandsHelper(grid, newx, newy);
			}
		}
	}

	private static int[][] sBoardDirection = {
			{-1, 0},
			{1, 0},
			{0, -1},
			{0, 1}
	};

	// https://leetcode.com/problems/bitwise-and-of-numbers-range/description/
	public int rangeBitwiseAnd(int m, int n) {
		if (m == 0) {
			return 0;
		}

		int result = m;
		for (int k = m + 1; k <= Math.min(n, m << 1); ++k) {
			result &= k;
		}

		return result;
	}

	// https://leetcode.com/problems/kth-largest-element-in-an-array/description/
	public int findKthLargest(int[] nums, int k) {
		Set<Integer> set = new HashSet<>();

		PriorityQueue<Integer> heap = new PriorityQueue<>(new Comparator<Integer>() {
			@Override
			public int compare(Integer o1, Integer o2) {
				return o1 - o2;
			}
		});

		int cnt = 0;
		for (int num : nums) {
			boolean existed = !set.add(num);
			if (existed) {
				continue;
			}
			if (cnt < k) {
				cnt++;
				heap.add(num);
			} else if (num > heap.peek()) {
				heap.poll();
				heap.add(num);
			}
		}

		return heap.poll();
	}

	// https://leetcode.com/problems/combination-sum-iii/description/
	public List<List<Integer>> combinationSum3(int k, int n) {
		LinkedList<List<Integer>> result = new LinkedList<>();

		if (k <= 0 || k >= 10) {
			return result;
		}

		combinationsSum3Helper(1, k, n, 0, 0, result, new LinkedList<>());

		return result;
	}

	private void combinationsSum3Helper(int index, int k, int n, int sum, int cnt, List<List<Integer>> result, List<Integer> one) {
		if (cnt > k || index >= 10) {
			return;
		}

		if (cnt == k && sum == n) {
			result.add(new LinkedList<>(one));
			return;
		}

		one.add(index);
		combinationsSum3Helper(index + 1, k, n, sum + index, cnt + 1, result, one);
		one.remove(one.size() - 1);
		combinationsSum3Helper(index + 1, k, n, sum, cnt, result, one);
	}

	// https://leetcode.com/problems/contains-duplicate-iii/description/
	public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
//		TreeSet
		return true;
	}

	// https://leetcode.com/problems/maximal-square/description/
	public int maximalSquare(char[][] matrix) {
		if (matrix == null || matrix.length <= 0
				|| matrix[0] == null || matrix[0].length <= 0) {
			return 0;
		}

		int[][] dp = new int[matrix.length][matrix[0].length];

		dp[0][0] = matrix[0][0] == '1' ? 1 : 0;
		int result = dp[0][0];

		for (int i = 1; i < matrix.length; ++i) {
			dp[i][0] = matrix[i][0] == '1' ? 1 : 0;
			result = Math.max(result, dp[i][0]);
		}

		for (int j = 1; j < matrix[0].length; ++j) {
			dp[0][j] = matrix[0][j] == '1' ? 1 : 0;
			result = Math.max(result, dp[0][j]);
		}

		for (int i = 1; i < matrix.length; ++i) {
			for (int j = 1; j < matrix[0].length; ++j) {
				dp[i][j] = matrix[i][j] == '0' ? 0 : Math.min(dp[i-1][j-1], Math.min(dp[i-1][j], dp[i][j-1])) + 1;
				result = Math.max(result, dp[i][j]);
			}
		}

		return result * result;
	}

	// https://leetcode.com/problems/majority-element-ii/description/
	public List<Integer> majorityElement(int[] nums) {
		List<Integer> result = new LinkedList<>();
		if (nums == null || nums.length <= 0) {
			return result;
		}

		if (nums.length == 1) {
			result.add(nums[0]);
			return result;
		}

		int a = 0, b = 0;
		int acnt = 0;
		int bcnt = 0;

		for (int i = 0; i < nums.length; ++i) {
			if (acnt == 0 && bcnt == 0) {
				a = nums[i];
				acnt = 1;
			} else if (acnt == 0 && bcnt != 0) {
				if (nums[i] == b) {
					bcnt++;
				} else {
					acnt++;
					a = nums[i];
				}
			} else if (acnt != 0 && bcnt == 0) {
				if (nums[i] == a) {
					acnt++;
				} else {
					b = nums[i];
					bcnt++;
				}
			} else {
				if (nums[i] == a) {
					acnt++;
				} else if (nums[i] == b) {
					bcnt++;
				} else {
					acnt--;
					bcnt--;
				}
			}
		}

		int realcnta = 0;
		int realcntb = 0;

		for (int num : nums) {
			if (acnt > 0 && num == a) {
				realcnta++;
			}
			if (bcnt > 0 && num == b) {
				realcntb++;
			}
		}

		if (realcnta > nums.length / 3) {
			result.add(a);
		}
		if (realcntb > nums.length / 3) {
			result.add(b);
		}

		return result;

	}

	// https://leetcode.com/problems/majority-element/description/
	public int majorityElementI(int[] nums) {
		if (nums == null || nums.length <= 0) {
			return -1;
		}

		int cnt = 1;
		int majority = nums[0];

		for (int i = 1; i < nums.length; ++i) {
			if (cnt == 0) {
				majority = nums[i];
			} else if (majority == nums[i]) {
				cnt++;
			} else {
				cnt--;
			}
		}

		return majority;
	}

	// https://leetcode.com/problems/product-of-array-except-self/description/
	public int[] productExceptSelf(int[] nums) {
		if (nums == null || nums.length <= 0) {
			return new int[0];
		}

		int[] preProduct = new int[nums.length];
		int[] result = new int[nums.length];

		preProduct[0] = nums[0];
		for (int i = 1; i < nums.length; ++i) {
			preProduct[i] = preProduct[i-1] * nums[i];
		}

		int suffixProduct = 1;
		for (int j = nums.length - 1; j >= 0; --j) {
			result[j] = (j == 0 ? 1 : preProduct[j - 1]) * suffixProduct;
			suffixProduct *= nums[j];
		}

		return result;
	}

	// https://leetcode.com/problems/h-index/description/
	public int hIndex(int[] citations) {
		return 0;
	}

	// https://leetcode.com/problems/perfect-squares/description/
	public int numSquares(int n) {
		if (n <= 0) {
			return 0;
		}

		int[] dp = new int[n+1];
		dp[0] = 0;

		for (int i = 1; i <= n; ++i) {
			final int sqrt = (int) Math.sqrt(i);
			int temp = Integer.MAX_VALUE;
			for (int j = 1; j <= sqrt; ++j) {
				temp = Math.min(temp, dp[i - j * j] + 1);
			}
			dp[i] = temp;
		}

		return dp[n];
	}

	// https://leetcode.com/problems/find-the-duplicate-number/description/
	public int findDuplicate(int[] nums) {

		if (nums == null || nums.length <= 1) {
			return -1;
		}

		// 找到环，就可以了
		int slowIndex = 0;
		int fastIndex = 0;

		do {
			slowIndex = nums[slowIndex];
			fastIndex = nums[nums[fastIndex]];
		} while(slowIndex != fastIndex);


		slowIndex = 0;

		while (slowIndex != fastIndex) {
			slowIndex = nums[slowIndex];
			fastIndex = nums[fastIndex];
		}
		// 找到环的位置
		return slowIndex;
	}

	// https://leetcode.com/problems/additive-number/description/
	public boolean isAdditiveNumber(String num) {
		for (int i = 0; i < num.length() - 1; ++i) {
			for (int j = i + 1; j < num.length() && (num.length() - (i + 1) - (j - i)) >= Math.max(i+1, j-i); ++j) {
				final int oneStart = 0;
				final int oneEnd = i;
				final int twoStart = oneEnd + 1;
				final int twoEnd = j;

				if (!getInvaliddNumber(num, oneStart, oneEnd)) {
					continue;
				}

				if (!getInvaliddNumber(num, twoStart, twoEnd)) {
					continue;
				}

				if (isAdditiveNumber(num, twoEnd + 1,
						num.substring(oneStart, oneEnd + 1),
						num.substring(twoStart, twoEnd + 1))) {
					return true;
				}
			}
		}

		return false;
	}

	private boolean getInvaliddNumber(CharSequence num, int start, int end) {
		final int length = end - start + 1;
		if (num.charAt(start) == '0' && length > 1) {
			return false;
		} else {
			return true;
		}
	}

	private boolean isAdditiveNumber(CharSequence num, int start, String one, String two) {
		while (start < num.length()) {
			final String next = addNumber(one, two);
			if (start + next.length() > num.length()) {
				return false;
			}

			int nextIdx = isAdditiveEqual(num, start, next);
			if (nextIdx == -1) {
				return false;
			}

			one = two;
			two = next;
			start = nextIdx;

			if (start == num.length()) {
				return true;
			}
		}

		return false;
	}

	private int isAdditiveEqual(CharSequence num, int start, String equal) {
		int i = start;
		int j = 0;

		if (start + equal.length() > num.length()) {
			return -1;
		}

		while (i < num.length() && j < equal.length()) {
			if (num.charAt(i) == equal.charAt(j)) {
				i++;
				j++;
			} else {
				return -1;
			}
		}

		return j == equal.length() ? i : -1;
	}

	private String addNumber(String one, String two) {
		String result = "";

		int i = one.length() - 1;
		int j = two.length() - 1;

		int add = 0;
		while (i >= 0 && j >= 0) {
			final char onec = one.charAt(i--);
			final char towc = two.charAt(j--);
			int newc = onec - '0' + towc - '0' + add;
			if (newc >= 10) {
				newc -= 10;
				add = 1;
			} else {
				add = 0;
			}


			result = String.valueOf(newc) + result;
		}


		while (i >= 0) {
			final char onec = one.charAt(i--);
			int newc = onec - '0' + add;
			if (newc >= 10) {
				newc -= 10;
				add = 1;
			} else {
				add = 0;
			}
			result = String.valueOf(newc) + result;
		}

		while (j >= 0) {
			final char onec = two.charAt(j--);
			int newc = onec - '0' + add;
			if (newc >= 10) {
				newc -= 10;
				add = 1;
			} else {
				add = 0;
			}
			result = String.valueOf(newc) + result;
		}

		if (add == 1) {
			result = "1" + result;
		}

		return result;
	}

	// https://leetcode.com/problems/super-ugly-number/description/
	public int nthSuperUglyNumber(int n, int[] primes) {
		return 0;
	}

	// https://leetcode.com/problems/maximum-product-of-word-lengths/description/
	public int maxProduct(String[] words) {
		return 0;
	}

	// https://leetcode.com/problems/wiggle-sort-ii/description/
	public void wiggleSort(int[] nums) {

	}

	// https://leetcode.com/problems/reconstruct-itinerary/description/
	private int itineraryCnt = 0;
	public List<String> findItinerary(String[][] tickets) {
		List<String> result = new LinkedList<>();
		if (tickets == null || tickets.length <= 0 || tickets[0] == null || tickets[0].length <= 0) {
			return result;
		}

		Map<String, LinkedList<String>> graph = new HashMap<>();
		for (String[] ticket : tickets) {
			LinkedList<String> tos = graph.get(ticket[0]);
			if (tos == null) {
				tos = new LinkedList<>();
				graph.put(ticket[0], tos);
			}
			tos.add(ticket[1]);
		}

		for (List<String> tos : graph.values()) {
			Collections.sort(tos);
		}

		itineraryCnt = tickets.length + 1;

		List<String> temp = new LinkedList<>();
		temp.add("JFK");

		findItineraryHelper(graph, "JFK", temp, result);

		return result;
	}

	private void findItineraryHelper(Map<String, LinkedList<String>> graph, String start, List<String> backStrace, List<String> result) {

		if (result.size() > 0) {
			return;
		}

		LinkedList<String> tos = graph.get(start);
		if (tos == null) {
			return;
		}

		for (int i = 0; i < tos.size(); ++i) {
			final String to = tos.removeFirst();

			backStrace.add(to);

			if (backStrace.size() == itineraryCnt) {
				result.addAll(backStrace);
				return;
			} else {
				findItineraryHelper(graph, to, backStrace, result);
			}

			backStrace.remove(backStrace.size() - 1);

			tos.addLast(to);
		}
	}

	// https://leetcode.com/problems/increasing-triplet-subsequence/description/
	public boolean increasingTriplet(int[] nums) {
		if (nums == null || nums.length < 3) {
			return false;
		}

		int min = Integer.MAX_VALUE;
		int min2 = Integer.MAX_VALUE;
		int cnt = 0;
		int next = Integer.MAX_VALUE;
		for (int num : nums) {
			if (num > next) {
				return true;
			}

			if (cnt == 0) {
				min2 = num;
				cnt = 1;
			} else if (cnt == 1) {
				if (num > min2) {
					min = min2;
					min2 = num;
					cnt = 2;
				} else {
					min2 = num;
				}
			} else if (min2 > num) {
				if (min >= num) {
					min2 = num;
					cnt = 1;
				} else {
					min2 = num;
				}
			}

			if (cnt == 2) {
				next = Math.min(next, min2);
			}

		}

		return false;
	}

	// https://leetcode.com/problems/integer-break/description/
	public int integerBreak(int n) {
		int[] dp = new int[n+1];
		dp[1] = 1;
		for (int i = 2; i <= n; ++i) {
			int temp = Integer.MIN_VALUE;
			for (int j = 1; j < i; ++j) {
				temp = Math.max(temp, Math.max(dp[j] * (i - j), j * (i - j)) );
			}
			dp[i] = temp;
		}

		return dp[n];
	}

	// https://leetcode.com/problems/count-numbers-with-unique-digits/description/
	public int countNumbersWithUniqueDigits(int n) {
		if (n == 0) {
			return 1;
		}
		if (n == 1) {
			return 10;
		}

		int pre = 10;
		int m = 1;
		for (int i = 2; i <= n; ++i) {
			m *= (10 - i + 1);
			pre = pre + 9 * m;
		}

		return pre;
	}

	// https://leetcode.com/problems/water-and-jug-problem/description/
	private int jugX, jugY, jugZ;
	public boolean canMeasureWater(int x, int y, int z) {
		jugX = x;
		jugY = y;
		jugZ = z;

		if (jugX + jugY < jugZ) {
			return false;
		}

		if (jugZ == jugX + jugZ || jugZ == jugX || jugZ == jugY) {
			return true;
		}

		Set<JugPair> set = new HashSet<>();
		set.add(new JugPair(0, 0));

		return canMeasureWaterHelper(0, 0, set);
	}

	private class JugPair {
		public int jugX;
		public int jugY;

		public JugPair(int x, int y) {
			jugX = x;
			jugY = y;
		}

		@Override
		public boolean equals(Object obj) {
			if (obj == null || !(obj instanceof JugPair)) {
				return false;
			}

			return jugX == ((JugPair) obj).jugX && jugY == ((JugPair) obj).jugY;
		}

		@Override
		public int hashCode() {
			return Objects.hash(jugX, jugY);
		}
	}

	private boolean canMeasureWaterHelper(int leftx, int lefty, Set<JugPair> set) {
		if (leftx == jugZ || lefty == jugZ || leftx + lefty == jugZ) {
			return true;
		}

		// x
		// 倒满
		if (canMeasureWater(leftx, lefty, jugX, lefty, set)) {
			return true;
		}

		// 清空
		if (canMeasureWater(leftx, lefty, 0, lefty, set)) {
			return true;
		}

		// 倒入y
		int toFullY = jugY - lefty;
		if (leftx > 0) {
			if (leftx < toFullY) {
				if (canMeasureWater(leftx, lefty, 0, lefty + leftx, set)) {
					return true;
				}
			} else {
				if (canMeasureWater(leftx, lefty, leftx - toFullY, jugY, set)) {
					return true;
				}
			}
		}

		// y
		if (canMeasureWater(leftx, lefty, leftx, jugY, set)) {
			return true;
		}

		// 清空
		if (canMeasureWater(leftx, lefty, leftx, 0, set)) {
			return true;
		}

		// 倒入x
		int toFullX = jugX - leftx;
		if (lefty > 0) {
			if (lefty < toFullX) {
				if (canMeasureWater(leftx, lefty, leftx + lefty, 0, set)) {
					return true;
				}
			} else {
				if (canMeasureWater(leftx, lefty, jugX, lefty - toFullX, set)) {
					return true;
				}
			}
		}

		return false;

	}

	private boolean canMeasureWater(int oldx, int oldy, int leftx, int lefty, Set<JugPair> set) {
		if (oldx == leftx && oldy == lefty) {
			return false;
		}

		boolean result = false;
		JugPair pair = new JugPair(leftx, lefty);

		if (set.add(pair)) {
			System.out.println("" + leftx + "," + lefty);
			result = canMeasureWaterHelper(leftx, lefty, set);
		}

		return result;
	}

	// https://leetcode.com/problems/find-k-pairs-with-smallest-sums/description/
	public List<int[]> kSmallestPairs(int[] nums1, int[] nums2, int k) {
		List<int[]> result = new LinkedList<>();
		if (nums1 == null || nums1.length <= 0 || nums2 == null || nums2.length <= 0 ||
				k <= 0 || k > nums1.length * nums2.length) {
			return result;
		}

		PriorityQueue<int[]> minHeap = new PriorityQueue<>(new Comparator<int[]>() {
			@Override
			public int compare(int[] o1, int[] o2) {
				return nums1[o1[0]] + nums2[o1[1]] - nums1[o2[0]] - nums2[o2[1]];
			}
		});

		for (int i = 0; i < nums1.length && i < k; ++i) {
			minHeap.add(new int[]{i, 0});
		}

		while (k-- > 0 && !minHeap.isEmpty()) {
			int[] min = minHeap.poll();
			result.add(new int[]{nums1[min[0]], nums2[min[1]]});
			if (min[1] == nums2.length - 1) {
				continue;
			}
			minHeap.add(new int[]{min[0], min[1] + 1});
		}

		return result;
	}

	// https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/description/
	public int kthSmallest(int[][] matrix, int k) {
		final int n = matrix.length;
		if (k == 1) {
			return matrix[0][0];
		} else if (k >= n * n) {
			return matrix[n-1][n-1];
		}

		PriorityQueue<int[]> minHeap = new PriorityQueue<>(new Comparator<int[]>() {
			@Override
			public int compare(int[] o1, int[] o2) {
				return matrix[o1[0]][o1[1]] - matrix[o2[0]][o2[1]];
			}
		});

		for (int j = 0; j < n; ++j) {
			minHeap.add(new int[]{0, j});
		}

		int result = 0;
		while (k-- > 0 && !minHeap.isEmpty()) {
			int[] min = minHeap.poll();
			result = matrix[min[0]][min[1]];
			if (min[0] >= n - 1) {
				continue;
			}
			minHeap.add(new int[]{min[0] + 1, min[1]});
		}

		return result;
	}

	// https://leetcode.com/problems/guess-number-higher-or-lower-ii/description/
	public int getMoneyAmount(int n) {
		if (n <= 1) {
			return 0;
		}

		int[][] dp = new int[n][n];

		for (int i = 0; i < n; ++i) dp[i][i] = 0;
		for (int i = 0; i < n - 1; ++i) dp[i][i+1] = i + 1;

		for (int k = 3; k <= n; k++) {
			for (int i = 0; i + k <= n; ++i) {
				int result = Integer.MAX_VALUE;
				for (int j = i; j < i + k; ++j) {
					result = Math.min(result, j+1 +
							Math.max(j > i ? dp[i][j-1] : 0,
									j < i + k - 1 ? dp[j+1][i+k-1] : 0));
				}
				dp[i][i+k-1] = result;
			}
		}

		return dp[0][n-1];
	}

	// https://leetcode.com/problems/wiggle-subsequence/description/
	public int wiggleMaxLength(int[] nums) {
		return 0;
	}

	// https://leetcode.com/problems/combination-sum-iv/description/
	private int tempTarget;
	private int tempCnt;
	public int combinationSum4(int[] nums, int target) {
		tempTarget = target;
		tempCnt = 0;
		combinationSum4Helper(nums, 0, 0, new LinkedList<>());
		return tempCnt;
	}

	private void combinationSum4Helper(int[] nums, int start, int sum, List<Integer> backs) {
//		if (sum > tempTarget) {
//			return;
//		}
//
//		if (sum == tempTarget) {
//			tempCnt++;
//			System.out.println(backs);
//			return;
//		}

		if (start == nums.length)
			System.out.println(backs);

		for (int i = start; i < nums.length; ++i) {
			swap(nums, start, i);

			int j;
//			for (j = 0; j <= (tempTarget - sum) / nums[start]; ++j) {
			backs.add(nums[start]);
			combinationSum4Helper(nums, start + 1, sum, backs);
			backs.remove(backs.size()-1);
//			}

//			while (--j >= 0) {
//				backs.remove(backs.size()-1);
//			}


			swap(nums, start, i);
		}
	}


	// https://leetcode.com/problems/lexicographical-numbers/description/
	public List<Integer> lexicalOrder(int n) {
		List<Integer> result = new ArrayList<>(n);

		lexicalOrder(1, n, result);

		return result;
	}

	private void lexicalOrder(int start, int n, List<Integer> result) {
		if (start > n) {
			return;
		}
		result.add(start);
		lexicalOrder(start * 10, n, result);
		if (start % 10 != 9) {
			lexicalOrder(start + 1, n, result);
		}
	}

	// https://leetcode.com/problems/mini-parser/description/
	public NestedInteger deserialize(String s) {
		Stack<Object> stack = new Stack<>();
		if (s == null || s.length() <= 0) {
			return null;
		}

		String temp = "";
		Stack<NestedInteger> back = new Stack<>();
		NestedInteger result = null;
		for (int i = 0; i < s.length(); ++i) {
			final char c = s.charAt(i);
			if (c == '[') {
				stack.push("[");
			} else if (c == ',') {
				if (temp.length() > 0) {
					stack.push(temp);
					temp = "";
				}
			} else if ((c >= '0' && c <= '9') || c == '-') {
				temp = temp + c;
			} else {
				if (temp.length() > 0) {
					stack.push(temp);
					temp = "";
				}

				NestedInteger nestedInteger = new NestedInteger();
				do {
					Object object = stack.pop();
					if (object instanceof String) {
						if (object.equals("[")) {
							while (!back.isEmpty()) {
								nestedInteger.add(back.pop());
							}
							stack.push(nestedInteger);
							break;
						} else {
							back.push(new NestedInteger(Integer.valueOf((String) object)));
						}
					} else if (object instanceof NestedInteger){
						back.push((NestedInteger) object);
					}
				} while (true);

				result = nestedInteger;
			}
		}

		if (temp.length() > 0) {
			return new NestedInteger(Integer.valueOf(temp));
		}

		return result;
	}

	// https://leetcode.com/problems/is-subsequence/description/
	public boolean isSubsequence(String s, String t) {
		if (s.length() <= 0) {
			return true;
		}

		final int scnt = s.length();
		final int tcnt = t.length();

		if (scnt > tcnt) {
			return false;
		}

		int end = -1;
		for (int i = 0; i < scnt; ++i) {
			final char c = s.charAt(i);

			if (scnt - i > tcnt - end - 1) {
				return false;
			}
			end = t.indexOf(c, end+1);
			if (end == -1) {
				return false;
			}
		}

		return true;
	}

	String s = "2[abc]3[cd]ef";
	// https://leetcode.com/problems/decode-string/description/
	public String decodeString(String s) {
		if (s == null || s.length() <= 0) {
			return "";
		}

		Stack<Object> stack = new Stack<>();

		StringBuilder builder = new StringBuilder();

		int time = 0;
		int strStart = -1;

		for (int i = 0; i < s.length(); ++i) {
			final char c = s.charAt(i);
			if (c == '[') {
				stack.push(time);
				time = 0;
			} else if (c == ']') {
				if (strStart != -1) {
					stack.push(s.substring(strStart, i));
					strStart = -1;
				}

				String temp = "";
				do {
					Object object = stack.pop();
					if (object instanceof String) {
						temp = ((String) object) + temp;
					} else {
						int times = (Integer) object;
						while (times-- > 0) {
							stack.push(temp);
						}
						break;
					}
				} while (true);

			} else if (c >= '0' && c <= '9') {
				if (strStart != -1) {
					stack.push(s.substring(strStart, i));
					strStart = -1;
				}
				time = time * 10 + c - '0';
			} else {
				if (strStart == -1) {
					strStart = i;
				}
			}
		}

		if (strStart != -1) {
			stack.push(s.substring(strStart));
		}

		while (!stack.isEmpty()) {
			builder.insert(0, stack.pop());
		}

		return builder.toString();
	}

	// https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/description/
	public int longestSubstring1(String s, int k) {
		if (k <= 1) {
			return s.length();
		}

		int[] hash = new int[26];

		int charKind = 0;
		int charCnt = 0;

		int result = 0;

		int start = 0;
		int end = 0;

		while (end < s.length()) {
			final int idx = s.charAt(end) - 'a';
			if (hash[idx] == 0) {
				charKind++;
			}
			if (hash[idx] < k) {
				charCnt++;
			}
			hash[idx]++;

			if (charCnt >= charKind * k) {
				result = Math.max(result, end - start + 1);
			}
		}

		return 0;
	}

	// https://leetcode.com/problems/evaluate-division/description/
	public double[] calcEquation(String[][] equations, double[] values, String[][] queries) {
		Map<String, Map<String, Double>> graph = new HashMap<>();

		for (int i = 0; i < values.length; ++i) {
			final String[] equation = equations[i];
			final double value = values[i];

			Map<String, Double> to = graph.get(equation[0]);
			if (to == null) {
				to = new HashMap<>();
				graph.put(equation[0], to);
			}

			to.put(equation[1], value);

			to = graph.get(equation[1]);
			if (to == null) {
				to = new HashMap<>();
				graph.put(equation[1], to);
			}

			to.put(equation[0], 1.0 / value);
		}

		double[] result = new double[queries.length];

		for (int i = 0; i < queries.length; ++i) {
			result[i] = calcEquation(graph, queries[i][0], queries[i][1]);
		}

		return result;
	}

	private class EquationNode {
		public final String divide;
		public final double value;
		public EquationNode(String divide, double value) {
			this.divide = divide;
			this.value = value;
		}
	}

	private double calcEquation(Map<String, Map<String, Double>> graph, String toDivide, String divide) {
		Map<String, Double> divides = graph.get(toDivide);

		if (divides == null) {
			return -1.0;
		}

		if (toDivide.equals(divide)) {
			return 1.0;
		}

		if (divides.containsKey(divide)) {
			return divides.get(divide);
		}

		Set<String> traveled = new HashSet<>();
		Queue<EquationNode> queue = new LinkedList<>();

		traveled.add(toDivide);
		for (Map.Entry<String, Double> entry : divides.entrySet()) {
			queue.add(new EquationNode(entry.getKey(), entry.getValue()));
		}

		while (!queue.isEmpty()) {
			EquationNode head = queue.poll();

			Map<String, Double> temp = graph.get(head.divide);

			if (temp != null) {
				for (Map.Entry<String, Double> entry : temp.entrySet()) {
					if (entry.getKey().equals(divide)) {
						return head.value * entry.getValue();
					}
					if (traveled.add(entry.getKey())) {
						queue.add(new EquationNode(entry.getKey(), head.value * entry.getValue()));
					}
				}
			}
		}

		return -1.0;
	}

	// https://leetcode.com/problems/arithmetic-slices/description/
	public int numberOfArithmeticSlices(int[] A) {
		if (A == null || A.length <= 2) {
			return 0;
		}

		int[] dp = new int[A.length];
		dp[0] = 0;
		dp[1] = 0;
		for (int i = 2; i < A.length; ++i) {
			if (A[i] - A[i-1] == A[i-1] - A[i-2]) {
				dp[i] = dp[i-1] + 1;
			}
		}

		int result = 0;
		int end = A.length - 1;
		while (end >= 0) {
			if (dp[end] == 0) {
				end--;
			} else {
				result += (dp[end] + 1) * dp[end] / 2;
				end = end - dp[end] - 1;
			}
		}

		return result;
	}

	// https://leetcode.com/problems/longest-repeating-character-replacement/description/
	public int characterReplacement(String s, int k) {
		if (s == null || s.length() <= 0) {
			return 0;
		}

		if (k >= s.length()) {
			return s.length();
		}

		int result = 0;
		for (char i = 'A'; i <= 'Z'; ++i) {
			result = Math.max(result, characterMaxLenth(s, i, k));
		}

		return result;
	}

	private int characterMaxLenth(CharSequence s, char compare, int maxDif) {
		int start = 0;
		int end = 0;

		int dif = 0;
		int result = 0;
		while (end < s.length()) {
			char c = s.charAt(end);
			if (c != compare) {
				dif++;

				while (dif > maxDif && start <= end) {
					if (s.charAt(start++) != compare) {
						dif--;
					}
				}
			}

			result = Math.max(result, end - start + 1);
			end++;
		}

		return result;
	}

	// https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/description/
	class Node {
		public int val;
		public Node prev;
		public Node next;
		public Node child;

		public Node() {}

		public Node(int _val,Node _prev,Node _next,Node _child) {
			val = _val;
			prev = _prev;
			next = _next;
			child = _child;
		}
	};

	private Node pre = null;
	public Node flatten(Node head) {
		pre = null;

		flattenHelper(head);

		return head;
	}

	private void flattenHelper(Node head) {
		if (head == null) {
			return;
		}

		final Node child = head.child;
		final Node next = head.next;

		if (pre != null) {
			pre.next = head;
			pre.child = null;
			head.prev = pre;
		}
		pre = head;

		flatten(child);
		flatten(next);
	}

	// https://leetcode.com/problems/non-overlapping-intervals/description/

	public int eraseOverlapIntervals(Interval[] intervals) {
		if (intervals == null || intervals.length <= 1) {
			return 0;
		}

		Arrays.sort(intervals, new Comparator<Interval>() {
			@Override
			public int compare(Interval o1, Interval o2) {
				if (o1.start < o2.start) {
					return -1;
				} else if (o1.start > o2.start) {
					return 1;
				} else if (o1.end < o2.end) {
					return -1;
				} else if (o1.end > o2.end) {
					return 1;
				} else {
					return 0;
				}
			}
		});

		int cnt = 0;
		int max = intervals[0].end;

		for (int i = 1; i < intervals.length; ++i) {
			if (intervals[i].start >= max) {
				max = intervals[i].end;
			} else {
				cnt++;
				max = Math.min(intervals[i].end, max);
			}
		}

		return cnt;
	}

	// https://leetcode.com/problems/find-right-interval/description/
	public int[] findRightInterval(Interval[] intervals) {
		if (intervals == null || intervals.length <= 0) {
			return new int[0];
		}

		Integer[] idxs = new Integer[intervals.length];
		for (int i = 0; i < idxs.length; ++i) {
			idxs[i] = i;
		}

		Arrays.sort(idxs, new Comparator<Integer>() {
			@Override
			public int compare(Integer o1, Integer o2) {
				if (intervals[o1].start < intervals[o2].start) {
					return -1;
				} else if (intervals[o1].start > intervals[o2].start) {
					return 1;
				} else if (intervals[o1].end < intervals[o2].end) {
					return -1;
				} else if (intervals[o1].end > intervals[o2].end) {
					return 1;
				} else {
					return 0;
				}
			}
		});

		int[] result = new int[intervals.length];

		for (int i = 0; i < result.length; ++i) {
			result[i] = findRightIntervalIdx(intervals, idxs, i);
		}

		return result;
	}

	private int findRightIntervalIdx(Interval[] intervals, Integer[] idxs, int find) {
		int left = 0;
		int right = idxs.length - 1;

		while (left <= right) {
			int middle = left + (right - left) / 2;
			if (intervals[idxs[middle]].start >= intervals[find].end) {
				right = middle - 1;
			} else {
				left = middle + 1;
			}
		}

		return left >= idxs.length ? -1 : idxs[left];
	}

	// https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/description/
	public int findMinArrowShots(int[][] points) {
		if (points == null || points.length <= 0) {
			return 0;
		}

		Arrays.sort(points, new Comparator<int[]>() {
			@Override
			public int compare(int[] o1, int[] o2) {
				return o1[1] - o2[1];
			}
		});

		int max = points[0][1];
		int cnt = 1;
		for (int i = 1; i < points.length; ++i) {
			if (points[i][0] > max) {
				cnt++;
				max = points[i][1];
			}
		}

		return cnt;
	}

	// https://leetcode.com/problems/132-pattern/description/
	public boolean find132pattern(int[] nums) {
		if (nums == null || nums.length < 3) {
			return false;
		}

		Stack<Integer> stack = new Stack<>();

		int s2 = Integer.MIN_VALUE;

		for (int i = nums.length - 1; i >= 0; --i) {
			if (nums[i] < s2) {
				return true;
			}
			while (!stack.isEmpty() && stack.peek() < nums[i]) {
				s2 = stack.pop();
			}
			stack.push(nums[i]);
		}

		return false;
	}

	// https://leetcode.com/problems/score-of-parentheses/description/
	public int scoreOfParentheses(String S) {
		if (S == null || S.length() <= 0) {
			return 0;
		}

		int score = 0;
		Stack<Integer> stack = new Stack<>();
		for (int i = 0; i < S.length(); ++i) {
			final char c = S.charAt(i);
			if (c == '(') {
				stack.push(-1);
			} else {
				int temp = 0;
				do {
					final int top = stack.pop();
					if (top == -1) {
						stack.push(temp == 0 ? 1 : 2 * temp);
						break;
					} else {
						temp += top;
					}
				} while (true);
			}
		}

		while (!stack.isEmpty()) {
			score += stack.pop();
		}

		return score;
	}

	// https://leetcode.com/problems/exclusive-time-of-functions/description/

	private class FuncNode {
		int id;
		int startTime;
		int endTime;
	}

	public int[] exclusiveTime(int n, List<String> logs) {

		int[] result = new int[n];

		Stack<FuncNode> stack = new Stack<>();
		for (String log : logs) {
			String[] split = log.split(":");
			final int curId = Integer.valueOf(split[0]);
			final int curTime = Integer.valueOf(split[2]);
			if (split[1].equals("start")) {
				FuncNode node = new FuncNode();
				node.startTime = curTime;
				node.id = curId;
				stack.push(node);
			} else {
				int otherConsume = 0;
				do {
					FuncNode top = stack.pop();
					if (top.id == curId) {
						final int consume = curTime - top.startTime + 1 - otherConsume;
						result[top.id] += consume;
						FuncNode node = new FuncNode();
						node.id = -1;
						node.startTime = top.startTime;
						node.endTime = curTime;
						stack.push(node);
						break;
					} else {
						otherConsume += top.endTime - top.startTime + 1;
					}
				} while (true);
			}
		}

		return result;
	}

	// https://leetcode.com/problems/minimum-moves-to-equal-array-elements-ii/description/
	public int minMoves2(int[] nums) {
		if (nums == null || nums.length <= 1) {
			return 0;
		}

		// 大的那一堆
		PriorityQueue<Integer> minHeap = new PriorityQueue<>();
		// 小的那一堆
		PriorityQueue<Integer> maxHeap = new PriorityQueue<>(new Comparator<Integer>() {
			@Override
			public int compare(Integer o1, Integer o2) {
				return o2 - o1;
			}
		});

		for (int num : nums) {
			// 每次都尝试往小的那一堆添加
			if (maxHeap.isEmpty()) {
				maxHeap.add(num);
			} else if (num > maxHeap.peek()) {
				minHeap.add(num);
				if (minHeap.size() > nums.length / 2) {
					maxHeap.add(minHeap.poll());
				}
			} else {
				maxHeap.add(num);
				if (maxHeap.size() > nums.length - (nums.length / 2)) {
					minHeap.add(maxHeap.poll());
				}

			}
		}

		final int middle = maxHeap.peek();
		int result = 0;
		for (int num : nums) {
			result += Math.abs(num - middle);
		}

		return result;
	}

	// https://leetcode.com/problems/minimum-moves-to-equal-array-elements/description/
	public int minMoves(int[] nums) {
		if (nums == null || nums.length <= 1) {
			return 0;
		}

		Arrays.sort(nums);
		int result = (nums.length - 1) * nums[0];
		for (int i = 1; i < nums.length; ++i) {
			result += nums[i];
		}

		return result;
	}
	public int rand7() {
		return 1 + (int) (Math.random() / 7);
	}

	// https://leetcode.com/problems/implement-rand10-using-rand7/description/
	public int rand10() {
		return 0;
	}

	// https://leetcode.com/problems/ones-and-zeroes/description/
	public int findMaxForm(String[] strs, int m, int n) {

		int[][] cnts = new int[strs.length][2];
		for (int i = 0; i < cnts.length; ++i) {
			int zeroCnt = 0;
			for (int j = 0; j < strs[i].length(); ++j) {
				if (strs[i].charAt(j) == '0') {
					zeroCnt++;
				}
			}

			cnts[i][0] = (i == 0 ? 0 : cnts[i-1][0]) + zeroCnt;
			cnts[i][1] = (i == 0 ? 0 : cnts[i-1][1]) + strs[i].length() - zeroCnt;
		}

		return findMaxForm(cnts, 0, m, n);
	}

	private class MaxFormCacheNode {
		int idx;
		int m;
		int n;

		public MaxFormCacheNode(int idx, int m, int n) {
			this.idx = idx;
			this.m = m;
			this.n = n;
		}

		@Override
		public int hashCode() {
			return Objects.hash(idx, m, n);
		}

		@Override
		public boolean equals(Object obj) {
			return obj != null
					&& (obj instanceof MaxFormCacheNode)
					&& (idx == ((MaxFormCacheNode) obj).idx)
					&& (m == ((MaxFormCacheNode) obj).m)
					&& (n == ((MaxFormCacheNode) obj).n);
		}
	}

	private Map<MaxFormCacheNode, Integer> mMaxFormCache = new HashMap<>();

	private int findMaxForm(int[][] cnts, int idx, int m, int n) {
		if (idx >= cnts.length || (m <= 0 && n <= 0)) {
			return 0;
		}



		final int maxZeros = cnts[cnts.length-1][0] - (idx == 0 ? 0 : cnts[idx-1][0]);
		final int maxOnes = cnts[cnts.length-1][1] - (idx == 0 ? 0 : cnts[idx-1][1]);

		if (m >= maxZeros && n >= maxOnes) {
			return cnts.length - idx;
		}

		MaxFormCacheNode node = new MaxFormCacheNode(idx, m, n);
		int value = mMaxFormCache.getOrDefault(node, -1);
		if (value != -1) {
			return value;
		}

		int result = findMaxForm(cnts, idx + 1, m, n);

		final int zeros = cnts[idx][0] - (idx == 0 ? 0 : cnts[idx-1][0]);
		final int ones = cnts[idx][1] - (idx == 0 ? 0 : cnts[idx-1][1]);
		if (m >= zeros && n >= ones) {
			result = Math.max(result, 1 + findMaxForm(cnts, idx + 1, m - zeros, n - ones));
		}

		mMaxFormCache.put(node, result);

		return result;
	}

	// https://leetcode.com/problems/total-hamming-distance/description/
	public int totalHammingDistance(int[] nums) {
		int result = 0;
		int next = 1;
		for (int i = 0; i < 32; ++i) {

			int oneCnt = 0;
			for (int num : nums) {
				if ((num & next) != 0) {
					oneCnt++;
				}
			}

			result += oneCnt * (nums.length - oneCnt);
			next <<= 1;
		}

		return result;

//		int result = 0;
//		for (int i = 0; i < nums.length - 1; ++i) {
//			for (int j = i + 1; j < nums.length; ++j) {
//				result += get1Cnt(nums[i] ^ nums[j]);
//			}
//		}
//
//		return result;
	}

	private int get1Cnt(int num) {
		int result = 0;
		while (num != 0) {
			num &= num - 1;
			result++;
		}
		return result;
	}

	// https://leetcode.com/problems/random-flip-matrix/description/

	private int mFlipTotal;
	private final int mFlipRows;
	private final int mFlipCols;
	private final Random mRandom;
	private final Map<Integer, Integer> mMap;
	public Solution(int n_rows, int n_cols) {
		mFlipRows = n_rows;
		mFlipCols = n_cols;
		mFlipTotal = mFlipCols * mFlipRows;
		mRandom = new Random();
		mMap = new HashMap<>();
	}

	public int[] flip() {
		int random = mRandom.nextInt(mFlipTotal--);
		int result = mMap.getOrDefault(random, random);
		mMap.put(random, mMap.getOrDefault(mFlipTotal, mFlipTotal));
		return new int[]{result / mFlipCols, result % mFlipCols};
	}

	public void reset() {
		mMap.clear();
		mFlipTotal = mFlipCols * mFlipRows;
	}

	// https://leetcode.com/problems/contiguous-array/description/
	public int findMaxLength(int[] nums) {
		if (nums == null || nums.length <= 1) {
			return 0;
		}

		int oneCnt = 0;
		int zeroCnt = 0;
		int result = 0;
		Map<Integer, Integer> map = new HashMap<>();
		map.put(0, -1);

		for (int i = 0; i < nums.length; ++i) {
			if (nums[i] == 0) oneCnt++;
			else zeroCnt++;

			int diff = oneCnt - zeroCnt;
			if (map.containsKey(diff)) {
				result = Math.max(result, i - map.get(diff));
			} else {
				map.put(diff, i);
			}
		}

		return result;
	}

	// https://leetcode.com/problems/beautiful-arrangement/description/
	public int countArrangement(int N) {
		mCountArrangement = 0;
		int[] nums = new int[N];
		for (int i = 0; i < N; ++i) nums[i] = i + 1;
		countArrangementHelper(nums, 0);
		return mCountArrangement;
	}

	private int mCountArrangement = 0;
	private void countArrangementHelper(int[] nums, int start) {
		if (start >= nums.length) {
			mCountArrangement++;
			return;
		}

		for (int i = start; i < nums.length; ++i) {
			swap(nums, i, start);

			if ((start + 1) % nums[start] == 0 || nums[start] % (start + 1) == 0 ) {
				countArrangementHelper(nums, start + 1);
			}
			swap(nums, i, start);
		}
	}

	// https://leetcode.com/problems/minesweeper/description/
	int[][] dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
	public char[][] updateBoard(char[][] board, int[] click) {
		int x = click[0], y = click[1];
		if (board[x][y] == 'M'){
			board[x][y] = 'X';
			return board;
		}
		dfs(board, x, y, new boolean[board.length][board[0].length]);
		return board;
	}
	public void dfs(char[][] board, int x, int y, boolean[][] visit){
		if (x < 0 || y < 0 || x >= board.length || y >= board[0].length || visit[x][y] || board[x][y] == 'M'){
			return;
		}
		visit[x][y] = true;
		char cur = getMines(board, x, y);
		board[x][y] = cur;
		if (cur == 'B'){
			for (int[] dir : dirs){
				int i = x + dir[0];
				int j = y + dir[1];
				dfs(board, i, j, visit);
			}
		}

	}
	public char getMines(char[][] board, int x, int y){
		int res = 0;
		for (int[] dir : dirs){
			int i = x + dir[0];
			int j = y + dir[1];
			if (i < 0 || j < 0 || i >= board.length || j >= board[0].length){
				continue;
			}
			res += board[i][j] == 'M' ? 1 : 0;
		}
		return res == 0 ? 'B' : (char)(res + '0');
	}

	// https://leetcode.com/problems/optimal-division/description/
	public String optimalDivision(int[] nums) {
		return null;
	}

	private int optimalDivisionMax(int[] nums, int start) {

		return nums[0] / optimalDivisionMin(nums, start + 1);
	}

	private int optimalDivisionMin(int[] nums, int start) {
		return nums[0] / optimalDivisionMax(nums, start + 1);
	}

	// https://leetcode.com/problems/next-greater-element-iii/description/
	public int nextGreaterElement(int n) {
		if (n <= 0) {
			return -1;
		}

		final int oldN = n;

		int[] nums = new int[13];
		nums[0] = n % 10;
		n /= 10;
		int i = 1;
		boolean hasGreat = false;
		while (n != 0) {
			nums[i] = n % 10;
			n /= 10;
			if (nums[i] < nums[i-1]) {
				hasGreat = true;
				break;
			}
			i++;
		}

		if (!hasGreat) {
			return -1;
		}

		for (int j = 0; j < i; ++j) {
			if (nums[j] > nums[i]) {
				swap(nums, i, j);
				break;
			}
		}

		Arrays.sort(nums, 0, i);

		int multi = 1;
		int result = 0;
		for (int j = i - 1; j >= 0; j--) {
			result += nums[j] * multi;
			multi *= 10;
		}


		result =  n * multi * 10 + result + nums[i] * multi;
		return result > oldN ? result : -1;
	}

	// https://leetcode.com/problems/delete-operation-for-two-strings/description/
	public int minDistance(String word1, String word2) {
		if (word1 == null && word2 == null) {
			return 0;
		}
		if (word1 == null) {
			return word2.length();
		} else if (word2 == null) {
			return word1.length();
		}

		int[][] dp = new int[word1.length() + 1][word2.length() + 1];
		for (int i = 1; i <= word1.length(); ++i) {
			for (int j = 1; j <= word2.length(); ++j) {
				int result = (word1.charAt(i-1) == word2.charAt(j-1)) ? dp[i-1][j-1] + 1: 0;

				result = Math.max(result, Math.max(dp[i][j-1], dp[i-1][j]));

				dp[i][j] = result;
			}
		}

		return word1.length() + word2.length() - 2 * dp[word1.length()][word2.length()];
	}

	// https://leetcode.com/problems/valid-triangle-number/description/
	public int triangleNumber(int[] nums) {
		if (nums == null || nums.length <= 2) {
			return 0;
		}

		Arrays.sort(nums);

		int result = 0;
		for (int i = nums.length - 1; i >= 2; --i) {
			result += twoSumGreatCnt(nums, 0, i - 1, nums[i]);
		}

		return result;
	}

	public int twoSumGreatCnt(int[] nums, int start, int end, int great) {
		int left = start;
		int right = end;

		int result = 0;
		while (left < right) {
			while (left < right && nums[left] + nums[right] <= great) {
				left++;
			}

			if (left < right) {
				result += right - left;
			}

			right--;
		}

		return result;
	}

	// https://leetcode.com/problems/solve-the-equation/description/
	public String solveEquation(String equation) {
		int[] leftValues = {0, 0};
		int rightIdx = solveEquation(equation, leftValues, 0);
		if (rightIdx == -1) {
			return "No solution";
		}

		int[] rightValue = {0, 0};
		solveEquation(equation, rightValue, rightIdx + 1);

		if (leftValues[0] == rightValue[0] && leftValues[1] == rightValue[1]) {
			return "Infinite solutions";
		} else if (leftValues[0] == rightValue[0]) {
			return "No solution";
		} else {
			return "x=" + (rightValue[1] - leftValues[1]) / (leftValues[0] - rightValue[0]);
		}
	}

	private int solveEquation(CharSequence equation, int[] result, int start) {
		int lastOp = 0;
		int value = -1;
		boolean lastValueX = false;

		for (int i = start; i <= equation.length(); ++i) {
			final char c = i == equation.length() ? '*' : equation.charAt(i);

			if (c == '+' || c == '-' || c == '=' || c == '*') {
				if (lastOp == 1) {
					if (lastValueX) {
						result[0] += value == -1 ? 1 : value;
					} else if (value != -1) {
						result[1] += value;
					}
				} else if (lastOp == 2) {
					if (lastValueX) {
						result[0] -= value == -1 ? 1 : value;
					} else if (value != -1) {
						result[1] -= value;
					}
				} else {
					if (lastValueX) {
						result[0] += value == -1 ? 1 : value;
					} else if (value != -1) {
						result[1] += value;
					}
				}
				lastOp = c == '+' ? 1 : 2;
				value = -1;

				if (c == '=') {
					return i;
				}
			} else if (c == 'x') {
				lastValueX = true;
			} else {
				lastValueX = false;
				if (value == -1) {
					value = c - '0';
				} else {
					value = value * 10 + (c - '0');
				}
			}
		}

		return -1;
	}

	// https://leetcode.com/problems/palindromic-substrings/description/
	public int countSubstrings(String s) {
		int result = 0;

		boolean[][] dp = new boolean[s.length()][s.length()];
		for (int i = 0; i < s.length(); ++i) {
			dp[i][i] = true;
			result++;
		}

		for (int k = 1; k < s.length(); k++) {
			for (int i = 0; i + k < s.length(); ++i) {
				if (s.charAt(i) == s.charAt(i+k)) {
					dp[i][i+k] = k == 1 || dp[i+1][i+k-1];
					if (dp[i][i+k]) {
						result++;
					}
				}
			}
		}

		return result;
	}

	// https://leetcode.com/problems/find-duplicate-subtrees/description/
	List<TreeNode> mDuplicateSubtreesResult = new LinkedList<>();
	Set<String> mRecordedSerialize = new HashSet<>();

	public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
		mDuplicateSubtreesResult.clear();
		mRecordedSerialize.clear();

		serializeBinaryTree(root, new HashMap<>());

		return mDuplicateSubtreesResult;
	}

	private String serializeBinaryTree(TreeNode root, Map<String, TreeNode> map) {
		if (root == null) {
			return "nu";
		}

		String result = String.valueOf(root.val) + "," + serializeBinaryTree(root.left, map) + "," + serializeBinaryTree(root.right, map);

		TreeNode node = map.get(result);
		if (node != null) {
			if (!mRecordedSerialize.contains(result)) {
				mDuplicateSubtreesResult.add(node);
				mRecordedSerialize.add(result);
			}
		} else {
			map.put(result, root);
		}

		return result;
	}

	// https://leetcode.com/problems/find-k-closest-elements/description/
	public List<Integer> findClosestElements(int[] arr, int k, int x) {
		List<Integer> result = new ArrayList<>(k);
		if (k <= 0 || arr == null || arr.length <= 0) {
			return result;
		}

		int left = 0;
		int right = arr.length - 1;
		boolean find = false;
		int middle = -1;
		while (left <= right) {
			middle = left + (right - left) / 2;
			if (arr[middle] == x) {
				find = true;
				break;
			} else if (arr[middle] > x) {
				right = middle - 1;
			} else {
				left = middle + 1;
			}
		}

		int[] startEnd = {0, 0};

		if (find) {
			findClosestElements(arr, k, x, middle - 1, middle, startEnd);
		} else if (right == -1) {
			startEnd[0] = 0;
			startEnd[1] = k - 1;

		} else if (left == arr.length) {
			startEnd[1] = arr.length - 1;
			startEnd[0] = arr.length - k;
		} else {
			findClosestElements(arr, k, x, left - 1, left, startEnd);
		}

		for (int i = startEnd[0]; i <= startEnd[1]; ++i) {
			result.add(arr[i]);
		}

		return result;
	}

	private void findClosestElements(int[] arr, int k, int x, int max, int min, int[] result) {
		result[0] = -1;
		result[1] = arr.length;

		int size = 0;
		// 合并操作
		while (max >= 0 && min <= arr.length - 1 && size++ < k) {
			if (Math.abs(arr[max] - x) <= Math.abs(arr[min] - x)) {
				result[0] = max;
				max--;
			} else {
				result[1] = min;
				min++;
			}

		}

		while (max >= 0 && size++ < k) {
			result[0] = max;
			max--;
		}

		while (min <= arr.length - 1 && size++ < k) {
			result[1] = min;
			min++;
		}

		if (result[0] == -1) {
			result[0] = result[1] - k + 1;
		} else if (result[1] == arr.length) {
			result[1] = result[0] + k - 1;
		}
	}

	// https://leetcode.com/problems/split-array-into-consecutive-subsequences/description/

	public boolean isPossible(int[] nums) {
		return false;
	}

	// https://leetcode.com/problems/beautiful-arrangement-ii/description/
	public int[] constructArray(int n, int k) {
		int[] result = new int[n];
		result[0] = 1;
		int i = 1;
		boolean add = true;
		while (k >= 1 && i < n) {
			result[i] = add ? result[i-1] + k : result[i-1] - k;
			add = !add;
			k--;
			i++;
		}

		while (i < n) {
			result[i] = i + 1;
			i++;
		}

		return result;
	}

	private int[] mConstructArray = null;
	public int[] constructArrayRecur(int n, int k) {
		mConstructArray = null;
		int[] nums = new int[n];
		for (int i = 0; i < n; i++) nums[i] = i + 1;
		constructArrayHelper(nums, k, 0, new HashMap<>());
		return mConstructArray;
	}

	public void swap(int[] nums, int i, int j) {
		int temp = nums[j];
		nums[j] = nums[i];
		nums[i] = temp;
	}

	private void constructArrayHelper(int[] nums, int k, int start, Map<Integer, Integer> map) {
		if (mConstructArray != null) {
			return;
		}
		if (start >= nums.length) {
			if (map.size() == k) {
				mConstructArray = new int[nums.length];
				System.arraycopy(nums, 0, mConstructArray, 0, nums.length);
			}
			return;
		}

		if (map.size() > k) {
			return;
		}

		for (int i = start; i < nums.length; ++i) {
			swap(nums, i, start);
			int key = -1;
			if (start >= 1) {
				key = Math.abs(nums[start] - nums[start-1]);
				map.put(key, map.getOrDefault(key, 0) + 1);
			}
			constructArrayHelper(nums, k, start + 1, map);
			swap(nums, i, start);

			if (key != -1) {
				int value = map.get(key);
				if (value == 1) {
					map.remove(key);
				} else {
					map.put(key, value - 1);
				}
			}
		}
	}

	// https://leetcode.com/problems/knight-probability-in-chessboard/description/
	private class KNight {
		int r;
		int c;
		int k;
		public KNight(int r, int c, int k) {
			this.r = r;
			this.c = c;
			this.k = k;
		}

		@Override
		public int hashCode() {
			return Objects.hash(r, c, k);
		}

		@Override
		public boolean equals(Object obj) {
			return (obj instanceof KNight) && ((KNight) obj).c == c
					&& ((KNight) obj).k == k
					&& ((KNight) obj).r == r;
		}
	}

	private Map<KNight, Double> mKnightCache = new HashMap<>();

	public double knightProbability(int N, int K, int r, int c) {
		return knightProbabilityHelper(N, K, r, c);
	}

	private double knightProbabilityHelper(int N, int K, int r, int c) {
		if (!isWithin(N, r, c)) {
			return 0;
		} else if (K == 0) {
			return 1;
		}

		KNight kNight = new KNight(r, c, K);
		double value = mKnightCache.getOrDefault(kNight, -1d);
		if (value != -1) {
			return value;
		}

		double result = 0;
		for (int i = 0; i < KNightDirections.length; ++i) {
			result += 0.125 * knightProbabilityHelper(N, K-1, r + KNightDirections[i][0], c + KNightDirections[i][1]);
		}

		mKnightCache.put(kNight, result);

		return result;
	}

	private int[][] KNightDirections = {
			{-2,-1}, {-1,-2}, {-2,1}, {-1,2}, {1,2}, {2,1}, {2,-1}, {1,-2}
	};

	private boolean isWithin(int N, int r, int c) {
		return r >= 0 && r < N && c >= 0 && c < N;
	}


	// https://leetcode.com/problems/max-area-of-island/description/
	public int maxAreaOfIsland(int[][] grid) {
		int result = 0;
		for (int i = 0; i < grid.length; ++i) {
			for (int j = 0; j < grid[0].length; ++i) {
				if (grid[i][j] == 1) {
					result = Math.max(result, maxAreaOfIsland(grid, i, j));
				}
			}
		}
		return result;
	}

	private int[][] IslandDir = {
			{0,1}, {0,-1}, {1,0}, {-1,0}
	};

	private int maxAreaOfIsland(int[][] grid, int r, int c) {
		grid[r][c] = 2;
		int result = 0;
		for (int[] dir : IslandDir) {
			final int newr = r + dir[0];
			final int newc = c + dir[1];
			if (newr >= 0 && newr < grid.length && newc >= 0 && newc < grid[0].length && grid[newr][newc] == 1) {
				result += maxAreaOfIsland(grid, newr, newc);
			}
		}

		return result + 1;
	}

	// https://leetcode.com/problems/insert-into-a-binary-search-tree/description/
	public TreeNode insertIntoBST(TreeNode root, int val) {
		if (root == null) {
			return new TreeNode(val);
		}

		insertIntoBSTHelper(root, val);

		return root;
	}

	private void insertIntoBSTHelper(TreeNode root, int val) {
		if (val > root.val) {
			if (root.right == null) {
				root.right = new TreeNode(val);
			} else {
				insertIntoBSTHelper(root.right, val);
			}
		} else {
			if (root.left == null) {
				root.left = new TreeNode(val);
			} else {
				insertIntoBSTHelper(root.left, val);
			}
		}
	}

	// https://leetcode.com/problems/accounts-merge/description/
	public List<List<String>> accountsMergeII(List<List<String>> accounts) {
		List<List<String>> result = new LinkedList<>();

		Map<String, List<Set<String>>> map = new HashMap<>();

		for (List<String> account : accounts) {
			final Iterator<String> it = account.iterator();
			final String name = it.next();
			final Set<String> newEmails = new TreeSet<>();
			while (it.hasNext()) {
				newEmails.add(it.next());
			}

			List<Set<String>> emails = map.get(name);
			if (emails == null) {
				emails = new LinkedList<>();
				emails.add(newEmails);
				map.put(name, emails);
				continue;
			}

			// 在每个 emails中查找是否存在
			final Iterator<Set<String>> setIterator = emails.iterator();
			while (setIterator.hasNext()) {
				Set<String> oldEmails = setIterator.next();
				boolean common = false;
				for (String email : oldEmails) {
					if (newEmails.contains(email)) {
						common = true;
						break;
					}
				}
				if (common) {
					newEmails.addAll(oldEmails);
					setIterator.remove();
				}
			}

			emails.add(newEmails);
		}

		for (Map.Entry<String, List<Set<String>>> entry : map.entrySet()) {
			for (Set<String> emails : entry.getValue()) {
				List<String> one = new LinkedList<>();
				one.add(entry.getKey());
				one.addAll(emails);
				result.add(one);
			}
		}

		return result;
	}

	private class AccountNode {
		String name;
		LinkedList<String> emails = new LinkedList<>();
		AccountNode parent;

		public AccountNode(String name) {
			this.name = name;
			this.parent = this;
		}


	}

	public AccountNode getParent(AccountNode node) {
		while (node.parent != node) {
			node.parent = node.parent.parent;
			node = node.parent;
		}
		return node;
	}

	public List<List<String>> accountsMerge(List<List<String>> accounts) {
		List<List<String>> result = new LinkedList<>();

		Map<String, AccountNode> map = new HashMap<>();
		List<AccountNode> nodes = new LinkedList<>();

		for (List<String> account : accounts) {
			final Iterator<String> it = account.iterator();
			final String name = it.next();
			final AccountNode node = new AccountNode(name);

			while (it.hasNext()) {
				String email = it.next();
				if (map.containsKey(email)) {
					AccountNode existed = map.get(email);
					AccountNode node_p = getParent(node);
					AccountNode node_e = getParent(existed);
					node_e.parent = node_p;
				} else {
					node.emails.add(email);
					map.put(email, node);
				}
			}
			nodes.add(node);
		}

		for (AccountNode node : nodes) {
			if (node.parent != node) {
				AccountNode parent = getParent(node);
				parent.emails.addAll(node.emails);
			}
		}

		for (AccountNode node : nodes) {
			if (node.parent == node) {
				List<String> one = new LinkedList<>();
				one.add(node.name);
				Collections.sort(node.emails);
				one.addAll(node.emails);
				result.add(one);
			}

		}

		return result;
	}

	//
	public int openLock(String[] deadends, String target) {
		Queue<String> queue = new LinkedList<>();

		queue.add("0000");

		Set<String> set = new HashSet<>();
		for (String deadend : deadends) {
			set.add(deadend);
		}

		if (set.contains("0000") || set.contains(target)) {
			return -1;
		}

		int level = 0;
		int cur = 1;
		int next = 0;

		Set<String> visited = new HashSet<>();
		visited.add("0000");

		while (!queue.isEmpty()) {
			String head = queue.poll();
			cur--;

			for (int i = 0; i < head.length(); ++i) {
				final int c = head.charAt(i) - '0';
				final int cadd = c + 1 == 10 ? 0 : c + 1;
				final String newLockAdd = head.substring(0,i) + cadd + head.substring(i+1);

				if (target.equals(newLockAdd)) {
					return level + 1;
				}

				if (!set.contains(newLockAdd) && visited.add(newLockAdd)) {
					queue.add(newLockAdd);
					next++;
				}

				final int cdel = c - 1 == -1 ? 9 : c - 1;
				final String newLockDel = head.substring(0,i) + cdel + head.substring(i+1);

				if (target.equals(newLockDel)) {
					return level + 1;
				}

				if (!set.contains(newLockDel) && visited.add(newLockDel)) {
					queue.add(newLockDel);
					next++;
				}
			}

			if (cur == 0) {
				cur = next;
				next = 0;
				level++;
			}
		}

		return -1;
	}

	// https://leetcode.com/problems/pyramid-transition-matrix/description/
	public boolean pyramidTransition(String bottom, List<String> allowed) {
		int[] nums = new int[bottom.length()];
		for (int i = 0; i < nums.length; i++) {
			nums[i] = bottom.charAt(i) - 'A';
		}

		Map<Integer, Set<Integer>> map = new HashMap<>();
		for (String allow : allowed) {
			final int key = (allow.charAt(0) - 'A') * 10 + allow.charAt(1) - 'A';
			Set<Integer> set = map.get(key);
			if (set == null) {
				set = new HashSet<>();
				map.put(key, set);
			}

			set.add(allow.charAt(2) - 'A');
		}

		return pyramidTransitionHelper(nums, 0, nums.length, map);
	}

	private boolean pyramidTransitionHelper(int[] nums, int start, int size, Map<Integer, Set<Integer>> map) {
		if (size == 1) {
			return true;
		}

		final int key = nums[start] * 10 + nums[start+1];
		final Set<Integer> set = map.get(key);

		if (set == null) {
			return false;
		}

		for (Integer integer : set) {
			final int old = nums[start];
			nums[start] = integer;

			if (start < size - 2 && pyramidTransitionHelper(nums, start + 1, size, map)) {
				return true;
			} else if (start >= size - 2 && pyramidTransitionHelper(nums, 0, size - 1, map)) {
				return true;
			}
			nums[start] = old;
		}

		return false;
	}

	// https://leetcode.com/problems/partition-labels/description/
	public List<Integer> partitionLabels(String S) {
		int[][] segments = new int[26][2];
		for (int i = 0; i < segments.length; ++i) {
			segments[i][0] = -1;
		}

		for (int i = 0; i < S.length(); ++i) {
			final int idx = S.charAt(i) - 'a';

			if (segments[idx][0] == -1) {
				segments[idx][0] = i;
				segments[idx][1] = i;
			} else {
				segments[idx][1] = i;
			}
		}

		Arrays.sort(segments, new Comparator<int[]>() {
			@Override
			public int compare(int[] o1, int[] o2) {
				return o1[0] - o2[0];
			}
		});

		int j = 0;
		while (segments[j][0] == -1) {
			++j;
		}

		List<Integer> result = new LinkedList<>();

		if (j >= segments.length) {
			return result;
		}

		int start = segments[j][0];
		int end = segments[j][1];

		for (int i = j + 1; i < segments.length; ++i) {
			if (segments[i][0] <= end) {
				end = Math.max(end, segments[i][1]);
			} else {
				result.add(end - start + 1);
				start = segments[i][0];
				end = segments[i][1];
			}
		}

		result.add(end - start + 1);

		return result;
	}

	// https://leetcode.com/problems/largest-plus-sign/description/
	public int orderOfLargestPlusSign(int N, int[][] mines) {
		return 0;
	}


	// https://leetcode.com/problems/reorganize-string/description/
	public String reorganizeString(String S) {
		if (S == null || S.length() <= 0) {
			return "";
		}

		int[] hash = new int[26];
		Integer[] indexs = new Integer[26];

		for (int i = 0; i < S.length(); ++i) {
			hash[S.charAt(i) - 'a']++;

		}

		for (int i = 0; i < indexs.length; ++i) {
			indexs[i] = i;
		}

		// 从大到小排列
		Arrays.sort(indexs, new Comparator<Integer>() {
			@Override
			public int compare(Integer o1, Integer o2) {
				return hash[o2] - hash[o1];
			}
		});

		final int segmentCnt = hash[indexs[0]];
		int sum = 0;
		for (int i = 1; i < indexs.length && hash[indexs[i]] > 0; ++i) {
			sum += hash[indexs[i]];
		}
		if (sum < segmentCnt - 1) {
			return "";
		}

		StringBuilder[] segments = new StringBuilder[segmentCnt];
		for (int i = 0; i < segmentCnt; ++i) {
			segments[i] = new StringBuilder();
			segments[i].append((char)(indexs[0] + 'a'));
		}
		int j = 0;
		for (int i = 1; i < indexs.length && hash[indexs[i]] > 0;) {
			while (hash[indexs[i]] > 0) {
				segments[j].append((char)(indexs[i] + 'a'));
				hash[indexs[i]]--;
				j++;
				if (j >= segmentCnt) {
					j = 0;
				}
			}
			i++;
		}

		StringBuilder result = new StringBuilder();
		for (StringBuilder builder : segments) {
			result.append(builder);
		}

		return result.toString();
	}

	// https://leetcode.com/problems/max-chunks-to-make-sorted/description/
	public int maxChunksToSorted(int[] arr) {
		if (arr == null) {
			return 0;
		} else if (arr.length <= 1) {
			return arr.length;
		}

		int start = -1;
		int end = -1;
		int result = 0;
		for (int i = 0; i < arr.length; ++i) {
			if (start == -1) {
				start = i;
			}
			end = Math.max(end, arr[i]);
			if (i == end) {
				start = -1;
				end = -1;
				result++;
			}
		}

		return result;
	}

	// https://leetcode.com/problems/global-and-local-inversions/description/
	public boolean isIdealPermutation(int[] A) {
		if (A == null || A.length <= 2) {
			return true;
		}

		int min = A[A.length-1];

		for (int i = A.length - 3; i >= 0; --i) {
			if (A[i] > min) {
				return false;
			} else {
				min = Math.min(min, A[i+1]);
			}
		}

		return true;
	}

	// https://leetcode.com/problems/swap-adjacent-in-lr-string/description/
	public boolean canTransform(String start, String end) {
		if (start.length() != end.length()) {
			return false;
		}


		return true;
	}

	// https://leetcode.com/problems/k-th-symbol-in-grammar/description/
	public int kthGrammar(int N, int K) {
		return kthGrammarHelper(N, K - 1);
	}

	private int kthGrammarHelper(int N, int K) {
		if (K == 0 || N == 1) {
			return 0;
		}

		int parent = K / 2;
		boolean left = parent * 2 == K;
		int pRes = kthGrammarHelper(N - 1, parent);

		if (left) {
			return pRes == 0 ? 0 : 1;
		} else {
			return pRes == 0 ? 1 : 0;
		}
	}

	// https://leetcode.com/problems/rabbits-in-forest/description/
	public int numRabbits(int[] answers) {
		if (answers == null || answers.length <= 0) {
			return 0;
		}

		Arrays.sort(answers);

		int result = 0;
		int lastVal = answers[0];
		int lastCnt = 1;
		int max = lastVal + 1;
		result += max;

		for (int i = 1; i < answers.length; ++i) {
			if (answers[i] == lastVal) {
				lastCnt++;
				if (lastCnt > max) {
					result += max;
					lastCnt = 1;
				}
			} else {
				lastCnt = 1;
				lastVal = answers[i];
				max = lastVal + 1;
				result += max;
			}
		}

		return result;
	}

	// https://leetcode.com/problems/is-graph-bipartite/description/
	public boolean isBipartite(int[][] graph) {
		if (graph == null) {
			return true;
		}
		int[] subsets = new int[graph.length];
		for (int i = 0; i < graph.length; ++i) {
			if (subsets[i] == 0 && !tipartiteHelper(graph, subsets, 1, i)) {
				return false;
			}
		}

		return true;

 	}

 	private boolean tipartiteHelper(int[][] graph, int[] subsets, int which, int from) {
		subsets[from] = which;

		final int[] tos = graph[from];
		if (tos == null || tos.length <= 0) {
			return true;
		}

		for (int to : tos) {
			if (subsets[to] == which) {
				return false;
			} else if (subsets[to] != -which) {
				if (!tipartiteHelper(graph, subsets, -which, to)) {
					return false;
				}
			}
		}
		return true;
	}

	// https://leetcode.com/problems/cheapest-flights-within-k-stops/description/
	public int findCheapestPrice(int n, int[][] flights, int src, int dst, int K) {
		if (src == dst) {
			return 0;
		}
		Map<Integer, Map<Integer, Integer>> map = new HashMap<>();
		Set<Integer> visited = new HashSet<>();

		for (int[] flight : flights) {
			Map<Integer, Integer> tos = map.get(flight[0]);
			if (tos == null) {
				tos = new HashMap<>();
				map.put(flight[0], tos);
			}
			tos.put(flight[1], flight[2]);
		}

		PriorityQueue<int[]> minHeap = new PriorityQueue<>(new Comparator<int[]>() {
			@Override
			public int compare(int[] o1, int[] o2) {
				return o1[0] - o2[0];
			}
		});

		minHeap.add(new int[]{0, src, -1});



		while (!minHeap.isEmpty()) {
			int[] head = minHeap.poll();
			if (head[1] == dst) {
				return head[0];
			}
			visited.add(head[1]);

			if (head[2] >= K) {
				continue;
			}
			Map<Integer, Integer> tos = map.get(head[1]);
			if (tos == null) {
				continue;
			}

			for (Map.Entry<Integer, Integer> entry : tos.entrySet()) {
				if (!visited.contains(entry.getKey())) {
					minHeap.add(new int[]{head[0] + entry.getValue(), entry.getKey(), head[2] + 1});
				}
			}
		}

		return -1;
	}

	public boolean escapeGhosts(int[][] ghosts, int[] target) {
		final int distance = Math.abs(target[0]) + Math.abs(target[1]);

		for (int[] ghost : ghosts) {
			int temp = Math.abs(ghost[0] - target[0]) + Math.abs(ghost[1] - target[1]);
			if (temp <= distance) {
				return false;
			}
		}

		return true;
	}

	// https://leetcode.com/problems/domino-and-tromino-tiling/description/
	public long numTilings(int N) {
		if (N == 0) {
			return 1;
		} else if (N == 1) {
			return 1;
		} else if (N == 2) {
			return 2;
		}

		long fn1 = 2;
		long fn2 = 1;
		long pn1 = 1;
		for (int i = 3; i <= N; ++i) {
			long fn = fn1 + fn2 + 2 * pn1;
			pn1 = fn2 + pn1;
			long temp = fn1;
			fn1 = fn;
			fn2 = temp;
		}

		return (fn1 % 1000000007);
	}

	// https://leetcode.com/problems/valid-tic-tac-toe-state/description/
	public boolean validTicTacToe(String[] board) {
		char[][] boards = new char[3][3];

		int xcnt = 0;
		int ocnt = 0;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				boards[i][j] = board[i].charAt(j);
				if (boards[i][j] == 'X') xcnt++;
				else if (boards[i][j] == 'O') ocnt++;
			}
		}

		if (ocnt > xcnt) {
			return false;
		}

		return validTicTacToe(boards, xcnt, ocnt, true);
	}

	public boolean validTicTacToe(char[][] board, int xcnt, int ocnt, boolean first) {
		if (ocnt == 0 && xcnt == 0) {
			return true;
		}

		if (ocnt == 0) {
            System.out.println(first);
        }


        if (first) {
            boolean result = false;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    if (board[i][j] == 'X') {
                        board[i][j] = 'Z';

                        if (isOverTicTacToe(board, i, j, 'Z')) {
                            if (ocnt == 0 && xcnt == 1) {
                                result = true;
                            }
                        } else {
                            result = validTicTacToe(board, xcnt - 1, ocnt, !first);
                        }

                        board[i][j] = 'X';
                    }

                    if (result) {
                        return true;
                    }
                }
            }
        } else {
                boolean result = false;
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {

                        if (board[i][j] == 'O') {
                            board[i][j] = 'Y';

                            if (isOverTicTacToe(board, i, j, 'Y')) {
                                if (xcnt == 0 && ocnt == 1) {
                                    result = true;
                                }
                            } else {
                                result = validTicTacToe(board, xcnt, ocnt - 1, !first);
                            }

                            board[i][j] = 'O';
                        }

                        if (result) {
                            return true;
                        }
                    }
                }
        }

		return false;
	}

	private boolean isOverTicTacToe(char[][] board, int r, int c, char compare) {

		boolean isOver = true;
		for (int i = 0; i < 3; ++i) {
			if (board[r][i] != compare) {
				isOver = false;
				break;
			}
		}

		if (isOver) {
			return true;
		}

		isOver = true;
		for (int i = 0; i < 3; ++i) {
			if (board[i][c] != compare) {
				isOver = false;
				break;
			}
		}

		if (isOver)
			return true;


		if (r == c) {
			isOver = true;
			for (int i = 0; i < 3; ++i) {
				if (board[i][i] != compare) {
					isOver = false;
					break;
				}
			}
			if (isOver) {
				return true;
			}
		}

		if (r + c == 2) {
			isOver = true;
			for (int i = 0; i < 3; ++i) {
				if (board[i][2 - i] != compare) {
					isOver = false;
					break;
				}
			}
			if (isOver) {
				return true;
			}
		}

		return false;
	}

	// https://leetcode.com/problems/number-of-subarrays-with-bounded-maximum/description/
    public int numSubarrayBoundedMax(int[] A, int L, int R) {
        if (A == null || A.length <= 0) {
            return 0;
        }

        int dp = 0;
        int result = 0;
        int lastLarger = -1;

        for (int i = 0; i < A.length; ++i) {
            if (A[i] >= L && A[i] <= R) {
                dp = i - lastLarger;
                result += dp;
            } else if (A[i] < L){
                if (dp > 0) {
                    result += dp;
                }
            } else {
                lastLarger = i;
                dp = 0;
            }
        }

        return result;
    }

    // https://leetcode.com/problems/champagne-tower/description/
    public double champagneTower(int poured, int query_row, int query_glass) {
        double[][] dp = new double[query_row + 1][query_row + 1];

        dp[0][0] = poured;

        for (int i = 1; i <= query_row; ++i) {
            if (dp[i-1][0] > 1) {
                dp[i][0] = (dp[i-1][0] - 1)/ 2.0d;
            }

            if (dp[i-1][i-1] > 1) {
                dp[i][i] = (dp[i-1][i-1] - 1) / 2.0d;
            }

            for (int j = 1; j < i; ++j) {
                dp[i][j] = (dp[i-1][j-1] > 1 ? (dp[i-1][j-1] - 1) / 2 : 0) +
                        (dp[i-1][j] > 1 ? (dp[i-1][j] - 1) / 2 : 0);
            }
        }

        return Math.min(dp[query_row][query_glass], 1);
    }

    // https://leetcode.com/problems/minimum-swaps-to-make-sequences-increasing/description/
    public int minSwap(int[] A, int[] B) {
        if (A == null || A.length <= 0 || B == null || B.length <= 0) {
            return 0;
        }

        int dp0; // 上一个不换的最小值
        int dp1; // 上一个换的最小值

        dp0 = 0;
        dp1 = 1;

        for (int i = 1; i < A.length; ++i) {
            // 不换的话,
            int temp0 = Integer.MAX_VALUE;
            // 前边也没换
            if (A[i-1] < A[i] && B[i-1] < B[i]) {
                temp0 = Math.min(temp0, dp0);
            }
            if (A[i-1] < B[i] && B[i-1] < A[i]) {
                temp0 = Math.min(temp0, dp1);
            }

            // 换的话
            int temp1 = Integer.MAX_VALUE;
            // 前边没换
            if (A[i-1] < B[i] && B[i-1] < A[i]) {
                temp1 = Math.min(temp1, dp0);
            }
            if (A[i-1] < A[i] && B[i-1] < B[i]) {
                temp1 = Math.min(temp1, dp1);
            }

            dp0 = temp0;
            dp1 = temp1 + 1;
        }

        return Math.min(dp0, dp1);
    }

    // https://leetcode.com/problems/find-eventual-safe-states/description/
    public List<Integer> eventualSafeNodes(int[][] graph) {
        List<Integer> result = new LinkedList<>();
        int[] all = new int[graph.length];

        for (int i = 0; i < graph.length; ++i) {
            if (all[i] == 0) {
                eventualSafeNodes(graph, i, new HashSet<>(), all);
            }
        }

        for (int i = 0; i < graph.length; ++i) {
            if (all[i] == 0) {
                result.add(i);
            }
        }

        return result;
    }

    private void eventualSafeNodes(int[][] graph, int start, Set<Integer> visited, int[] result) {
        if (visited.contains(start)) {
            for (int visit : visited) {
                result[visit] = 1;
            }
            return;
        }

        if (result[start] == 1) {
            for (int visit : visited) {
                result[visit] = 1;
            }
            return;
        }

        visited.add(start);

        for (int to : graph[start]) {
            eventualSafeNodes(graph, to, visited, result);
        }

        visited.remove(start);
    }

    // https://leetcode.com/problems/max-increase-to-keep-city-skyline/description/
    public int maxIncreaseKeepingSkyline(int[][] grid) {
        int[] maxColumn = new int[grid[0].length];
        int[] maxRow = new int[grid.length];

        for (int i = 0; i < grid.length; ++i) {
            int temp = Integer.MIN_VALUE;
            for (int j = 0; j < grid[0].length; ++j) {
                temp = Math.max(temp, grid[i][j]);
            }
            maxRow[i] = temp;
        }

        for (int i = 0; i < grid[0].length; ++i) {
            int temp = Integer.MIN_VALUE;
            for (int j = 0; j < grid.length; ++j) {
                temp = Math.max(temp, grid[j][i]);
            }
            maxColumn[i] = temp;
        }

        int result = 0;
        for (int i = 0; i < grid.length; ++i) {
            for (int j = 0; j < grid[0].length; ++j) {
                result += Math.min(maxRow[i], maxColumn[j]) - grid[i][j];
            }
        }

        return result;
    }

    // https://leetcode.com/problems/largest-sum-of-averages/description/
    public double largestSumOfAverages(int[] A, int K) {
        int[] sums = new int[A.length];
        sums[0] = A[0];
        for (int i = 1; i < A.length; ++i) {
            sums[i] = sums[i-1] + A[i];
        }

        return largestSumOfAverages(A, sums, 0, K);
    }

    private double largestSumOfAverages(int[] A, int[] sums, int start, int K) {
	    if (start >= A.length) {
	        return 0;
        }

	    K = Math.min(K, A.length - start);

	    if (K == 1) {
	        return (sums[A.length-1] - (start == 0 ? 0 : sums[start-1])) * 1.0d / (A.length - start);
        } else {
	        double result = Double.MIN_VALUE;
	        for (int i = start; i < A.length; ++i) {
	            result = Math.max(result, largestSumOfAverages(A, sums, i+1, K - 1)
                    + ((sums[i] - (start == 0 ? 0 : sums[start-1]))) * 1.0d / (i - start + 1)
                );
            }

            return result;
        }
    }

    // https://leetcode.com/problems/binary-tree-pruning/description/
    public TreeNode pruneTree(TreeNode root) {
        return pruneTreeHelper(root, new boolean[1]);
    }

    public TreeNode pruneTreeHelper(TreeNode root, boolean[] allZero) {
	    if (root == null) {
	        allZero[0] = true;
	        return null;
        }

        boolean[] leftZero = {true};
	    boolean[] rightZero = {true};

	    TreeNode left = pruneTreeHelper(root.left, leftZero);
	    TreeNode right = pruneTreeHelper(root.right, rightZero);

	    if (leftZero[0] && rightZero[0]) {
            if (root.val == 0) {
                allZero[0] = true;
                return null;
            } else {
                allZero[0] = false;
                root.left = root.right = null;
                return root;
            }

        } else if (leftZero[0]) {
	        allZero[0] = false;
	        root.left = null;
	        root.right = right;
	        return root;
        } else if (rightZero[0]) {
	        allZero[0] = false;
	        root.left = left;
	        root.right = null;
	        return root;
        } else {
	        allZero[0] = false;
	        return root;
        }
    }

    // https://leetcode.com/problems/linked-list-components/description/
    public int numComponents(ListNode head, int[] G) {
        if (head == null || G == null || G.length <= 0) {
            return 0;
        }

        Set<Integer> set = new HashSet<>();
        for (int g : G) {
            set.add(g);
        }

        ListNode temp = head;
        boolean hasPre = set.contains(temp.val);

        int result = 0;
        while (temp.next != null) {
            boolean hasNext = set.contains(temp.next.val);
            if (!hasNext && hasPre) {
                result++;
            }

            hasPre = hasNext;
            temp = temp.next;
        }

        if (hasPre) {
            result++;
        }

        return result;
    }

    // https://leetcode.com/problems/short-encoding-of-words/description/
    private class EncodeTrieNode {
	    boolean isLeaf;
	    int deep;
	    EncodeTrieNode[] next = new EncodeTrieNode[26];
    }

    private void addWord(EncodeTrieNode root, String word, Set<EncodeTrieNode> leafs) {
	    if (word == null || word.length() <= 0) {
	        return;
        }

        for (int i = word.length() - 1; i >= 0; --i) {
	        final char c = word.charAt(i);
	        final int idx = c- 'a';
	        if (root.next[idx] == null) {
	            root.next[idx] = new EncodeTrieNode();
	            if (i == 0) {
	                root.next[idx].isLeaf = true;
	                root.next[idx].deep = word.length();
	                leafs.add(root.next[idx]);
                }
            } else if (i != 0 && root.next[idx].isLeaf) {
	            root.next[idx].isLeaf = false;
	            leafs.remove(root.next[idx]);
            }
            root = root.next[idx];
        }
    }

    public int minimumLengthEncoding(String[] words) {
        EncodeTrieNode root = new EncodeTrieNode();
        Set<EncodeTrieNode> leafs = new HashSet<>();
        for (String word : words) {
            addWord(root, word, leafs);
        }

        int result = 0;
        for (EncodeTrieNode node : leafs) {
            result += node.deep + 1;
        }

        return result;
    }

    // https://leetcode.com/problems/card-flipping-game/description/
    public int flipgame(int[] fronts, int[] backs) {
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < fronts.length; i++) {
            if (fronts[i] == backs[i]) set.add(fronts[i]);
        }
        int res = Integer.MAX_VALUE;
        for (int i = 0; i < fronts.length; i++) {
            if (!set.contains(fronts[i]) && fronts[i] < res) {
                res = fronts[i];
            }
            if (!set.contains(backs[i]) && backs[i] < res) {
                res = backs[i];
            }
        }
        return res == Integer.MAX_VALUE ? 0 : res;
    }

    // https://leetcode.com/problems/binary-trees-with-factors/description/
    public int numFactoredBinaryTrees(int[] A) {
        if (A == null || A.length <= 0) {
            return 0;
        }

        long result = 0;
        Arrays.sort(A);

        long[] dp = new long[A.length];
        dp[0] = 1;
        result += dp[0];
        Map<Integer, Integer> map = new HashMap<>();
        map.put(A[0], 0);

        for (int i = 1; i < A.length; ++i) {
            long temp = 1;
            Set<Integer> set = new HashSet<>();
            for (int j = i - 1; j >= 0; --j) {
                if (!set.contains(A[j]) && A[i] % A[j] == 0) {
                    final int shang = A[i] / A[j];
                    if (shang == A[j]) {
                        temp += dp[j] * dp[j] % 1000000007;
                    } else {
                        int idx = map.getOrDefault(shang, -1);
                        if (idx != -1) {
                            set.add(shang);
                            temp += 2 * dp[j] * dp[idx] % 1000000007;
                        }
                    }
                }
            }

            dp[i] = temp % 1000000007;
            map.put(A[i], i);
            result = (result + dp[i]) % 1000000007;
        }

        return (int) result;
    }

    // https://leetcode.com/problems/friends-of-appropriate-ages/description/
    public int numFriendRequests(int[] ages) {
	    if (ages == null || ages.length <= 1) {
	        return 0;
        }

        int[] hash = new int[121];
	    for (int age : ages) {
	        hash[age]++;
        }

        int result = 0;
	    for (int i = 1; i < hash.length; ++i) {
	        for (int j = i; j < hash.length; ++j) {
	            if (hash[i] == 0 || hash[j] == 0) {
	                continue;
                }

                if (i == j) {
	                if (i >= 15) {
	                    result += hash[i] * (hash[i] - 1);
                    }
                    continue;
                }

                if (j + 14 < 2 * i && (j >= 100 || i <= 100)) {
	                result += hash[j] * hash[i];
                }
            }
        }

        return result;
    }


    public int maxProfitAssignment(int[] difficulty, int[] profit, int[] worker) {
        if (difficulty == null || difficulty.length <= 0 || worker == null ||
                worker.length <= 0) {
            return 0;
        }

        Integer[] idxs = new Integer[difficulty.length];
        for (int i = 0; i < idxs.length; ++i) {
            idxs[0] = 0;
        }

        Arrays.sort(idxs, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return difficulty[o1] - difficulty[o2];
            }
        });

        Arrays.sort(worker);

        int result = 0;
        int i = 0;
        int j = 0;
        int max = Integer.MIN_VALUE;
        while (j < worker.length) {
            while (i < idxs.length && difficulty[idxs[i]] <= worker[j]) {
                max = Math.max(max, profit[idxs[i]]);
                i++;
            }
            if (max != Integer.MIN_VALUE) {
                result += max;
            }
            j++;
        }

        return result;
    }

    // https://leetcode.com/problems/image-overlap/description/
    public int largestOverlap(int[][] A, int[][] B) {
	    int result = 0;
        for (int i = 0; i < A.length; ++i) {
            for (int j = 0; j < A.length; ++j) {
                result = Math.max(result, largestOverlapHelper(A, B, i, j));
            }
        }

        return result;
    }

    public int largestOverlapHelper(int[][] A, int[][] B, int leftShift, int downShift) {
	    int result = 0;
	    for (int i = 0; i < A.length; ++i) {
	        for (int j = 0; j < A.length; ++j) {
	            if (A[i][j] == 1) {
	                final int b = B[(i + downShift) % A.length][(j + leftShift) % A.length];
	                if (b == 1) {
	                    result++;
                    }
                }
            }
        }

        return result;
    }

    // https://leetcode.com/problems/new-21-game/description/
    public double new21Game(int N, int K, int W) {

        return new21GameHelper(N, K, W, new HashMap<>());
    }

    private double new21GameHelper(int N, int K, int W, Map<Integer, Double> cache) {
	    if (K == 0) {
	        return 1;
        }
	    if (K == 1) {
	        return W <= N ? 1.0d : N * 1.0d / W;
        }

        final double c = cache.getOrDefault(K, -1.0);
	    if (c != -1.0) {
	        return c;
        }

        double res = 0.0d;
	    int i = 1;
        for (; i <= W && i <= K - 1; ++i) {
	        res += 1.0d / W * new21GameHelper(N-i, K-i, W, cache);
        }

        if (i <= Math.min(W, N)) {
            res += (Math.min(W, N) - K + 1) * 1.0d / W;
        }

        cache.put(K, res);

        return res;
    }

    // https://leetcode.com/problems/push-dominoes/description/
    public String pushDominoes(String dominoes) {
        if (dominoes == null || dominoes.length() <= 0) {
            return "";
        }

        StringBuilder builder = new StringBuilder(dominoes.length());

        int lastPos = -1;
        int lastOp = 0; // 0 stay, 1 left， 2 right
        for (int i = 0; i <= dominoes.length(); ++i) {
            final char c = i == dominoes.length() ? 'R' : dominoes.charAt(i);

           if (c == 'L') {
                // 右推
                if (lastOp == 2) {
                    final int cnt = i - lastPos + 1;
                    final int half = cnt >> 1;
                    int j = 0;
                    while (j++ < half) {
                        builder.append('R');
                    }
                    if ((cnt & 1) == 1) {
                        builder.append('.');
                    }
                    j = 0;
                    while (j++ < half) {
                        builder.append('L');
                    }
                    lastOp = 0;
                    lastPos = i;
                } else {
                    // 不用管其他的
                    for (int j = lastPos + 1; j <= i; ++j) {
                        builder.append('L');
                    }
                    lastOp = 0;
                    lastPos = i;
                }
            } else if (c == 'R') {
                if (lastOp == 2) {
                    for (int j = lastPos; j < i; ++j) {
                        builder.append('R');
                    }
                    lastOp = 2;
                    lastPos = i;
                } else {
                    builder.append(dominoes.substring(lastPos + 1, i));
                    lastOp = 2;
                    lastPos = i;
                }
            }
        }

        return builder.toString();
    }

    // https://leetcode.com/problems/split-array-into-fibonacci-sequence/description/
    private final String FibonacciMaxInteger = String.valueOf(Integer.MAX_VALUE);

    public List<Integer> splitIntoFibonacci(String S) {
        List<Integer> result = new LinkedList<>();
        for (int i = 0; i < S.length() - 2 && i < FibonacciMaxInteger.length() && (i == 0 || (i >= 1 && S.charAt(0) != '0')); ++i) {
            final String oneStr = S.substring(0, i+1);
            if (oneStr.length() == FibonacciMaxInteger.length() && FibonacciMaxInteger.compareTo(oneStr) < 0) {
                continue;
            }

            final int one = Integer.valueOf(oneStr);

            for (int j = i + 1; j < S.length() - 1 && j - i <= FibonacciMaxInteger.length() && (j == i+ 1 || (j >= i + 2 && S.charAt(i+1) != '0')); ++j) {
                final String twoStr = S.substring(i+1, j+1);
                if (twoStr.length() == FibonacciMaxInteger.length() && FibonacciMaxInteger.compareTo(twoStr) < 0) {
                    continue;
                }

                final int two = Integer.valueOf(twoStr);

                result.add(one);
                result.add(two);

                if (splitIntoFibonacciHelper(S, one, two, j + 1, result)) {
                    return result;
                } else {
                    result.clear();
                }
            }
        }

        return result;
    }

    private boolean splitIntoFibonacciHelper(String s, int one, int two, int start, List<Integer> result) {
	    if (start == s.length()) {
	        return true;
        }

        final int three = one + two;
	    if (three < one || three < two) {
	        return false;
        }

        final String threeStr = String.valueOf(three);
	    if (s.startsWith(threeStr, start)) {
	        result.add(three);
	        return splitIntoFibonacciHelper(s, two, three, start + threeStr.length(), result);
        }
        return false;
    }

    public int longestMountain(int[] A) {
        if (A == null || A.length <= 2) {
            return 0;
        }

        int[] dp = new int[A.length];
        dp[0] = 0;

        for (int i = 1; i < A.length; ++i) {
            if (A[i] > A[i-1]) {
                dp[i] = dp[i-1] + 1;
            } else {
                dp[i] = 1;
            }
        }

        int result = 0;
        int pre = 1;
        for (int i = A.length - 2; i >= 1; --i) {
            if (A[i] > A[i+1]) {
                pre = pre + 1;
            } else {
                pre = 1;
            }

            if (pre >= 2 && dp[i] >= 2) {
                result = Math.max(result, pre + dp[i] - 1);
            }
        }

        return result;
    }

    public boolean isNStraightHand(int[] hands, int W) {
        if (hands == null || hands.length == 0) {
            return true;
        }

        if (W == 1) {
            return true;
        }

        final int length = hands.length;
        if (length % W != 0) {
            return false;
        }

        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (int hand : hands) {
            minHeap.add(hand);
        }

        while (!minHeap.isEmpty()) {

            final int start = minHeap.poll();
            final int end = start + W - 1;
            int temp = start + 1;
            while (temp <= end) {
                if (!minHeap.remove(temp)) {
                    return false;
                }
                temp++;
            }
        }

        return true;
    }

    public int[] loudAndRich(int[][] richer, int[] quiet) {
        Map<Integer, Set<Integer>> map = new HashMap<>();
        for (int[] rich : richer) {
            Set<Integer> set = map.get(rich[1]);
            if (set == null) {
                set = new HashSet<>();
                map.put(rich[1], set);
            }
            set.add(rich[0]);
        }

        int[] result = new int[quiet.length];
        for (int i = 0; i < result.length; ++i) {
            result[i] = -1;
        }

        for (int i = 0; i < result.length; ++i) {
            if (result[i] == -1) {
                loudAndRichHelper(map, i, quiet, result);
            }
        }

        return result;
    }

    private int loudAndRichHelper(Map<Integer, Set<Integer>> map, int from, int[] quiet, int[] result) {
        if (result[from] != -1) {
            return result[from];
        }

        int idx = from;

        Set<Integer> set = map.get(from);
        if (set != null) {
            for (int to : set) {
                int newIdx = loudAndRichHelper(map, to, quiet, result);
                if (quiet[newIdx] < quiet[idx]) {
                    idx = newIdx;
                }
            }
        }

        result[from] = idx;

        return idx;
    }

    public int carFleet(int target, int[] position, int[] speed) {
        if (position == null || position.length <= 0) {
            return 0;
        }

        Integer[] idxs = new Integer[position.length];
        for (int i = 0; i < idxs.length; ++i) {
            idxs[i] = i;
        }

        Arrays.sort(idxs, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return position[o1] - position[o2];
            }
        });

        Stack<Integer> stack = new Stack<>();

        for (int idx : idxs) {
            while (!stack.isEmpty() && canFleet(idx, stack.peek(), target, position, speed)) {
                stack.pop();
            }

            stack.push(idx);
        }

        return stack.size();
    }

    private boolean canFleet(int before, int after, int target, int[] position, int[] speed) {
        if (speed[before] == speed[after]) {
            if (position[before] == position[after]) {
                return true;
            } else  {
                return false;
            }
        } else if (speed[before] > speed[after]) {
            return false;
        } else {
            double fleetPos = position[before] +
                    speed[before] * 1.0d / (speed[after] - speed[before]) * (position[before] - position[after]);

            return fleetPos <= target;
        }
    }

	public int mirrorReflection(int p, int q) {
		if (q == 0) {
			return 0;
		} else if (p == q) {
			return 1;
		}

		final int gcd = gcd(p, q);

		final int height = q / gcd;
		final int width = p / gcd;

		boolean hou = (height & 1) == 0;
		boolean wou = (width & 1) == 0;

		if (hou) {
			return 0;
		} else if (wou) {
			return 2;
		} else {
			return 1;
		}
	}

	private int gcd(int p, int q) {
    	if (p == q) {
    		return p;
		} else if (q == 0) {
    		return p;
		} else {
    		return gcd(q, p % q);
		}
	}

	public int matrixScore(int[][] A) {
		if (A == null || A.length <= 0 || A[0] == null || A[0].length <= 0) {
			return 0;
		}

		// 先把第一列全部换成1
		int result = 0;
		for (int i = 0; i < A.length; ++i) {
			if (A[i][0] == 0) {
				for (int j = 0; j < A[0].length; ++j) {
					A[i][j] = A[i][j] ^ 1;
				}
			}
			result += 1 << (A[0].length - 1);
		}

		// 针对后面的每一列，统计1的个数
		for (int j = 1; j < A[0].length; ++j) {
			int onecnt = 0;
			for (int i = 0; i < A.length; ++i) {
				if (A[i][j] == 1) {
					onecnt++;
				}
			}

			onecnt = onecnt >= (1 + A.length) / 2 ? onecnt : (A.length - onecnt);
			result += onecnt * (1 << (A[0].length - 1 - j));
		}

		return result;
	}

	public List<Integer> distanceK(TreeNode root, TreeNode target, int K) {

		List<Integer> result = new LinkedList<>();
    	if (K == 0) {
    		result.add(target.val);
    		return result;
		}

		Map<TreeNode, TreeNode> map = new HashMap<>();
		distanceHelper(root, null, map);

		Set<TreeNode> visited = new HashSet<>();

		Queue<TreeNode> queue = new LinkedList<>();

		int cur = 1;
		int next = 0;
		int level = 1;

		queue.add(target);
		visited.add(target);

		while (!queue.isEmpty() && level <= K) {
			TreeNode head = queue.poll();
			cur--;

			if (head.left != null && visited.add(head.left)) {
				queue.add(head.left);
				next++;

				if (level == K) {
					result.add(head.left.val);
				}
			}

			if (head.right != null && visited.add(head.right)) {
				queue.add(head.right);
				next++;

				if (level == K) {
					result.add(head.right.val);
				}
			}

			final TreeNode parent = map.get(head);
			if (parent != null && visited.add(parent)) {
				queue.add(parent);
				next++;

				if (level == K) {
					result.add(parent.val);
				}
			}


			if (cur == 0) {
				cur = next;
				next = 0;
				level++;
			}
		}

		return result;

	}

	private void distanceHelper(TreeNode root, TreeNode parent, Map<TreeNode, TreeNode> map) {
    	if (root == null) {
    		return;
		}

		map.put(root, parent);
    	distanceHelper(root.left, root, map);
    	distanceHelper(root.right, root, map);
	}

	public TreeNode subtreeWithAllDeepest(TreeNode root) {
    	int[] deep = {0};
		return subtreeWithAllDeepestHelper(root, deep);
	}

	private TreeNode subtreeWithAllDeepestHelper(TreeNode root, int[] deep) {
    	if (root == null) {
    		deep[0] = 0;
    		return root;
		}

		int[] deepLeft = {0};
    	int[] deepRight = {0};
		TreeNode left = subtreeWithAllDeepestHelper(root.left, deepLeft);
		TreeNode right = subtreeWithAllDeepestHelper(root.right, deepRight);

		// 相等的话，返回哪个
		if (deepLeft[0] == deepRight[0]) {
			deep[0] = deepLeft[0] + 1;
			return root;
		} else if (deepLeft[0] > deepRight[0]) {
			deep[0] = deepLeft[0] + 1;
			return left;
		} else {
			deep[0] = deepRight[0] + 1;
			return right;
		}
	}

	public boolean reorderedPowerOf2(int N) {
		char[] num = String.valueOf(N).toCharArray();

		return reorderedPowerOf2(num, 0);
	}

	private boolean reorderedPowerOf2(char[] num, int start) {
    	if (start >= num.length) {
    		int result = 0;
    		for (char c : num) {
    			result = result * 10 + (c - '0');
			}
			if ((result & (result - 1)) == 0) {
    			return true;
			} else {
    			return false;
			}
		}

		for (int i = start; i < num.length; ++i) {
    		if (start == 0 && num[i] == '0') {
    			continue;
			}
			reorderedPowerOf2Swap(num, start, i);

    		boolean res = reorderedPowerOf2(num, start + 1);

    		reorderedPowerOf2Swap(num, start, i);
    		if (res) {
    			return true;
			}

		}

		return false;
	}

	private void reorderedPowerOf2Swap(char[] num, int from, int to) {
    	char c = num[from];
    	num[from] = num[to];
    	num[to] = c;
	}

	public int[] advantageCount(int[] A, int[] B) {
		TreeMap<Integer, Integer> set = new TreeMap<>();
		for (int i = 0; i < A.length; ++i) {
			set.put(A[i], set.getOrDefault(A[i], 0) + 1);
			A[i] = -1;
		}

		for (int i = 0; i < A.length; ++i) {
			Map.Entry<Integer, Integer> entry = set.higherEntry(B[i]);
			if (entry != null) {
				A[i] = entry.getKey();
				if (entry.getValue() == 1) {
					set.remove(entry.getKey());
				} else {
					set.put(entry.getKey(), entry.getValue() - 1);
				}
			} else {
				Map.Entry<Integer, Integer> first = set.firstEntry();
				A[i] = first.getKey();
				if (first.getValue() == 1) {
					set.remove(first.getKey());
				} else {
					set.put(first.getKey(), first.getValue() - 1);
				}
			}
		}

		return A;
	}

	public int lenLongestFibSubseq(int[] A) {
		if(A == null || A.length == 0)
			return 0;
		Map<Integer,Integer> map = new HashMap<>();
		for(int i=0; i<A.length; ++i)
			map.put(A[i],i);
		int maxLen = 0;
		for(int i=0; i<A.length; ++i){
			for(int j=i+1; j<A.length; ++j){
				int left = i, right = j, count = 0;
				while(map.containsKey(A[left] + A[right])){
					int temp = right;
					right = map.get(A[left] + A[right]);
					left = temp;
					count ++;
				}
				// if exists a sequence, add first 2 nums
				if(count != 0){
					count += 2;
					maxLen = Math.max(maxLen,count);
				}
			}
		}
		return maxLen;
	}

	public int minEatingSpeed(int[] piles, int H) {
    	int max = -1;
    	for (int pile : piles) {
    		max = max < pile ? pile : max;
		}
		int left = 1;
		int right = max;

		while (left <= right) {
			int middle = left + (right - left) / 2;

			int sum = minEatingSpeedHelper(piles, middle);

			if (sum > H) {
				left = middle + 1;
			} else {
				right = middle - 1;
			}
		}

		return left;
	}

	private int minEatingSpeedHelper(int[] piles, int k) {
    	int sum = 0;

    	for (int pile : piles) {
    		sum += (pile % k == 0 ? pile / k : (pile / k) + 1);
		}

		return sum;
	}

	public String decodeAtIndex(String S, int K) {
		return null;
	}

	public int numRescueBoats(int[] people, int limit) {
		if (people == null || people.length <= 0) {
			return 0;
		}

		Arrays.sort(people);

		int start = 0;
		int end = people.length - 1;

		int result = 0;
		while (start <= end) {
			if (start == end) {
				result++;
				return result;
			} else {
				if (people[start] + people[end] <= limit) {
					result++;
					start++;
					end--;
				} else {
					end--;
					result++;
				}
			}
		}

		return result;
	}

	public TreeNode constructFromPrePost(int[] pre, int[] post) {
		if (pre == null || pre.length <= 0) {
			return null;
		}

		Map<Integer, Integer> map = new HashMap<>();
		for (int i = 0; i < post.length; ++i) {
			map.put(post[i], i);
		}

		return constructFromPrePostHelper(pre, 0, pre.length - 1, 0, pre.length - 1, map);
	}

	private TreeNode constructFromPrePostHelper(int[] pre, int pres, int pree, int posts, int poste, Map<Integer, Integer> map) {
    	if (pres == pree) {
    		return new TreeNode(pre[pres]);
		} else if (pres > pree) {
    		return null;
		}

		TreeNode root = new TreeNode(pre[pres]);

		final int next = pre[pres + 1];
    	final int postIdx = map.get(next);
		final int length = postIdx - posts + 1;
		final int preNewEnd = pres + length;

		TreeNode left = constructFromPrePostHelper(pre, pres + 1, preNewEnd,  posts, postIdx, map);
		TreeNode right = constructFromPrePostHelper(pre, preNewEnd + 1, pree, postIdx + 1, poste - 1, map);

		root.left = left;
		root.right = right;

		return root;
	}

	public List<TreeNode> allPossibleFBT(int N) {
		List<TreeNode> result = new LinkedList<>();

		if (N == 0) {
			return result;
		} else if (N == 1) {
			result.add(new TreeNode(0));
			return result;
		}

		for (int i = 1; i < N - 1; i = i+2) {
			List<TreeNode> lefts = allPossibleFBT(i);
			List<TreeNode> rights = allPossibleFBT(N - 1 - i);

			for (TreeNode left : lefts) {
				for (TreeNode right : rights) {
					TreeNode root = new TreeNode(0);

					root.left = left;
					root.right = right;

					result.add(root);
				}
			}
		}

		return result;
	}

	public int subarrayBitwiseORs(int[] A) {
		return 0;
	}

	public int totalFruit(int[] tree) {
		if (tree == null || tree.length <= 0) {
			return 0;
		}

		int kinds = 0;
		int start = 0;
		int end = 0;

		int[] hash = new int[40001];

		int result = 0;
		while (end < tree.length) {
			final int type = tree[end];
			if (hash[type] == 0) {
				kinds++;
			}
			hash[type]++;

			while (start <= end && kinds > 2) {
				if (hash[tree[start]] == 1) {
					kinds--;
				}
				hash[tree[start]]--;
				start++;
			}

			result = Math.max(result, end - start + 1);

			end++;
		}

		return result;
	}

	public int  sumSubarrayMins(int[] A) {
		if (A == null || A.length <= 0) {
			return 0;
		}

		Stack<Integer> idxs = new Stack<>();

		long result = 0;
		for (int i = 0; i < A.length; ++i) {
			while (!idxs.isEmpty() && A[idxs.peek()] > A[i]) {
				final int idxTop = idxs.pop();
				result = result % 1000000007 + (long)A[idxTop] * (idxTop - (idxs.isEmpty() ? -1 : idxs.peek())) * (i - idxTop)% 1000000007;
			}

			idxs.push(i);
		}

		while (!idxs.isEmpty()) {
			final int idxTop = idxs.pop();
			result = result % 1000000007 + (long) A[idxTop] * (idxTop - (idxs.isEmpty() ? -1 : idxs.peek())) * (A.length - idxTop) % 1000000007;
		}

		return (int) (result % 1000000007);
	}

	public int snakesAndLadders(int[][] board) {
		Queue<Integer> queue = new LinkedList<>();

		queue.add(1);

		Set<Integer> visiteds = new HashSet<>();
		visiteds.add(1);

		int cur = 1;
		int level = 0;
		int next = 0;

		final int resultDest = board.length * board[0].length;

		while (!queue.isEmpty()) {
			int head = queue.poll();
			cur--;

			for (int i = 1; i <= 6; ++i) {
				final int dest = head + i;
				final int destRowReverse = ((dest - 1) / board[0].length);
				final int destRow = board.length - 1 - destRowReverse;
				final int destColumn = ((destRowReverse & 1) == 0) ? (dest - 1) % board[0].length : (board[0].length - 1 - (dest - 1) % board[0].length);


				int finalDest = dest;
				if (board[destRow][destColumn] != -1) {
					finalDest = board[destRow][destColumn];
				}

				if (dest >= resultDest) {
					return level + 1;
				}

				if (visiteds.add(finalDest)) {
					queue.add(finalDest);
					next++;
				}
			}
			if (cur == 0) {
				cur = next;
				next = 0;
				level++;
			}
		}

		return -1;
	}

	public int partitionDisjoint(int[] A) {
		if (A == null || A.length <= 0) {
			return 0;
		}

		int premax = A[0];
		int curMax = A[0];
		int idx = 0;

		for (int i = 1; i < A.length; ++i) {
			if (A[i] >= premax) {
				curMax = Math.max(curMax, A[i]);
			} else {
				premax = curMax;
				idx = i;
			}
		}

		return idx == A.length - 1 ? idx : idx + 1;
	}

	public List<String> wordSubsets(String[] A, String[] B) {
		ArrayList<String> result = new ArrayList<>();

		int[] hash = new int[26];
		int[] temp = new int[26];
		for (String subset : B) {
			Arrays.fill(temp, 0);
			for (int i = 0; i < subset.length(); ++i) {
				temp[subset.charAt(i) - 'a']++;
			}

			for (int i = 0; i < hash.length; ++i) {
				hash[i] = Math.max(hash[i], temp[i]);
			}
		}

		int[] wordHash = new int[26];

		for (String word : A) {
			Arrays.fill(wordHash, 0);
			for (int i = 0; i < word.length(); ++i) {
				wordHash[word.charAt(i) - 'a']++;
			}

			boolean valid = true;
			for (int i = 0; i < hash.length; ++i) {
				if (wordHash[i] < hash[i]) {
					valid = false;
					break;
				}
			}

			if (valid) {
				result.add(word);
			}
		}

		return result;
	}

	public int maxSubarraySumCircular(int[] A) {
		if (A == null || A.length <= 0) {
			return 0;
		}

		int minIdx = 0;
		for (int i = 1; i < A.length; ++i) {
			if (A[minIdx] >= A[i]) {
				minIdx = i;
			}
		}

		int result = A[minIdx];
		int pre = A[minIdx];
		for (int i = 1; i < A.length; ++i) {
			final int realIdx = (i + minIdx) % A.length;
			if (pre > 0) {
				pre = pre + A[realIdx];
			} else {
				pre = A[realIdx];
			}
			result = Math.max(pre, result);
		}

		return result;
	}

	public int minAddToMakeValid(String S) {
		int result = 0;
		int balance = 0;

		for (int i = 0; i < S.length(); ++i) {
			balance += S.charAt(i) == '(' ? 1 : -1;
			if (balance < 0) {
				balance++;
				result++;
			}
		}

		return result + balance;
	}

	public int threeSumMulti(int[] A, int target) {
		long[] hash = new long[101];

		for (int num : A) {
			hash[num]++;
		}

		int result= 0;
		//
		for (int i = 0; i < hash.length; ++i) {
			if (hash[i] == 0) {
				continue;
			}

			// 处理第一个只包含一个
			int newTarget = target - i;

			// 处理另外两个值不同
			int start = i + 1;
			int end = hash.length - 1;
			while (start < end && end < hash.length) {
				if (start + end > newTarget) {
					end--;
				} else if (start + end < newTarget) {
					start++;
				} else {
					if (hash[start] != 0 && hash[end] != 0) {
						result = result % 1000000007 + (int)(hash[start] * hash[end] * hash[i]) % 1000000007;
					}
					start++;
					end--;
				}
			}

			// 处理另外两个值相同
			start = i + 1;
			end = hash.length - 1;

			while (start <= end) {
				if (hash[start] != 0 && start * 2 == newTarget) {
					result = result % 1000000007 + (int)(hash[start] * (hash[start] - 1) / 2 * hash[i] % 1000000007);
				}
				start++;
			}


			// 处理第一个包含两个
			if (hash[i] >= 2) {
				newTarget = target - 2 * i;

				start = i + 1;
				end = hash.length - 1;

				while (start <= end) {
					if (hash[start] != 0 && start == newTarget) {
						result = result % 1000000007 +  (int)(hash[i] * (hash[i] - 1) / 2 * hash[start] % 1000000007);
					}
					start++;
				}
			}

			if (hash[i] >= 3) {
				newTarget = target - 3 * i;
				if (newTarget == 0) {
					result = result % 1000000007 + (int)(hash[i] * (hash[i] - 1) * (hash[i] - 2) / 6 % 1000000007);
				}
			}
		}

		return result;
	}

	public int minFlipsMonoIncr(String S) {
		if (S == null || S.length() <= 0) {
			return 0;
		}

		int zero = 0;
		int one = 0;

		for (int i = 0; i < S.length(); ++i) {
			if (S.charAt(i) == '0') {
				one = Math.min(zero, one) + 1;
			} else {
				one = Math.min(zero, one);
				zero = zero + 1;
			}
		}

		return Math.min(zero, one);
	}

	public int numSubarraysWithSum(int[] A, int S) {
		int start = 0;
		int end = 0;

		int one = 0;
		int result = 0;
		while (end < A.length) {
			final int c = A[end];
			if (c == 1) {
				one++;
			} else if (one == S) {
				result++;
			}

			while (one > S && start < end) {
				if (A[start] == 1) {
					one--;
				}
				start++;
			}

			while (one == S && start <= end && A[start] == 0) {
				result++;
				start++;
			}

			end++;
		}

		return result;
	}

	public int minFallingPathSum(int[][] A) {
		if (A == null || A.length <= 0 || A[0] == null || A[0].length <= 0) {
			return 0;
		}

		int[][] dp = new int[A.length][A[0].length];

		for (int j = 0; j < A[0].length; ++j) {
			dp[0][j] = A[0][j];
		}

		for (int i = 1; i < A.length; ++i) {
			for (int j = 0; j < A[0].length; ++j) {
				int min = dp[i-1][j];
				if (j > 0) {
					min = Math.min(min, dp[i-1][j-1]);
				}

				if (j < A[0].length - 1) {
					min = Math.min(min, dp[i-1][j + 1]);
				}
				dp[i][j] = min + A[i][j];
			}
		}

		int result = Integer.MAX_VALUE;
		for (int j = 0; j < A[0].length; ++j) {
			result = Math.min(result, dp[A.length-1][j]);
		}

		return result;
	}


	Map<Integer, int[]> memo;
	public int[] beautifulArray(int N) {
		memo = new HashMap();
		return f(N);
	}

	public int[] f(int N) {
		if (memo.containsKey(N))
			return memo.get(N);

		int[] ans = new int[N];
		if (N == 1) {
			ans[0] = 1;
		} else {
			int t = 0;
			for (int x: f((N+1)/2))  // odds
				ans[t++] = 2*x - 1;
			for (int x: f(N/2))  // evens
				ans[t++] = 2*x;
		}
		memo.put(N, ans);
		return ans;
	}

	private int[][] bridgeDir = {
			{1,0},
			{-1, 0},
			{0, 1},
			{0, -1}
	};

	private int getPostion(int row, int column) {
		return (row << 16) | column;
	}

	private int getRow(int pos) {
		return pos >> 16;
	}

	private int getColumn(int pos) {
		return pos & 0x0000FFFF;
	}

	public int shortestBridge(int[][] A) {
		markBridge(A, 2);
		markBridge(A, 3);

		int result = Integer.MAX_VALUE;
		for (int i = 0; i < A.length; ++i) {
			for (int j = 0; j < A[0].length; ++j) {
				if (A[i][j] == 0) {
					boolean[] adjusts = {false, false};
					checkAdjustBridge(A, i, j, adjusts, 2, 3);
					if (adjusts[0] && adjusts[1]) {
						return 1;
					} else if (adjusts[0]) {
						result = Math.min(result, shortestBridge(A, i, j, 2, 3));
						if (result == 1) {
							return 1;
						}
					}
				}
			}
		}

		return result;
	}

	private int shortestBridge(int[][] A, int x, int y, int from, int to) {

		Queue<Integer> queue = new LinkedList<>();
		Set<Integer> visited = new HashSet<>();
		final int position = getPostion(x, y);
		queue.add(position);
		visited.add(position);

		int level = 1;
		int cur = 1;
		int next = 0;
		while (!queue.isEmpty()) {
			final int head = queue.poll();
			final int row = getRow(head);
			final int column = getColumn(head);
			cur--;

			for (int[] dir : bridgeDir) {
				final int nx = row + dir[0];
				final int ny = column + dir[1];
				if (nx >= 0 && nx < A.length && ny >= 0 && ny < A[0].length && A[nx][ny] == 0) {
					final int newPos = getPostion(nx, ny);
					if (visited.add(newPos)) {
						boolean[] adjusts = {false, false};
						checkAdjustBridge(A, nx, ny, adjusts, from, to);
						if (adjusts[1]) {
							return level + 1;
						} else if (!adjusts[0]) {
							queue.add(newPos);
							next++;
						}
					}
				}
			}

			if (cur == 0) {
				cur = next;
				next = 0;
				level++;
			}
		}

		return Integer.MAX_VALUE;
	}

	private void checkAdjustBridge(int[][] A, int x, int y, boolean[] adjusts, int from, int to) {
		for (int[] dir : bridgeDir) {
			final int nx = x + dir[0];
			final int ny = y + dir[1];

			if (nx >= 0 && nx < A.length && ny >= 0 && ny < A[0].length) {
				if (A[nx][ny] == from) {
					adjusts[0] = true;
				} else if (A[nx][ny] == to) {
					adjusts[1] = true;
				}
			}
		}
	}


	private void markBridge(int[][] A, int which) {
		for (int i = 0; i < A.length; ++i) {
			for (int j = 0; j < A[0].length; ++j) {
				if (A[i][j] == 1) {
					shortestBridgeHelper(A, i, j, which);
					return;
				}
			}
		}
	}
	private void shortestBridgeHelper(int[][] A, int x, int y, int which) {
		A[x][y] = which;

		for (int[] dir : bridgeDir) {
			final int nx = x + dir[0];
			final int ny = y + dir[1];

			if (nx >= 0 && nx < A.length && ny >= 0 && ny < A[0].length && A[nx][ny] == 1) {
				shortestBridgeHelper(A, nx, ny, which);
			}
		}
	}

	public int rangeSumBST(TreeNode root, int L, int R) {
		if (root == null) {
			return 0;
		}

		if (root.val >= L && root.val <= R) {
			return root.val + rangeSumBST(root.left, L, root.val) + rangeSumBST(root.right, root.val, R);
		} else if (root.val < L) {
			return rangeSumBST(root.right, L, R);
		} else {
			return rangeSumBST(root.left, L, R);
		}
	}

	public int minIncrementForUnique(int[] A) {
		if (A == null || A.length <= 1) {
			return 0;
		}

		TreeMap<Integer, Integer> map = new TreeMap<>();
		for (int num : A) {
			map.put(num, map.getOrDefault(num, 0) + 1);
		}

		int nextCompare = -1;
		int result = 0;

		List<Integer> keys = new ArrayList<>(map.keySet());
		for (int key : keys) {
			nextCompare = Math.max(nextCompare, key + 1);
			int times = map.get(key);
			while (times-- > 1) {
				while (map.containsKey(nextCompare)) {
					nextCompare++;
				}

				result += nextCompare - key;
				map.put(nextCompare, 1);

				nextCompare++;
			}

		}

		return result;
	}

	public boolean validateStackSequences(int[] pushed, int[] popped) {
		Stack<Integer> stack = new Stack<>();

		int i = 0;
		int j = 0;
		while (i < pushed.length && j < popped.length) {
			while (!stack.isEmpty() && stack.peek() == popped[j]) {
				stack.pop();
				j++;
			}
			stack.push(pushed[i]);
			++i;
		}

		while (j < popped.length) {
			if (popped[j] != stack.pop()) {
				return false;
			}
			j++;
		}

		return true;
	}

	public int removeStones(int[][] stones) {
		if (stones == null || stones.length <= 0) {
			return 0;
		}

		Map<Integer, Set<Integer>> rows = new HashMap<>();
		Map<Integer, Set<Integer>> columns = new HashMap<>();

		int[] union = new int[stones.length];

		for (int i = 0; i < stones.length; ++i) {
			final int[] stone = stones[i];
			final int row = stone[0];
			final int column = stone[1];
			rows.computeIfAbsent(row, key -> new HashSet<>()).add(i);
			columns.computeIfAbsent(column, key -> new HashSet<>()).add(i);
		}

		int result = 0;
		int id = 1;
		for (int i = 0; i < union.length; ++i) {
			if (union[i] != 0) {
				continue;
			}

			result += unionStones(rows, columns, stones, i, union, id++);
		}

		return result;
	}

	private int unionStones(Map<Integer, Set<Integer>> rows, Map<Integer, Set<Integer>> columns, int[][] stones,
							int idx, int[] unions, final int id) {
		// bfs
		int result = 0;

		Queue<Integer> queue = new LinkedList<>();
		queue.add(idx);
		unions[idx] = id;
		result = 1;

		while (!queue.isEmpty()) {
			int head = queue.poll();

			final int row = stones[head][0];
			final int column = stones[head][1];

			Iterator<Integer> it = rows.get(row).iterator();
			while (it.hasNext()) {
				int adjacent = it.next();
				if (unions[adjacent] == 0) {
					unions[adjacent] = id;
					result++;
					queue.add(adjacent);
					it.remove();
				}
			}

			it = columns.get(column).iterator();
			while (it.hasNext()) {
				int adjacent = it.next();
				if (unions[adjacent] == 0) {
					unions[adjacent] = id;
					result++;
					queue.add(adjacent);
					it.remove();
				}
			}
		}

		return result - 1;
	}

	public int bagOfTokensScore(int[] tokens, int P) {
		if (tokens == null || tokens.length <= 0) {
			return 0;
		}

		int result = 0;

		int gains = 0;

		Arrays.sort(tokens);

		int left = 0;
		int right = tokens.length - 1;

		while (left <= right) {
			while(left <= right && P >= tokens[left]) {
				P -= tokens[left];
				gains++;
				left++;
			}

			result = Math.max(result, gains);

			if (gains > 0) {
				P += tokens[right];
				right--;
				gains--;
			} else {
				break;
			}
		}

		return result;
	}

	public int[] deckRevealedIncreasing(int[] deck) {
		int N = deck.length;
		Deque<Integer> index = new LinkedList();
		for (int i = 0; i < N; ++i)
			index.add(i);

		int[] ans = new int[N];
		Arrays.sort(deck);
		for (int card: deck) {
			ans[index.pollFirst()] = card;
			if (!index.isEmpty())
				index.add(index.pollFirst());
		}

		return ans;
	}

	public boolean flipEquiv(TreeNode root1, TreeNode root2) {
		if (root1 == root2) {
			return true;
		}

		if (root1 == null && root2 == null) {
			return true;
		} else if (root2 == null) {
			return false;
		} else if (root1 == null) {
			return false;
		}

		if (root1.val != root2.val) {
			return false;
		}

		return (flipEquiv(root1.left, root2.left) && flipEquiv(root1.right, root2.right)) ||
				(flipEquiv(root1.left, root2.right) && flipEquiv(root1.right, root2.left));
	}

	public boolean canReorderDoubled(int[] A) {
		if (A == null || A.length <= 0) {
			return true;
		}

		Map<Integer, Integer> map = new HashMap<>();
		for (int num : A) {
			map.put(num, map.getOrDefault(num, 0) + 1);
		}

		int[] keys = new int[map.keySet().size()];
		int i = 0;
		for (int key : map.keySet()) {
			keys[i] = key;
			i++;
		}



		for (int num : keys) {
			if (num == 0) {
				int value = map.getOrDefault(0, -1);
				if (value != -1) {
					if ((value & 1) != 0) {
						return false;
					} else {
						map.remove(0);
					}
				}
			} else  {
				while (map.containsKey(num)) {
					if (map.containsKey(num * 2)) {
						decreaseValue(map, num * 2);
						decreaseValue(map, num);
					} else if (num % 2 == 0 && map.containsKey(num / 2)) {
						decreaseValue(map, num / 2);
						decreaseValue(map, num);
					} else {
						return false;
					}
				}
			}
		}

		return map.isEmpty();
	}

	private void decreaseValue(Map<Integer, Integer> map, int key) {
		int value = map.get(key);
		if (value == 1) {
			map.remove(key);
		} else {
			map.put(key, value - 1);
		}
	}

	public int minDeletionSize(String[] A) {
		if (A == null || A.length <= 0 || A[0].length() <= 0) {
			return 0;
		}

		int N = A.length;
		int W = A[0].length();

		int anser = 0;

		int[] needCompare = new int[A.length];

		for (int i = 0; i < W; ++i) {
			final int[] nextCompare = new int[A.length];
			final int compare = isSorted(A, i, needCompare, nextCompare);
			if (compare > 0) {
				anser++;
			} else if (compare < 0) {
				return anser;
			} else {
				needCompare = nextCompare;
			}
		}

		return anser;
	}

	// needCompare 0需要比较，1不需要比较
	private int isSorted(String[] A, int idx, int[] needCompare, int[] nextCompare) {
		boolean hasEqual = false;
		for (int i = 0; i < A.length - 1; ++i) {
			if (needCompare[i] == 0) {
				final int diff = A[i].charAt(idx) - A[i+1].charAt(idx);
				if (diff == 0) {
					hasEqual = true;
				} else if (diff > 0) {
					return 1;
				} else {
					nextCompare[i] = 1;
				}
			} else {
				nextCompare[i] = 1;
			}
		}

		return hasEqual ? 0 : -1;
	}

	public int[] prisonAfterNDays(int[] cells, int N) {
		if (N <= 0) {
			return cells;
		}

		Map<Integer, Integer> map = new HashMap<>();
		int first = 0;
		for (int i = 0; i < 8; ++i) {
			if (cells[i] == 1) {
				first += (1 << (7 - i));
			}
		}

		Map<Integer, Integer> mapReverse = new HashMap<>();
		map.put(first, 0);
		mapReverse.put(0, first);

		int[] preCells = cells;
		int[] nextCells = new int[cells.length];
		int[] temp;
		for (int i = 1; i <= N; ++i) {
			final int next = prisonAfterNDaysNext(preCells, nextCells);
			final Integer preValue = map.put(next, i);
			if (preValue != null) {
				int periodStart = preValue;
				int periodEnd = i;

				int index = (N - periodEnd) % (periodEnd - periodStart);
				int result = mapReverse.get(periodStart + index);
				final int[] res = new int[8];
				int idx = 7;
				while (result != 0) {
					res[idx--] = result & 1;
					result >>= 1;
				}

				return res;
			} else {
				temp = preCells;
				preCells = nextCells;
				nextCells = temp;
				mapReverse.put(i, next);
			}
		}

		return preCells;
	}


	public int prisonAfterNDaysNext(int[] preCells, int[] nextCells) {
		int result = 0;
		nextCells[0] = nextCells[7] = 0;
		for (int i = 1; i < 7; ++i) {
			nextCells[i] = preCells[i-1] == preCells[i+1] ? 1 : 0;
			if (nextCells[i] == 1) {
				result += (1 << (7 - i));
			}
		}

		return result;
	}

	public boolean isCompleteTree(TreeNode root) {
		boolean end = false;
		Queue<TreeNode> queue = new LinkedList<>();
		queue.add(root);
		while(!queue.isEmpty()) {
			TreeNode cur = queue.poll();
			if(cur == null) end = true;
			else{
				if(end) return false;
				queue.add(cur.left);
				queue.add(cur.right);
			}
		}
		return true;
	}

	public int maxWidthRamp(int[] A) {
		Integer[] B = new Integer[A.length];
		for (int i = 0; i < A.length; ++i) {
			B[i] = i;
		}

		Arrays.sort(B, new Comparator<Integer>() {
			@Override
			public int compare(Integer o1, Integer o2) {
				return A[o1] - A[o2];
			}
		});

		int minIdx = Integer.MAX_VALUE;
		int result = 0;
		for (int i = 0; i < B.length; ++i) {
			if (B[i] > minIdx) {
				result = Math.max(result, B[i] - minIdx);
			}
			minIdx = Math.min(B[i], minIdx);
		}

		return result;
	}

	public String[] spellchecker(String[] wordlist, String[] queries) {
		if (wordlist == null || queries == null) {
			return null;
		}

		Map<String, Integer> exactMatch = new HashMap<>();
		for (int i = 0; i < wordlist.length; ++i) {
			final String word = wordlist[i];
			exactMatch.putIfAbsent(word, i);
		}

		Map<String, Integer> capitlizationMatch = new HashMap<>();
		for (int i = 0; i < wordlist.length; ++i) {
			final String lower = wordlist[i].toLowerCase();
			capitlizationMatch.putIfAbsent(lower, i);
		}

		Map<String, Integer> vowelMatch = new HashMap<>();
		for (int i = 0; i < wordlist.length; ++i) {
			final String vowelWord = toFirstVowel(wordlist[i]);
			vowelMatch.putIfAbsent(vowelWord, i);
		}

		String[] result = new String[queries.length];
		int i = 0;
		for (String query : queries) {
			Integer exact = exactMatch.get(query);
			if (exact != null) {
				result[i] = wordlist[exact];
			} else {
				final Integer cap = capitlizationMatch.get(query.toLowerCase());
				if (cap != null) {
					result[i] = wordlist[cap];
				} else {
					final Integer vowel = vowelMatch.get(toFirstVowel(query));
					if (vowel != null) {
						result[i] = wordlist[vowel];
					}
				}
			}

			if (result[i] == null) {
				result[i] = "";
			}
			i++;
		}

		return result;
	}

	private String toFirstVowel(String word) {
		word = word.toLowerCase();

		String result = "";
		for (int i = 0; i < word.length(); ++i) {
			final char c = word.charAt(i);
			if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
				result += 'a';
			} else {
				result += c;
			}
		}
		return result;
	}

	public int[] numsSameConsecDiff(int N, int K) {
		if (N == 1) {
			int[] result = new int[10];
			for (int i = 0; i < 10; ++i) {
				result[i] = i;
			}
			return result;
		} else {
			int[] nums = numsSameConsecDiff(N - 1, K);
			ArrayList<Integer> listResult = new ArrayList<>();
			for (int num : nums) {
				if (num != 0) {
					int last = num % 10;
					if (K == 0) {
						listResult.add(num * 10 + last);
					} else {
						if (last + K <= 9) {
							listResult.add(num * 10 + last + K);
						}
						if (last - K >= 0) {
							listResult.add(num * 10 + last - K);
						}
					}
				}
			}

			int[] result = new int[listResult.size()];
			for (int i = 0; i < result.length; ++i) {
				result[i] = listResult.get(i);
			}

			return result;
		}
	}

	public List<Integer> pancakeSort(int[] A) {
		List<Integer> result = new ArrayList<>();
		for (int i = A.length; i >= 1; --i) {
			pancakeSort(A, i, result);
		}
		return result;
	}

	public void pancakeSort(int[] A, int toSort, List<Integer> result) {
		int idx = -1;
		for (int i = 0; i < toSort; ++i) {
			if (A[i] == toSort) {
				idx = i;
				break;
			}
		}

		if (idx + 1 == toSort) {
			return;
		}

		// 先把idx弄到第一个
		if (A[0] != toSort) {
			result.add(idx + 1);
			pancakeSortReverse(A, 0, idx);
		}

		//
		result.add(toSort);
		pancakeSortReverse(A, 0, toSort - 1);
	}

	private void pancakeSortReverse(int[] A, int start, int end) {
		while (start < end) {
			int temp = A[start];
			A[start] = A[end];
			A[end] =  temp;

			start++;
			end--;
		}
	}

	public List<Integer> flipMatchVoyage(TreeNode root, int[] voyage) {
		return flipMatchVoyage(root, voyage, 0, voyage.length - 1);
	}

	private List<Integer> flipMatchVoyage(TreeNode root, int[] voyage, int start, int end) {
		List<Integer> result = new ArrayList<>();
		if (root == null) {
			return result;
		}
		if (voyage[start] != root.val) {
			result.add(-1);
			return result;
		}

		if (root.left != null && root.right != null) {
			if (root.left.val == voyage[start + 1]) {
				// 查找跟right相等的值
				int idx = -1;
				for (int i = start + 2; i <= end; ++i) {
					if (voyage[i] == root.right.val) {
						idx = i;
						break;
					}
				}

				if (idx == -1) {
					result.add(-1);
					return result;
				}

				List<Integer> leftRes = flipMatchVoyage(root.left, voyage, start + 1, idx - 1);
				List<Integer> rightRes = flipMatchVoyage(root.right, voyage, idx, end);

				if ((leftRes.size() == 1 && leftRes.get(0) == -1) ||
						(rightRes.size() == 1 && rightRes.get(0) == -1)) {
					result.add(-1);
					return result;
				}

				result.addAll(leftRes);
				result.addAll(rightRes);

				return result;
			} else if (root.right.val == voyage[start + 1]) {
				// 查找跟right相等的值
				int idx = -1;
				for (int i = start + 2; i <= end; ++i) {
					if (voyage[i] == root.left.val) {
						idx = i;
						break;
					}
				}

				if (idx == -1) {
					result.add(-1);
					return result;
				}

				List<Integer> leftRes = flipMatchVoyage(root.right, voyage, start + 1, idx - 1);
				List<Integer> rightRes = flipMatchVoyage(root.left, voyage, idx, end);

				if ((leftRes.size() == 1 && leftRes.get(0) == -1) ||
						(rightRes.size() == 1 && rightRes.get(0) == -1)) {
					result.add(-1);
					return result;
				}

				result.add(root.val);
				result.addAll(leftRes);
				result.addAll(rightRes);

				return result;
			} else {
				result.add(-1);
				return result;
			}
		} else if (root.left == null && root.right == null) {
			if (start == end) {
				return result;
			} else {
				result.add(-1);
				return result;
			}
		} else if (root.right == null) {
			if (root.left.val == voyage[start + 1]) {
				List<Integer> leftRes = flipMatchVoyage(root.left, voyage, start + 1, end);
				if (leftRes.size() == 1 && leftRes.get(0) == -1) {
					result.add(-1);
					return result;
				}
				result.addAll(leftRes);

				return result;
			} else {
				result.add(-1);
				return result;
			}
		} else {
			if (root.right.val == voyage[start + 1]) {
				List<Integer> leftRes = flipMatchVoyage(root.right, voyage, start + 1, end);
				if (leftRes.size() == 1 && leftRes.get(0) == -1) {
					result.add(-1);
					return result;
				}
				result.addAll(leftRes);

				return result;
			} else {
				result.add(-1);
				return result;
			}
		}
	}

	public int subarraysDivByK(int[] A, int K) {
		Map<Integer, Integer> map = new HashMap<>();
		int result = 0;
		int sum = 0;
		map.put(0, 0);

		for (int num : A) {
			sum += num;
			int left = sum % K;
			int old = map.getOrDefault(left, -1);
			if (old == -1) {
				if (left > 0) {
					result += map.getOrDefault(left - K, 0);
				} else {
					result += map.getOrDefault(K + left, 0);
				}

				map.put(left, 1);

			} else {
				map.put(left, old + 1);
				result += old;
				if (left == 0) {
					result += 1;
				}
			}
		}

		return result;
	}

	public int maxTurbulenceSize(int[] A) {
		if (A == null || A.length <= 0) {
			return 0;
		}

		int lastOp = 0;
		int result = 1;

		int res = 1;

		for (int i = 1; i < A.length; ++i) {
			if (A[i] == A[i-1]) {
				lastOp = 0;
				result = 1;
			} else if (A[i-1] < A[i]) {
				if (lastOp == 0 || lastOp == 2) {
					result++;
				} else {
					result = 2;
				}
				lastOp = 1;
			} else {
				if (lastOp == 0 || lastOp == 1) {
					result++;
				} else {
					result = 2;
				}
				lastOp = 2;
			}

			res = Math.max(res, result);
		}

		return res;
	}

	public int distributeCoins(TreeNode root) {
		distributeCoinsHelper(root);
		return distributeCoinsResult;
	}

	int distributeCoinsResult = 0;
	public int distributeCoinsHelper(TreeNode root) {
		if (root == null) {
			return 0;
		}

		int leftMore = distributeCoinsHelper(root.left);
		int rightMore = distributeCoinsHelper(root.right);

		distributeCoinsResult = Math.abs(leftMore) + Math.abs(rightMore);

		return leftMore + rightMore + root.val - 1;
	}

	public int mincostTickets(int[] days, int[] costs) {
		if (days == null || days.length <= 0) {
			return 0;
		}

		int[] dp = new int[days.length];

		dp[days.length - 1] = Math.min(costs[0], Math.min(costs[1], costs[2]));

		for (int i = days.length - 2; i >= 0; --i) {
			final int curDay = days[i];
			int result = costs[0] + dp[i+1];

			int nextIdx = binarySearch(days, curDay + 6);
			result = Math.min(result, costs[1] + (nextIdx < days.length - 1 ? dp[nextIdx + 1] : 0));

			nextIdx = binarySearch(days, curDay + 29);
			result = Math.min(result, costs[2] + (nextIdx < days.length - 1 ? dp[nextIdx + 1] : 0));

			dp[i] = result;
		}

		return dp[0];
	}

	private int mincostTicketsHelper(int[] days, int[] costs, int idx) {
		if (idx >= days.length) {
			return 0;
		}

		// 取三个最小的
		int result = Integer.MAX_VALUE;

		result = Math.min(costs[0] + mincostTicketsHelper(days, costs, idx + 1), result);

		int nextIdx = binarySearch(days, days[idx] + 6);

		result = Math.min(costs[1] + mincostTicketsHelper(days, costs, nextIdx + 1), result);

		nextIdx = binarySearch(days, days[idx] + 29);

		result = Math.min(costs[2] + mincostTicketsHelper(days, costs, nextIdx + 1), result);

		return result;

	}

	// 二分查找
	private int binarySearch(int[] arrays, int timestamp) {
		int left = 0;
		int right = arrays.length - 1;

		while (left <= right) {
			int middle = left + (right - left) / 2;

			if (arrays[middle] == timestamp) {
				return middle;
			} else if (arrays[middle] > timestamp) {
				right = middle - 1;
			} else {
				left = middle + 1;
			}
		}

		return right;
	}

	public Interval[] intervalIntersection(Interval[] A, Interval[] B) {
		if (A == null || A.length <= 0 || B == null || B.length <= 0) {
			return new Interval[0];
		}

		ArrayList<Interval> result = new ArrayList<>();

		int i = 0;
		int j = 0;

		while (i < A.length && j < B.length) {
			if (A[i].end < B[j].start) {
				i++;
			} else if (B[j].end < A[i].start) {
				j++;
			} else if (A[i].end >= B[j].start && B[j].end >= A[i].start) {
				Interval interval = new Interval(
						Math.max(A[i].start, B[j].start),
						Math.min(A[i].end, B[j].end));

				result.add(interval);


				if (A[i].end > B[j].end) {
					j++;
				} else if (A[i].end < B[j].end) {
					i++;
				} else {
					i++;
					j++;
				}
			}
		}

		Interval[] res = new Interval[result.size()];
		return result.toArray(res);
	}


	private class VerticalTree {
		int mYPos;
		int mValue;

		VerticalTree(int yPos, int value) {
			mYPos = yPos;
			mValue = value;
		}
	}

	public List<List<Integer>> verticalTraversal(TreeNode root) {
		List<List<Integer>> result = new ArrayList<>();
		if (root == null) {
			return result;
		}

		TreeMap<Integer, ArrayList<VerticalTree>> map = new TreeMap<>();
		verticalTraversalHelper(root, 0, 0, map);

		for (Map.Entry<Integer, ArrayList<VerticalTree>> entry : map.entrySet()) {
			final ArrayList<VerticalTree> list = entry.getValue();
			list.sort(new Comparator<VerticalTree>() {
				@Override
				public int compare(VerticalTree o1, VerticalTree o2) {
					if (o1.mYPos < o2.mYPos) {
						return -1;
					} else if (o1.mYPos > o2.mYPos) {
						return 1;
					} else {
						return o1.mValue - o2.mValue;
					}
				}
			});

			ArrayList<Integer> listResult = new ArrayList<>(list.size());
			for (VerticalTree verticalTree : list) {
				listResult.add(verticalTree.mValue);
			}

			result.add(listResult);
		}

		return result;
	}

	private void verticalTraversalHelper(TreeNode root, int x, int y, TreeMap<Integer, ArrayList<VerticalTree>> map) {
		if (root == null) {
			return;
		}

		map.computeIfAbsent(x, new Function<Integer, ArrayList<VerticalTree>>() {
			@Override
			public ArrayList<VerticalTree> apply(Integer integer) {
				return new ArrayList<>();
			}
		}).add(new VerticalTree(y, root.val));

		verticalTraversalHelper(root.left, x - 1, y + 1, map);
		verticalTraversalHelper(root.right, x + 1, y + 1, map);
	}

	String result = null;
	public String smallestFromLeaf(TreeNode root) {
		smallestFromLeaf(root, "");

		return result == null ? "" : result;
	}

	private void smallestFromLeaf(TreeNode root, String prefix) {
		if (root == null) {
			return;
		}

		if (root.left == null && root.right == null) {
			final String word = new StringBuilder((prefix + ((char)('a' + root.val)))).reverse().toString();
			if (result == null) {
				result = word;
			} else if (result.compareTo(word) > 0){
				result = word;
			}

			return;
		}

		smallestFromLeaf(root.left, prefix + ((char)('a' + root.val)));
		smallestFromLeaf(root.right, prefix + ((char)('a' + root.val)));
	}


	private int[] EquationsRoot;
	private int[] EquationsWight;

	public boolean equationsPossible(String[] equations) {
		EquationsRoot = new int[26];
		EquationsWight = new int[26];
		for (int i = 0; i < EquationsRoot.length; ++i) {
			EquationsRoot[i] = i;
			EquationsWight[i] = 1;
		}

		for (int i = 0; i < equations.length; ++i) {
			final String equation = equations[i];

			if (equation.charAt(1) == '=') {
				int p = equation.charAt(0) - 'a';
				int q = equation.charAt(3) - 'a';
				equationsUnion(p, q);
			}
		}

		for (int i = 0; i < equations.length; ++i) {
			final String equation = equations[i];

			if (equation.charAt(1) != '=') {
				int p = equation.charAt(0) - 'a';
				int q = equation.charAt(3) - 'a';
				if (equaionsFindUnion(p, q)) {
					return false;
				}
			}
		}

		return true;
	}

	private boolean equaionsFindUnion(int p, int q) {
		if (p == q) {
			return true;
		}
		return equationsFindParent(p) == equationsFindParent(q);
	}

	private void equationsUnion(int p, int q) {
		int pParent = equationsFindParent(p);
		int qParent = equationsFindParent(q);

		if (pParent == qParent) {
			return;
		}

		if (EquationsWight[pParent] < EquationsWight[qParent]) {
			EquationsRoot[qParent] = pParent;
			EquationsWight[pParent] += EquationsWight[qParent];
		} else {
			EquationsRoot[pParent] = qParent;
			EquationsWight[qParent] += EquationsWight[pParent];
		}
	}

	private int equationsFindParent(int p) {
		while (EquationsRoot[p] != p) {
			p = EquationsRoot[p];
		}

		return p;
	}

	public int brokenCalc(int X, int Y) {
		if (X >= Y) {
			return X - Y;
		}

		if ((Y & 1) == 0) {
			return 1 + brokenCalc(X, Y >> 1);
		} else {
			return 2 + brokenCalc(X, (Y + 1) >> 1);
		}
	}

	public TreeNode insertIntoMaxTree(TreeNode root, int val) {
		mMaxTreeRoot = root;
		insertIntoMaxTreeHelper(root, val, true);
		return mMaxTreeRoot;
	}

	private TreeNode mMaxTreeRoot;
	private void insertIntoMaxTreeHelper(TreeNode cur, int val, boolean root) {
		if (cur == null) {
			mMaxTreeRoot = new TreeNode(val);
			return;
		} else if (cur.val < val && root) {
			mMaxTreeRoot = new TreeNode(val);
			mMaxTreeRoot.left = cur;
			return;
		}
		if (cur.right == null) {
			cur.right = new TreeNode(val);
		} else if (val > cur.right.val) {
			TreeNode oldLeft = cur.right;
			cur.right = new TreeNode(val);
			cur.right.left = oldLeft;
		} else {
			insertIntoMaxTreeHelper(cur.right, val, false);
		}
	}

}