import java.nio.channels.NonWritableChannelException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

public class Twitter {
	
	// interger -> userId, TweetNode -> tweetId list, save tweetIds of all users
	private final Map<Integer, TweetNode> mMapUserTweets;
	// integer -> userid, integer -> userids of follows
	private final Map<Integer, Set<Integer>> mMapUserFollows;
	
	private final Comparator<TweetNode> mComparator = new Comparator<TweetNode>() {
		@Override
		public int compare(TweetNode o1, TweetNode o2) {
			// TODO Auto-generated method stub
			return o2.mPostId - o1.mPostId;
		}
	};
	
	private int mPostId = 0;
	
	/** Initialize your data structure here. */
    public Twitter() {
        mMapUserFollows = new HashMap<>();
        mMapUserTweets = new HashMap<>();
    }
    
    /** Compose a new tweet. */
    public void postTweet(int userId, int tweetId) {
        addTweetToHead(userId, tweetId);
    }
    
    /** Retrieve the 10 most recent tweet ids in the user's news feed. 
     * Each item in the news feed must be posted by users who the user followed or by the user herself. 
     * Tweets must be ordered from most recent to least recent. */
    public List<Integer> getNewsFeed(int userId) {
        PriorityQueue<TweetNode> queue = new PriorityQueue<>(mComparator);
        
        TweetNode head = mMapUserTweets.get(userId);
        if (head != null) {
        		queue.add(head);
        }
        Set<Integer> followees = mMapUserFollows.get(userId);
        if (followees != null) {
        		for (Integer followee : followees) {
        			head = mMapUserTweets.get(followee);
        			if (head != null) {
        				queue.add(head);
        			}
        		}
        }
        
        List<Integer> result = new LinkedList<>();
        while (!queue.isEmpty()) {
        		TweetNode curMaxNode = queue.poll();
        		result.add(curMaxNode.mTweetId);
        		
        		if (result.size() >= 10) {
        			break;
        		}
        		
        		if (curMaxNode.mNext != null) {
        			queue.add(curMaxNode.mNext);
        		}
        }
        
        return result;
    }
    
    /** Follower follows a followee. If the operation is invalid, it should be a no-op. */
    public void follow(int followerId, int followeeId) {
    		if (followeeId == followerId) return;
        Set<Integer> followees = mMapUserFollows.get(followerId);
        if (followees == null) {
        		followees = new HashSet<>();
        		mMapUserFollows.put(followerId, followees);
        }
        followees.add(followeeId);
    }
    
    /** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
    public void unfollow(int followerId, int followeeId) {
    		if (followeeId == followerId) return;
        Set<Integer> followees = mMapUserFollows.get(followerId);
        if (followees != null) {
        		followees.remove(followeeId);
        }
    }
    
    private void addTweetToHead(int userId, int tweetId) {
    		TweetNode head = mMapUserTweets.get(userId);
    		TweetNode post = new TweetNode(tweetId, mPostId++);
    		if (head == null) {
    			head = post; 
    		} else {
    			post.mNext = head;
    			head = post;
    		}
    		mMapUserTweets.put(userId, head);
    }
    
    private class TweetNode {
    		final int mPostId;
    		int mTweetId;
    		TweetNode mNext;
    		public TweetNode(int tweetid, int postId) {
    			mTweetId = tweetid;
    			mPostId = postId;
		}
    }
    
    
}
