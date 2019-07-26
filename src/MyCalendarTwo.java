public class MyCalendarTwo {
    private class Node {
        Node left;
        Node right;
        Node intersect;

        int start;
        int end;

        public Node(int start, int end) {
            this.start = start;
            this.end = end;
        }
    }

    private Node mRoot;

    public MyCalendarTwo() {

    }

    public boolean book(int start, int end) {
        if (mRoot == null) {
            mRoot = new Node(start, end);
            return true;
        }

        Node root = mRoot;
        int intersect = 1;
        while (intersect <= 3) {
            if (start >= root.end) {
                if (root.intersect != null && !(start >= root.intersect.end || end <= root.intersect.start)) {
                    intersect += 1;
                    if (intersect >= 3) {
                        return false;
                    }
                }
                if (root.right == null) {
                    root.right = new Node(start, end);
                    return true;
                } else {
                    root = root.right;
                }
            } else if (end <= root.start) {
                if (root.intersect != null && !(start >= root.intersect.end || end <= root.intersect.start)) {
                    intersect += 1;
                    if (intersect >= 3) {
                        return false;
                    }
                }
                if (root.left == null) {
                    root.left = new Node(start, end);
                    return true;
                } else {
                    root = root.left;
                }
            } else {
                if (root.intersect == null)
                    return false;
            }
        }

        return false;
    }


}
