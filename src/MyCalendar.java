public class MyCalendar {

    private class Node {
        Node left;
        Node right;
        int start;
        int end;

        public Node(int start, int end) {
            this.start = start;
            this.end = end;
        }
    }

    private Node mRoot;

    public MyCalendar() {

    }

    public boolean book(int start, int end) {
        if (mRoot == null) {
            mRoot = new Node(start, end);
            return true;
        }

        Node root = mRoot;
        while (true) {
            if (start >= root.end) {
                if (root.right == null) {
                    root.right = new Node(start, end);
                    return true;
                } else {
                    root = root.right;
                }
            } else if (end <= root.start) {
                if (root.left == null) {
                    root.left = new Node(start, end);
                    return true;
                } else {
                    root = root.left;
                }
            } else {
                return false;
            }
        }
    }

}
