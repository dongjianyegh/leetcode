import java.util.List;

public class NestedInteger {

    // Constructor initializes an empty nested list.
    public NestedInteger(){}

    public NestedInteger(int value){}

    // @return true if this NestedInteger holds a single integer, rather than a nested list.
    public boolean isInteger(){
        return false;
    }

    // @return the single integer that this NestedInteger holds, if it holds a single integer
    // Return null if this NestedInteger holds a nested list

    public Integer getInteger(){
        return 0;
    }

    // @return the nested list that this NestedInteger holds, if it holds a nested list
    // Return null if this NestedInteger holds a single integer
    public List<NestedInteger> getList(){
        return null;
    }

    public void setInteger(int value){}

    public void add(NestedInteger ni){}
}
