import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Represents a state in the river-crossing problem for the A* search algorithm.
 */
class State implements Comparable<State> {

    private int f, h, g; // Evaluation function components
    private int totalTime; // Total time elapsed in the crossing
    private State father; // Parent state
    private int numberOfPeople; // Total number of people in the problem

    private List<Integer> rightSide; // Times taken by people on the right side
    private List<Integer> leftSide; // Times taken by people on the left side

    private boolean lanternOnRight; // Indicates whether the lantern is on the right side or on the left side

    // Constructor for initial state
    public State(int numberOfPeople, int totalTime, List<Integer> i_times) {
        // Initialize state properties
        this.f = 0;
        this.h = 0;
        this.g = 0;
        this.father = null;
        this.totalTime = totalTime;
        this.numberOfPeople = numberOfPeople;
        this.lanternOnRight = true; // True if the lantern is on the right side, false otherwise
        this.rightSide = new ArrayList<>(i_times);
        this.leftSide = new ArrayList<>();
    }

    // copy constructor
    public State(State s) {
        // Copy values from another state
        this.f = s.f;
        this.h = s.h;
        this.g = s.g;
        this.father = s.father;
        this.totalTime = s.totalTime;
        this.numberOfPeople = s.numberOfPeople;
        this.lanternOnRight = s.lanternOnRight;
        this.rightSide = new ArrayList<>(s.rightSide);
        this.leftSide = new ArrayList<>(s.leftSide);

    }

    // Getters and setters for various properties

    public boolean isLanternOnRight() {
        return lanternOnRight;
    }

    public void setLanternOnRight(boolean lanternOnRight) {
        this.lanternOnRight = lanternOnRight;
    }

    public int getF() {
        return this.f;
    }

    public int getG() {
        return this.g;
    }

    public int getH() {
        return this.h;
    }

    public State getFather() {
        return this.father;
    }

    public int getTotalTime() {
        return this.totalTime;
    }

    public List<Integer> getRightside() {
        return this.rightSide;
    }

    public List<Integer> getLeftside() {
        return this.leftSide;
    }

    public int getN() {
        return this.numberOfPeople;
    }

    public void setF(int f) {
        this.f = f;
    }

    public void setG(int g) {
        this.g = g;
    }

    public void setH(int h) {
        this.h = h;
    }

    public void setFather(State f) {
        this.father = f;
    }

    public void setTotalTime(int time) {
        this.totalTime = time;
    }

    public void setRightside(List<Integer> rightSide) {
        this.rightSide = new ArrayList<>(rightSide);
    }

    public void setLeftside(List<Integer> leftSide) {
        this.leftSide = new ArrayList<>(leftSide);
    }

    public void setN(int N) {
        this.numberOfPeople = N;
    };

    // Evaluate the state based on the selected heuristic
    public void evaluate(int heuristic) {
        switch (heuristic) {
            case 1:
                this.heuristic1();
                break;
            default:
                break;
        }

        setF(this.getG() + this.getH());

    }

    // Heuristic 1: Find the maximum time on the right side
    public void heuristic1() {
        int maxTime = Integer.MIN_VALUE; // minimum possible value for type int
        for (int time : rightSide) {
            if (time > maxTime) {
                maxTime = time;
            }
        }
        setH(maxTime);
    }

    // Print the state along with move details
    public void printMoveDetails(State previousState) {
        StringBuilder output = new StringBuilder();

        // Construct the string for the left side if it is not empty
        if (!leftSide.isEmpty()) {
            StringBuilder leftStrBuilder = new StringBuilder("Left side: ");
            for (int time : leftSide) {
                leftStrBuilder.append(time).append(", ");
            }
            // Remove the last comma and space.
            leftStrBuilder.delete(leftStrBuilder.length() - 2, leftStrBuilder.length());
            output.append(leftStrBuilder).append(" ");
        }

        // Construct the string for the right side if it is not empty
        if (!rightSide.isEmpty()) {
            StringBuilder rightStrBuilder = new StringBuilder("Right side: ");
            for (int time : rightSide) {
                rightStrBuilder.append(time).append(", ");
            }
            // Remove the last comma and space
            rightStrBuilder.delete(rightStrBuilder.length() - 2, rightStrBuilder.length());
            output.append(rightStrBuilder).append(" ");
        }

        // Compose the torch position string and append it
        output.append("Torch position: ").append(isLanternOnRight() ? "Right" : "Left").append(" ");

        // Create the elapsed time string and append it.
        output.append("Time taken: ").append(this.getG());

        // Check if this state is different from the previous one
        if (!this.equals(previousState)) {
            // Print details of the move
            output.append(" (Move: ");
            if (isLanternOnRight()) {
                output.append("Person from Right to Left)");
            } else {
                output.append("Person from Left to Right)");
            }
        }

        // Output the complete status of the crossing.
        System.out.println(output);
    }



    public List<State> getChildren(int heuristic) {
        List<State> children = new ArrayList<>();

        if (lanternOnRight) {
            generateChildrenFromRight(children);
        } else {
            generateChildrenFromLeft(children);
        }

        return children;
    }

    private void generateChildrenFromRight(List<State> children) {
        for (int i = 0; i < rightSide.size(); i++) {
            for (int j = i + 1; j < rightSide.size(); j++) {
                State child = moveElementsToOtherSide(i, j);
                children.add(child);
            }
        }
    }

    private void generateChildrenFromLeft(List<State> children) {
        for (int i = 0; i < leftSide.size(); i++) {
            State child = moveElementToOtherSide(i);
            children.add(child);
        }
    }

    private State moveElementsToOtherSide(int index1, int index2) {
        List<Integer> right = new ArrayList<>(rightSide);
        List<Integer> left = new ArrayList<>(leftSide);

        int timeToCross1 = right.get(index1);
        int timeToCross2 = right.get(index2);

        left.add(timeToCross1);
        left.add(timeToCross2);

        right.remove(Math.max(index1, index2));
        right.remove(Math.min(index1, index2));

        State child = new State(this);
        child.setRightside(right);
        child.setLeftside(left);
        child.setG(child.getG() + Math.max(timeToCross1, timeToCross2));
        child.setTotalTime(child.getTotalTime() - Math.max(timeToCross1, timeToCross2));
        child.setLanternOnRight(false);

        return child;
    }

    private State moveElementToOtherSide(int index) {
        List<Integer> right = new ArrayList<>(rightSide);
        List<Integer> left = new ArrayList<>(leftSide);

        int timeToCross = left.get(index);
        right.add(timeToCross);
        left.remove(index);

        State child = new State(this);
        child.setRightside(right);
        child.setLeftside(left);
        child.setG(child.getG() + timeToCross);
        child.setTotalTime(child.getTotalTime() - timeToCross);
        child.setLanternOnRight(true);

        return child;
    }

    // Check if the state is the final/goal state
    public boolean isFinal() {
        return rightSide.isEmpty() && isLanternOnRight() == false;
    }

    // Equals method for comparing two states
    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        State state = (State) obj;
        return lanternOnRight == state.lanternOnRight &&
                totalTime == state.totalTime &&
                numberOfPeople == state.numberOfPeople &&
                Objects.equals(rightSide, state.rightSide) &&
                Objects.equals(leftSide, state.leftSide);
    }

    // Hash code method for generating hash codes for states
    @Override
    public int hashCode() {
        return Objects.hash(lanternOnRight, totalTime, numberOfPeople, rightSide, leftSide);
    }

    // Compare two states based on their 'f' values
    @Override
    public int compareTo(State s) {
        return Integer.compare(this.f, s.getF());
    }
}
