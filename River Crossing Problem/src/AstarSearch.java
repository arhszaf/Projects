import java.util.*;

/**
 * A class implementing the A* search algorithm.
 */
class AstarSearch {
    private PriorityQueue<State> frontier; // Priority queue for managing states based on their evaluation function
    private HashSet<State> closedSet; // Set to keep track of states that have been explored

    /**
     * Constructor to initialize the frontier and closedSet.
     */
    AstarSearch() {
        this.frontier = new PriorityQueue<>();
        this.closedSet = new HashSet<>();
    }

    /**
     * A* search algorithm to find the best path to the goal state.
     *
     * @param initialState The initial state of the search.
     * @param heuristic    The heuristic function to estimate the cost to reach the
     *                     goal.
     * @return The list of states representing the best path to the goal.
     */
    ArrayList<State> BestFSClosedSet(State initialState, int heuristic) {

        // Step 0: Check if the initial state is already the goal state
        if (initialState.isFinal()) {
            ArrayList<State> path = new ArrayList<>();
            path.add(initialState);
            return path;
        }

        // Step 1: Put the initial state in the frontier.
        this.frontier.add(initialState);

        // Step 2: Check for an empty frontier.
        while (!this.frontier.isEmpty()) {

            // Step 3: Get the first state out of the frontier.
            State currentState = this.frontier.poll();

            // Step 4: If the current state is the goal state, reconstruct and return the
            // path.
            if (currentState.isFinal()) {
                return reconstructPath(currentState);
            }
            if(initialState.getTotalTime()<=currentState.getG()){
                System.out.print("There is no possible solution!");
                return null;
            }

            // Step 5: If the current state is not in the closed set, explore it.
            if (!this.closedSet.contains(currentState)) {
                this.closedSet.add(currentState);

                // Get and evaluate children of the current state
                List<State> children = currentState.getChildren(heuristic);
                for (State child : children) {
                    // If the child state is not in the closed set or frontier, add it to the
                    // frontier
                    if (!this.closedSet.contains(child) && !this.frontier.contains(child)) {
                        child.evaluate(heuristic);
                        child.setFather(currentState); // Set the parent to reconstruct the path later
                        this.frontier.add(child);
                    }
                }
            }
        }
        return null; // If the goal state is not found, return null
    }

    /**
     * Reconstruct the path from the initial state to the given state.
     *
     * @param state The goal state.
     * @return The list of states representing the path.
     */
    private ArrayList<State> reconstructPath(State state) {
        ArrayList<State> path = new ArrayList<>();
        State current = state;
        while (current != null) {
            path.add(current);
            current = current.getFather();
        }
        Collections.reverse(path); // Reverse the list to get the correct order
        return path;
    }
}
