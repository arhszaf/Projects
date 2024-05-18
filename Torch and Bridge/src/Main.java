import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

/**
 * Main class to execute the river-crossing problem using the A* search
 * algorithm.
 */
public class Main {
    static long maxAllowedTime = 60000; // maxAllowedTime = 60 seconds
    static int totalTime;
    static int N;
    static List<Integer> individualTimes = new ArrayList<>();

    public static void main(String[] args) {

        Scanner scanner = new Scanner(System.in);

        // Input validation for the number of people (N)
        do {
            System.out.print("Enter the number of people: ");
            if (scanner.hasNextInt()) {
                N = scanner.nextInt();
                if (N <= 0) {
                    System.out.println("Error: Please enter a positive number for the number of people.");

                }
            } else {
                System.out.println("Error: Please enter a valid integer for the number of people.");
                scanner.next(); // consume invalid input
            }
        } while (N <= 0);

        // Input validation for the total time
        do {
            System.out.print("Enter the total time (in minutes): ");
            if (scanner.hasNextInt()) {
                totalTime = scanner.nextInt();
                if (totalTime <= 0) {
                    System.out.println("Error: Please enter a positive number for the total time.");
                }
            } else {
                System.out.println("Error: Please enter a valid integer for the total time.");
                scanner.next(); // consume invalid input
            }
        } while (totalTime <= 0);

        System.out.println("Enter the time required for each person to cross the river (in minutes):");

        // Input for individual times for each person
        for (int i = 0; i < N; i++) {
            boolean inputValid = true;
            while (inputValid) {
                System.out.print("Person " + (i + 1) + ": ");
                if (scanner.hasNextInt()) {
                    int time = scanner.nextInt();
                    if (time > 0) {
                        if (i == 0 || !individualTimes.contains(time)) {
                            individualTimes.add(time);
                            inputValid = false;
                        } else {
                            System.out.println("Error: The time already exists for another person.");
                        }
                    } else {
                        System.out.println("Error: Please enter a positive number for the individual time.");
                    }
                } else {
                    System.out.println("Error: Please enter a valid integer for the individual time.");
                    scanner.next(); // consume invalid input
                }
            }
        }

        scanner.close();

        // Create the initial state based on user input
        Collections.sort(individualTimes);
        State initialState = new State(N, totalTime, individualTimes);

        // Perform A* search
        AstarSearch searcher = new AstarSearch();
        long start = System.currentTimeMillis();

        // Choose heuristic (in this case, heuristic value is hardcoded as 1)
        ArrayList<State> path = searcher.BestFSClosedSet(initialState, 1);

        long end = System.currentTimeMillis();
        long total = end - start;

        // Print the result
        if (path == null || path.isEmpty())
            System.out.println("Could not find a solution.");
        else {
            path.get(0).printMoveDetails(null); // Print the first state without previousState
            for (int i = 1; i < path.size(); i++) {
                path.get(i).printMoveDetails(path.get(i - 1)); // Print the remaining states

            }
        }

        System.out.println();
        if (total > maxAllowedTime) {
            System.out.println("The program exceeded the maximum allowed time.");
        }
        System.out.println("Search time: " + (double) (total) / 1000 + " seconds."); // total time of searching in
        // seconds
    }
}
