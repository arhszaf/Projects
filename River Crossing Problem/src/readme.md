# A* Search Algorithm for River Crossing Problem 
 
This project implements the A* search algorithm to solve the river crossing problem. The program 
allows users to input the number of people,the total allowed crossing time, and the individual 
crossing times for each person. It then finds the optimal sequence of crossings to get everyone 
across the river within the specified time using the A* algorithm.

## Overview
The river crossing problem involves getting a group of people across a river with a limited amount of time and a boat that can carry a limited number of people at a time. 
Each person has a different crossing time, and the goal is to find the optimal sequence of crossings to minimize the total time taken.
## Features
* Implements the A* search algorithm to solve the river crossing problem.
* Allows user input for the number of people, total allowed crossing time, and individual crossing times.
* Validates user input for positive integers and unique crossing times.
* Provides the optimal solution if it exists within the specified time.

## Usage
1. When prompted, enter the number of people.
2. Enter the total allowed crossing time in minutes.
3. Enter the individual crossing times for each person. Each time must be a positive integer and unique.
4. The program will output the optimal sequence of crossings if a solution exists within the specified time.
## Implementation Details
### **Main Class**
The `Main` class handles user input and initializes the AstarSearch object. It also prints the solution path if found.

### **AstarSearch Class**
The `AstarSearch` class implements the A* search algorithm. It maintains a priority queue for the frontier and a closed set for explored states.

### **State Class**
The `State` class represents a state in the river crossing problem. It includes methods for generating child states, evaluating the heuristic, and printing move details.

### **Heuristics**
* **Heuristic 1** : Estimates the cost to reach the goal by finding the maximum time on the right side.
