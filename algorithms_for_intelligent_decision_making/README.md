# **CS4210 Algorithms for Intelligent decision making** - notes :robot:
_TU Delft_

These notes are best viewed with MathJax [extension](https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima) in chrome.


---
### Part II: Game Theory and Mechanism Design
- Prisoner's dilema
    - *Pareto* - Improve for one player without harming any other player. (Minimum one pareto optimal solution exists)
    - *Nash* - Not even a single player has an incentive to move! (**Given you have other player's strategy!**)
    - *Dominant Strategy* - (**Independent of what the others do**, this is the best course of action)
- Given a strategy profile of all other players, a pure best response of a player is his pure strategy profile.
    - *Mixed strategy profile* assigns a probability to each strategy.
- First Price Auction:
    - 

---
### Part I: Constraint Programming
- In a constraint problem, values have to be found for unknowns (*variables*) making sure that they satisfy all the given **constraints**. 
    - Optionally a given objective function on the variables has an optimal value (*min cost* or *max profit*).
- **Search space** = All possible solutions
    - Solution to a satisfaction problem = **feasible**
    - Optimal solution to an optimization problem = **feasible and optimal**
- `MiniZinc`:
    - Parameters - (Constants that don't change) `int: budget = 1000;`
    - Variables - (Variables that you want to compute) `var 0..1000: a;`
    - Constraints - (Constraints on variables) `constraint a + b < budget;`
    - Solver - (Solve the optimization problem) `solve maximize 3*a + 2*b;` or `solve satisfy`
- More MiniZinc Tricks:
    - `enum COLOR = {RED, GREEN, BLUE}` then vars can be of type `COLOR` like `var COLOR: a;`
    - To get all solutions run with flag `minizinc --all-solutions test.mzn`
    - Parameters can be given a value later on using a file with extension `.dzn` like `minizinc test.mzn values.dzn` (The model MUST be provided with the values of the parameters from a single / multiple dzn files)
    - Array declarations for parameters:
        - `array[range/enum] of int: arr1`
        - 2D `array[enum1, enum2] of float: consumption`
        - Generator operations on arrays:
            - `constraint forall(i in enum)(arr1[i] >= 0)`
            - `solve maximize sum(i in enum)(arr1[i]*2) < 10`
    - Array Comprehensions: `[expression | generator, generator2 where condition]`
        - example: `[i+j | i,j in 1..4 where i < j]`
    - TLDR:
        - **enums to name objects**
        - **arrays to capture information about objects**
        - **comprehensions for building constraints**
    - `If else` can be used inside forall blocks and outside the nested forall blocks.
    - `if condition then <blah> else <blah2> endif`

