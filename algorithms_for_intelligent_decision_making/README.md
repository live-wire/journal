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
    - Parameters
    - Variables
    - Constraints
    - Solver

