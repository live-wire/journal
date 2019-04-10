# **CS4210 Algorithms for Intelligent decision making** - notes :robot:
_TU Delft_

These notes are best viewed with MathJax [extension](https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima) in chrome.

---
### Exam time:
##### Part 1: Constraint Programming
- Explain what is a constraint program, and the concepts of constraint programming.
- List some common applications of CP, i.e., show awareness of when CP is an appropriate technology in terms of modelling and/or solving.
- Write a CP model for an optimisation problem (in MiniZinc), including demonstrating com-petency is good modelling practice. See the lecture slides for details.
- Solve a CP model written in MiniZinc (using a solver) – this is assessed primarily through the assignment, but the exam might ask you to solve a small model by hand.
- Summarise the main ideas in CP solving. See the lecture slides for details.2


---
### Part III: Reinforcement Learning and Planning
- Markov Decision Process - Mathematical model for modelling decision making in situations where outcomes are partly random and partly under control of a decision maker. Useful for studying optimization problems.
- Partially observable Markov Decision Process


---
### Part II: Game Theory and Mechanism Design
- Prisoner's dilema
    - *Pareto* - Improve for one player without harming any other player. (Minimum one pareto optimal solution exists)
    - *Nash* - Not even a single player has an incentive to move! (**Given you have other player's strategy!**)
    - *Dominant Strategy* - (**Independent of what the others do**, this is the best course of action)
- Given a strategy profile of all other players, a pure best response of a player is his pure strategy profile.
    - *Mixed strategy profile* assigns a probability to each strategy.
- Following this [Advanced Game Theory Course](https://www.coursera.org/learn/game-theory-2/) on coursera.
###### Lecture 5:
- **Mixed strategy nash equilibrium**: Pure Nash doesn't exist! What is nash ? see point below!
    - first equate prob of going left and right to obtain the value of $\sigma_u$ (going up/down)
    - Then do the same for going up and down to obtain $\sigma_l$ (going left and right)
    - Strategy with probabilities!
    - In every game there is one Nash! Pure or Mixed! :bomb:
- **Pareto efficienct**: Going up or down or left or right from that state is not possible without harming atleast one player!
- **Nash**: Moving from that position gives loss to BOTH PLAYERS. `Fix one player and see the other`, You know what the other player is doing!
- **First price sealed bid auction**- Highest bidder has to pay the amount!
    - Proof: Bayes Nash equilibrium is when all bid v1/2, v2/2 as:
        - S1 >= V2/2
        - Integrate (0, 2S1) [v1 - s1] dv2
        - 2s1v1 - 2s1^2
        - Maximise w.r.t. s1 so, 2v1 - 4s1 = 0 
        - s1 = v1/2
    - For N bidders, Sn = ((N-1)/N) * Vn 
- **Second price sealed bid auction**- Highest bidder has to pay the second highest bid amount! 
    - Proof: No use trying to be dishonest!
- **Mixed auction**: Set a hard and a soft floor!

###### Lecture 6:
- Find rule G such that it Maximizes the social sum of utilities = equilibrium.
- No use lying to G bitches. Direct stategy-proof mechanism implementing social choice function. When g == f => equilibrium. DSIC = Direct strategy-proof incentive compatible mechanism.
- There is also mechanism design with transfers!
    - For each row (theta) (agent), add utilities of other agents (transfers) to it! to calculate that agent's total utility!
    - Quasi Linear representation
- Groves mechanism: If there is Quasi linear preferences IS `strategy proof and efficient`.
    - Common sense show agents the global picture! 
- **VCG**: Budget balance:
    - Transfers are not just added!, The ex: t1 = v2 - maxUtil(v2)
    - Therefore, t1 can be negative.
    - Bass!
    - What is subtracted, is a part of the _socially maximizing outcome_!
- VCG is no good for min makespan!


###### Lecture 7: Voting schemes
- Our setting:
    - **Outcomes** / Alternatives
    - Agents have **preferences**
    - Goal: **Social choice function** - Mapping from preference profiles to outcomes
- Voting Schemes:
    - *Plurality*: Pick outcome that is most prefferred by most people 
    - *Cumulative Voting*: Distribute 5 votes each (Possible to vote same candidate multiple times)
    - *Approval Voting*: Vote as many candidates as you like (Stackoverflow rating)
    - *Plurality with elimination*: Outcome with fewest votes is eliminated (Repeat till there is a majority/winner) (French Elections)
    - *Borda count*: If there are n candidates, each agent will give n-1 votes to top preference,n-2 to second .. and 0 to the last one. (Sports Evaluations)
    - *Successive/Pairwise Elimination*: Pairwise elimination. (Like Hackjunction Budapest) (Sensitive to order in which you run it)
    - *Copeland/Schulze*: Pairwise majority between 2 candidates and then assign +1 to winner and -1 to loser. Do this for all pairs. (Always selects the condorcet winner).
- **Condorcet Consistency**:
    - Means there is a clear winner (a candidate always wins in a pairwise majority(50% votes)).
    - NOTE: There is not always a condorcet winner!
- Every voting scheme can run into problems! Paradoxical outcomes:
- Pareto efficiency- 
    - Social welfare also should have b at last.
    - If b is always top or always last, social welfare function also should have b at top or last.
- Monotonicity: If outcome was the winner, It must stay the winner if support for it in a preference profile is increased.
- Arrow's impossibility theorem: If preferences are considered, the election will always fail one of the following properties:
    - Non- Dictatorship
    - Unanimity
    - Independence of irrelevant alternatives
- None of the positional scoring rules are condorcet consistent:
    - Borda, Plurality(with/without elimination)
- Kendal Tau: Concordant pairs = C, Discordant pairs = D , Formula = (C-D)/(C+D) (range -1 to 1)

###### Lecture 8: Deferred Acceptance
`Two sided matching` - student prefs vs teacher prefs!
- Stable and strategy-proof for one side. No algo is stable and strategy proof (students can't mess with the system) for both sides.
- Don't accept anyone! Just keep rejecting and when there are no rejections, stop!
    - Student doesn't go anywhere else till he gets rejected!

`One sided matching`
- Simple Serial assignment! Strategy proof and Pareto efficient! (But how do you decide who gets a vote first)
- Top Trading Cycle Algorithm: 
    - Create a graph! And give em what they want whenever there is a cycle!


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

