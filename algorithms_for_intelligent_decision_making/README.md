# **CS4210 Algorithms for Intelligent decision making** - notes :robot:
_TU Delft_

These notes are best viewed with MathJax [extension](https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima) in chrome.

---
### Exam time:
##### Part 1: Constraint Programming
- Explainwhat is a constraint program, and the concepts of constraint programming.
- Listsome common applications of CP, i.e., show awareness of when CP is an appropriatetechnology in terms of modelling and/or solving.
- Writea CP model for an optimisation problem (in MiniZinc), including demonstrating com-petency is good modelling practice. See the lecture slides for details.
- Solvea CP model written in MiniZinc (using a solver) – this is assessed primarily throughthe assignment, but the exam might ask you to solve a small model by hand.
- Summarisethe main ideas in CP solving. See the lecture slides for details.2

##### Part 2: Game Theory and Mechanism Designorganized per lecture:
###### Lecture 5:
- Games/auctions: English auction, Prisoner’s dilemma, matching pennies, battle of the sexes,sealed-bid (first/second-price Vickrey) auction
- Definitions: solution concepts, normal / strategic form game, strategy profile, utility func-tion, preferences, Pareto efficiency, dominant strategy, dominant-strategy equilibrium (DSE),(pure) best response, (pure) Nash equilibrium, best-response correspondence, Bayes-Nashequilibrium (BNE)
- Proofs: Second-price DSE, 2 player first-price BNE
- Results: Nash’s theorem, revenue-equivalence

###### Lecture 6:
- Games/auctions: combinatorial auction, auction, Groves mechanism, with Clarke pivot rule,VCG, critical-value payment (CVP), AGV / d’AGVA, median rule, knapsack auction withCVP, min-makespan scheduling with "payment identity"
- Definitions: social welfare, incentive compatible, strategy-proof, implementing, direct mech-anism, quasi-linear utility function, transfer/payment, budget balance, individual rationality,single-parameter domain, monotonicity, critical value payment/price, single-peaked prefer-ences
- Proofs: Revelation principle. Groves mechanisms are strategy-proof and efficient.
- Results: Gibbard-Satterthwaite. Groves are the only mechanisms (Green-Lafont) / taxationprinciple. Bilateral trading (HGL). Myerson-Satterthwaite.

###### Lecture 7:
- Voting Rules: majority, plurality, plurality-with-elimination, single transferable votes, Borda,Copeland, Schulze, Kemeny
- Definitions:  social choice/welfare function, Condorcet winner, weighted majority graph,Condorcet-consistent, Non-dictatorship, Unrestricted Domain, Independence of IrrelevantAlternatives, Anonymity, Neutrality, Weak monotonicity, Non-constancy, Consistency, Con-tinuity, positional-scoring rule, Kendall-tau distance, Kemeny score, Condorcet criterion,Mallows model
- Proofs: Copeland always selects the Condorcet winner. The Schulze rule is Condorcet con-sistent. No positional-scoring rule is Condorcet-consistent.
- Results: Arrow’s theorem. Identification of positional-scoring rules.

###### Lecture 8:
- Mechanisms: deferred acceptance (DA), serial dictatorship (SD), top trading cycle algorithm(TTC), max-cardinality matching with 2-cycles,
- Definitions:  Two-sided One-to-one Matching, blocking pair, stable matching, achievable,student-optimal, House allocation (HA), assignment problem, housing market, the core,kidney-paired donation,
- Proofs: DA is stable.  DA is strategy-proof for one side.  No mechanism is both stableand strategy-proof for both sides. SD is strategy-proof and Pareto efficient for HA. max-cardinality matching with priority-based tie-breaking is strategy-proof (and consistent).
- Results: Existence of stable matching. Properties of TTC.3

##### Part 3: Reinforcement Learning
- Explainwhat Markov decision processes are, including partially observable, multi-objectiveand constrained versions.
- Constructsuch models for simple applications.
- Explainand apply basic solution algorithms.
- Summarisethe key concepts behind these algorithms.

---
### Part III: Reinforcement Learning and Planning
- Markov Decision Process
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

##### Voting schemes
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
- **Condorcet Consistency**:
    - Means there is a clear winner (a candidate always wins in a pairwise majority(50% votes)).
    - NOTE: There is not always a condorcet winner!
- Every voting scheme can run into problems! Paradoxical outcomes:
- Pareto efficiency- 
    - Social welfare also should have b at last.
    - If b is always top or always last, social welfare function also should have b at top or last.
- Monotonicity: If outcome was the winner, It must stay the winner if support for it in a preference profile is increased.

##### Mechanism Design
- Can we design a game that yields a particular outcome.


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

