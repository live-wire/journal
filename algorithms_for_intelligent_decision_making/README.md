# **CS4210 Algorithms for Intelligent decision making** - notes :robot:
_TU Delft_

These notes are best viewed with MathJax [extension](https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima) in chrome.

---
### Part III: Reinforcement Learning and Planning
`MDP, POMDP, MOMDP`
- Markov Decision Process - Mathematical model for modelling decision making in situations where outcomes are partly random and partly under control of a decision maker. Useful for studying optimization problems.
- Partially observable Markov Decision Process
- MOMDP is difficult because number of joint actions is exponential in number of agents
- Value iteration requires a model of the environment.
- Point based POMDP = scalable!
- Q learning learns from experience, no information about model is required. Greedy exploration!
- POMDP value function is piecewise linear and convex!
###### MoMDP
- Multi Objective: 
    - Motivation: when the rward is a vector of rewards! Every policy has a value and the value is a scalar(reward)! But in MoMDP, every value is a vector of rewards.
        - $R = SxAxS = R^n$
        - Can't we just scalarize it ?
        - It is not always feasible/ desirable/ impossible to a priori serialize it.
    - Scalarization function projects multi objective value to a scalar.
        - It's a weighted sum of Values! (usually between 0 and 1)
        - $\sum V_i^{\pi} w_i$
    - Unknown weight scenario - Weights revealed only in the execution phase!
    - Decision support scenario - User selection scenario! Which of the following outcomes you prefer!
- Stationary policy: Maps states to probability of taking an action
- Deterministic stationary policy: maps states directly to actions.

- Instead of $V^{\pi}(s) = E[R_t | \pi, s_t = s]$, it is now $V^{\pi} = E[\sum_{k=0}^{\infty} \gamma^k r_{k+1} | \pi, s_t = s]$.
    - Partial ordering (comparing vectors will be weird), optimality not clear!
    - Solution: Problem Taxonomy
- Problem Taxonomy - Utility based approach:
    - While execution, one policy will obviously followed! (That follows some weight!)
    - Deduce solution set from 3 factors:
        - Multi objective - (known weights(single policy) / Unknown weights(multiple policies))
        - Properties of scalarization function - (Linear / Monotonically increasing) (We can know something about it)
        - Allowable policies - (Deterministic/Stochastic)
    - Solution space is a grid which considers all combinations of the above three factors!
- Solved by using convex hull value iteration instead of regular value iteration!
    - On Each iteration, the value sets are merged! They are then merged with the R and discounted previous state values! (separately for each action) and then merged together! 
        - CPrune is used to get rid of useless vectors in this merged set!
- Another approach = Optimistic Linear support! Move a horizontal bar up the inverted triangle solution that you've found! :confused:
- Cool application = Epidemic control! - MOPOMDP!
![Problem Taxonomy](https://live-wire.github.io/images/taxonomy.png)

- `Constrained MDP`
- Taxonomy Depends on 2 factors:
    - Offline(communication)/ Online(no communication)
    - Hard constraint/ Soft constraint
- Possible grid of solution spaces = 
    - offline: Hard(Deterministic preallocation), Soft(Stochastic preallocation)
    - online: Hard(Coordination online/Online reallocation), Soft(Coordination online/Online reallocation)
- With resource constraint L:
    - $K_i$ = all policies, $C_i(s,a)$ = resource consumption, $R_i(s,a)$ = reward function, $x_{i,\pi_j}$ = probability of following a polciy __for agent i__.
    - maximize $\sum_{agent-i} \sum_{\pi \in K_i} E[R_{\pi_i}].x_{j,\pi_j}$
    - s.t. 
        - $\sum_{agent-i} \sum_{\pi \in K_i} E[C_{\pi_j}].x_{i,\pi_j} <= L$ (constraint)
        - $\sum_{\pi_j \in K_i} x_{i,\pi_j} = 1$ (sum of probs = 1)
        - $x_{i,\pi_j} >= 0$ (Probabilities >= 0 duh)


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

