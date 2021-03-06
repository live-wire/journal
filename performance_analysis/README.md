# IN4341 Performance Analysis of Complex Networks :space_invader:

These notes are best viewed with MathJax [extension](https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima) in chrome.

> "Only what you do today, determines your tomorrow" - Me (inspired by Markov Chains :robot:)


---
`Lecture 3`
#### Markov Discrete :pager:
- Markov Process is called a Markov chain if its state space is discrete.
- Markov process can be entirely defined by the **transition probabilities**:
- $P[X(t_{n+1})=x_{n+1}|X(t_0)=x_0, X(t_1)=x_1,.. X(t_n)=x_n] = P[X(t_{n+1})=x_{n+1}|X(t_n)=x_n]$
- For the transition probability matrix it holds that: $\sum_{j=1}^N P_{ij} = 1$
    - For each row, sum of transition probabilities = 1
    - vector $Pu = u$ (u here is 1s column vector)
    - Duh! Because it is 100% certain that the state makes a transition with each timestep. Be it back to itself!
- Even after n time steps, the markov chain must be in one of the N states, $\sum_{j=1}^N (P^n)_{ij} = 1$
    - Iteratively multiply $Pu = u$ with P on both sides to get $P^nu = u$
- If every state is reachable from every other state, the Markov chain is called **irreducible**.
- A = Adjacency matrix. Between i,k: Total k hop walks = $(A^k)_{ij}$
- **Equivalence Class** (set of states that can communicate (back and forth reachable)). If the equivalence relations result in a single class, the markov chain is irreducible. If there is a no-way-back-link from one equivalence class to another, the chain is *reducible*.
- An irreducible chain is **periodic** $(P^n)_{jj} > 0$ if period (d > 1) else it is *aperiodic* with period (d = 1) which means a self loop.
- Infinite iterations for Markovs: **recurrent** state if it will ever occur again ? For a markov initiated at i, will it come to j ?
    - $r_{ij} = P[T_j < \infty | X_0 = i]$ if $r_{ij} = 1$ it is recurrent and **transient** if P < 1.
    - A finite Markov chain must have **at least one reccurent state**.
    - If i is reachable from j (back path available) and i is recurrent, then j is also recurrent! Duh!
- **Steady State of Markov Chains** - 
    - For **aperiodic and irreducible chains**, the steady state is _unique_!
        - It is a vector like a row i P. $\pi =lim_{k \to \infty} s[k]$
        - $\pi = \pi P$ Therefore $\pi$ is a row vector where each element of pi is: $\pi_j = \sum_{k=1}^N P_{kj} \pi_k$
        - Two ways of computing $\pi$ for irreducible chains:
            - $lim_{k \to \infty} (P^k)ij = \pi_j$
            - Solve matrix $P^T\pi^T = [0,0,0...1]^T$ where the last row of $P^T$ is all ones and R.H.S is a column vector with the last row as 1!
        - There has to be one $\pi_j != 0$ otherwise there is no stationary probability distribution.
    - Steady state is independent of where the chain began!
    - **Only recurrent states** (not transients) have a non-zero steady state probability $\pi_j$
    - Steady state _also occurs for periodic chains_: $\pi = [\frac{p}{p+q}, \frac{q}{p+q}]$ where p = q = 1 will be [1/2, 1/2]
- `PROBLEMS TAKEAWAY`: 
    - If there is an absorbing state, steady state will usually be 1 for that and 0 for the other transient states.
    - For a 2 state markov, steady-state: $\pi = [\frac{p}{p+q}, \frac{q}{p+q}]$ where p and q are probabilities of moving to the other state!

---
`Lecture 2`
#### Poisson :snake:
- MIT Opencourseware video [link](https://www.youtube.com/watch?v=jsqSScywvMc&index=53&t=16s&list=PLUl4u3cNGP60A3XMwZ5sep719_nh95qOe)
- Beautiful [video](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/poisson-distribution/v/poisson-process-1) by Khan Academy.
- Derived Poisson.
- Poisson is just Bernoulli trials with n (number of trials) (Granularity) approaching infinity.
    - Success = Arrival
    - Memoryless
    - Independent from what happened before
- Assumptions:
    - Intervals of same length behave probabilistically identically.
- **Interarrival times**
    - Prob that arrival occurs between t and $t+\delta$ = k-1 arrivals from [0,t] and 1 arrival from [$t,t+\delta$]
        - $f_{Y_k}(t)\delta = \frac{\lambda^{k-1} e^{-\lambda y}}{(k-1)!} . \lambda \delta$
        - Kth arrival = $f_{Y_k} = \frac{\lambda^{k} e^{-\lambda y}}{(k-1)!}$
        - Time of first arrival = $\lambda . e^{\lambda.y}$ (Exponential distribution)

#### Piet Van Mieghem's poisson
- Poisson ~ counting process.
- Probability of Number of occurences (k) in a small time interval (h). $P[X(h + s) - X(s) = k]$ = $\frac{(\lambda h)^k e^{-\lambda.h}}{k !}$
- These interval probabilities are conditionally independent!
- Expected Number of occurences in interval (h) = $\lambda. h$
- Sum of two poisson processes is also a poisson with $\lambda s$ added.
- **Uniform distribution**: If an event of a Poisson Process occurs during interval [0, t], the time of occurance of this event is uniformly distributed over [0, t].
- `PROBLEMS TAKEAWAY`: 
    - Don't forget the law of total probabilities (Summation) for calculating PDFs with a **variable** input etc. There can be questions with a mixture of Poisson (Probability that Count of something is k) and Bernoulli (Probability that number of successes is k) distributions.
    - From a combination of poissons, **Given an arrival:** prob that arrival was from p1 = $\frac{\lambda_1}{\lambda_1 + \lambda_2}$ 
        - This will be just a product if **not given** an arrival. (Bayes! Duh!)
        - **A joint flow can also be decomposed** with rates $\lambda.p$ and $\lambda.(1-p)$ (Instead of mixing it up with a binomial).
    - Maximising winning probability = equate first derivative to 0 and solve!
    - For calculating the prob from x to infinity, don't forget you can always compute it as [ 1 - P(0 to x)].

---
`Lecture 1`
#### Self Evaluation Prep
- Binomial P[X=number of successes k] = $C^N_k p^k (1-p)^{N-k}$
- Geometric Distribution: Distribution of number of trials needed to get _the first success_ in independent bernouli trials. First success k = $(1-p)^{k-1}.p$
- Poisson Distribution: (mean = $\lambda$ Number other than mean k) $\frac{\lambda^k e^{-\lambda}}{k!}$
- Gaussian Distribution: $P[X=x] = \frac{e^{-\frac{(x-\mu)^2}{2.\sigma^2}}}{\sqrt{2.\pi.\sigma^2}}$

#### Random Variables and Distributions
- $Pr[A]=\frac{n_A}{n}$ where $n_A$ = favourable outcomes and n = all possible outcomes
- For mutually exclusive events: $Pr[A \cup B] = Pr[A] + Pr[B]$
- Mutually exclusive (means $A \cap B = \phi$) events: $Pr[A \cup B] = Pr[A] + Pr[B] - Pr[A \cap B]$
- **Independent events:** $Pr[A \cap B] = Pr[A]Pr[B]$
- **Conditional:** $Pr[A \cap B] = Pr[A|B]Pr[B]$ therefore for independent events, $Pr[A|B] = Pr[A]$ Duh!
- **Law of total probability**: $Pr[A] = \sum_i Pr[A|B_i]Pr[B_i]$ where $B_i \cap B_j = \phi$

#### Discrete Random Variables
- Countable outcomes (set)
- Discrete probability space. PDF = $Pr[X = x]$
- $\Omega = {x_1, x_2, ...}$, $P[\Omega] = 1$
- Expectation of a discrete random variable $E[X] = \sum_x xPr[X=x] = \mu$
- Moment of a random variable:
$E[X^n] = \sum_x x^n Pr[X=x]$
- Variance of X is $Var[X] = E[(X - E[x])^2] = E[X^2] - \mu^2$
- Probability generating function: $E[g(x)] \sum_x g(x) Pr[X=x]$ therefore: $\phi(z) = E[z^x]$
- Derivative of **pgf** = $\frac{d}{dx}E[z^x]$, now since Expectation is linear, derivative can move inside. = $E[xz^{x-1}]$, for z= 1 = E[x]
