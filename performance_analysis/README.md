# IN4341 Performance Analysis :space_invader:

---
`Lecture 1`
#### Random Variables and Distributions
- $Pr[A]=\frac{n_A}{n}$ where $n_A$ = favourable outcomes and n = all possible outcomes
- For mutually exclusive events, $Pr[A \union B] = Pr[A] + Pr[B]$
- Mutually exclusive (means $A \cap B = \o$) events: $Pr[A \cup B] = Pr[A] + Pr[B] - Pr[A \cap B]$
- **Independent events:** $Pr[A \cap B] = Pr[A]Pr[B]$
- **Conditional:** $Pr[A \cap B] = Pr[A|B]Pr[B]$ therefore for independent events, $Pr[A|B] = Pr[A]$ Duh!
- **Law of total probability**: $Pr[A] = \sum_i Pr[A|B_i]Pr[B_i]$ where $B_i \cap B_j = \o$

#### Discrete Random Variables
- Countable outcomes (set)
- Discrete probability space. PDF = $Pr[X = x]$
- $\ohm = {x_1, x_2, ...}$, $P[\ohm] = 1$
- Expectation of a discrete random variable $E[X] = \sum_x xPr[X=x] = \mu$
- Moment of a random variable:
$E[X^n] = \sum_x x^n Pr[X=x]$
- Variance of X is $Var[X] = E[(X - E[x])^2] = E[X^2] - \mu^2$
- Probability generating function: $E[g(x)] \sum_x g(x) Pr[X=x] $ therefore: $\phi(z) = E[z^x]$

