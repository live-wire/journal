# **EE4C06 Networking** - notes :scroll:
_TU Delft_

These notes are best viewed with MathJax [extension](https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima) in chrome.

> Average hop-count of the planet <= 4 :cyclone:

> Any eigenvector is orthogonal to any other one! (dot product = 0)

> Calculating Mean and Variance of a distribution is easy! But calculating a tail probability is difficult!

`NOTE: Apparently these notes don't contain information about Reiman zeta function (Power law graphs) and the best Spreader.`

---
`Exam time`
- Spectral radius = Largest Eigen Value
	- Eigen values are bounded by $-d_{max}, d_{max}$ (non inclusive)
	- Largest Eigen value is bounded by $d_{average}, d_{max}$
	- $d_{average} = 2L/N \sqrt{1 + \frac{Var(d)}{E[D]^2}}$
- Shortest paths passing through the link: Expected Betweenness = E[B] = maxLinks/L * Expected Hopcount = $\frac{(N(N-1)/2)}{L}E[H]$ Average hop count is in the scale O(log(N)) for small world graphs.
- When calculating R (robustness value),$R = \sum_k s_km_k$ all the metrics(m) being considered are orthogonal to each other.
- Number of k-hop walks = All elements of $A^k$ which is $u^TA^ku$. Closed walks = Diagonals of this array
- Sum of graphs in Erdos Renyi will be Union so p1 + p2 - p1p2
- Sum of degrees will always be <= 2L
- Multiplicity of eigenvalues = some value occurs again.
	- Eigen values of a complete graph are: 0 and N(same for all N-1 nodes)
- Odd power of A (Adjacency) matrix for a bipartite graph, the diagonal elements are 0.
- Zero or 2 odd degree nodes in a graph! 
- Solve electrical circuits without matrices!
- If there are x nodes with degree y, there are nodes with degree 1 >= $x(y-2) + 2$
- Effective graph resistance = sum of all elements of $\omega$/2 = $N. trace(Q^{-1})$
- For a star graph, effective graph resistance = $(n-1)\sum_i^n r_{ni}$  where n is the hub (sum of resistances connected to the hub * (n-1))
- For a fully connected circuit, add a node in the center and make it a star!
- For any graph
	- Atleast two nodes with same degree
	- Even odd degree nodes
- Eigen stuff:
	- A and $A^T$ have same eigen values but not necessarily the same eigen vectors
	- Eigen vectors are linearly independent
	- If $P^{-1}$ exists then $P^{-1}AP$ has the same eigen values as A but the eigen vector is Px
- Of Adjacency matrix
	- If main eigen vector has elements > 0 and eigen value > 0, Graph is connected!
    - sum of eigen values = 0, sum of squared = sum of degrees = 2L, sum of $\lambda^k$ = Trace of $A^k$
    - Triangles = 1/N*$\sum \lambda^3$
- Of Laplacian
	- Partition links to separate graphs = $1/4 \sum_j^N \alpha^2_j \mu_j$ (where $\alpha = y^Tx$) (where y = +1/-1 based on graph and x = eigenvector of Q, $\mu$ = eigen values of Q)
	- Minimum number of links to separate = second smalles eigen value: $1/4\alpha^2_{N-1}\mu_{N-1}$
- Erdos Renyi
	- link density, clustering coefficient = p.
	- $\mu=(N-1)p$ (Maximum degree * p), $\sigma^2=(N-1)p(1-p)$
	- critical density(p) = logN/N
- Electrical:
	- $By=f$, $y = B^Tv$ y=current in link(Lx1), B = incidence matrix, v = voltage vector(Nx1)
	- $f=Qv$, $Q^{-1}f = v - v_{average}u$ where $v_{average} = u^Tv/N$
	- Effective resistance:
		- $\omega = zu^T + uz^T - 2Q^{-1}$ (z is diagonal elements of $Q^{_1}$)
		- Effective graph resistance = sum of omega elements /2 = $(u^T \omega u)/2$ = $N.Trace(Q^{-1})$ = $N.\sum \frac{1}{\mu}$
	- Injected nodal current = f (x some places). By = f or Qv = f
- Turan Graph: $L = \frac{(r-1)}{2r}N^2$ r = number of partitions

---
`Lecture 8`
#### Scheduling
- Multiclass queueing system:
	- $N_k(t)$ - Number of class k packets at time t
	- $\mu_k(t)$ - Service rate of class k packets at time t
	- For stability, FlowRate / ServiceRate $\frac{\lambda}{\rho}$ should always be <= 1.
- Priority Systems:
	- Head of the Line - _Packets always sorted_ (Logical implementation, different queues for each class) (Bad for low priority!) 
	- Partial Buffer Sharing - _Packets above a Threshold are not sorted_
	- Push out buffer - Pushes out a low priority buffer if buffer is full (FiFO or LiFO or Random Out)
- Token Bucket: [] Burstiness constraint = $L(u,t) = \lambda(t-u) + \sigma$
	- Each packet needs a token to enter the network
	- Bucket rate = $\lambda$ = token generation rate
	- Bucket depth = $\sigma$ = Burst!
	- Packet is discarded if token bucket is empty (Packets arriving too fast!)
- Connection Admission Control: (Regulated flow w.r.t. loss of packets) (Non-realtime!)
	- Check if Sum of all input burstinesses $\sum_i \sigma_i $ is <= G (Buffer positions) Guaranteed Loss-less multiplexing!
- Regulated flow w.r.t. delay (Real-time)
	- service rate = $\mu$, Packet arrival rate = $\lambda$ , $\kappa = \frac{\mu}{\sum_i \lambda_i}$ (which is > 1 for stability)
	- Again Connection Admission Control: Maximum Delay < $\frac{\sigma_k}{\lambda_k}$
- Wow! Just check $\sigma$ and $\lambda$ You can guarantee loss-lessness and maximum delay!
- No free lunch: `work hard to get rich`
	- Loss-less multiplexing vs Statistical Multiplexing
	- $N_u = G$ and $N_s = \frac{2}{ln(clr)}G^2$
	- For buffer size 100 ratios of $N_u$ and $N_s$ are 100 vs 865, and for buffer size 1000, they are 1000 vs 86500

> Once resources are finite (Moore's law is not exponential anymore) we'll need resource planning like the use of ATM (instead of the Internet!)


---
`Lecture 7`
#### Function of a network
Function of a network is to transport the items over its underlying graph.
- Asynchronous Transfer Mode - (ATM)
- Natural Information rate - Rate at which a source is generating information
- Statistical multiplexing - Increases the capacity of network by reducing bandwidth
- Constant bit rate- Not constant in real life situations (Usually distributed gaussianly)
- PCR = Peak Cell Rate
- SCR = Sustainable Cell Rate, burst tolerance can represent the packets distributed over the on-off(incoming packets are flowing/not) periods homogeneously.
- **Arrival Rate**: (Instantaneous arrival rate = $\lambda(t)$)
	- In interval u,t, $L(u, t) = \int_u^t \lambda(\tau)d\tau$
	- It is bounded by $L(u,t) <= max(\lambda(\tau))(t - u)$ because duh!
	- $E[\lambda] = L(u,t)/(t-u)$
- **Burstiness Constraint**
	- Flow rate $L(u,t) = \int_u^t \lambda(\tau)d\tau <= \overline{\sigma} + \overline{\lambda}(t-u)$
	- This was deducible from substituting $\overline{\sigma}$ to zero in the above equation and $\overline{\lambda}$ to max.
	- Divide both sides by (t - u) to get a relation with $E[\lambda]$ as well
	- If you plot Rate $\overline{\lambda}$ vs tolerance $\overline{\sigma}$ , for each source, there exists a convex curve.
- Input control - Rate can't exceed more than what was agreed upon before. (Network policing)
- Quality of Service, CAC- Connection Admission Control - Connection acceptance rules for new requests in order to guarantee QoS.
- Congestion - Offered load vs Carried Load, Curve bends after like 80%. 
	- System buffers fill up! -> Retransmissions -> More packet loss -> System collapse
	- Controlling:
		Preventives:
		- CAC
		- Traffic Shaping
		- Priority Control
		Reactive:
		- Selective Cell Discard
		- Forward Congestion indication
- If eigen values of A are evenly distributed around 0, it means the Graph is a tree.

---
`Lecture 6`
#### Robustness of a Network
- Probability distribution of the magnitude of a disaster often follows a power-law $Pr[D=k] = k^{-a}$ where a is called the `avelanche exponent`
	- Energy blackouts, earthquakes, twitter retweets etc.
- Framework for robustness:
	- Express robustness by a real number R (is between 0(not robust) and 1(robust af))
	- Define R value, then keep modifying the graph such that the graph achieves the desired R value threshold.
	- $R = \sum_{k=1}^m s_kt_k$ where s is the weight/service vector and t is the topology vector containing graph metrics like (clustering coefficient, average degree, algebraic connectivity etc.)
	- Assuming deterministic and Linear (Calculating E[R] is easy!)
	- _Issues_:
		- Linearity - Why ? Too simple ?
		- Dimension m - (Number of metrics)
		- Dependence of metrics on each other (correlation) Solution: Use eigenvectors
		- Normalization: Compare graphs with different number of nodes and links
		- Service decomposition: How to find weight corresponding to a metric?
- _Robustness Envelope_
	- Resillience - network's capability to withstand perturbations(removal/rewiring of links).
	- Find probability contours(envelope) of the R value on introducing n perturbations.
	- Looking at R vs attacks graph:
		- Look at area under the curve (R value) (higher the better)
		- Sensitivity: how spread out the contour plots are around the line. (lower/smaller the better)
	- Try out strategies when introducing perturbations like removing nodes with highest degree/betweenness/eigenvectors etc.

---
`Lecture 5`
- Create a graph from the circuit provided and find directions of links in the graph (direction of current) Find currents $y_{ij}$ and node voltages $v_i$
- Recap: $u^TB = 0$, $Qu=0$
- Current Laws:
	- _Kirchhoff's Law_ - Sum of currents in each node = 0 $\sum_{j \in neighbours(i)} y_{ij} = 0$
	$By = 0$ (and if you inject current $f_i$ in the node i): $By = f$
	- _Ohm's Law_ - Voltage difference = current * Resistance => $v_i - v_j = r_{ij}y_{ij}$
	$y = B^Tv$ where v is the voltage vector (Nx1) and y is the current vector (Lx1)
	general form with resistance not always 1 = $y = diag(1/r_{ij})B^Tv$
	- Combining laws gives: $f = Qv$ so we can deduce:
		- if no injected current, Qv = 0 (Means the v = all 1s vector and eigen value 0)
		- If injected current, Qv = f (But determinant of Q is non zero) (Substitute a reference voltage (Remove one node from the graph and solve))
		- Another way would be to compute pseudo-inverse of Q. 
			- Laplacian is symmetric so can be decomposed as $ZMZ^T$ (Z are orthogonal eigen vectors and M is eigen value diagonal matrix)
			- $Q = \sum_k\mu_kz_kz_k^T$ and $Q_p^{-1} = \sum_k \frac{1}{\mu_k}z_kz_k^T$ pseudo inverse
			- $QQ_p^{-1} = I - \frac{1}{N}J$ where $J = uu^T$ all one matrix (The eigen vector corresponding to the 0 eigen value)
			(Basically doing the same thing fancily)
		- Now $f = Qv$ => $Q_p^{-1}f = (I - (1/N)J)v$ => $Q_p^{-1}f = v - v_{average}u$
		- $v_{average} = u^Tv/N$ (u = all 1s vector)
	- Effective resistance: 
		- $v = Q_p^{-1}f$ = $v = IQ_p^{-1}(e_a - e_b)$
		- Aim: $v_1 - v_2 = I \omega_{12}$
		- Finally Effective resistance = $\omega = (e_a - e_b)^T Q_p^{-1} (e_a - e_b)$
		- Can also be written as :  $\omega = uz^T + zu^T - 2Q_p^{-1}$ (where z = diagonal elements of $Q_p^{-1}$)
			- Omega is symmetric
			- diagonal elements are zero
			- Effective graph resistance = sum of all elements of $omega/2$ $1/2 (u^T \omega u)$ also = $N. trace(Q_p^{-1})$ = $N. u^T z$ = $N. \sum (1/ \mu)$ (Note that diagonal elements of pseudo inverse are 1/eigenvalues of the original matrix)
			- Remember both techniques above `important`
			- ELements of this matrix can be verified by manually calculating resistance between two nodes:
				- Series = R1 + R2
				- 1/Parallel = 1/R1 + 1/R2

---
`Lecture 4`
#### Network Models
- Deterministic networks (Don't change over time)
- Types of graphs:
    - **Random Graphs - Erdos Renyi**
    	- Each node pair is connected independently with probability p (randomly)
    	- Average number of links in graph = number of links * p
    	- Complement of such a graph is also Erdos Renyi! (With probability of 1-p duh!)
    	- Link density = $L/L_{max}$ = p
    	- Clustering coefficient = p
    	- Degree distribution = Binomial = Pr[D=k] = $C^{N-1}_kp^k(1-p)^{N-1-k}$ very close to a Gaussian
    		- $\mu = E[D] = (N-1)p$
    		- $\sigma^2 = Var[D] = (N-1)p(1-p)$
    	- If N is very large, the randomness goes away as the ratio of $\sigma / \mu$ grows to zero! (Regular graph) (Becomes deterministic).
    	- For very large N, the binomial distribution tends to a Poisson distribution if E[D] is kept constant. (Degree is kept constant)
    	- Critical density p = $log N/N$ (Above which, the graph will be connected) (`phase transition`)
	- **Small world**
		- Take a regular graph and randomly rewire a few links
			- Clustering coefficient decreases rapidly (Triangles)
			- Diameter (Hop counts) decreases rapidly
		- Spectrums of Real world graphs show: (Eigen value ranges of the adjacency matrix)
			- Sum of eigen values = 0 (Trace of A = 0)
			- Broadness of peak = randomness
			- If it is even, it is tree like
			- Any tree can be represented as a bipartite graph
		
	- **Scale-free graph - Barabasi Albert**
		- Power law graphs
			- $P[D=k] = ck^{-\beta}$ here most important parameter = $\beta$ the exponent!
		- Scale Freeness - $Pr[D=ak] = a^{-t}Pr[D=ak]$
		- Near a phase transition, many properties of a physical distribution are power-law distributed!
		- Prepare your own power-law graph: courtesy Barabasi-Albert
			- Start with n nodes
			- Attach a node with m links to each node proportionally to its degree
			- Repeat untill desirable size(N) is achieved.
		- Power law has 3 critical points as $\tau$ increases:
			- 1-2 -> Degree grows fast, E[H] constant (No large powerlaw network can exist here)
			- 2-3 -> $D_{max} = N^{1/(\tau-1)}$, E[H] ~ log(log(N)) Ultra small world (Scale free regime)
			- 3+ -> Small world, $D_{max} = N^{1/(\tau-1)}$, E[H] = logN/logE[D] (Random regime)
	- Properties of real world networks:
		- Small world: Average Hop count E[H] = O(Log(N)) is short compared to size of network N.
		- Scale free- Non Gaussian behaviour
		- Robustness to random node failures is not that awesome in real world powerlaw unlike Erdos-Renyi.
		
---
`Lecture 3`

- **Eigen** things: $Ax = \lambda x$
	- x is a non-zero vector, for the eigen value $\lambda$
	- n x n matrix has n eigen values (not all distinct possibly)
	- A and $A^T$ have same eigen values but not necessarily the same eigen vectors
	- Eigen vectors are linearly independent
	- If $P^{-1}$ exists then $P^{-1}AP$ has the same eigen values as A but the eigen vector is Px
- More cool stuff
	- Any real symmetric matrix can be written as $X \delta X^T$ (X = matrix containing eigenvectors as columns and orthogonal ($X^T = X^{-1}$ or $XX^T = I$), $delta$ = diagonal matrix with real eigen values as diagonal elements)
	- therefore If we use it with the eigen equation: $AX = X\delta $ => $A = X\delta X^{-1}$
	- If symmetric, A = $A^T = X\delta X^T = \sum_k^N\lambda_kx_kx_k^T$
	- Therefore $A^m = \sum_k^N\lambda_k^mx_kx_kT$ - Remember this can be used to find number of m-hop paths from i to j using this technique.

#### Linear Algebra on Graphs
`Important`
- **Spectrum of Adjacency** - All eigenvalues lie in the interval $(-d_{max}, d_{max})$ (degree contains information about range of eigen values)
- Sum of eigen values of A = 0 $\sum\lambda = 0$
	- $\sum\lambda^2$ = 2L = sum of degree of all nodes
	- $\sum\lambda^k$ = Trace(A^k)
	- _Peron Frobenius theory_: For a connected graph, the main eigen vector has elements > 0 and eigen value is > 0 
	- Number of triangles = $1/N \sum\lambda^3$
- **Largest eigenvalue** of a symmetric matrix
  - For a symmetric matrix, $A^kx = \lambda^kx$ Notice how eigen vectors remain unchanged and power gets passed on to the eigen value.
  - Power method to find largest eigen value of a huge matrix!
  - $E[d]=2L/N <= \lambda_{largest} <= d_{max}$
  	- Better Bound: $\lambda_{largest} >= \frac{2L}{N}\sqrt{1 + \frac{Var(d)}{E(d)^2}}$
- Interlacing : Eigenvalues of a matrix obtained by removing some nodes from the original graph are always bounded by the eigen values of the original graph
- **Eigenvector Centrality**
	- Rank the importance of nodes according to the eigenvalues
	- Principal eigenvector $x_1$ is often used. $(x_1)_i$ is the eigenvector centrality of node i. Which is proportional to the number of walks through $node_i$
	- Sooo powerful - we just need to calculate the principal eigen vector to get the importance(rank) of a node! :bomb:
- **Spectrum of Laplacian** - Laplacian always has one eigen value of all ones with eigen value 0
	- _Algebraic connectivity_: _high means strongly connected_, >0 means connected. (Second smallest eigen value of Laplacian($BB^T$ where B = incidence matrix)).
	- `Graph partitioning` into disjoint subgraphs $G_1$ and $G_2$
	  - Laplacian can also be written as : $\delta - A$ where $\delta$ = diagonal matrix containing degrees.
	  - One eigen value will always be 0 (corresponding to vector: all ones u)
	  - All eigen values will be +ve
	  - Laplacian = $Q = BB^T$, Quadratic form = $z^TQz = z^TBB^Tz = ||B^Tz||^2_2$
	  - Number of links between G_1 and G_2 = 1/4 $\sum (y_{1}^+ - y_{-1}^-)^2$ (where y = 1 or -1 based on if the node is in G1 or G2)
- **Fiedler vector** - Split graph based on threshold. (Sum of eigen vector components)
- Notations for exam:
	- $\mu$ = eigen values of laplacian
	- $\alpha_k = y^Tx_k$ (x => eigen value of laplacian (linear combination)) and y is -1 or 1 based on where node belongs
	- Partition links to separate graphs = $R = 1/4 \sum_j^N \alpha_j^2\mu_j$
	- Smallest number of links == $R >= 1/4 \alpha_{N-1}^2 \mu_{N-1}$ (Second smallest eigen value after zero!)
- Degree preserving rewiring (crazy! so the effects of a node going down are minimized!)


---
`Lecture 2`

**Graph Metrics** - Measures or quantifies graph properties
- Walk succession of links, Path - A walk in which all hops are different nodes.
	- betweek i, j, k-hop walks = $(A^k)_{ij}$
	- total k-hop walks = $u^T A^k u$ (Sum of all elements of $A^k$)
	- means that all the diagonal elements of $A^k$ will be the k-hop-walks that end on the same node.
_Types of properties_: local (properties of the surrounding of a node), global.
- **Degree** $\sum_j^N d_j = 2L$ (Local metric)
	- Average Degree: $E[D] = \frac{1}{N}\sum_j^N d_j = \frac{2L}{N}$
	Soo for a connected graph, $Tree < E[D] < Complete$ = $2(N-1)/N < E[D] < (N-1)$
	- Degree vector(contains the degree of all nodes) = $d = Au$ where A = Adjacency matrix and u = all one vector (column)
	- Therefore sum = $u^Td = 2L$ => means $u^TAu = 2L$ (Which is the sum of all elements of the adjacency matrix)
	- And also fun take away: $d = (A)^2_{jj}$ (Diagonal elements of $A^2$)
	- In any graph:
		- There are at least 2 nodes with the same degree
		- Number of nodes with odd degree is even
	- Find out _degree distribution_ from the degrees by plotting a histogram (`mean` and `variance`)
		- Many networks have a power law degree distribution.
- **Clustering Coefficient** (How well connected are the node's neighbors (local density around a node))(between 0 and 1)
	- For a node v, Number of actual links between neighbors of a node = y, number of neighbors = $d_v$ (Max possible links = $d_v(d_v - 1)/2$)
	- Clustering coefficient $c_G(v) = \frac{2y}{d_v(d_v - 1)}$ $c_G = 0$ if $d_v = 1$
	- For the entire graph = $\frac{1}{N} \sum_v^N c_G(v)$ (Average of all nodes)
- **Hopcount** - Length of shortest path from i to j
	- Diameter of G = Length of longest shortest path.
	- Average hopcount E[H] = Efficiency of transport in G.
	- Calculating diameter from the adjacency matrix - Keep claculating A, A^2, A^3 till you have all non zero elements. the power is then the diameter as discussed in the formula above (betweek i, j, k-hop walks = $(A^k)_{ij}$)
- **Betweenness** of a node/link
	- Number of shortest paths passing throught the node/link.
	- Degree is correlated to betweenness (like most metrics)
	- Formula of linear correlation for variables X, Y = $\frac{E[XY] - E[X]E[Y]}{std(X)std(Y)}$ (-1 to 1)
	- E[B] = MaxLinks/L (E[Hopcount])
- **Degree Assortativity** - How are degrees on both sides of **a link** related
	- Can be used for load balancing
	- Correlation between the degrees of the nodes on left and right sides of the link
- **Connectivity of a Graph**
	- Complement of a disconnected graph is always connected - (Converse not true!)


---
`Lecture 1`

- `Konigsberg Bridge problem` The Graph needs to have all nodes with even degree and only zero or 2 nodes with odd degree for the _Eulerian Walk_ to be possible. (Same as being able to draw a figure without lifting the pencil or drawing on the same line again)
- Types of Graph representations:
	- List of neighbors
	- List of links
	- Adjacency matrix (A) - (Always symetric --> Real Eigenvalues/vectors)
	- Incidence matrix (B) - (Gives directions of the links `1 ==> -1`) $u^TB = 0$ where u is (1,1 ... ,1,1) all one vector
		- Rows = nodes
		- Columns = links
	- Laplacian matrix (Q)- ($Q =BB^T$) (Property -  $Qu = 0$)
		- Rows = nodes
		- Columns = nodes
		- Because of the property $Qu = 0$ we now know that u i.e. the all 1 vector is an eigen vector of Q (with eigen value 0)
- Classes of Networs:
	- Connected graphs:
		- Tree (N-1 links)
		- Cycle (N links)
		- Complete graph (Links = N(N-1) / 2) ==> (Binomial = N*(N-1))
	- Other Graphs:
		- Null Graph (0 links)
		- Complement $G^C$ of a graph G
		- Line Graph I(G) of G - (Each link in G is a node in I(G) - Hence two nodes in I(G) are connected if the corresponding links in G have a common node in them).
- _Eigen vectors tell you something about a node in the Graph_
- _Eigen vectors of the Line graph tell you something about a link in the Graph_
- Subgraph A = [<table><tr><td>_</td><td>_</td></tr><tr><td>$A_S$</td><td>B</td></tr><tr><td>C</td><td>$A_{G/S}$</td></tr></table>]
	- If G is undirected, $B = C^T$




