# **EE4C06 Networking** - notes :scroll:
_TU Delft_

---
`Lecture 2`
**Graph Metrics** - Measures or quantifies graph properties
- Walk succession of links, Path - A walk in which all hops are different nodes.
	- betweek i, j, k-hop walks = $(A^k)_ij$
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
	- Clustering coefficient $c_G(v) = \frac{2y}{d_v(d_v - 1)}$ $c_G = 0 if d_v = 1$
	- For the entire graph = $\frac{1}{N} \sum_v^N c_G(v)$ (Average of all nodes)
- **Hopcount** - Length of shortest path from i to j
	- Diameter of G = Length of longest shortest path.
	- Average hopcount E[H] = Efficiency of transport in G.
	- Calculating diameter from the adjacency matrix - Keep claculating A, A^2, A^3 till you have all non zero elements. the power is then the diameter as discussed in the formula above (betweek i, j, k-hop walks = $(A^k)_ij$)
- **Between-ness** of a node/link
	- Number of shortest paths passing throught the node/link.
	- Degree is correlated to betweenness (like most metrics)
	- Formula of linear correlation for variables X, Y = $\frac{E[XY] - E[X]E[Y]}{std(X)std(Y)}$ (-1 to 1)
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




