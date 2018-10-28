# **EE4C06 Networking** - notes :scroll:
_TU Delft_

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
	- Laplacian matrix (Q)- ($Q =BB^T) (Property - $Qu = 0$)
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
- Subgraph A = [<table><tr><td>$A_S$</td><td>B</td></tr><tr><td>C</td><td>A_{G/S}</td></tr></table>]
	- If G is undirected, $B = C^T$
