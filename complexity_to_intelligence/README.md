# From Complexity to Intelligence

---
`Lecture 1` 
**Nov 19, 2018**
#### Plain Complexity
- [Link](https://perso.telecom-paristech.fr/pamurena/courses/FCI/lecture0.pdf) to the lecture.
- Perception of randomness is related to the absence of structure
- *Cognitive science* is statistically how the majority would respond to something.
- Simplicity = minimum complexity
- **Kolmogorov complexity** of an object = minimum length of a computer program that can produce that object.
    - Depends on the chosen language (Python, Java etc.)
- `Alan Turing` came up with the:
    - **Turing Machine** - 0s and 1s on a tape and a set of actions based on state and pointer.
        - **UTM** - (Universal turing machine) - That can simulate any turing machine.
    - **Turing test** - That Google Assistant almost just beat in the recent IO.
- Complexity: $C_M(x) = min_{p \in P_M}\{l(p):p() = x\}$ - Shortest program on M producing an output x. Here p() means empty input-tape for the turing machine.
- Conditional Complexity: $C_M(x|y) = min_{p \in P_M} \{l(p): p(y) = x\}$
- Invariance Theorem: $|C_{M_2}(x) - C_{M_1}(x)| < c_{M_1 M_2}$ (Bounded) Doesn't depend on the output sequence x
- $\pi$ is not complex: $\frac{\pi}{4} = 1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7}...$
- Complexity of integer = $C(i) = log_2(i)$
- `Turing Machines are like humans` - Complexity of an item in the list depends on the position of element in the list.
- Properties: 
    - C(x) <= l(x) + c (+ c for the invariance theorem (Python to c etc.))
    - C(x|y) <= C(x) + c
    - C(x, y) <= C(x) + C(y) + 2log(min(C(x), C(y))) (This extra log term is to add separators between when x starts and y begins etc.)
