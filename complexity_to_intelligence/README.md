# From Complexity to Intelligence
**Pierre Alexandre Murena**


---
`Lecture 1` 
**Nov 19, 2018**
#### Plain Complexity
- [Link](https://perso.telecom-paristech.fr/pamurena/courses/FCI/lecture0.pdf) to the lecture.
- Perception of randomness is related to the absence of structure
- *Cognitive science* is statistically how the majority would respond to something.
- Simplicity = minimum complexity
- **Kolmogorov complexity** of an object = minimum length of a computer program that can produce that object. (The same as the size of the shortest unambiguous binary description of the object.) (The same as the length (in bits) of the output of an ideal compressor (like an ideal ‘Zip’ program) acting on the object) :bomb:
    - Depends on the chosen language (Python, Java etc.)
- `Alan Turing` came up with the:
    - **Turing Machine** - 0s and 1s on a tape and a set of actions based on state and pointer.
        - **UTM** - (Universal turing machine) - That can simulate any turing machine.
    - **Turing test** - That Google Assistant almost just beat in the recent IO.
- Complexity: $C_M(x) = min_{p \in P_M}\{l(p):p() = x\}$ - Shortest program on M producing an output x. Here p() means empty input-tape for the turing machine.
- Conditional Complexity: $C_M(x|y) = min_{p \in P_M} \{l(p): p(y) = x\}$
- Invariance Theorem: $|C_{M_2}(x) - C_{M_1}(x)| < c_{M_1 M_2}$ (Bounded) Doesn't depend on the output sequence x
- $\pi$ is not complex: $\frac{\pi}{4} = 1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7}...$
- Description length of an integer = $C(i) = log_2(i)$
    - Description length of a list = description length of the integer: len(list)
- `Turing Machines are like humans` - Complexity of an item in the list depends on the position of element in the list.
- Properties: 
    - C(x) <= l(x) + c (+ c for the invariance theorem (Python to c etc.))
    - C(x|y) <= C(x) + c
    - C(x, y) <= C(x) + C(y) + 2log(min(C(x), C(y))) (This extra log term is to add separators between when x starts and y begins etc.)
- The most probable hypothesis is the one that minimizes the description complexity of the couple.

---
`Lecture 2` 
**Nov 20, 2018**
#### Codes
- A code is called a **prefix** if and only if no codeword is the prefix of another. (This makes them _uniquely decodable_)
- Technique to find out if the code is prefix is to draw a table of 0s and 1s like Murena drew on the board. If a cell is selected, it's children(on the right) can't be selected.
- For each layer, $\frac{1}{2^l}$ is added and the sum should always be <= 1
- **Kraft inequality**: $\sum_r 2^{-l_r} <= 1$ - All decodable codes have to obey this constraint, but converse may not be true.
- Shannon Claude:
    - Father of Information Theory. 
    - Also invented the flip the switch game
    - L = min $\{ \sum_x P(x)l_x: C is a prefix code \}$ Where P is the probability that a word x is produced by the source
    - Shannon Fano code: $l_x = \[ \frac{1}{log(p(x))} \]$
- _Zipf law_: Rank of the word is linearly correlated to the frequency of occurance.
    - Means rank can be used instead of actual probability

#### Compression
- Kolmogorov complexity measures how compressible an object can be.
- There is no universal compression. If one word gets compressed, there is some word that gets expanded.
- Techniques:
    - Lempel-Ziv - Build a dictionary of already encountered patterns. - Used for GIFs!

#### Complexity, Randomness (For finite sequences) (For uniform distribution)
- Randomness is related to compressibility. Incompressibility = Passing the Randomness tests.
- Universal P-Test is a randomness test with respect to a distribution P.
- There is also a function depending on complexity that depicts radomness.

#### Positional coding
- Old encoding is just take the log of the number.
- Positional is different:
    - it is as follows 0-_, 1-0, 2-1, 3-00, 4-01, 5-10, 6-11, 7-000 etc.
    - It can store $2^{N+1} - 2$ numbers which mean each integer can be represented in $log(n + 2) - 1$ bits
    - Formula = 0/1 [Alternative (round numbers)/ Standard] 0/1 [Add/subtract] xxx [Number poscode] xxx [number to be added/subtracted]
    - This formula is the number of bits required and is the complexity!!!
- **Zipping sequences**: replace the recurring sequence with r(37, 9) r = repeat(occured before on the left), 37 = 37 items before, 9 = length.

