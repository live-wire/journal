# Questions from `paper_11` :robot: 

**Q: Which of the following options, according to this paper, is false?**
- Self-attention layers are slower than recurrent layers when the sequence length n is smaller than the representation dimensionality d, in terms of computational complexity.
- Self-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.
- One of the motivations behind the use of self-attention is the amount of computation that can be parallelized.
- The Transformer, the first sequence transduction model based entirely on attention, can be trained significantly faster than architectures based on convolutional layers.
* `[ A ]`


---

**Q: Why do we need attention mechanisms?**
- it is the only way to process sequences
- to better deal with very long input sequences
- to allow the use on mini batches in RNNs
- All of the above
* `[ B ]`


---

**Q: Which statement is false?**
- Dot-product attention is identical to our algorithm, except for the scaling factor. Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.
- In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence.
- An encoder layer takes in one argument and outputs at least more than two argument.
- In a self-attention layer all of the keys, values and queries come from the same place.
* `[ C ]`


---

**Q: What statement about the Transformer as described in the paper 'Attention is all you need' is FALSE?**
- It is the first sequence transduction model based entirely on attention
- It replaces the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self attention
- The Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers
- The Transformer can only be used for a very select set of language translations
* `[ D ]`


---

**Q: In the “Transformer” model presented in the paper “Attention Is All You Need”, what is used instead of the convolutional layer?**
- Sequence aligned RNNs
- A “Self-attention” mechanism
- An “attention mechanisms” used in conjunction with a recurrent network
- None of above
* `[ B ]`


---

**Q: Which of the following is NOT a possible advantage of self-attention compared to convolutional or recurrent layers?**
- Lower total computational complexity per layer.
- Shorter length of longest paths between any two positions in the network.
- Models with self-attention naturally contain positional information. 
- A larger amount of information can be parallelized.
* `[ C ]`


---

**Q: Can a automated transformer generalize well to other tasks using limited training data also?
 **
- Yes
- can be possible
- Never
- None of above are correct
* `[ B ]`


---

**Q: Which sentence best describes the relationship between additive and dot-product attention?**
- Dot-product attention is faster and more space-efficient.
- Their theoretical complexity is vastly different.
- Dot-product always outperforms additive attention.
- Additive attention uses a feed-forward network with multiple layers to calculate the compatibility function.
* `[ A ]`


---

**Q: What is the main problem with recurrent models having longer sequence lengths?**
- The calculation of the Hessian becomes way too difficult.
- Memory constraints limit batching across examples.
- Truncation error.
- The calculation of the Jacobian becomes way too difficult.
* `[ B ]`


---

**Q: The Transformer transduction model architecture set forth in the article "Attention Is All You Need"**
- consists of encoders and decoders, both in turn consisting of attention module(s) and feed forward network(s).
- has multi-head attention modules consisting of several layers running in parallel.
- is significantly faster than architectures based on recurrent or conventional layers.
- All of the above are true.
* `[ D ]`


---

**Q: Choose the correct statement:**
- In the Transformer model, constant number of operations is needed to relate signals in distant positions.
- Recurrent layer connects all positions with a constant number of sequentially executed operations.
- Self-attention decreases computational complexity at sequences of arbitrary length.
- Positional encoding uses exponential functions so the model can make us of the order of the sequence by knowing the absolute position of the tokens.
* `[ A ]`


---

**Q: What is self-attention?**
- A function which increases the output of certain nodes, thereby increasing the attention for certain features
- A mechanism which relates different positions in a single sequence in order to compute a reperesentation of the entire sequence
- A maping from a set of key-value pairs to an output consisting of vectors
- None of the above
* `[ B ]`


---

**Q: What kind of tasks work particularly well with the Transformer?**
- Language translation
- Image processing
- Text to speech
- None of the above.
* `[ A ]`


---

**Q: What is not one of the steps an attention function takes for translating speech?**
- Mapping a query and a set of key-value pairs to an output
- Computing the output as a weighted sum of the values
- Normalizing the values using the sequence length
- Computing the weight of each value by a compatibility function
* `[ C ]`


---

**Q: What are the successes of the Transformer network?**
- It is very fast
- It obtains very good results
- Both A and B
- Neither A or B
* `[ C ]`


---

**Q: Which of the following is not a way through which a transformer neural network makes use of multi-head attention to its different layers?**
- Every positions in the decoder can attend over all positions in the input sequence
- Each positions in the encoder can attend to all positions in the previous layer of the encoder
- Each positions in the encoder can attend to all positions in the decoder layer
- Each position inthe decoder can attend to all positions in the previous layer of the decoder 
* `[ C ]`


---

**Q: Which of the following attention types allows to jointly attend information from different representation subspaces at different positions?**
- Dot product attention
- Scaled dot product of attention
- Additive attention
- Multi-head attention
* `[ D ]`


---

**Q: What is the main idea of attention?**
- take more time determining the architecture
- longer training
- connect encoder directly to decoder
- all of the above
* `[ C ]`


---

**Q: Where does the transformer uses multi-head attention?**
- In "encoder-decoder attention" layers
- In self-attention layers
- In every hidden layer
- Both a and b
* `[ D ]`


---

**Q: What is the attention mechanism introduced to RNN?**
- Attention mechanisms determine the region of focus in for example the visual input
- It is introduced to because RNNs like children, have very short attention span.
- RNNs are bad at understanding context and attention mechanisms create context
- There is no such thing and the Earth is flat
* `[ A ]`


---

**Q: What is true about the translation model proposed in the article “Attention Is All You Need”?**
- It does not use a recurrent neural net
- It does not use convolution
- Both of the above
- None of the above.
* `[ C ]`


---

**Q: Which component is not used in Transformer?**
- Encoder and decoder
- Mutli-head attention
- residual block
- LSTM
* `[ D ]`


---

**Q: In the paper: Attention is all you need. The author presented the Transformer model. Which of the following statements is correct:**
- Transformer model can be trained significantly faster than a regular recurrent network. But cannot achieve the same level of accuracy.
- It does not know the >>concept<< of an encoder or decoder. 
- Allows parallelization by using multiheaded attention
- The transform model is more complex, which will cause a vanishing gradient problem to occur when longer sequences are used.
* `[ C ]`


---

**Q: In paper 11 the authors propose a network architecture based on attention mechanisms. What was the purpose of the attention function?**
- To map a query and a set of key-value pairs to an output (queries, keys, values and outputs are all vectors);
- To map a query and a set of key-value pairs to an input (queries, keys, values and outputs are all vectors);
- To pre-process the queries received from the encoder;
- To pre-process the queries that need to be further send to the network.
* `[ A ]`


---

**Q: The task of "sequence transduction", in which RNNs are applied commonly, is the task of**
- transforming sequences of inputs into related sequences of outputs
- translating a sentence from English to some other language
- translating a sentence from any language to English
- transforming sequences of inputs into matrices
* `[ A ]`


---

**Q: ‘FFN’  stands for ‘feed forward neural network’ and ‘CNN’ stands for ‘convolutional neural network’.
The attention-based network is better able to learn long-range dependencies by:**
- Replacing the use of CNNs with FFNs, which preserves more gradient throughout the stacks of the model.
- Having a design that can be trained in parallel, thus being able to train a deeper model.
- For each layer, mapping the input representations to the another representation is done using a dot product between the two, thus reducing the total distance to be a constant.
-  Stacking multiple FFNs on different parts of an input representation, which shortens the connection between those parts.
* `[ C ]`


---

**Q: What is \textbf{not} a reason to use self attention in a neural network**
- The computational complexity per layer is relatively low
- Work can easily be parallelized.
- The path length between any combination of positions is shorter.
- Short-range dependencies can be more easily learned.
* `[ D ]`


---

**Q: What is an advantage of using multi-headed self-attention instead of recurrent layers in encoder-decoder architectures?**
- A Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers
- Transformers achieve better results on translation tasks
- Self-attention and thus Transformers could yield more interpretable models
- Answers A-C are all correct
* `[ D ]`


---

**Q: What is true about the Transformer network**
- It's a family of RNN
- Attention is used in the encoder and decoder
- Transformers don't need regularization anymore
- Transformers are still hard to parallelize
* `[ B ]`


---

**Q: The authors of "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" are trying to prove that:**
- the closer in time the RNNs look for predicting events, the better the results
- the TCNs (temporal convolutional networks) can have better performances with respect to RNNs on sequential input data
- the main difference between TCNs and RNNs is the "memory" element, that makes RNNs better in analyzing sequential data
- RNNs and TCNs are not comparable because the input the operate onto is too different
* `[ B ]`


---

**Q: What does "Attention" achieve in a network performing sequence modeling?**
- Maintains an optimal cell state to handle long short term memory.
- Detects important nodes in the network layer, to minimise convergence time.
- Detects important non-trivial subspaces in data based on the context.
- All of the above.
* `[ C ]`


---

**Q: What is NOT true about sequence transduction models based solely on attention mechanisms?**
- These models require less time to train than traditional models.
- These models have a higher performance accuracy than traditional models.
- Reducing the attention key size improves the model quality.
- These models are more parallelizable than traditional models.
* `[ C ]`


---

**Q: What is the correct informal definition of Attention in deep neural networks?**
- The mechanism equips a neural network with the ability to focus on a subset of its inputs.
- It’s  a function that takes as input a vector of K real numbers, and normalizes it into a probability distribution consisting of K probabilities.
- It’s a method to encapsulate the information for all input elements in order to help the decoder make accurate predictions.
- None of the above.
* `[ A ]`


---

**Q: Consider the following two statements about the Transformer network architecture:

1.	An attention function can be described as mapping a query and a set of key-value pairs to an output,
where the query, keys, values, and output are all vectors.

2.	The proposed transformer uses multi-head attention in “encoder-decoder attention” layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. 

Which of the statements is true?**
- Both are true
- Both are false
- 1 is true and 2 is false
- 1 is false and 2 is true
* `[ A ]`


---

**Q: Which of the following is true:**
- The output of the attention function is the weighted sum of the key values
- Additive attention is much faster than dot-product attention in practice
- The dot-product attention outperforms additive attention
- The downside of multi-head attention is that it has a much higher computational cost compared to that of single-head attention
* `[ A ]`


---

**Q: Which of the following statements is not true about the Transformer?**
- It generalizes well with respect to its task.
- Its the first sequence transduction model based on attention.
- It uses recurrent layers.
- It is designed as a new simple network architecture.
* `[ C ]`


---

**Q: Which of the following is \emph{not} part of the Transformer - model architecture:**
- Multi-Head Attention.
- Point-wise, fully connected layers.
- Encoder-decoder structure.
- Convolutional layers.
* `[ D ]`


---

**Q: According to the paper, what is self-attention?**
- Rearranging the order of the sequence in order to put the most significant positions first
- Finding similarities between multiple sequences in order to extract the general meaning
- Linking the order of the words in order to compute a representation of the sequence
- Relating different positions of a single sequence in order to compute a representation of the sequence
* `[ D ]`


---

**Q: Which two parts does the Transformer consist of?**
- Self-Attenion and Feed Forward Neural Network
- Self-Attention and RNN
- Self-Attention and Decoder
- Decoder and Encoder
* `[ A ]`


---

**Q: Statement 1: Recurrent models typically factor computation along the symbol positions of the input and output sequences. 
Statement 2: The encoder is composecd of a stack of N=6 identical layers and each layer has two sub-layers, namely a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. **
- Both statements are true
- Statement 1 is true
- Statement 2 is true
- None of the statements are true
* `[ A ]`


---

**Q: How are Q (Queries), K (Keys) and V (Values) used in the self-attention mechanism?**
- They are used to store values for future hidden states. V are the values stored, Q and V are used to index them.
- If a given query closely matches a given key, their dot product will be of high value, meaning the value is selected.
- If a value is deemed of high importance, its corresponding query is passed to the next layer, so it may be used to retrieve the value again.
- If a value is deemed of high importance, its corresponding key is passed to the next layer, so it may be used to retrieve the value again.
* `[ B ]`


---

**Q: The paper poses the Transformer architechture for sequence tasks. What is very attractive about this network?**
- The network is much cheaper to train than the canonical RNN architecture
- This network is proofed to be the optimal encoder
- The architecture is position invariant
- The architecture is length invariant
* `[ A ]`


---

**Q: Which statement is false?**
- Dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder
- The best performing sequence transduction models also connect the encoder and decoder through an attention mechanism
- The Transformer is solely based on attention mechanisms
- Dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include multiple autoencoders
* `[ D ]`


---

**Q: What is the effect the paths size between combination of positions in the input and output sequences?**
- Shorter paths means easier to learn long-range dependencies
- Longer paths means easier to learn long-range dependencies
- Shorter paths means easier to learn short-range dependencies
- Longer paths means easier to learn long-range dependencies
* `[ A ]`


---

**Q: Which of the following is not one of the biggest motivations to use self-attention?**
- Low computational complexity per layer
- High generalization to unseen training data
- Large amount of parallelized computation
- Low path-length for long-term dependencies
* `[ B ]`


---

**Q: What is meant by Attention layers in the proposed Architechture**
- A layer that focuses the input
- A layer that has a query as input
- A layer that maps a query to key-value pairs
- A layer that reflect on the entire model
* `[ C ]`


---

**Q: Which of the following was not a motivating factor for the authors, in order to use a transduction model relying
entirely on self-attention to compute representations of its input and output?**
- The smaller total computational complexity per layer.
- The biggest amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.
- The shorter path length between long-range dependencies in the network.
- The reduction in the number of weight parameters.
* `[ D ]`


---

**Q: What is attention function?**
- It maps query vector and key values vector to an output vector
- It maps query vector and key values vector to an output scalar
- It maps query scalar and key value scalar to an output vector
- It maps query scalar and key value scalar to an output scalar
* `[ A ]`


---

**Q: Select the motivation for using self-attention layers.**
- Computational complexity
- Amount of parallelized computation
- Path length between long-range dependencies
- All of the above
* `[ D ]`


---

**Q: Which one is not the advantage of the Transformer based on attention mechanisms ?**
- In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d. 
- Self-attention mechanism of the Transformer could yield more interpretable models. 
- The transformer allows for significantly more parallelization. However, the recurrent models precludes parallelization within training examples. 
-  To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size r in the input sequence centered around the output position. This wouldn't increase the maximum path length. 
* `[ D ]`


---

**Q: What is a reason that  long short-term memory  is hard in the neural networks?**
- It is hard for humans thus hard for computers
- It isn't hard at all
- As information gets lost by the network over time it is difficult to learn the connections over time.
- Due to the layered network.
* `[ C ]`


---

**Q: Why is self attention preferred over RNN and CNN?**
- Total computational complexity per layer is less
- No.of parallelizable computations is more
- Shorter path length between long-range dependencies in the network
- All of the above
* `[ D ]`


---

**Q: Why is self attention preferred over RNN and CNN?**
- Less computational complexity per layer
- The amount of computation that can be parallelized is high
- Smaller path length between long-range dependencies in the network
- All of the above
* `[ D ]`


---

**Q: Which is NOT advantage of attention layer compared with recurrent layer?**
- Total computational complexity is smaller
- The amount of computation that can be parallelized is larger
- Path length between long-range depencies in the network is shorter
- Avoid vanishing of gradient
* `[ D ]`


---

**Q: Choose whether each statement is true or false. 1) The Transformer is based solely on attention mechanisms. 2) The Transformer takes much longer to train for translation tasks than other architectures based on recurrent networks.**
- 1- True. 2- True.
- 1- False. 2- False.
- 1- True. 2- False.
- 1 - False. 2- True
* `[ C ]`


---

**Q: What is self-attention?**
- An attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence
- An attention mechanism to map a query and a set of key-value pairs to an output
- An attention mechanism relating different positions of a multiple sequences in order to compute a representation a set of sequences.
- None of the above.
* `[ A ]`


---

**Q: With the same training time, which of these networks is currently most suitable for translation tasks?**
- RNN
- CNN
- GNN
- Transformer
* `[ D ]`


---

**Q: The Transformer uses multi-head attention in three different ways, which of the following is not one if these ways?**
- A way where queries come from the previous decoder layer,
and the memory keys and values come from the output of the encoder.
- A way where all values and queries come from the same place.
- A way where self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position.
- A way where queries come from the previous encoder layer,
and the memory keys and values come from the output of the decoder.
* `[ D ]`


---

**Q: What is the computational complexity per layer of Self-Attention? (n is the sequence length, d is the representation dimension and k is the kernel size of convolutions)**
- O(n^2 * d^2)
- O(n^2 * d)
- O(n * d^2)
- O(k * n * d^2)
* `[ B ]`


---

**Q: Given that the Transformer architecture is not a RNN but a CNN, how does it deal with the lack of information about the relative or absolute position of the tokens?**
- Positional encodings are added to word embeddings. 
- In processes the input from left to right.
- An additional array is created and the network access it dynamically.
- TCN do not take into account position information.
* `[ A ]`


---

**Q: What is the goal of an attention mechanism?**
- Weighting the importance of previous items in a sequence by how important they are in the output
- Drawing global dependencies from input to output
- Watching the change in the objective function during training and tuning hyperparameters accordingly
- Remembering information from a point very far back into a sequential input
* `[ B ]`


---

**Q: What is the main problem that attention mechanisms trying to solve?**
- Short-term dependencies.
- Overfitting.
- Long-range dependencies.
- Underfitting
* `[ C ]`


---

**Q: The Transformer is a new model architecture elaborated by Google researchers to perform translation tasks between two different languages. Why is it different with respect to the previous state-of-the-art models?**
- Because it exploits recurrency in RNNs to achieve better results
- Because it is able to introduce the so called "attention" in a RNN, speeding up training
- Because its architecture relies completely on self attention during this task
- Because it uses convolutional layers in a very innovative way, achieving higher performances in word recognition
* `[ C ]`


---

**Q: Which of the following statements is FALSE?**
- Attention mechanisms allow the modeling of dependencies in sequences without regard to their distance
- Attention mechanisms are often used in conjunction with a RNN
- Because Attention is used for sequence modeling, it is impossible to use without also using some form of RNN
- For translation tasks, the Transformer (self-attention based architecture) can be trained significantly faster then RNN or CNN architectures
* `[ C ]`


---

**Q: Which of the following is not true regarding the paper?**
- Residual dropout regularization is used 
- For translation tasks, the transformer can be trained  faster than recurrent or convolutional layers
- Self attention is not used in the architecture
- The experiment model does not contain any recurrence or convolution
* `[ C ]`


---

**Q: Which of these statements is not a motivating factor to use self-attention?**
- The total complexity per layer will decrease
- Self-attention will cause the amount of computations that can be parallelised to increase
- Self-attention will cause the path lengths between long-range dependencies in the network to shorten
- Self-attention increases the regularisation of the model
* `[ D ]`


---

**Q: Which of the following statement is NOT TRUE?**
- Memory contains remain in RNN models.
- The Transformer allows for more parallelization by using multi-head self-attention.
- The attention mechanism in Transformer helps reduce the number of operations required to relate signals from two arbitrary input and output positions. 
- All of above.
* `[ D ]`


---

**Q: What is the purpose of attention in a neural network**
- Attention is an algorithm that allows the neural network to attend to previous inputs or outputs and extract useful information
- attention allows the neural network to see the previous hidden state
- attention is a form of recurrence
- None of the above
* `[ A ]`


---

**Q: Where is Attention used in a Transformer model?**
- Encoder-decoder attention layer, where every position in the decoder attends over all positions in the input sequence
- Each position in the encoder attends to all positions in the previous layer
- Each position in the decoder attends to all positions in the decoder
- All of the above
* `[ D ]`


---

**Q: What is FALSE about the Transformer architecture developed in Vaswani et al. 2018? **
- Both the encoder and the decoder have some layers of Positional encoding
- the encoder and the decoder have the same number of sub-layers
- it is applied a residual connection between Multi-head attention layer and Positional encoding layer
- compered to the encoder, the decoder has an additional Multi-head attention sub-layer in each layer
* `[ B ]`


---

**Q: Q: In the paper "Attention is All You Need", which are NOT motivations for why self-attention layers might prove better than more traditional recurrent or convolutional layers?**
- Reduced computational complexity per layer
- Increase in the number of computations which can be parallelized
- Decrease in path length between forward and backward signals traveling in the network
- None of the above (They are all valid motivations)
* `[ D ]`


---

**Q: How does the transformer preserves it auto-regressive property, without making use of a RNN structure?**
- The decoder of the transformer makes use self-attention and masked attention which allows each position in the decoder to attend to all positions in the decoder up to and including that position without including future words
- The encoder of the transformer makes use self-attention and masked attention which allows each position in the encoder to attend to all positions in the encoder up to and including that position without including future words
- Both the encoder and decoder of the transformer makes use of input and output embeddings, ensuring the position of each word using positional encoding and thereby making sure each position can attend to all positions in both the encoder and decoder
- The transformer processes whole sequences at once making it not auto-regressive
* `[ A ]`


---

**Q: In the “Attention is all you need” paper the Transformer architecture is introduced. In what ways does the Transformer use multi-head attention?**
- In the “encode-decoder attention” layers
- In the encoder, as self-attention layers
- In the decoder
- In all three ways presented above
* `[ D ]`


---

**Q: What's an advantage of using Self-Attention over the conventional sequential setup of RNN's?**
- It reduces the length of paths between long-range dependencies in the network.
- It allows for parallelization.
- Both A and B are correct.
- None of the above.
* `[ C ]`


---

**Q: Which of following statements is/are correct?
(1)	Using Attention Mechanism, the output of each decoder depends on a summary of the hidden states of the encoder.
(2)	Attention mechanism can be used in conjunction with a recurrent network but do not have to.**
- -1
- -2
- (1) & (2)
- None of them is correct.
* `[ C ]`


---

**Q: When using a self-attention layer as a layer in a neural network which of the following statistics would result in the highest computation time? Where n is the sequence length and d is the representation dimension. **
- Low n, high d.
- Low n, low d.
- High n, low d.
- None of the above
* `[ C ]`


---

**Q: Which of the following statements is false?**
- An attention function can be described as mapping a query and a set of key-value pairs to an output. 
- Multi-head attention consists of several attention layers running in parallel.
- The ‘Transformer’ is the first sequence transduction model based entirely on attention.
- In terms of computational complexity, self-attention layers are slower than recurrent layers.
* `[ D ]`


---

**Q: What is the path length of a recurrent network**
- O(n)
- O(1)
- O(logk(n))
- O(n/r)
* `[ A ]`


---

**Q: Which of the following statements about self-attention is FALSE?**
- An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.
- Two commonly used attention functions are additive attention and dot-product attention. 
- Motivations for the use of self-attention include: the total computational complexity per layer.; the amount of computation that can be parallelized; and no need to regularize.
- Self-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. 
* `[ C ]`


---

**Q: Which of the following is true in context of not using recurrent and convolutional mechanisms, but attention mechanisms for sequence modelling?**
- We can increase the paralellization of the computations involved as there are less sequential operations like in a recurrent mechanism.
- Convolutional mechanisms are in general less computationally heavy, but cannot do long-term dependencies like attention mechanism can
- Attention models cannot use an encoder-decoder architecture, thus limiting their representational capabilities.
- None
* `[ A ]`


---

**Q: What are reasons of using self-attention in layers of a network?**
- The first reason is to make certain layers less impactful. The second reason is that with self-attention the amount of computational power over the entire network will be reduced. With the reason is that you are more clearly able to see short-range dependencies in the network and can act upon that.
-  The first reason is to reduce the total computational complexity per layer. The second reason is that with self-attention is that the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required. With the third reason is the ability to learn long-range dependencies in the network.
- The first reason is to make the network less complex. The second reason is that with self-attention there is a greater chance to obtain long-range dependencies between nodes in a network. The third reason is that you need self-attention for translation tasks, because otherwise it will not work.
- There is no reason to use it
* `[ B ]`


---

**Q: Which line in the following description of an attention function is not correct?**
- An attention function can be described as mapping a query to a set of key-value pairs
- The output of the attention function is calculated using values from the key-value pairs from the function
- The weight assigned to each value is computed using a compatibility function
- This compatibility function takes in the query and the key corresponding to the value
* `[ A ]`


---

**Q: What is the main advantage of attention mechanisms compared to the recurrent way of modelling sequences?**
- Attention allows modelling dependencies without regard to their distance in the input or output sequences.
- Attention is parallelizable, and thus way faster to train.
- Both A and B are correct.
- None of these answers are correct.
* `[ C ]`


---

**Q: Why is self-attention used, where the best performing model also connects the encoder and decoder through an attention mechanism?**
- total computational complexity per layer
- amount of computations that can be parallelized
- path lengths between long range dependencies in the network 
- all of the above
* `[ D ]`


---

**Q: What is the obstacle of using RNNs/CNNs for sequence modelling?**
- They tend to forget long-term dependencies, that is why we need another approach that will not consider distances in the input/output 
- With the advanced settings long-term dependencies is not already the problem but it does not eliminate the fundamental limit
- Both are correct
- Both are incorrect
* `[ C ]`


---

**Q: How can we described attention function?**
- As mapping a query and a set of key-value pairs to an output, where only output is vector.
- As mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.
- As mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all matrices.
- As selective narrowing or focusing of consciousness and receptivity.
* `[ B ]`


---

**Q: Consider the following statements
a)In sequential transduction tasks, the path lengths that the forward and backward signals have to traverse have no effect on the learning ability of the network
b)A benefit of self-attention is that it could yield more interpretable models**
- Both statements are wrong
- Statement (a) is correct; Statement (b) is wrong
- Statement (a) is wrong; Statement (b) is correct
- Both statements are correct
* `[ C ]`


---

**Q: Which advantage does the Transformer proposed in the paper has over regular RNNs?**
- By eschewing recurrence, it attains a smoother gradient.
- By eschewing recurrence, it gains a better long-time memory.
- By eschewing recurrence, it is able to achieve a higher level of parallelization.
- By eschewing recurrence, it requires less training data.
* `[ C ]`


---

**Q: what is self attention?**
- a way for rnns to improve retention
- a way for rnns to modify their weights
- an attention mechanism
- None of the above
* `[ C ]`


---

**Q: What's a typical RNN structure used for machine transaltion?**
- Encoder-Decoder
- LSTM
- Gated RNN
- all of the above
* `[ D ]`


---

**Q: Why does (generally speaking) an attention model perform better at translation tasks than for example RNN (both using a encoder-decoder)?**
- An attention model creates shortcuts between the context vector and the entire source input
- An attention model builds a single context vector from the encoders last hidden state
- A RNN is not able to handle sequential data
- A RNN creates shortcuts between the context vector and the entire source input
* `[ A ]`


---

**Q: Which of the following statements is NOT an improvement for the Temporal Convolutional Network (TCN) compared to other networks (CNN & RNN's)?**
- An input sequence can processed as a whole instead of sequentially
- The backpropagation path is different from the temporal direction of the input sequence and exploding/vanishing gradients are avoided
- During evaluation, the sequential input can be processed as a whole and less memory is required
- The receptive field of a TCN is highly flexible 
* `[ A ]`


---

**Q: Which of the following is true**
- The Transformer is the first transduction problem relying entirely on self-attention.
- Most competitive tansduction models have an encoder-decoder structure
- Residual Dropout and Label Smoothing are types of regularization
- All of the above
* `[ D ]`


---

**Q: Which statements about the “Attention Is All You Need” paper are true?:

Statement1: To the author’s best knowledge, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence aligned RNNs or convolution.

Statement2: The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs

Statement3: Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

Statement4: Convolutional type of network (compared to Self-attention, Recurrent, Self-attention restricted) has the highest complexity per-layer. **
- 1 and 4
- 2 and 3
- 1 2 and 3 
- All statements are true
* `[ D ]`


---

**Q: Which of the following statements is true:
1. An attention function can be described as mapping a query and a set of key-value pairs to an output, where all in and output are vectors
2. Networks with attention can outperform RNNs in sequential tasks**
- 1
- 2
- both
- neither
* `[ C ]`


---

**Q: What is not a benefit of the Attention model used in the Transformer Network as developed by Google Brain?**
- It is computationally more efficient due to higher parellelism compared to LSTM RNNs.
- It is better able to model long term dependencies by reducing path length.
- It increases overfitting slightly in general in order to achieve a lower BLUE score.
- It generalizes well to other tasks than Neural Machine Translation.
* `[ C ]`


---

**Q: Which of the following statement is not a reason to motivate the use of self-attention?**
- the total computational complexity per layer
- the amount of computation that can be parallelized
- the position-wise feed forward networks
- the path length between long-range dependencies in the network
* `[ C ]`


---

**Q: What is the type of cognitive task the Transformer was trained on?**
- Image recognition
- Speech recognition
- Translation tasks
- Movie suggestions
* `[ C ]`


---

**Q: Which of the following is considered desirable in choosing self-attention to recurrent and convolution layers?**
- Total computational complexity per layer, self-attention layers are fatser.
- The amount of computation that can be parallelised, self-attention requires a constant number of sequentially executed operations
- The path length between long-range dependencies in the network
- All
* `[ D ]`


---

**Q: Which statement is correct?**
- The Transformer reduces the number of sequential operations to relate two symbols from input/output sequences to a constant O(1) number of operations.
- The Multi-Head Attention is several attention layers stacked in parallel, with different linear transformations of the same input.
- The Transformer uses Multi-Head Attention in three different ways
- All of the statements are correct
* `[ D ]`


---

**Q: What is TRUE about the Transformer Architecture?**
- It uses convolutional layers with kernel size of 1
- It contains fully connected layers in both encoder and decoder 
- It contains trainable embeddings layers
- It achieves state-of-the-art results in terms of BLEU score, but at a cost of higher computational demands
* `[ B ]`


---

**Q: Consider the following statements
a)In sequential transduction tasks, the path lengths that the forward and backward signals have to traverse have no effect on the learning ability of the network
b)A benefit of self-attention is that it could yield more interpretable models**
- Both statements are wrong
- Statement (a) is correct; Statement (b) is wrong
- Statement (a) is wrong; Statement (b) is correct
- Both statement are correct
* `[ C ]`


---

**Q: For Transformer, which description is correct?**
- It uses sequence-aligned RNNs and convolution
- It uses sequence-aligned RNNs but not convolution
- It does not use sequence-aligned RNNs but convolution
- It does not use sequence-aligned RNNs and convolution
* `[ D ]`


---

**Q: A single convolutional layer with a kernal width that is smaller than the number of symbol representations:**
- Does not connect all the pairs of input and output positions. The amount of convolutional layers is: O(n/k) for contiguous kernels and O(log_k(n)) for dilated convolutions.
- Connects all the pairs of input and output positions and amount of convolutional layers is: O(n/k) for contiguous kernels and O(log_k(n)) for dilated convolutions.
- Does not connect all the pairs of input and output positions. The amount of convolutional layers is: O(log_k(n)) for contiguous kernels and O(log_k(n/k)) for dilated convolutions.
- Does not connect all the pairs of input and output positions. The amount of convolutional layers is: O(log_k(n)) for contiguous kernels and O(log_k(n/k)) for dilated convolutions.
* `[ A ]`


---

**Q: Which of the following is NOT a reason to prefer convolutional networks over RNN**
- Convolutional networks are easier to train.
- Convolutional networks are conceptually simpler
- Convolutional networks can take arbitrary length inputs
- Convolutional networks can be trained more efficiently in parallel
* `[ C ]`


---

**Q: Which of the following statements is true about causal and dilated convolutions:**
- Causal convolutions are convolutions where an output is only convolved with elements from earlier timesteps in the previous layer.
- A disadvantage of causal convolutions is that very deep networks are needed to achieve long history
- Dilation introduces a fixed step between adjecent filter taps and expand the receptive field of CNNs
- All answers are true
* `[ D ]`


---

**Q: which statement is wrong?**
- An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.
- Self-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.
- Self-attention could yield more interpretable models.
- Non of them.
* `[ D ]`


---

**Q: What is an advantage of using attention over classic sequential learning?**
- You need less data.
-  Attention can be done in parallel
- It learns better by processing data step by step
- None of the above
* `[ B ]`


---

**Q: Which of the two following statements about self-attention network architecture, the Transformer, as discussed by the paper by Vaswani et al. (2017), is correct?
1)	The decoder in the system can attend over all positions in the input sequence, compared to a regular RNN that can only attend one position.
2)	In a self-attention layer all of the keys, values and queries come from the same place, say the output of the previous layer in the encoder.**
- a) Both are correct
- b) Statement 1 is correct, Statement 2 is false
- c) Statement 1 is false, Statement 2 is correct
- d) Both are false
* `[ A ]`


---

**Q: What is NOT the pros of the Transformer network?**
- It's very innovative that does not use traditional RNN and CNN but has a quite good performance.
- It makes the distance between any two words constant which is also easy for GPU parallelization.
- Transformer could be extended to other research field rather than only in NLP.
- It has more power of capturing local features compared to CNN and RNN.
* `[ D ]`


---

**Q: In the paper: "Attention Is All You Need", the authors propose a model architecture for sequential transduction which does not use recurrent modules. Which of the following might be an advantage of this model**
- It has smaller memory requirements for storing previous states
- It has a higher level of parallelism as the largely sequential component of RNN is not being used
- It is less computationally expensive the RNN architectures
- None of the above
* `[ B ]`


---

**Q: Consider attention mechanisms using RNNs. What advantage does this networks have compared to other neural networks?**
- The number of operations required to relate signals from two arbitrary input or output positions is smaller than other NN.
- The number of operations required to relate signals from two arbitrary input or output positions is larger than other NN.
- Attention mechanisms helps to make training more parallel, thus these networks can be trained faster.  
- none of the above
* `[ A ]`


---

**Q: Which of the following does not occur in a  neural sequence transduction model?**
- Encoding symbol representations to a sequence of continuous representations.
- At each step consuming the previously generated symbols as additional input when generating the next.
- Using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.
- None of the above
* `[ D ]`


---

**Q: Which of the following is NOT TRUE about transformer used as multi-head attention?**
- the queries from the previous decoder layer and memory keys of values come from the output of the encoder
- encoder contain self-attention layers where all the keys, values and queries come from the same place. 
- decoder contain self-attention layers where all the keys, values and queries come from the same place.
- decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position
* `[ C ]`


---

**Q: What statement is true?**
- The number of sequential operations needed in a self-attention layer is comparable to that in a convolutional layer
- The computational complexity of a self-attention layer is lower than that of a recurrent layer
- The maximum path length between any two input and output positions in a self-attention layer is comparable to that in a convolutional network
- All of the above
* `[ B ]`


---

**Q: Pick the correct one**
- Generally RNN are more expensive than CNN by a factor of k
- Generally CNN are more expensive than RNN by a factor of k
- Generally CNN are more expensive than RNN by a factor of k^2
- Generally RNN are more expensive than CNN by a factor of k^2
* `[ B ]`


---

**Q: Which of the following sentences are true for both recurrent and self-attention models?**
- Recurrent networks’ layer connects all positions with a constant number of sequentially executed operations, while a self-attention layer requires O(n) sequential operations.
- Recurrent networks are faster than the self-attention systems in terms of computational complexity
- Convolutional layers are generally more expensive than recurrent layers, by a factor of k
- The complexity of the Sequential operations are the same for Recurrent and Convolution neural networks
* `[ C ]`


---

**Q: Which of the following is/are true about the transformer?**
- Relies entirely on self-attention to compute representations of it’s input and output. 
- Follows the overall architecture using stacked self-attention and point-wise, fully connected layers.
- A and B.
- None of the above. 
* `[ C ]`


---

**Q: What are the advantages of "Self-attention" at layers?**
- They could yield more interpret-able models.
- attention distributions from models could be inspected easily
- Learning related to exhibit behavior to the syntactic and semantic structure of the sentences can be easily done
- All of the above
* `[ D ]`


---

**Q: Why does this paper scale the dot-product attention with 1/sqrt(dk)**
- Because otherwise they did the same as other research
- For large dk values the soft-max function will probably be pushed into regions with extreme small gradients
- For small dk values the soft-max function will probably be pushed into regions with extreme small gradients
- For small dk values the soft-max function will probably be pushed into regions with extreme large gradients
* `[ B ]`


---

**Q: What is the BLEU score.**
- The accuracy of machine-translation according to the dictionary.
- How close the machine-translation is to a human translation.
- The distance from the predicted vector to the actual translation vector, similar to the L2 norm.
- How efficiently a RNN encodes knowledge.
* `[ B ]`


---

**Q: Why use self-attention instead of recurrent networks?**
- Self-attention has a shorter dependency chain
- Self-attention can uncover more features in the data
- Self-attention does data augmentation
- Self-attention has fewer parameters
* `[ A ]`


---

**Q: According to the author, which of the statements are true regarding the transformer?**
- For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. 
- For translation tasks, the Transformer takes longer to train than architectures based on recurrent or convolutional layers.
- For translation tasks, the Transformer always gives better result when compared to architectures based on recurrent or convlotional layers.
- None of the above.
* `[ A ]`


---

**Q: What is a side-benefit of self-attention?**
- Results are more interpretable.
- The complexity is reduced.
- The performance is increased.
- None of the above
* `[ A ]`


---

**Q: The biggest constraint in recurrent neural networks after the authors of paper 11 is...**
- ...sequential computation
- ...parallel computation
- ...low consumption of GPU power
- ...high consumption of GPU power
* `[ A ]`


---

**Q: In sequence transduction tasks, which of the following bottlenecks are encountered in recurrent neural network, causing sub-par performance as compared to other state of the art representation models?**
- Parallelism, as measured by the minimum number of sequential operations required
- Learning long-range dependencies in the network
- Vanishing and Exploding gradients
- All of above
* `[ D ]`


---

**Q: What is the change proposed by the Transformer in the architecture of sequence neural networks?**
- Adding a different way of computing attention using recurrent neural networks
- Get rid of the recurrence and using only attention mechanisms
- Using convolutions instead of recurrence to model transduction tasks
- None of the above
* `[ B ]`


---

**Q: What is NOT a reason to use self attention?**
- Total computational complexity per layer
- Path length between long-range dependencies in the network
- The amount of computation that can
be parallelized
- Intuitiveness for the programmer
* `[ D ]`


---

**Q: What is the purpose of “self-attention" in the Transformer architecture?**
- It is used to compute a vector representation of an input sequence
- It is used to compute a vector representation of the output sequence
- it is used to speed up the gradient descent algorithm 
- None of the above 
* `[ A ]`


---

**Q: -**
- -
- -
- --
- -
* `[ A ]`


---

**Q: Which option below is an advance achieved by attention mechanism based model(such as Transformer), compared with generic RNN**
- More parallelization
- Faster model training 
- Better performance in longer sequence task
- All of the three
* `[ D ]`


---

**Q: In RNN the attention mechaninc is responsible for?**
- A transfer function ($z^-1$) between layers addressing how far back in time the networks remembers
- The networks ability to map inputs to an in-between encoder-decoder "C" vector
- Mapping a query and a set of key-value pairs to an output. All being vectors.
- A outputs field of vision to the neighboring (past or present) outputs
* `[ C ]`


---

**Q: Which of these statements is/are true?
1: Methods using attention are, like RNN's, especially suitable for processing sequential data
2: The Transformer model that is proposed by the authors outperforms classical approaches with respect to computational time**
- Only statement 1 is true
- Only statement 2 is true
- Both statements are true
- Both statements are false
* `[ C ]`


---

**Q: What is the advantage of using an attention mechanism eschewing recurrence?**
- global dependency between input and output
- allows for significantly more parallelisation
- can reach a new state of the art in translation quality
- all of the above
* `[ D ]`


---

**Q: What is(are) the advantage(s) of using the Transformer architecture (with attention) for text translation applications over the use of RNNs or CNNs?**
- It can associate parts of a sequence that are semantically related but separated in the sequence by shortening the operations needed to connect/relate them. 
- It mimics consciousness and gives machines the ability to reason like humans. 
- It is an special case of Highway Networks that can control information flow in the network.
- Both A and C.
* `[ A ]`


---

**Q: Which of the following statements about the attentional approach of the Transformer is false**
- The sequence transformation from input to output does not make use of any recursive schemes
- The training of the Transformer requires multi-step backpropagation
- The decoder identifies different components of information from the source sentence. Each of these components is represented by a feature (key) and a relevant value.
- The decoder, based on the content of the generated part of the output sentence, makes queries and identifies which key aligns with a query. The value that corresponds to the key helps the decoder to detect the most probable next output of the target sentence
* `[ B ]`


---

**Q: What statement is correct
1) Self-attention is a mechanism that computes a representation of a sequences by relating different position in the sequence
2) Self-attention is not likely to have weight convergence or divergence, since everything is connect by fixed number of layers**
- 1 & 2 are correct
- 1 is correct
- 2 is correct
- none are correct
* `[ A ]`


---

**Q: Which of the following is wrong**
- Transformer can replace recurrent layers in encoder-decoder system.
- Transformer can be trained significantly faster than architectures based
on recurrent or convolutional layers
- encoder -decoder architecture can not control the input size in neural network
- Attention mechanism allows modeling of dependencies without regard to their distance in the input or output sequences
* `[ C ]`


---

