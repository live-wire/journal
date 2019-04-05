# Questions from `rnns` :robot: 

**Q: RNNs are usually not used for which of the following?**
- Character recognition.
- Image classification.
- Machine translation.
- Emotion classification.
* `[ B ]`


---

**Q: Which of the following is not a disadvantage of a recurrent neural network?**
- RNNs are not able to keep track of long-term dependencies.
- Large amount of training required.
- RNNs has poor performance.
- RNNs cannot be stacked into very deep models.
* `[ C ]`


---

**Q: Which of the following is an example of one-to-many problem?**
- classification of a CIFAR-10 image
- classifying a certain sentence with positive/negative sentiment
- captioning ad image with a descriptive sentence
- None of the above
* `[ C ]`


---

**Q: Which of the following is an issue with RNNs?**
- too many weights to learn, because there is a weight matrix for each step
- they cannot deal with variable sequence lengths
- training phase cannot be parallelized within a sequence
- none of the above
* `[ C ]`


---

**Q: What is not true when we talk about RNN’s?**
- They share the same weights across several time steps.
- They share some information on different time steps.
- Regardless of sequence length, the learned model always has the same input size.
- The transition function f needs new (updates) parameters at every time step.
* `[ D ]`


---

**Q: What kind of sequential data ‘transmission’ is video classification on the frame level?**
- one to one
- one to many
- many to one
- many to many
* `[ D ]`


---

**Q: Which of the following is NOT a correct example for the different types of sequential data?**
- One to one - Number recognition: image to multi-digit number
- One to many - Image captioning: image to sequence of words
- Many to one - Emotion classification: sequence of words to emotion
- Many to many - Machine translation: sequence of words to sequence of words
* `[ A ]`


---

**Q: Which of the following statements is FALSE?**
- RNN's use parameter sharing
- The problem of vanishing or exploding gradients can be dealt with using gated RNNs
- RNNs allow for the prediction of whole input sequences (bidirectional RNN)
- RNN stands for Residual Neural Network
* `[ D ]`


---

**Q: When RNNs are better then CNNs?**
- When we have grids of values
- When we have sequential data
- When we have different input size
- When we are not able to calculate the gradient of the loss
* `[ B ]`


---

**Q: What is the problem with long-term dependencies computing h(t) in a RNN?**
- The gradient could vanish
- The gradient could explode
- Might take a very long time to learn long-term dependencies
- All the previous answers
* `[ D ]`


---

**Q: Which of the following is an example for a many-to-one RNN model?**
- Translating a single word to French.
- Returning a caption for an image.
- Labeling a sequence of frames from a video.
- Translating a sentence to Dutch.
* `[ C ]`


---

**Q: Which of the following is NOT an advantage of RNNs?**
- Regardless of sequence length, the learned model always has the same input size.
- The same transition function f with the same parameters can be used at every time step.
- RNNs can generalize to sequence lengths unseen during training.
- The forward propagation can be highly parallelized. 
* `[ D ]`


---

**Q: Which statement is correct?**
- RNN is designed for sequence prediction problems
- RNN is designed for image processing 
- both A and B are correct
- None of A and B are correct
* `[ A ]`


---

**Q: Which statement is correct?**
- In One-to-Many RNN, An observation as input mapped to a sequence with multiple steps as an output.
- In many-to-Many RNN, An observation as input mapped to a sequence with multiple steps as an output.
- In many-to-one RNN, An observation as input mapped to a sequence with multiple steps as an output.
- none 
* `[ A ]`


---

**Q: The emotion classification task corresponds to which type of sequential data?**
- One to one.
- One to many.
- Many to one.
- Many to many.
* `[ C ]`


---

**Q: Which of the following statements is true about Recurrent Neural Networks?**
- They are ideal for sequential data.
- The same transition function with the same parameters can be used at every time step.
- They are trained using forward and backward propagation.
- All of the above.
* `[ D ]`


---

**Q: Consider following two statements:

\begin{enumerate}
    \item Recurrent Neural Networks are mainly used for grid applications.
    \item Convolutional Neural Networks are mainly used for speech recognition applications.
\end{enumerate}

Which one(s) are true or false?**
- Both true.
- 1 true and 2 false.
- 1 false and 2 true.
- Both false.
* `[ D ]`


---

**Q: One of the fundamental parts of a RNN is the recurrence formula:

\begin{equation}
    h_{t} = f_{w}(h_{t-1},x_{t})
\end{equation}

Where does the $h$ stands for in the formula?**
- The state.
- The feedforward function.
- The back propagation function.
- The control function.
* `[ A ]`


---

**Q: A Recurrent Neural Network (RNN) does not**
- share its parameters across time steps.
- share its resilience to noise across positions in time and sequence lengths.
- have stable gradients (in a traditional mode of operation).
- generalize to data points with sequence lengths different from training data points.
* `[ C ]`


---

**Q: Which of the following statements about RNNs does not unequivocally hold?**
- Long-term dependencies in traditional RNNs are difficult to learn.
- The forward pass of a traditional RNN is necessarily sequential.
- Gated RNNs help transfer information from the distance past to the present.
- States computed during the forward pass in a traditional RNN can be discarded after the forward pass.
* `[ D ]`


---

**Q: Decide which of the following statements is true, regarding recurrent networks.
Statement 1: In recurrent networks, the size of the learned model is not dependent of the sequence length because the history of states is normalized to a given length.
Statement 2: A single transition function f can be used at every timestep, but the parameter values must change.**
- 1: true, 2: true
- 1: true, 2: false
- 1: false, 2: true
- 1: false, 2: false
* `[ D ]`


---

**Q: Choose the correct statement about recurrent neural networks (RNNs):**
- Bidirectional RNNs use information from the future to compute results.
- RNNs can only map an input sequence to an output sequence of the same length.
- The learned model provides separate functions that operate on specific timesteps in an input sequence.
- Gated RNNs speed up learning models with long-term data dependencies by using addressable memory block.
* `[ A ]`


---

**Q: Which of these statements about recurrent networks are true?
I: The same transition function can be used at every time step, as long as parameters are changed
II: The input size is constant, regardless of sequence length**
- Both I and II are true
- Only I is true
- Only II is true
- Both I and II are false
* `[ C ]`


---

**Q: Which of the following gates is not found in a LSTM**
- Forget gate
- Output gate
- Remember gate
- All are part of a LSTM
* `[ C ]`


---

**Q: What problems are well suited for Recurrent Neural Networks (RNN)s?**
- Problems where solving sub problems is important.
- Problems where information becomes relevant again after a long time.
- Problems where information is immediately reused.
- None of the above.
* `[ C ]`


---

**Q: What is an encoder-decoder or sequence-to-sequence useful for?**
- For dealing with differences in length between input and output.
- To translate between different datatypes.
- To translate solutions from to other problems.
- None of the above.
* `[ A ]`


---

**Q: What contrast between images and speech makes convolutional networks a good method for handling for the former but not so good for the latter?**
- Grid of values vs. sequence of values
- Pixels vs. text
- Constant color strength vs. different voice volumes
- Convolutional networks also work well on speech
* `[ A ]`


---

**Q: Which of the following answers in not part of a recurrence formula for a RNN?**
- A function f_W
- Lenth of the sequence s_{1:t}
- The old state h_{t-1}
- An input vector x_t
* `[ B ]`


---

**Q: What is meant with parameter sharing with regards to recurrent neural networks?**
- The parameters used for training and testing are the same
- The parameters for the weight matrix are the same at each timestep
- For each timestep in the network, the input data for all the timesteps is used
- None of the above
* `[ B ]`


---

**Q: What is the advantage of a Long Short Term Memory?**
- It learns when to remove old data
- It learns when to add new data
- It removes long-term dependencies on previous data inputs
- All of the Above
* `[ D ]`


---

**Q: What is one of the main advantages of recurrent neural networks?**
- They are able to perform backpropagation far faster than classical neural networks due to their shallowness
- In comparison to other neural networks they in general require little memory to run
- RNN's are effective when the input is sequential data 
- Due to the same transition function being used at every time step the model will never overfit
* `[ C ]`


---

**Q: Review the following two statements:
\begin{enumerate}
    \item An RNN learns a seperate model $g^(t)$ for each time step
    \item An RNN can generalize well to sequence lengths not received as input during the training phase
\end{enumerate}
Which of the two statements are true?**
- Statement 1 and 2 are true
- Statement 1 is true and statement 2 is false
- Statement 1 is false and statement 2 is true
- Statement 1 and 2 are false
* `[ C ]`


---

**Q: Which of the following is FALSE in topic of sequential data?**
- Character recognition is an example of one to one model
- Emotion classification from sentence is example of many to one model
- Machine translation is many to many model 
- Image captioning (image to sequence of words) is one to one model
* `[ D ]`


---

**Q: What are NOT the strengths of RNNs?**
- It learns only one model f which operates on all time steps and all sequence lengths
- Generalize to sequence lengths unseen during training
- Share statistical strengths across different sequence lengths and time positions.
- Use connectivity patterns between its neurons, inspired by the organization of the animal visual cortex
* `[ D ]`


---

**Q: Statement 1: RNN’s can use the same weights for all time steps. Statement 2: The input size depends on the sequence length.**
- both statements are true
- statement 1 is true, statement 2 is false
- statement 1 is false, statement 2 is true
- both statements are false
* `[ B ]`


---

**Q: LSTMs (Long Short Term Memory) contain which of the following gates: (1) input gate, (2) forget gate, (3) output gate? **
- 1 and 2
- 1 and 3
- 2 and 3
- 1, 2 and 3
* `[ D ]`


---

**Q: In what category does machine translation falls into?**
-  One to one
- One to many
- Many to one
- Many to many
* `[ D ]`


---

**Q: What is a strength of an RNN?**
- Learning a single model $f$ that operates on all timesteps and all sequence lengths
- Generalization to sequence lengths unseen during training 
- Sharing statistical strength across different sequence lengths and across different positions in time
- All of the above
* `[ D ]`


---

**Q: What is the strength of a Recurrent Neural Netwok (RNN) over a Convolution network?**
- RNNs Learn a model that operates on all time steps and all sequence lengths rather than a separate model for all possible time steps
- RNNs reduce redundant computation
- RNNs are faster than Convolution networks because of their simplicity
- None of the above
* `[ A ]`


---

**Q: What is the strength of a Recurrent Neural Netwok (RNN) over a Convolution network?**
- Sharing statistical strength across different sequence lengths and across different positions in time
- RNNs reduce redundant computation
- RNNs are faster than Convolution networks because of their simplicity
- None of the above
* `[ A ]`


---

**Q: What is true about a recurrent neural net when compared to a regular neural net?**
- It directly uses older training samples to predict the output
- It is a form of weight sharing
- Both of the above
- None of the above
* `[ B ]`


---

**Q: What is not a typical application for an RNN?**
- Generating a caption for an image.
- Translating text
- Object detection in an image
- Image to speech
* `[ C ]`


---

**Q: Which following is not true for recurrent neural network?**
- It is specialized for processing sequential data.
- It uses separate parameters for different input data.
- It applies the recurrence formula at each time step.
- It generalizes well to sequence lengths unseen during training.
* `[ B ]`


---

**Q: Which following is nor true for variants of RNN?**
- Bidirectional RNN is proposed based on the idea that the output of network not only depends on previous element but also future elements.
- Bidirectional RNN can be used to predict a missing word in a sequence.
- Long Short Term Memory(LSTM) solves the problem of exploding and vanishing gradients of typical RNN.
- Output gate in LSTM is used to determine what new information we want to store in the cell state.
* `[ D ]`


---

**Q: Which of the following systems (assume default variants of these systems) are not able to deal with variable length inputs?
- Encoder Decoder
- MLP
- CNN
- GRU**
- 2,4
- 1,4
- 2,3
- 3,4
* `[ C ]`


---

**Q: What can be said about training RNNs?**
-  It is generally expensive. The forward pass can be parallelized but the backward pass needs to be done sequentially.
- Memory consumption is low because we can throw away the computed states in the forward pass.
- RNN have a non-differentiable loss function. Making it hard to optimize them.
- None of the above
* `[ D ]`


---

**Q: When training an RNN the total loss function for a given sequence of x values paired with a sequence of y training values would be:**
- The sum of the losses over all the time steps;
- The sum of losses over the most recent time steps;
- The loss of the current time step;
- The sum of the losses of the current time step and the previous time step.
* `[ A ]`


---

**Q: The gradient of the loss function is calculating by using:**
- A forward propagation pass followed by a backward propagation pass;
- A backward propagation pass followed by a forward propagation pass;
- A forward propagation pass;
- A backward propagation pass.
* `[ A ]`


---

**Q: Say you have a Recurrent Neural Network that is trained to receive a name character-by-character and detect its language of origin. This would be an example of an RNN that is **
- many to one
- many to many
- one to one
- one to many
* `[ A ]`


---

**Q: One thing that distinguishes Recurrent Neural Networks from other network architectures is that**
- sequential input can be handled by introducing hidden states
- given some input, the output is computed through recurrence formulas instead of matrix multiplications
- training the network requires access to the previous training samples
- the gradients are calculated with a recurrence formula given by the backprop algorithm
* `[ A ]`


---

**Q: Which of the following is false:**
-  An RNN’s hidden state can in theory store information about an infinite number of observed time steps.
- When the number of time-steps in the input changes, the same learned transition function can be re-used.
- Internal states need to be stored for the backward pass, which can make RNNs memory expensive.
- RNNs can be trained in parallel by calculating the output from different starting time points.
* `[ D ]`


---

**Q: Which of the following is true:**
- A LSTM network, without operations to modify the internal state, functions exactly like a regular RNN.
- A LSTM network can memorize information about entities across multiple texts, with multiple sentences each, easily.
- Long Short Term Memory networks maintain two hidden states; one for short-term information and one for long-term information.
- The size of a hidden state for a LSTM network changes as more information gets added or removed.
* `[ A ]`


---

**Q: What is a strength of an RNN?**
- It learns multiple models f that operate on all time steps
and all sequence lengths rather than a separate g(t) for all
possible time steps
- The generalization to sequence lengths unseen during
training
- It supports many to one networks
- It always has a different input size
* `[ B ]`


---

**Q: Give an example of a network type for which only RNN's are used**
- Encoder-decoder RNN
- Convolutianal RNN
- Forward pass RNN
- One directional RNN
* `[ A ]`


---

**Q: What is NOT a strength of RNNs?**
- Generalization to sequence lengths unseen during training
- The same transition function with the same parameters can be used at every time step
- The learned model always has the same input size, since the sequence length is determined in terms of state transitions
- An RNN can access any historical piece of information individually
* `[ D ]`


---

**Q: What is the purpose of the cell state in an LSTM?**
- Sum up information of all cells and transport it directly to the output layer
- Store the complete state of one cell and pass it over to the next cell
- It carries information from cell to cell. Each cell can add and remove information from the cell state via gates 
- It keeps a complete history of all information from the previous cells.
* `[ C ]`


---

**Q: Why are the weights shared in a RNN?**
- Improve generalization of sequence length data
- Improve the feasibility of the computation
- Otherwise it would be just a regular feed forward network
- All of the above
* `[ D ]`


---

**Q: What is NOT true?**
- RNN suffers heavily from gradient vanishing/explosion
- LSTM are mainly constructed to solve the problem of A
- LSTM solved the gradient problems for the most part
- Cells are storing states while the gated decided on what is stored in this state
* `[ C ]`


---

**Q: Which of the following statements regarding RNNs is TRUE?**
- At a time step t, the output is a function depending only on the input of the same time step
- They perform the best with grid-like data
- The same function and the same set of parameters are used at every time step
- The architecture can be one-to-many, but not many-to-one
* `[ C ]`


---

**Q: What is one of the major problems of RNNs?**
- They can be used only with grid-like data
- The use of a uniqe function employed at each time step
- The problem of vanishing or exploding gradients
- None of the above
* `[ C ]`


---

**Q: Long Short Term Memory networks achieve memory of historical data by using, **
- Gated networks to maintain and protect cell state.
- Feedback network to decide which memory to protect and maintain.
- Residual networks to skip connections on trivial data.
- All of the above.
* `[ A ]`


---

**Q: What is the most necessary component of Neural network performing sequence modelling?**
- Feedback Loops.
- Back Propagation and Gradient Descent.
- Same input and input length.
- Causal connections
* `[ D ]`


---

**Q: What is NOT true about recurrent neural networks (RNNs)?**
- RNNs use the same function at every time step.
- Regardless of sequence length, the learned model always has the same input size.
- RNNs have difficulties with sequence lengths unseen during training.
- RNNs share the same weights across several time steps.
* `[ C ]`


---

**Q: What is NOT true about training recurrent neural networks (RNNs)?**
- The total loss for a given sequence of values paired with a sequence of training values would be the sum of the losses over all the time steps.
- RNNs are not expensive in time since the forward pass is sequential.
- The gradient of the loss is calculated using a forward propagation pass followed by a backward propagation pass and is also used for retraining the RNN.
- RNNs are expensive in memory since states computed in the forward pass must be stored until they are reused during the backward pass.
* `[ B ]`


---

**Q: What is the NOT benefit of general RNN in comparison with traditional neural networks, like CNN?**
- Regardless of the sequence length, the learned model always ha the same input size.
- Can remember (a large amount of)  their former inputs and affects the output.
- It saves resource usage, especially in terms of memory usage.
- All of the above are true benefits of RNN over traditional neural networks.
* `[ C ]`


---

**Q: What is a possible application for a many-to-many RNN?**
- Image classification.
- Image captioning which takes an image and outputs a sentence of words.
- Sentiment analysis where a given sentence is classified as expressing positive or negative sentiments.
- Machine translation where it reads a sentence i.e. english and ten outputs a sentence i.e. French.
* `[ D ]`


---

**Q: Which of the following statements on Recurrent Network Processes is incorrect?**
- An Recurrent Neural Network processes a sequence of vectors x by applying a recurrence formula at every time step: h_t = f_w(h_{t-1},x_t), with for h_t the new state, f_w some function with parameter W, h_{t-1} the old state, x_t the input vector at some time step. 
- An RNN learns a separate f^(t) for all possible time steps rather than a single model f that operates on all time steps and all sequence lengths.
- RNN has no generalization to sequence lengths during training. 
- The RNN method share statistical strength across different sequence lengths and across different positions in time. 
* `[ B ]`


---

**Q: Consider the following two statements about Recurrence Network Processes:
1.	Regardless of the sequence length, the learned model always has the same input size, because sequence length is specified in terms of state transitions rather than a variable-length history of states. 
2.	The same transition function f with the same parameters can be used at every time step. 
Which of the statements is true?**
- Both are true. 
- Both are false. 
- 1 is true and 2 is false. 
- 1 is false and 2 is true. 
* `[ A ]`


---

**Q: What recommends RNNs over conventional deep neural networks:**
- RNNs are less complex
- RNNs have a lower demand for memory
- An RNN does not require to learn a different model for each time step
- RNNs do not suffer from the exploding/vanishing gradients problem
* `[ C ]`


---

**Q: Which of the following is false:**
- There are RNN architectures that allow using future information to better estimate the current one
- Bidirectional RNNs are well suited for speech recognition due to their robustness over different pronunciations
- LSTMs allow discarding information that is not considered useful
- LSTMs use memory cells to store data
* `[ D ]`


---

**Q: Emotion classification is a type of sequential data that can be classified as:**
- One to one
- One to many
- Many to one
- Many to many
* `[ C ]`


---

**Q: Which of the following statements is not true about gated RNNs?**
- They are designed to remember information for long periods of time.
- They scale information from distant past to the present more efficiently than RNNs.
- They transfer information from distant past to the present more efficiently than RNNs.
- They operate on a single scale.
* `[ D ]`


---

**Q: Which of the following are major advantages of recurrence in RNNs:

\begin{enumerate}
\item Regardless of sequence length, the learned model always has the same input size.
\item Different transition functions can be used at different time steps.
\end{enumerate}**
- Only 1. 
- Only 2.
- Both 1 and 2.
- Neither
* `[ A ]`


---

**Q: Which of the following are true about training an RNN compared to regular neural networks:

\begin{enumerate}
\item It is computationally expensive to compute the forward pass.  
\item In order to perform the backward pass a lot of memory is necessary. 
\end{enumerate}**
- Only 1.
- Only 2.
- Both 1 and 2.
- Neither
* `[ C ]`


---

**Q: How do the parameters of function f change between recurrent layers in an RNN?**
- They generally tend to increase
- They generally tend to decrease
- The parameters are re-initialized for every recurrent iteration
- The parameters stay the same for every time step
* `[ D ]`


---

**Q: What is the main benefit of a gated RNN?**
- It is designed to 'remember' information for long periods of time
- It filters information more effectively
- It trains faster than a regular RNN
- It can better deal with long input sequences
* `[ A ]`


---

**Q: What is NOT the effective solution to solve the long-term dependecies?**
- gated RNNs
- use Bidirectional RNN
- Designed the model to remember information for long periods of time
- Scales and transfers information from the distant past to the present more efficiently
* `[ B ]`


---

**Q: How many extra layers does LSTM RNN compared with original RNN**
- 2
- 3
- 4
- 5
* `[ C ]`


---

**Q: Statement 1: The same transition function with the the same parameters can not always be used at every time step. 
Statement 2: Regardless of sequence length, the learned model always has the same input size, because sequence length is specified in terms of state transitions rather than variable-length history of states. **
- Both statements are true
- Statement 1 is true
- Statement 2 is true
- None of the statements are true
* `[ C ]`


---

**Q: Statement 1: An RNN can learn a single model f that operates on all time steps and all sequence lengths rather than a separate g(t) for all possible time steps. 
Statement 2: An RNN can share statistical strength across different sequence lengths and accross different positions in time. **
- Both statements are true
- Statement 1 is true
- Statement 2 is true
- None of the statements are true
* `[ A ]`


---

**Q: Which of these is a disadvantage present in the use of RNNs?**
- The sharing of parameters across time steps.
- They are not parallelizable.
- They can not be made to access "future" information.
- The memory cost of storing previous hidden states.
* `[ B ]`


---

**Q: Choose the correct option regarding the forget gate in an LSTM unit.**
- Based on the previous cell state and the previous hidden state decides whether to forget the information in the input data.
- Based on the previous cell state and the previous hidden state decides whether to forget the information in the input data.
- Based on the previous cell state and the input data decides whether to forget the information of the previous hidden state.
- Based on the previous hidden state and the input data decides whether to forget the information of the previous cell state.
* `[ D ]`


---

**Q: What is not typical for an RNN?**
- The same weights are used at every recurrent step
- Regardless of sequence length, the learned model always has the same input size
- Sharing statistical strength across different sequence lengths
- The output must be smaller than the input
* `[ D ]`


---

**Q: A Long Short Term Memory architecture has several interacting layers. Which task is not represented by such a layer?**
- Forgetting information
- Convolving information
- Adding information
- Outputting information
* `[ B ]`


---

**Q: What is the problem when using a CNN?**
- CNN's don’t have enough weights to capture context in an image
- CNN's are good to be deployed on grids. However, speech to text is an sequential task at which it has bad perfomance
- CNN's don’t have enough weights to capture detail in an image
- CNN's overfit often
* `[ B ]`


---

**Q: What is a disadvantage of using an RNN?**
- Regardless of sequence length the learned model will always have same input size
- Same transition function can be used at every time step
- It shares statistical strength across different sequence length and across different positions in time
- An RNN has more convolutional layers such that it is better at image classification than a CNN
* `[ D ]`


---

**Q: What is not a strength of an RNN? **
- Learns a single model for all possible time steps
- Generalization to sequence lengths unseen during training
- Sharing statistical strength across different sequence lengths
- Not a lot of memory is needed
* `[ B ]`


---

**Q: What is an example of a one to many network?**
- character recognition
- image captioning
- motion classification
- machine translation
* `[ B ]`


---

**Q: There are strengths of recurrent neural networks, which of the following is not?**
- Generalization to unseen sequence lengths
- Single model that operates on all time steps
- Sharing of stastitical strength
- Fast learning of long-term dependencies
* `[ D ]`


---

**Q: Which of the following is/are correct?

1. A bidirectional RNN is useful for speech recognition as a sound can be dependent of future sounds.
2. An encoder-decoder architecture is useful when input and output sequences differ in size.**
- Only 1 is correct
- Only 2 is correct
- Both are correct
- Both are incorrect
* `[ C ]`


---

**Q: What does change over time in an RNN**
- The input value
- The transition function
- The model parameters
- The input dimension
* `[ A ]`


---

**Q: Why is training a RNN generally more expensive than training a CNN**
- Due to the fact that dimension might change over time
- Due to the fact that not every step has an output
- Due to the fact that he forward step has to be processed sequentially 
- Do to the fact that training sets are bigger
* `[ C ]`


---

**Q: Which of the following statements is not a true about a RNN?**
- Unlike a traditional deep neural network, which uses different parameters at each layer, a RNN shares the same parameters across all steps.
- The gradient of the loss is calculated using a forward propagation pass followed by a backward propagation pass and used for retraining the RNN.
- RNNs are the best models for sequential data.
- It is a feedforward neural network.
* `[ D ]`


---

**Q: Which of the following is not a RNN architecture?**
- Long Short-Term Memory Networks (LSTM)
- Gated Recurrent Unit Neural Networks (GRU)
- Neural Turing Machines (NTM)
- Deep Belief Network (DBN) 
* `[ D ]`


---

**Q: Which of the following does the task of character recognition fall into?**
- One to one
- One to many
- Many to one
- Many to many
* `[ A ]`


---

**Q: Which of the following does machine translation fall into?**
- One to one
- One to many
- Many to one
- Many to many
* `[ D ]`


---

**Q: Which is the strength of an RNN?**
- Learn a single model
- Generalization to the sequence length
- Sharing statistical strength
- All of the above
* `[ D ]`


---

**Q: Which type of gated RNNs is used for storing information in the cell state?**
- Forget gate
- Input gate
- Output gate 
- Gated recurrent units
* `[ B ]`


---

**Q: Which of the following is not true about the basic RNN networks ?**
- In theory, RNNs are absolutely capable of handling long-term dependencies. However, in practice, RNNs aren't able to learn long-term dependencies. 
-  A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor. 
- RNNs is the idea that they are able to connect previous information to present task. 
-  In some cases, where the gap between the relevant information and the place that it's needed is small, RNNs can learn to use the past information. 
* `[ C ]`


---

**Q: Which of the following is not true about the structure of  LSTM networks ?**
- An LSTM has three gates to protect and control the cell states. The gates are composed of a sigmoid neural net layer and a pointwise multiplication operation. The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. 
- The new cell state for one LSTM is the sum of the old cell state and the the new candidate values. The tanh layer called the "input gate layer" decides which values we'll update. 
-  LSTMs are explicitly designed to avoid the long-term dependency problem. 
-   A LSTM has four neural network layer interacting in a very special way. 
* `[ B ]`


---

**Q: Recurrent neural networks specialized for?**
- Images
- Text and speech
- Speech
- Optimazion
* `[ B ]`


---

**Q: What is true: about Recurrent Neural Network**
- Hides state
- Convolution based
- Extremely easy to train
- Great for images
* `[ A ]`


---

**Q: Which of these is TRUE about an RNN**
- The learned model always has the same input size
- The sequence length is specified in terms of state transitions
- The same transition function with the same parameters can be used at every time step
- All of the above
* `[ D ]`


---

**Q: What is the problem with long term dependencies of RNN**
- Vanishing gradients
- Exploding gradients
- Expensive in terms of time
- All of the above
* `[ D ]`


---

**Q: What kind of RNN network will be used in image captioning?**
- Many to One.
-  One to Many.
- One to One.
- Many to Many.
* `[ B ]`


---

**Q: Which network is most suitable for predicting the 7th character in the input "Networ" (which is a 6 character word)?**
- Fully connected.
-  Convolutional neural network.
- Recurrent neural network.
- Perceptron
* `[ C ]`


---

**Q: Which is a solution to learnlong-term dependencies?**
- LSTM
- LSAM
- LSUM
- LSZM
* `[ A ]`


---

**Q: Why do we use gated RNN?**
- To remember information for long periods of time
- To remember information for short periods of time
- To remember weight for long periods of time
- To remember weight for short periods of time
* `[ A ]`


---

**Q: If you're making a deep learning model for 1) Transcribing speech into text and 2) Performing image recognition, which artificial neural network would be more beneficial for each task?**
- 1- Recurrent neural network 2- recurrent neural network
- 1- Convolutional neural network 2- recurrent neural network
- 1- Recurrent neural network 2- Convolutional neural network
- 1- Convolutional neural network 2- Convolutional neural network
* `[ C ]`


---

**Q: If you're training a recurrent neural network based model with a training set "y". How would you expect the total loss to behave when changing the training set size?**
- If your training set size increases, the total loss would decrease.
- If your training set size increases, the total loss would remain constant.
- If your training set size decreases, the total loss would decrease.
- If your training set size decreases, the total loss would remain increase.
* `[ C ]`


---

**Q: The network that involves backward links from output to the input and hidden layers is called:**
- Recurrent neural network
- Convolutional neural network
- Multi-layered perceptron
- None of the above
* `[ A ]`


---

**Q: When is it better to use an RNN over a feed-forward network?**
- When the sequence of the input matters: RNNs allows you to capture temporal relationships.
- When the sequence of the input does not matter: RNNs are optimized to not capture temporal relationships.
- When the input data contains a high amount of noise.
- None of the above.
* `[ A ]`


---

**Q: What is not necessarily a strength of an RNN?**
- Learns a single model f that operates on all time steps and all sequence lengths rather than a separate g(t) for all possible time steps 
- Generalization to sequence lengths unseen during training
- No compression is needed
- Sharing statistical strength across different sequence lengths and across different positions in time
* `[ C ]`


---

**Q: What kind of information processing is difficult for a deep net?**
- Differentiating between different animals
- Estimating the distribution of a dataset
- Finding a different representation for a dataset
- Listing specific facts about an object; 'a cat is an animal'
* `[ D ]`


---

**Q: Which of the following is an example of many to one?**
- sequence of words to emotion
- sequence of words to sequence of words
- video classification on the frame level
- image to sequence of words
* `[ A ]`


---

**Q: What are, according to the slides,  the two major advantages of recurrence? **
- Regardless of sequence length, the learned model always has the same input size \& Recurrent networks are faster
- Recurrent networks are more memory efficient & Recurrent networks are easier to tune.
- Regardless of sequence length, the learned model always has the same input size \& The same transition function f with the same parameters can be used at every time step
- Recurrent networks are easier to tune & Architecture selection is less complicated for RNN's
* `[ C ]`


---

**Q: What does happens with LSTMs?**
- Exploding gradients
- Removing information from the input or the history
- Vanishing gradients
- It is not able to do many-to-many tasks
* `[ B ]`


---

**Q: What is a major advantage of regular RNNs?**
- Regardless of sequence length, the learned model always has the same input size
- Good at processing large amounts of info to predict implicit information
- Regularisation is not necessary
- Invariant to temporal translations
* `[ A ]`


---

**Q: What is the main problem of “vanilla” RNNs as opposed to their most used variant, LSTM?**
- They struggle to learn long term dependencies. 
- They cannot handle many-to-many tasks.
- They cannot be combined with other architectures such as CNNs.
- They are way more expensive in terms of computation and memory usage.
* `[ A ]`


---

**Q: In the LSTM, there are gates by which information must go through. These are composed out of a sigmoid neural net layer and a point-wise multiplication operation. What is the function of these gates?**
- They combine information streams coming from previous modules. 
- They normalize the input to avoid the internal covariate shift.
- They describe how much of each component of the input should be let through. 
- They prepare the input to be stored in the learned data store (attention mechanism).
* `[ C ]`


---

**Q: Which of the following are reasons for the high cost of training RNNs (in time and memory)?
I: The forward pass is sequential
II: Architectures must store parameters for the computation of multiple items in an input sequence
III: States must be saved during the forward pass so that it can be used during the backward pass**
- I & II
- II & III
- I & III
- I, II, & III
* `[ C ]`


---

**Q: Which of the following input/output relationships has no meaningful applications in RNNs?**
- One-to-one
- One-to-many
- Many-to-one
- Many-to-many
* `[ A ]`


---

**Q: What is the target application of RNNs?**
- Sequential data processing
- Image processing.
- Unsupervised learning.
- 1-to-1 mapping.
* `[ A ]`


---

**Q: What is the encoder-decoder architecture for in RNNs?**
- To make it able to work with bi-directional sequences.
- Prediction of variable length sequences.
- Compression of input and output data.
- Normalizing inputs.
* `[ B ]`


---

**Q: Which of the following features is the most relevant when speaking of recurrent neural networks?**
- They are based on a symmetric kernel, that allows for faster image recognition
- They can have multiple outputs based on just one input, and viceversa 
- They work with a fuction characterized by the same set of parameters at each time step
- They can work with a lot of distinct hidden layers
* `[ C ]`


---

**Q: What is a gated RNN NOT able to do?**
- Selectively forget some informations coming from its past time steps
- Remember informations for longer time then a classical RNN
- Transferring informations from the past to the present very efficiently
- Exploit its characteristical encoder-decoder structure to compress relevant information over time
* `[ D ]`


---

**Q: What problem of 'vanilla' RNNs do LSTMs try to address?**
- Long term dependencies are hard to handle due to the vanishing gradients problem
- Long term dependencies are hard to handle due to the activation function used in standard RNNs
- Short term dependencies are not learned, being replaced with long term dependencies
- Short term dependencies are hard to handle due to the vanishing gradients problem
* `[ A ]`


---

**Q: Which of the following is NOT an advantage of recurrence in RNNs?**
- Because the network is defined in terms of recurrence, the same transition function can be used at every timestep. This requires just one set of shared parameters.
- Because the network is defined in terms of recurrence, input sequences can be arbitrarily long, given that the model works on the basis of state transitions, and not the entire history of states
- Because the network is defined in terms of recurrence, the architecture does not suffer from the vanishing gradients problem
- Because the network is defined in terms of recurrence, the architecture needs less memory at vallidation time, since we do not have to load in the entire sequence but instead load the elements in one by one
* `[ C ]`


---

**Q: Which of the following matchings regarding the different types of RNN input/output relationships and examples is false?**
- One to many  - image to sequence of words
- Many to one - Sequence of words to emotions 
- Many to many - Sequence of words to sequence of words
- One to one - Video classification
* `[ D ]`


---

**Q: Which of the following is false regarding RNNs?**
- The gradient of loss is calculated using both forward and backward propagation
- The same transition function f with the same parameters can be used at every time step
- Bidirectional RNNs are useful in speech recognition where the
pronunciation of a sound depends on the future sounds
- Different weight matrices are used in RNNs
* `[ D ]`


---

**Q: Which of these is not a strength of an RNN?**
- Can learn a single model that operates on all time steps and sequence lengths rather than on a separate function for all possible time steps
- An RNN can generalize to sequence lengths that were not seen during training
- An RNN shares statistical strength across different sequence lengths and across different positions in time
- An RNN does not have to store states computed in the forward pass
* `[ D ]`


---

**Q: Which of these is statements concerning RNNs is incorrect?**
- An RNN processes a sequence of vectors by applying a recurrence formula at    every time step
- Regardless of sequence length, the learned model always has the same input size
- Large parts of the forward pass of an RNN can be done in parallel
- The same transition function with the same parameters can be used at every time step
* `[ C ]`


---

**Q: Which of the following is WRONG?**
- RNN can only take a fixed number of sequence as input.
- The hidden layers (recurrence) in RNNs are able to memorize past information.
- Memorizing all of the past information is expensive in computation as well as in memory for RNNs.
- RNNs use shared parameters across the network.
* `[ A ]`


---

**Q: Which of the following is WRONG?**
- Long Short Term Memory (LSTM) is able to remember history information for a long period of time.
- LSTM is also capable of forgetting irrelevant information by using gates.
- LSTM can also add information to inputs.
- All of above.
* `[ D ]`


---

**Q: How is recurrence used in a neural network**
- Estimated outputs are fed back into the input
- The previous state is used to generate the current state
- previous inputs are used to generate the current state
- None of the above
* `[ B ]`


---

**Q: Which of the following statements is true:
I. In an RNN, the same transition function wit hthe same parameters can be used at every time step.
II. Training an RNN is fairly inexpensive due to the high amount of computations that can be parallelized**
- I. is true
- II. is true
- I and II are both correct
- I and II are both incorrect
* `[ A ]`


---

**Q: How does recurrent neural networks (RNN) share parameters?**
- Through application of the same convolution kernel at each time step
- By reusing the same weight matrix at every time step
- Each member of the output is produced using the same update rule applied to the previous outputs
- B and C are correct
* `[ D ]`


---

**Q: What is a possible useful case for bidirectional recurrent neural networks?**
- Automatic speech recognition, where the pronunciation of a sound can be dependent on future sounds due to coarticulation
- Question answering, where the input and output sequences in the training set are generally not of the same length
- Machine translation, where a language may not have a standardized written text format
- None of the above
* `[ A ]`


---

**Q: What is FALSE about Recurrent Neural Network(RNN) ?**
- in emotion classification is used a many-to-one architecture
- in each time step the parameters used in the input and in the hidden layer are the same
- in any time step in the hidden layer is applied the same function f
- usually is used tanh as activation function and softmax as output function
* `[ B ]`


---

**Q: What is TRUE about LSTM network (Long Short Term Memory) ?**
- it is composed by 2 gates: update and forget gate
- an attention mechanism is applied in the sequence output
- each unit in the LSTM network is called Gated recurrent unit (GRU)
- it can solve the vanishing/exploding gradient problem
* `[ D ]`


---

**Q: Consider a basic, uni-directional Recurrent Neural Network. Which values does the RNN reuse when making predictions on samples later in the sequence given the previous elements of the sequence?**
- The real input value(s) of the previous element in the sequence
- The activation values of the network for the previous element
- The real output value for the previous element
- The predicted class value for the previous element
* `[ B ]`


---

**Q: How do you deal with vanishing gradients in Recurrent Neural Networks?**
- Batch Normalisation
- Feature-wise normalisation of input vector
- Gated Recurrent Units
- There is no vanishing graident issue with RNNs
* `[ C ]`


---

**Q: What is an advantage and disadvantage of a Recurrent neural network regarding sequential data?**
- A RNN always uses the same input size, regardless of the sequence length. However, it has trouble remembering information because of exploding or vanishing gradients
- A RNN always uses the same input size, regardless of the sequence length. 
However, it is computational very expensive when the sequence length is to long
- A RNN uses the same transition function for each time step. However, the parameters of the transition function depends on the last time step it made
- A RNN uses the same transition function for each time step. However, it is computational very expensive when the sequence length is to long
* `[ A ]`


---

**Q: You are given a task to create a neural network which should do machine translations. What would be the most suitable architecture given the following 4 options?**
- A Feed-Forward Neural Network
- A Convolutional Neural Network
- A Bidirectional Recurrent Neural Network
- An Encoder-Decoder Recurrent Neural Network
* `[ D ]`


---

**Q: Consider a one to many architecture for sequential data. What could be an example of an application domain considering this setting?**
- Character recognition
- Machine translation
- Image captioning
- Emotion classification
* `[ C ]`


---

**Q: When it comes to the strong points of Residual Neural Networks (RNNs), which of the following statements hold?**
- RNNs learn a single model that operates on all time steps and all sequence lengths.
- RNNs can generalize to sequence lengths unseen during training.
- RNNs can share statistical strength across different sequence lengths and positions in time.
- All of the above.
* `[ D ]`


---

**Q: What's a downside of the recurrent architecture of Recurrent Neural Networks?**
- The entire history of inputs needs to be considered for the current hidden state.
- For each timestep, the parameters of the network need to be computed.
- Vanishing gradients occur because of the various, different, activation functions used at different timesteps.
- None of the above.
* `[ D ]`


---

**Q: Which of the following is a downside to training Recurrent Neural Networks?**
- The forward pass is expensive: it is necessarily sequential.
- The backward pass is expensive: all intermediate states must be stored until they are reused during the backward pass.
- Both A and B are correct.
- None of the above.
* `[ C ]`


---

**Q: What is true about RNNs?**
- Regarding a many-to-many input-output pattern, the total loss of the input sequence is the average of all the losses over all time steps
- The forward propagation pass can be parallelized
- The order of the input (x(1),…, x(T)) matters e.g. the output of input (x(1),…, x(T)) differs significantly from the output of (x(T),…, x(1))
- Recurrent and Recursive Neural Networks are not the same
* `[ D ]`


---

**Q: What is true about Recurrent Neural Networks**
- A bidirectional RNN demanding 2D inputs consists of four sub-RNNs
- A regular RNN can handle differences in length between input and output
- A problem of RNNs can be vanishing or exploding gradients 
- Training a RNN is computational expensive as there are T different (hidden-to-hidden) weight matrixes, with T being the total number of time steps
* `[ D ]`


---

**Q: Which of the following sequential data classifications would be preferable in video classification on frame level?**
- Many to one
- One to one
- One to Many
- Many to Many
* `[ D ]`


---

**Q: Which of the following properties is not a property for RNNs?**
- RNNs learn a single model that operates on all time steps
- RNNs computation times are sequential and thus slow.
- RNNs don’t require a lot of memory
- RNNs can be altered to make use of future information by using backpropagation in between layers. 
* `[ C ]`


---

**Q: What is not a strength of a recurrent neural network (RNN)?**
- Generalization to sequence lengths unseen during training.
- Learn a single model $f$ that operates on all time steps and all sequence lengths rather than a separate $g^{(t)}$ for all possible time steps.
- Sharing statistical strength across different sequence lengths and across different positions in time.
- Long-term dependencies can be learned easily
* `[ D ]`


---

**Q: What is not a RNN architecture?**
- Encoder-decoder
- Sequence-to-sequence
- Multidirectional RNN
- Bidirectional RNN
* `[ C ]`


---

**Q: What is the benefit of encoder**
- can be pretrained
- challange the hidden layers
- None of the above
-  All of the above 
* `[ A ]`


---

**Q: What is the problem with these long term dependecies**
- weights can be smaller and smaller
- on exploding gradients (weights are getting to big and explodes)
- none of the above
- all of the above 
* `[ D ]`


---

**Q: Which of the following statements about the BENEFITS of RNNs is FALSE?**
- Learns a single model $f$ that operates on all time step and all sequence lengths rather than a separate $g^{(t)}$ for all possible time steps.
- Can perform convolutions in parallel.
- Can generalize to sequence lengths unseen during training.
- Shares statistical strength across different sequence lengths and across different positions in time.
* `[ B ]`


---

**Q: Which of the following is incorrect?**
- Training an RNN is expensive In memory: states computed in the forward pass must be stored until they are reused during the backward pass.
- Long-term dependencies introduce the problem of vanishing or exploding gradients.
- The repeating module in a Long Short Term Memory (LSTMs) contains a dual (often tanh and sigmoid) layer.
- Training an RNN is expensive in time: forward pass is necessarily sequential.
* `[ C ]`


---

**Q: Which of the following is a false statement?**
- RNNs can take in an input sequence of variable length
- RNNs can use the same function f, that updates your hidden-state - h_t using input x_t for every time step
- A function f, shall have different parameters for every time-step of the input sequence
- None
* `[ C ]`


---

**Q: Which of these is false in context of gated RNNs?**
- They are designed to remember information for long periods of time
- They were designed to optimized the gradient issues that a RNN can face
- Gated RNNs focus on what information should be discarded so that gradient does not explode
- None
* `[ C ]`


---

**Q: For a Recurrent Neural Network (RNN) you have the following recurrence in the function: $s^{(t)} = f(s^{(t-1)}; \theta) $. What do you need to to do in order to have a $s^{(t)}$ without recurrence?**
- You cannot do anything as this is a recurrent function.
- You take for example t = 3 and unfold the equation like this: $s^{(3)} = f(s^{(2)}; \theta) = f((s^{(1)}; \theta);\theta) $
- This will only without recurrence if you take t = 1
- None of the above answers is correct
* `[ B ]`


---

**Q: What is the main difference in using Recurrent neural networks (RNN) and Convolutional neural networks (CNN) in terms of where they are used for?**
- CNN and RNN are used for he same processing such as sequential data or grid of values both CNN and RNN are specialized for this
- CNN are mostly specialized for processing a grid of values and RNN are Specialized for processing sequential data
- CNN are Specialized for processing sequential data and RNN are mostly specialized for processing a grid of values
-  None of the above answers is correc
* `[ B ]`


---

**Q: Which statement on RNNs is not correct?**
- An RNN learns a single model that operates on all timesteps and all sequence lengths rather than a seperate model for each timestep
- An RNN generalizes to a sequence length that is unseen during training
- An RNN specifies its sequence length in terms of the history of states
- An RNN shares statistical strength across different sequence lengths and across different positions in time
* `[ C ]`


---

**Q: Which of the following statements on the output gate of LSTMs is not correct**
- The tanh function ensures values between -1 and +1
- The forget gate that is part of the output gate "forgets" a small subset of the input
- The sigmoid layer is used to decide which parts of the cell state are going to be output to the next layer
- The output of the sigmoid gate is multiplied with the tanh output
* `[ B ]`


---

**Q: How does a RNN allow for input sequences of arbitrary length?**
- A RNN keeps a hidden state vector that is propagated through time, and essentially keeps a summary of the input it has seen so far. Each time new input is seen, it's information gets added to the state vector.
- A RNN accepts vectors of arbitrary length as input
- A RNN can convolve it's weight matrix over inputs of arbitrary size.
- This is not true, RNNs have a fixed input size.
* `[ A ]`


---

**Q: What is the downside of keeping track of state the way a standard RNN does?**
- You can only fit so much information in a fixed size vector, keeping track of long term dependencies is difficult for a standard RNN.
- It takes a lot of memory to keep track of all the state vectors during training.
- Because of the temporal dependency of the next step on the previous step it is impossible to parallelize calculations.
- All of the above.
* `[ D ]`


---

**Q: Unfolding RNN into computational graphs has the following advantage:**
- sharing statistical strengths across different sequence lengths and across different positions in time.
- regardless of the sequence length, the learned model always has the same input size
- the use of specialized transitions functions at every time step
- all of the above
* `[ B ]`


---

**Q: What is the alternative for training a RNN that maps an input sequence to a corrresponding output sequence?**
- Back-propagation through time
- teacher forcing
- RNN with directional graphs 
- none of the above
* `[ B ]`


---

**Q: What is the reason for using RNN for machine translation?**
- It is always more powerful than Convolutional Neural Network
- It works well when the input and the output should be sequencial data
- It works well when the input and the output have the same length as seen during the training, but otherwise RNN is nor suitable
- All the variants
* `[ B ]`


---

**Q: Why is it important to deal with vanising gradients in RNNS?**
- Bacause we should capture long-term dependecies, for exaple, in text, which is impossible with vanishing gradient
- It speeds up the training
- Both variants
- None of the variants
* `[ A ]`


---

**Q: What is one of the major advantage of recurrence in RNN?**
- The same transition function f with the same parameters can be used at every time step.
- The similiar transition function f with the same parameters can be used at every time step.
- The different transition function f with the same parameters can be used at every time step.
- None of the above.
* `[ A ]`


---

**Q: What is NOT strenght of RNN?**
- Learns a single model f that operates on all time steps and all sequence lengths rather than a separate g(t) for all possible time steps.
- Sharing statistical strength across different sequence lengths and across different positions in time.
- Parallelize multiple sequence lengths on one time step.
- Generalization to sequence lengths unseen during training.
* `[ C ]`


---

**Q: Consider the following statements:
a) RNNs suffer from vanishing gradient problem because of the involvement of chained matrix multiplication
b) RNNs are not expensive in both time and memory as it learns only a single model that operates on all time steps and sequence lengths
Choose the correct option**
- Statement (a) is correct; Statement (b) is wrong
- Statement (a) is wrong; Statement (b) is correct
- Both statements are wrong
- Both statement are correct
* `[ A ]`


---

**Q: Given below are two application where an variant of RNN has to be used. Choose the most suitable option
a) Speech recognition - pronunciation of a sound can vary across a set of sentences spoken
b) Machine translation - different length for output and inputs**
- For both (a) and (b) use Bidirectional RNN
- For both (a) and (b) use RNN with encoder-decoder
- For (a) use Bidirectional RNN; for (b) use RNN with encoder-decoder
- For (a) use RNN with encoder-decoder  ; for (b) use Bidirectional RNN
* `[ C ]`


---

**Q: What is the main advantage of a bidirectional RNN?**
- It is more resistant to noise
- For certain sequences, values are dependent on their successors as well as their predecessors, which it takes into account.
- It creates additional smoothing
- It is more robust for sequences in which is some values have been swapped in location
* `[ B ]`


---

**Q: What is the main advantage of a encoder/decoder RNN?**
- It does not care about the input and output size
- It is more robust for sequences in which is some values have been swapped in location
- For certain sequences, values are dependent on their successors as well as their predecessors, which it takes into account.
- It can swap the encoder/decoder parts easily for different solutions
* `[ A ]`


---

**Q: can encoders and decoders for RNNs be trained independently from eachother**
- yes
- no
- yes, if they share a common "language"
- None of the above
* `[ C ]`


---

**Q: what groups of problems can RNNS be applied to**
- one to one
- one to many
- many to many
- all of the above
* `[ D ]`


---

**Q: Machine translation neural network architecture is of type:**
- one to many
- many to one
- many to many 
- one to one
* `[ C ]`


---

**Q: What is one of the biggest advantage of RNNs over different networkd structures?**
- states computed in the forward pass must be stored until they are reused during the backward pass
- sharing weights
- being able to process sequence of inputs
- memory saving
* `[ C ]`


---

**Q: Which of the following answers is not part of the recurrence formula?**
- The next input vector
- The current state
- The old state
- The current input vector
* `[ A ]`


---

**Q: Which of the following gates I not part of a LSTM RNN?**
- Forget gate
- Input gate
- Output gate
- Memory gate
* `[ D ]`


---

**Q: Given are four different types (A-B-C-D) of sequential data. Which network structure (1-2-3-4) fits the best?

A - machine translation
B - image captioning
C - character recognition
D - emotion classification

1 - one to one
2 - one to many
3 - many to one
4 - many to many**
- A - 1, B - 2, C - 3, D - 4
- A - 4, B - 3, C - 1, D - 2
- A - 4, B - 2, C - 1, D - 3
- A - 1, B - 2, C - 4, D - 3
* `[ B ]`


---

**Q: What is NOT a downside of the recurrent neural networks (including LSTM & GRUs) ?**
- A RNN has long term dependencies which can lead to vanishing or exploding gradients during training
- These networks struggle to memorize facts that did not occur as implicit information in the training data
- The amount of information that can be remembered is only in the order of hundreds
- These networks can not operate on multiple time scales
* `[ D ]`


---

**Q: Image captioning is and example of what type of sequence**
- One-to-one
- One-to-many
- Many-to-one
- Many-to-many
* `[ B ]`


---

**Q: Which of the following is true about RNNs?**
- A strength of RNNs is that they can generalize to sequence length during training
- States computed in the backward pass makes RNNs expensive in memory
- Gradient problems associated with long-term dependencies can be dealt with gated RNNs.
- LSTMs are more powerful than RNNs
* `[ C ]`


---

**Q: Which statements about RNN are true?

Statement1: A recurrent neuron stores the state of a previous input and combines with the current input. Further, thereby preserving some relationship of the current input with the previous input.

Statement2: As the time steps increase, the unit gets influence by a larger neighbourhood. With that information, recurrent networks can watch large regions in the input space.

Statement3: The recurrent connections increase the network depth while they keep the number of parameters low by weight sharing.

Statement4: . The big advantage is that RNN offers some kind of memory, which can use in many applications. **
- 1 and 3
- 2 and 4
- 1 2 and 3
- All statements are true
* `[ D ]`


---

**Q: Which statements related to RNNs are true?

Statement1: Long Short-Term Memory (LSTM) networks are an extension for recurrent neural networks, which basically extends their memory. Therefore it is well suited to learn from important experiences that have very long time lags in between.

Statement2:  Exploding gradients is an issue in standard RNN architectures.  But fortunately, this problem can be easily solved if you truncate or squash the gradients.

Statement3: Vanishing Gradients  is an issue in standard RNN architectures. Fortunately, this problem can be solved through the concept of LSTM.

Statement4: In an LSTM you have three gates: input, forget and output gate. These gates determine whether or not to let new input in (input gate), delete the information because it isn’t important (forget gate) or to let it impact the output at the current time step (output gate).**
- 1 and 2
- 2 and 3
- 3 and 4
- All statements are true
* `[ D ]`


---

**Q: An RNN uses a recurrence formula that takes as input the input vector and the old state. What is not a strength of this model?**
- it learns a single model f that operates on all time steps and all sequence lengths rather than a separate g(t) for all possible time steps
- It can generalise to sequence lengths unseen during training
- It is a relatively small model that can take in arbitrarily large inputs by taking them sequentially
- It shares statistical strength across different sequence lengths and across different positions in time
* `[ C ]`


---

**Q: which of the following statements is false**
- LSTMs are the most powerful RNNs
- Gated RNNS can solve the problem of exploding gradients 
- Long term dependencies can take a very long time to learn using RNNs
- RNN's cannot be used in one-to-many type tasks
* `[ D ]`


---

**Q: Why are RNNs suitable for modeling sequence data?**
- RNNs are able to compute the full hidden representation in a single step, which makes them computationally efficient.
- RNNs, compared to other models, require less training data.
- RNNs have state of the art performance on Neural Machine Translation (NMT), such as English to French translation.
- RNNs can take into account the hidden representation from previous elements in the sequence.
* `[ D ]`


---

**Q: What do LSTM RNNs offer over the standard RNN architecture?**
- LSTM reduces computation intensity, which is a large problem with standard RNNs.
- LSTM RNNs are useful for differences in length between input and output.
- LSTM RNNs offer a solution for long term dependencies in sequences that can be prevalent in sequence-to-sequence models.
- LSTM RRNs are the best models for sequential data. 
* `[ C ]`


---

**Q: Which of the following statement for RNN is not true?**
- Generalization to sequence lengths unseen during training
- Sharing statistical strength across different sequence lengths and across different positions in time
- In time: forward pass is necessarily sequential
- In memory: states computed in the forward pass do not need to be stored until they are reused during the backward pass
* `[ D ]`


---

**Q: Which of the following statement for LSTMs is not true?**
- It is an unrolled recurrent neural network
- The repeating module in a standard RNN contains a single (often tanh) layer
- Adding and removing information to Ct via gates: sigmoid+ pointwise multiplication
- Adding information from the input or the history
* `[ D ]`


---

**Q: Which of the following is not a main strength of an RNN?**
- Learns a single model f that operates on all time steps and all sequence lengths rather than a separate g(t) for all possible time steps.
- Generalization to sequence lengths unseen during training.
- Sharing statistical strength across different sequence lengths and across different positions in time.
- RNNs have long term dependencies.
* `[ D ]`


---

**Q: What is the purpose of image2speech?**
- Generate a spoken description of an image
- Generate an image of a spoken description
- Generate a spoken description and translate it
- Users submit a spoken description to create a database for images.
* `[ A ]`


---

**Q: What drawbacks of RNNs can be dealt with using gated RNNs for sequential data?**
- Hard to train
- Suffer from vanishing gradients, exploding gradients
- Hard to remember values from long way in the past
- All
* `[ D ]`


---

**Q: While training a RNN:**
- Forward pass is necessarily sequential, hence expensive in terms of time.
- States computed in the forward pass must be stored until they are reused during the backward pass, hence expensive in terms of memory.
- both
- none
* `[ C ]`


---

**Q: Which statement is wrong?**
- Difference between feedforward network and a recurrent network: A feedforward network has no cyclic activation flows.
-  RNN reuse the same weights, so they only need one of the iterations to have gradients for the weights at all iterations to get updated.
- RNN and CNN are two distinct  models, with their own advantages and disadvantages.
- None
* `[ D ]`


---

**Q: What is the probable approach when dealing with Exploding Gradient in RNNs?**
-  Use modified architectures for example LSTM , GRUs
- Dropout
- Threshold the gradient values at a specific point
- None
* `[ C ]`


---

**Q: Looking at the unrolled RNN structure, what operation/s can be parallelized (Consider W_ih, W_hh and W_ho for input-hidden, hidden-hidden and hidden-output)?**
- The multiplication between x = [x_0, x_1, …, x_t] and W_ih 
- The multiplication between h = [h_0, h_1, …, h_t] and W_hh
- The multiplication between h = [h_0, h_1, …, h_t] and W_ho
- All of the above operations
* `[ A ]`


---

**Q: What is the usage of sigmoid function in the LSTM cell internal structure?**
- Having values between -1 and 1, it makes sure that the state vector Ct values are able to both increase and decrease
- Having values between 0 and 1, it acts as a gating function of what to remember and what to forget 
- Having only positive values, it makes convex optimization more efficient
- Having a large range of values, it makes the architecture capable of remembering information for long sequences
* `[ B ]`


---

**Q: Consider the following statements:
a) RNNs suffer from vanishing gradient problem because of the involvement of chained matrix multiplication
b) RNNs are not expensive in both time and memory as it learns only a single model that operates on all time steps and sequence lengths
Choose the correct option**
- Statement (a) is correct; Statement (b) is wrong
- Statement (a) is wrong; Statement (b) is correct
- Both statements are wrong
- Both statement are correct
* `[ A ]`


---

**Q: Given below are two application where an variant of RNN has to be used. Choose the most suitable option
a) Speech recognition - pronunciation of a sound can vary across a set of sentences spoken
b) Machine translation - different length for output and inputs**
- For both (a) and (b) use Bidirectional RNN
- For both (a) and (b) use RNN with encoder-decoder
- For (a) use Bidirectional RNN; for (b) use RNN with encoder-decoder
- For (a) use RNN with encoder-decoder  ; for (b) use Bidirectional RNN
* `[ C ]`


---

**Q: What is the work flow of RNN?**
- Big matrix to small matrix
- Big matrix to small matrix
- Matrix to vector
- Vector to vector
* `[ D ]`


---

**Q: What is not a challenge of RNN?**
- Long-term dependencies
- Sequence-to-sequence model design
- Long short term memory
- Various solution
* `[ B ]`


---

**Q: Given a RNN output y^(t) = softmax(c + V * tanh(b + W * h(t-1) + U * x(t))), what are the functions of b and c?**
- b and c are bias vectors, where b functions as input bias, and c functions as output bias.
- b and c are bias vectors which prevent uniform weight initialization.
- b and c are weight vectors, where b functions as weight of the input layer and c functions as weight from the output layer.
- b and c are in- and output vectors, where b maps the input vector using the activation function and c contains the output-firing threshold.
* `[ A ]`


---

**Q: When considering a learned model of a recurrent neural network, which one of the following statements is true?**
- An input size independent of the sequence length since the input is specified by state transistions in stead of states.
- An input size that depends on the relation that is learned, e.g. one-to-one or one-to-many.
- An input size that depends on the number of the number of network layers.
- An input size independent on the number of the transition function parameters.
* `[ A ]`


---

**Q: Which of the following is NOT a strength of RNN's**
- It is capable of learning a single model that operates on all time steps and sequence lengths
- It generalizes to sequence lengths unseen during training
- It shares statistical strength across different sequence lengths and positions in time
- It is less affected by the vanishing gradient problem than non-RNN architectures
* `[ D ]`


---

**Q: On what key properties does an LSTM differ from a tanh based RNN cell**
- It is possible to forget information from the previous cell state
- It is possible to share multiple pieces of information on a single cell state
- It can be unrolled, unlike the traditional tanh RNN cell
- It is differentiable, which doesn't hold for the traditional cell
* `[ A ]`


---

**Q: Which of the following statements about RNNs is true:**
- RNNs need to keep track of all of the past hidden states ($h_{t-1}, ..., h_{0}) to come to an prediction
- RNNs use the same set of parameters for each timestep
- Input sequences to RNNs need to be of the same length during training and testing
- All statements are false
* `[ A ]`


---

**Q: Which of the following statements is NOT correct about training RNNs:**
- The total loss of an output sequence given some input sequence is equal to the multiplication of the losses over all time steps
- The forward pass in RNNs is sequential
- Training RNNs is both time- and space-expensive
- All statements are true
* `[ A ]`


---

**Q: which statement is wrong about RNNs?**
-  A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor
-  They allow us to operate over sequences of vectors
- RNN models perform their mapping using a fixed amount of computational steps.
-  RNNs have two sources of input, the present and the recent past.
* `[ C ]`


---

**Q: which statement is false about Long Short-Term Memory ?**
- LSTMs  help preserve the error that can be backpropagated through time and layers.
- LSTMs contain information outside the normal flow of the recurrent network in a gated cell.
- By maintaining a more constant error, they allow recurrent nets to continue to learn over many time steps 
- Non of them.
* `[ D ]`


---

**Q: How do transition functions change between time steps in RNNs?**
- The parameters of the function update
- The weights decrease according to the gradient
- The function is updated according to its derivative
- The function and its parameters don’t change
* `[ D ]`


---

**Q: What of the following is true about training a RNN?**
- It is expensive in memory
- Training is generally done in parallel
- There is no need for backpropagation
- None of the above
* `[ A ]`


---

**Q: Let’s say the data you want to process is sequential data, e.g. speech, and you want to convert this to text by training a neural network. What type of neural network will be most suitable for that?**
- a) A Convolutional Neural Network
- b) A Recurrent Neural Network
- c) A Feed Forward Neural Network
- d) None of the above are suitable
* `[ B ]`


---

**Q: Which of the two following statements regarding a Recurrent Neural Network are correct?
1)	Regardless of sequence length, the learned model always has the same input size.
2)	At every time step, a different transition function f with different parameters has the be used.**
- a) Both are correct
- b) Statement 1 is correct, Statement 2 is false
- c) Statement 1 is false, Statement 2 is correct
- d) Both are false
* `[ B ]`


---

**Q: What is NOT an application of RNN?**
- Image captioning which translates an image to a sequence of words.
- Emotion classification which translates a sequence of words to emotion.
- Machine translation which translates sequence of words to another sequence of words.
- Image classification which splits an image to many segmented images.
* `[ D ]`


---

**Q: What is NOT the strength of an RNN?**
- Learns a single model f that operates on all time steps and all sequence lengths.
- Generalization to sequence lengths unseen during training.
- Sharing statistical strength across different sequence lengths and across different positions in time.
- It takes a very short time to learn long-term dependencies
* `[ D ]`


---

**Q: Which of the following is not a disadvantage of Representative Recurrent Neural Networks**
- Computing the forward pass is a necessarily sequential process
- Previously computed states must be stored until they are used during back propagation
- Representative RNN always has a fixed input size despite a variable length sequence of inputs
- None of the above
* `[ C ]`


---

**Q: Recurrent mouldes a standard LSTM network typically have four neural network layers, which of these layers is typically used to "forget" past information?**
- A sigmoid layer
- A residual layer with skip connections
- A tanh layer
- None of the above
* `[ A ]`


---

**Q: What is the feature of speech that makes it more suitable to process with RNN instead of different neural network architectures (e.g. CNN)?**
- speech recognition is a sequential task
- speech is a grid type task
- speech is parallel task
- none of the above
* `[ A ]`


---

**Q: Consider recurrent neural networks. What holds true?**
- Regardless of  sequence length, the learned model always has the same input size
- Regardless of  sequence length, the learned model always has different input size
- It cannot generalize to sequence lengths unseen during training
- none of the above
* `[ A ]`


---

**Q: Which of the following is true for a RNN?**
- The learned model always has the same input size.
- Sequence length is specified in terms of state transitions.
- The same transition function with the same parameters can be used at each time step
- All the above
* `[ D ]`


---

**Q: Training a RNN is**
- Expensive in time.
- Expensive in memory.
- Both a and b.
- Isn't expensive in terms of time and memory
* `[ C ]`


---

**Q: Which of the following is TRUE about strengths of RNN?**
- Learns a single model f that operates on all time steps and all sequence lengths rather than a  separate g(t) for all possible time steps 
- Generalisation to sequence length unseen during training
- Sharing statistical strength across different sequence lengths and across different positions in time
- All of the above
* `[ D ]`


---

**Q: Which is the following is supposedly NOT TRUE on the advantages of Recurrence?**
- The same transition function f with the same parameters can be used at every time step
- The same transition function f with the same parameters can be used at alternate step
- Regardless of sequence length, the learned model always has the same input
- sequence length is specified in terms of state transitions rather than a variable-length history of states
* `[ B ]`


---

**Q: What is not a strength of an RNN?**
- Training can be done relatively fast
- Learns a single model that operates on all time steps
- Learns a single model that operates on all sequence lengths
- Generalizes to sequence lengths unseen during training
* `[ A ]`


---

**Q: Translating one word in a sentence from one language to another is an example of..**
- One to one
- One to many
- Many to one
- Many to many
* `[ C ]`


---

**Q: 1. RNN share the hyperparameters and it has to be passed only once
2. RNN can take variable length inputs**
- 1 is wrong and 2 is correct
- 2 is wrong and 1 is correct
- both 1 and 2 are correct
- both 1 and 2 are wrong
* `[ C ]`


---

**Q: Gated RNN is _____ remembering old data as it ________ tackle exploding/vanishing gradient**
- bad , cannot
- good , can
- bad , can
- good , cannot
* `[ B ]`


---

**Q: Which sentence is true about RNNs and CNNs respectively?**
- CNNs learns a single model f that operates on all time steps and all sequence lengths rather than a separate $g^(t)$ for all possible time steps
- RNNs usually performs well for image classification tasks
- RNNs share statistical strength across different sequence lengths and across different positions in time
- For RNNs the states that are computed in the forward pass don’t have to be stored until they are reused during the backward pass 
* `[ C ]`


---

**Q: RNNs performs well in applications like:**
- language translation
- machine comprehension
-  feature extraction for image captioning
-  both for a), b)
* `[ D ]`


---

**Q: Which of the following is/are true about RNN’s?**
- Irrespective of the sequence length, the learned model always has the same input size. 
- Same transition function ‘f’ with the same parameters can be used at every time step .
- Shares statistical strength across different sequence lengths and across different positions in time. 
- All of the above.
* `[ D ]`


---

**Q: Which of the following is/are true about gated RNNs?**
- Designed to remember information for long periods of time. 
- Scales and transfers information from the distant past to the present more efficiently. 
- A and B
- None of the above
* `[ C ]`


---

**Q: For which of the following tasks is many-to-one RNN appropriate?**
- Character recognition
- Speech recognition
- Sentiment classification from text
- Image captioning
* `[ C ]`


---

**Q: Exploding gradient problem occurs when gradient gets large and loss goes to infinity and explodes. Which of the following can be used as a solution to exploding gradient problem?**
- Gradient clipping
- Setting gradient to zero after a threshold
- Dropout
- Regularization
* `[ A ]`


---

**Q: Which statement is true in the discussion CNN vs RNn**
- Convolutions are well-suited for grids of values, where RNN are more suited for sequential task.
- Convolutions are better for speach recongnitions.
- The main application of RNN is in image processing
- RNNs are well-suited for grids of values, where CNN are more suited for sequential task.
* `[ A ]`


---

**Q: What is not a Strength of an RNN**
- Learns a single model f that operates on all time steps
and all sequence lengths rather than a separate g^(t) for all
possible time steps
- Generalization to sequence lengths unseen during
training
- Sharing statistical strength across different sequence
lengths and across different positions in time
- It can deal vary well with exploding gradients in grid style applications
* `[ D ]`


---

**Q: Why is training a RNN memory instensive?**
- You need to save a weights matrix for all different lengths of sequences the network may see.
- In order to train the states need to be saved until the backwards pass.
- The forwards pass needs to be sequential.
- Older states need to be stored for time dependent series.
* `[ B ]`


---

**Q: Why does being able to forget increase the performance of RNNs.**
- It works are a form of regularization
- It counteracts any noise that could be in signals such as voice recordings.
- A sentence could change the expected subject as it progresses.
- It is a time based form of dropout.
* `[ C ]`


---

**Q: What data property do RNNs make use of?**
- Temporal correlation
- Spatial correlation
- Temporal and spatial correlation
- Correlation between training examples
* `[ A ]`


---

**Q: How many gates in an LSTM cell?**
- 2
- 3
- 4
- 5
* `[ C ]`


---

**Q: Which of the statements are true regarding RNN?**
- RNNS give best results for image classification problems.
- RNNs are specialised for processing sequential data. 
- Both A and B
- None of the above
* `[ B ]`


---

**Q: The gradient of the loss in RNNs can be calculated by**
- using a forward propagation pass followed by a backward propagation pass and used for retraining the RNN
- using a backward propagation pass followed by a forward propagation pass and used for retraining the RNN
- Both A and B
- None of the above
* `[ A ]`


---

**Q: What is an advantage of RNNs over CNNs?**
- They can deal with variable length input.
- They do not suffer from exploding gradient.
- They use weight sharing.
- All of the above
* `[ A ]`


---

**Q: What can be said about the time complexity and space complexity of RNNs compared to CNNs?**
- RNNs are more computationally expensive to train in both time and space.
- RNNs are more computationally expensive to train in time but not in space.
- RNNs are less computationally expensive to train in both time and space.
- RNNs are less computationally expensive to train in time but not in space.
* `[ A ]`


---

**Q: The loss of a recurrent neural network is...**
- ...the sum of all losses over all time steps.
- ...the sum of all losses over all time steps in the past until present.
- ...the product of all losses over all time steps.
- ...the product of all losses over all time steps in the past until the present.
* `[ A ]`


---

**Q: Recurrent neural networks are so computationally expensive because...**
- ...there is no hardware optimized for this operation.
- ...the forward pass is forcibly sequential. 
- ...every calculation has to be done at the same time.
- ...there are not more steps but every calculation step in a recurrent neural network is computationally more expensive
* `[ B ]`


---

**Q: Which of the following is an example of a “One to Many” sequential decision problem?**
- Character Recognition (Image to Letter)
- Image Captioning (Image to Caption)
- Machine Translation (Language Translation)
- Sentiment Analysis
* `[ B ]`


---

**Q: Which of the following approaches DOES NOT suffer with vanishing gradient or exploding gradient?**
- Recurrent Neural Network
- Convolutional Neural Networks
- Gated Recurrent Units
- None of above
* `[ C ]`


---

**Q: How does a recurrent neural network compute a step forward?**
- It uses a different set of weights and function on every step
- It does not use the input x to compute the forward step, but uses the computed step before and the same weights and function every time
- It uses the same function and the same set of parameters at every time step
- None of the above
* `[ C ]`


---

**Q: What does the sigmoid multiplication to the state Ct-1 means?**
- 1. It squashes the cell state from 0 to 1 for easier computation
- 2. It means how much to remember from state Ct-1 to Ct since the sigmoid has a range between 0 and 1
- It means how much of the gradient to remember from one state to the other.
- None of the above
* `[ B ]`


---

**Q: Why are convolution nets not good for speech to text tasks?**
- Speech to text is a sequential task.
- Speech to text can only be learned by hidden layers
- Speech to text can only be learned by shallow nets
- Convolution layers would take too much memory for speech to text tasks.
* `[ A ]`


---

**Q: What is NOT a strength of RNN's?**
- Sharing statistical strength across different sequence lengths and across different positions in time
- Generalization to sequence lengths unseen during training
- Learns a single model f that operates on all time steps and all sequence lengths rather than a separate g(t) for all possible time steps
- Less memory intensive during training
* `[ D ]`


---

**Q: which of the following is not a problem that is typically solved using a RNN?**
-  Language translation
-  DNA sequence analysis 
-  Music classification
-  Image classification
* `[ D ]`


---

**Q: Which of the following is false regarding RNNs?**
-  RNNs can be seen as a feedforward NN when unwrapped over time.
-  A recurrent neural network shares the same parameters in all time steps
-  RNNs are not subject to the vanishing gradient problem.
-  RNNs use a modified version of back propagation called back propagation through time.
* `[ C ]`


---

**Q: Imagine a model for transcribing speech into text, what is the problem when using a convolutional network?**
- Convolutions are well-suited for grids of values, whereas speech to text is a sequential task.
- Convolutions are poor-suited for grids of values, whereas speech to text is a sequential task.
- Convolutions are well-suited for grids of values, whereas speech to text is a parallel task.
- Convolutions are poor-suited for grids of values, whereas speech to text is a parallel task.
* `[ A ]`


---

**Q: What can be said about the following statements? A: Regardless of sequence length, the learned model always has the same input size. B: At every time step a different transition function f is used.**
- A: true, B: true
- A: false, B: true
- A: true, B: false
- A: false, B: false
* `[ C ]`


---

**Q: Which option below is not an advantage of RNN?**
- Same imput size 
- Same trasition function f
- Well-suited for grids of values
- Generalization to sequence lengths unseen during training
* `[ C ]`


---

**Q: Which option below is not true while traing an RNN**
- RNN training is expensive in time and memory
- There is a forward pass and a backward pass in training RNN
- The gradient of the loss is used for retraining RNN
- The gradient of the loss is calculated in the backward pass
* `[ D ]`


---

**Q: An Encoder to decoder RNN architecture is recomendend when?**
- Encoder and decoders are used for music recommendation systems.
- encoder to decoder are recommended when there is no clear similarity-measure between several input features.
- Input and output have difference tensor size.
- When the input tensor is equal or larger than the output tensor.
* `[ C ]`


---

**Q: What are LSTMs 4 interacting layers?**
- (1) Long term [L], (2) Short term [S], (3) Combination [LS-M] & (4) Update
- (1) Conveyor belt of information ,(2) Forget gate, (3) Adding information & (4) Output gate
- (1) Pooling filter, (2) Time-step gate, (3) Neighbor comparison & (4) Combination
- (1) Node activators, (2) Neighbor evaluation, (3) Local similarity & (4) Output control
* `[ B ]`


---

**Q: which of these statements is/are true?
1: RNN (recurrent neural networks) are used specifically for problems with sequential data.
2: The transition function differs for each step in a RNN**
- Only statement 1 is true
- Only statement 2 is true
- Both statements are true
- Both statements are false
* `[ A ]`


---

**Q: the training of an RNN is expensive in:**
- Time
- Memory
- Both time and memory
- Neither time nor memory
* `[ C ]`


---

**Q: Why do we use an RNN instead of CNN for automatic speech recognition?**
- CNN is for grid values
- CNN has high latency in this case
- CNN can be too deep
- None of the above
* `[ A ]`


---

**Q: What is NOT the advantage of using RNN?**
- Sharing information across time steps
- Specialised for sequential data
- Regardless of sequence length, the learned model always has the same input size
- Cannot be generalised to sequence lengths unseen during training
* `[ D ]`


---

**Q: What is the advantage of Bi-directional RNNs over traditional RNNs?**
- Bi-directional can learn backward dependencies in sequences too.
- They can easily flip the temporal order of sequences.
- They can have explicit memory for keeping track of prominent events.
- None of the above.
* `[ A ]`


---

**Q: What makes LSTMs special?**
- It can choose what to remember.
- It can choose what to forget.
- It can learn how to choose what to learn or forget.
- All of the above.
* `[ D ]`


---

**Q: Which of the following applications can not be facilitated by the use of Bidirectional approaches to the RNNs?**
- A speech to text converter
- A text autocorrector
- A decision support tool for self driving cars
- Context annotation of a video
* `[ C ]`


---

**Q: Consider a more abstract approach to the principles of a RNN:
Assume an input sequence $X$ whose samples $x_{i}$  belong to the set $V_{x}=\big\{w^{x}_{1},,\dots,w^{x}_{N}\big\}$  such that $X(t)=\bigcup\limits_{i=1}^{n} x_{i}$ with $x_{i} \in V_{x}$. 
\\Assume also a RNN $\hat{M}$ that maps such input  $X$ into a new sequence $\Psi (t)= \bigcup\limits_{j=1}^{m}y_{j}$ where $y_{j} \in V_{y}=\big\{ w^{y}_{1}\dots w^{y}_{M} \big\}$.
Which of the following statements is false?**
-  A context function $C$ can be approached as a semantic mapping function that given $X$ we can define a process $\Pi$ with $\Pi(C)=\Psi$ such that the conditional probability $p\left(\Psi \lvert X \right)$ is maximal
- If the output $\Psi$ does not depend  on  time  but rather on the underlying relationship of the input components $x_{i}$ such that $\Psi(t+\tau)=\hat{M}\left(X(t+\tau)\right)=\hat{M}\left(X(t)\right)=\Psi(t)$ where $X$ is the same, then the conditional probability between input and output is stationary & deterministic. This property can help decrease exponentially the number of parameters that $\hat{M}$ needs in order to model the relationship between $X$,$\Psi$
- A causal RNN will produce outputs such that $p(\Psi(t)\lvert X(t+N), X(t+ N-1), \dots X(1) )$ is maximal
-  A teacher forcing training requires the computation of $p\left(\Psi(t+1),\Psi(t),\dots \Psi(1) \lvert X(t+1),X(t),\dots,X(1) \right)$
* `[ C ]`


---

**Q: Which statement is incorrect?**
- The auto-correct option in your phone is an example of a one-to-many RNN
- language recognition is an example of a many-to-one RNN
- Video classification is an example of a one-to-one RNN
- Sentence translation is an example of a many-to-many RNN
* `[ C ]`


---

**Q: What gates are used in a LSTM (long short term memory) Node.**
- The cell state gate, the forget gate, the input gate and the output gate
- The information gate, the removal gate, the input gate and the output state
- The cell state gate, the forget gate, the update gate
- The information gate, the input gate and the output gate
* `[ A ]`


---

**Q: Which of the following about RNN is right**
- deep RNN may encounter problems like gradient exploding and vanishing
- RNN perform better than CNN when data is sequential
- parameter is not shared in RNN
- Regardless of   sequence length, the learned model always has the same input size
* `[ C ]`


---

**Q: Which of the following is not right about LSTM**
- LSTM'gradient vanishing problem is worse than RNN
- compared with regular RNN, it can remember information longer
- It contains input gate, forget gate and an output gamt
- sigmoid is used cause it's value is in [0,1], which can represent forget and remember
* `[ A ]`


---

