# Questions from `paper_12` :robot: 

**Q: Which of the following options, according to the background and empirical results in this paper, is false?**
- TCNs exhibit shorter memory than recurrent architectures with the same capacity.
- TCNs don’t have the problem of exploding or vanishing gradients.
- TCN models substantially outperform generic recurrent architectures.
- TCNs can take, like RNNs, input with variable lengths in a recurrent way.
* `[ A ]`


---

**Q: Which of the following is an advantage of a TCN over a RNN?**
- TCN does not have to be deep
- Training can be parallelized, due to TCN not being recurrent
- TCN has significantly less parameters to learn
- TCN is extremely faster in testing phase, and uses less memory.
* `[ B ]`


---

**Q: What is no principle where temporal convolutional networks are based on?**
- The fact that the network produces an output of the same length as the input.
- That it is causal, so there will be no leakage from the future onto the past.
- The fact that they are forces to use a hug amount of history sizes which stimulates the ability for the network to look very far into the past to make a prediction.
- All of the named principles above are principles where temporal convolutional networks are based on. 
* `[ C ]`


---

**Q: What is TRUE about Convolutional Networks?**
- In practice they are not so powerful for sequence modeling as Recurrent Neural Networks
- Due to the introduction of architectural elements such as dilated convolutions and residual connections, convolutional architectures have become much stronger for sequence modeling
- Recurrent Neural Networks are preferred over Convolutional Networks, because of their simpler architecture
- In practice they perform worse than Recurrent Neural Networks on sequence modeling, because they do not have 'infinite memory'
* `[ B ]`


---

**Q: What is causal convolution?**
- It is a convolution in which an output at time t is convolved with all the elements at time t from previous layer
- It is a convolution in which an output at time determine the value at time t-1
- It is a convolution in which an output at time t is convolved only with elements from time t and earlier in the previous layer
- None of Above
* `[ C ]`


---

**Q: Which of the following is NOT a possible advantage of a temporal convolutional network (TCN) in sequence modeling?**
- Convolutions can be done in parallel since the same filter is used in each layer.
- The problem of exploding/ vanishing gradients is avoided.
- Lower memory requirement than RNNs during evaluation. 
- A TCN can change its receptive field size in various ways, such as by increasing the filter size.
* `[ C ]`


---

**Q: Which statement is correct?**
-  temporal convolutional network (TCN) can be used for tabular data
-  temporal convolutional network (TCN) can be used for image processing
- both A and B are correct
- None of above are correct
* `[ D ]`


---

**Q: Which of the following constitutes an advantage of a Temporal Convolution Network (TCN) over a Recurrent Neural Network (RNN)?**
- A TCN only needs to maintain a hidden state.
- A TCN offers more opportunities for parallel computing compared to an RNN.
- A TCN can be transferred to a different domain without changing any parameter.
- A TCN takes inputs of fixed length.
* `[ B ]`


---

**Q: If possible, complete the following statement: "Temporal Convolutional Networks exhibit ... memory than recurrent architectures with the same capacity."**
- the same amount of
- shorter
- longer
- None of the above answers is correct.
* `[ C ]`


---

**Q: Which of the following two statements is true? 1) A causal convolution has the following property: an output at time t is convolved only with elements from time t and earlier in the previous layer. 2) A dilated convolution introduces fixed steps between considered inputs.**
- 1) true 2) true
- 1) false 2) true
- 1) true 2) false
- 1) false 2) false
* `[ A ]`


---

**Q: Which statement is NOT correct regarding Temporal Convolutional Networks:**
- The convolutions in the architecture are always casual.
- The architecture maps an arbitrary length of input sequence to an output sequence of a fixed length.
- The architecture supports parallelism so the lon ginput sequences can be processed as a whole.
- TCNs use relatively low memory compared to RNNs.
* `[ A ]`


---

**Q: Which is not a distinguishing characteristic of a TNC**
- Domain transfers where memory availability changes have little impact on the performance of a TNC
- Any sequence of length N will be mapped to an output sequence of length N
- There is no infomation leakage from future to past
- All are distinguishing characteristics
* `[ A ]`


---

**Q: On what kind of tasks can convolutional networks NOT outperform recurrent networks?**
- Machine translation.
- Audio synthesis.
- Image recognition.
- None of the above.
* `[ C ]`


---

**Q: Which of the following architectures is the most common starting point when dealing with sequence modeling tasks?**
- Convolutional
- Long short-term memory
- Multilayer Kernel
- Recurrent
* `[ D ]`


---

**Q: Regarding a task with sequential data, what kind of model should be a starting point?**
- Long Shot Term Memory Networks
- Convolutional Neural Networks
- Temporal Convolutional Networks
- Recurrent Neural Networks
* `[ C ]`


---

**Q: Which of the following is not an advantage of temporal convolutional networks?**
- Convolutions can be done in parallel since the same filter is used in each layer
- TCN's can change their receptive fields in multiple ways.
- TCN's can store a summary of the entire history, keeping memory usage low during evaluation.
- They can avoid the problem of vanishing gradients due to using a different backpropagation path.
* `[ C ]`


---

**Q: TCN is simply:**
- Combination of LSTM and GRU
- 1Dimensional Fully Connected Network and casual convolutions
- variation of Recurrent Neural Network which uses dot-product attention
- Combination of variations of CNN and RNN
* `[ B ]`


---

**Q: What works better for sequence modeling: convolutional or recurrent networks?**
- convolutional nets
- recurrent nets
- not much difference
- totally dependent on application
* `[ A ]`


---

**Q: What is a disadvantage of using TCNs for sequence modeling?**
- Parallelism 
- Flexible receptive field size
- Stable gradients
- Data storage during evaluation
* `[ D ]`


---

**Q: The distinguishing characteristics of Temporal Convolutional Networks (TCN) are: 1) the convolutions in the architecture are causal, meaning that there is no information “leakage” from future to past; 2) the architecture can take a sequence of any length and map it to an output sequence of the same length, just as with an RNN.**
- Both 1) and 2) are TRUE
- 2) is FALSE because RNN’s can not work with temporal data
- 1) is FALSE because information can always leak, NSA is watching!
- Both 1) and 2) are FALSE because of the aforementioned reasons
* `[ A ]`


---

**Q: What is true about a TCN with respect to a LSTM?**
- A TCN is based around convolution whereas a LSTM is based around recurrence.
- A TCN is based around recurrence whereas a LSTM is based around convolution.
- A LSTM performs better at sequential asks than a TCN.
- None of the above
* `[ A ]`


---

**Q: What is the goal of dilated convolutions?**
-  Increase the receptive field
- Make the input and output in same dimensions
- Use information only from the past
- Increase the depth of the network
* `[ A ]`


---

**Q: Temporal convolutional networks TCN can change their receptive field in multiple ways. What is not effective to increase the receptive field (history) of temporal convolutional networks TCNs:**
-  Use a larger dilation 
-  Increase the network depth
- Adding casual convolutions
- Add an LSTM layer
* `[ D ]`


---

**Q: The temporal convolutional networks presented in the paper is composed of two parts: 1D full-convolutional network architecture and the causal convolutions. What do causal convolutions mean?**
- Convolution where an output at time t is convolved only with elements  from time t and earlier in the previous layer;
- Convolution where an output at time t is convolved only with the elements from time t-1;
- Convolution inside the hidden layers;
- Convolution where an output at time t is convolved only with elements from times t to t/2.
* `[ A ]`


---

**Q: One of the major benefits that can be introduced by using CNNs in sequence modeling tasks is**
- saving memory, as CNNs do not need to store intermediate results like LSTMs/GRUs do (or instead the bigger batch sizes)
- the interpretability of the results, as one can easily understand what features maps the CNN computes
- the reproducibility of the experiments, as RNNs are non-deterministic
- the speedup that one gets, because CNNs can be trained with GPUs and RNNs can only be trained in CPUs
* `[ A ]`


---

**Q: In this question ‘TCN’ refers to the ‘Temporal Convolutional Network’, as described in the paper.
Considering the following statements, which one is true?**
- Recurrent neural networks retain information of longer sequences better than TCNs, as they can retail informations through sequences of unlimited length.
-  In order to deal with longer histories, simple convolution is replaced with dilated convolution in TCNs.
- For every RNN implementation there exists an TCN implementation that can outperform it.
- TCNs gain advantage over RNNs by using information of inputs that are observed after the previously observed inputs.
* `[ B ]`


---

**Q: Which of the following is true?**
- A TCN can change its receptive field by using smaller dilation factors
-  TCN avoids the problem of exploding/vanishing gradients 
- The partial results of LSTMs and GRUs use a small amount of memory 
- TCN has no need the raw sequence up to a certain effective history length
* `[ B ]`


---

**Q: Which practice has NOT directly made Temporal Convolutional Networks more competitive when compared to  Recurrent architectures?**
- Dilated Convolutions
- Residual Connections
- ReLUs
- Causal Convolutions
* `[ C ]`


---

**Q: What is needed to make a convolution architecture process sequential data?**
- We need padding
- Dilation convolution
- The output of the hidden layer must be equal to the length of the input
- All of the above
* `[ D ]`


---

**Q: In "Attention is all you need" the authors replace the usual recurrent layers in encoder-decoder architectures with:**
- multi-headed self-attention
- scaled dot-product attention
- additive attention
- none of the above
* `[ A ]`


---

**Q: Temporal ConvNet (TCN) is a form of ConvN which Sequence Modeling using, **
- Dilation Convolution and Residual Connections.
- Dilation Convolution and Gated Recurrent.
-  Gated Recurrent and Residual Connections.
-  Gated-LSTM and Residual Connections.
* `[ A ]`


---

**Q: What is NOT true about temporal convolutional networks (TCNs)?**
- TCNs often outperform canonical recurrent networks on sequence modelling tasks.
- TCNs exhibit longer memory than recurrent architectures.
- TCNs are more complex than recurrent architectures.
- The convolutions in TCNs are causal.
* `[ C ]`


---

**Q: Which of the following is NOT a method to increase the receptive field (RF) of a convolutional network?**
- Make larger kernel sizes.
- Use higher stride or pooling.
- Using dilation factors in the kernel.
- All of the above are correct methods to increase the RF.
* `[ D ]`


---

**Q: Consider the following two statements about sequence modeling:
1.	The TCN (convolutional) architecture is more accurate than cnonical recurrent networks such as LSTMs and GRUs. 
2.	However the TCN architecture is much more complex and adjective.
Which of the statements is true?**
- Both are true
- Both are false
- 1 is true and 2 is false
- 1 is false and 2 is true
* `[ C ]`


---

**Q: Temporal Convolutional Networks:**
- Need more memory than RNNs for training
- Need more memory than RNNs for evaluation
- Present a bidirectional dependency between the future and the past
- Add the output directly to the input, the same as RNN
* `[ B ]`


---

**Q: Which of the following statements is not one of the main points from the presented empirical evaluation?**
- A simple convolutional architecture outperforms conical recurrent networks.
- Convolutional networks should be regarded as a natural starting point for sequence modeling tasks.
- The infinite memory advantage of RNNs has a significant influence on the performance in practice.
- A simple convolutional network outperforms generic recurrent architectures.
* `[ C ]`


---

**Q: Which of the following correctly characterises Temporal Convolutional Networks (TCN):

\begin{enumerate}
\item The convolutions are causal
\item An input sequence of any length can be mapped to an output sequence of any length
\end{enumerate}**
- Only 1.
- Only 2.
- Both 1 and 2.
- Neither.
* `[ A ]`


---

**Q: What is NOT a named advantage for TCN over RNN?**
- No parameter changes are needed for domain transfer
- Flexible receptive field size, thus have better control over the model's memory size
- Easier parallelization, a long input sequence can be processed as a whole
- Gradients are more stable compared with an RNN
* `[ A ]`


---

**Q: Which of the following is the principle of TCN?**
- the network produces an output of the same length as the input
- there can be little leakage from the future into the past.
- the architecture can take a sequence of certain length
- the architecture can map it to an output sequence of the almost the same length
* `[ A ]`


---

**Q: Statement 1: Basic RNN architectures very easy to train.
Statement 2: A characteristic of a TCN is that the convolutions in the architecture are causal, meaning that there is no information 'leakage' from future to past. **
- Both statements are true
- Statement 1 is true
- Statement 2 is true
- None of the statements are true
* `[ C ]`


---

**Q: Which of these is NOT an advantage that TCNs have over RNNs?**
- Parallelism.
- Stable gradients.
- Variable length inputs.
- Flexible receptive field size.
* `[ C ]`


---

**Q: What do the authors give as an explanation of the succes of the CNN for sequential tasks, noting that the RNN was before the accepted "better choice"?**
- Architectural elements such as dilated convolution and residual connections have enhanced CNN's tremendously
- Former researchers were not critical enough in their research and claimed the succes of RNN's
- The CNN's described, uses features from future moments in the sequence, giving the CNN more information than the RNN
- The research was performed on the worst cases of RNN's. So it is not strange that CNN's can win here
* `[ A ]`


---

**Q: What are the advantages of a TCN for sequence modelling?**
- Less convolutional layers, no learning rate to tune
- Short input sequence, fixed receptive field size, unstable gradients
- Long input sequence can be processed, ability to change the receptive field size, stable gradients
- Less input data necessary
* `[ C ]`


---

**Q: Are Parallelism and Data storage during evaluation an advantage or disadvantage respectively for TCNs? **
- Advantage & Advantage  
- Disadvantage & Advantage 
- Advantage & Disadvantage 
- Disadvantage & Disadvantage 
* `[ C ]`


---

**Q: Which of the following is not an advantage of the temporal convolutional network?**
- Parallelism
- Stable gradients
- Low memory requirements during training and evaluation
- Ability to deal with variable input lengths
* `[ C ]`


---

**Q: What is not a metric that the paper reviews architectures on**
- Parallelism
- Memory usage
- Classification performance
- Power usage
* `[ D ]`


---

**Q: Which of the following statements, regarding the empirical evaluation of a simple temporal convolutional network (TCN)  and generic recurrent architectures across a comprehensive suite of sequence modeling tasks, is false?**
- TCN models substantially outperform generic recurrent architectures such as LSTMs and GRUs.
- TCNs exhibit longer memory than recurrent architectures with the same capacity.
- Before the introduction of architectural elements such as dilated convolutions and residual connections,  convolutional architectures were indeed weaker.
- RNNs should still be regarded as a natural starting point and a powerful toolkit for sequence modeling, due to their comparable clarity and simplicity.
* `[ D ]`


---

**Q: Which of the following networks needs low memory requirements for training?**
- TCN
- LSTM
- GRU
- None of the above
* `[ A ]`


---

**Q: Which is the disadvantage of using the temporal convolutional network (TCN)?**
- Adaptive to the transfer of domains 
- Flexible receptive field size
- Low memory requirement for training
- Parallelism.
* `[ A ]`


---

**Q: Which of the following is not the advantage of the TCN (Temporal Convolutional Networks) ?**
- In both training and evaluation, a long input sequence can be processed as a whole in TCN, instead of sequentially as in RNN. The predictions for later timesteps can be processed parallelly with their predecessors. 
- TCNs can afford better control of the model's memory size. 
-   TCNs can avoid the problem of exploding/vanishing gradients, which is a major issue for RNNs. 
- TCNs outperform RNNs in data storage during evaluation. 
* `[ D ]`


---

**Q: Recurrent  networks  are  dedicated as**
- Sequence  models  
- Matrix models
- AI 
- Neural nets
* `[ A ]`


---

**Q: How does TCN produce an output of the same length as the input**
- Using Dilated Convolutions
- Using Causal Convolutions
- Using a 1D fully-convolutional network (FCN) architecture
- Using Residual Connections
* `[ C ]`


---

**Q: Select the incorrect statement.**
-  LSTMs and GRUs substantially outperforms temporal convolutional networks(TCN).
- TCNs exhibit longer memory than recurrent architectures with the
same capacity.
- A simple convolutional architecture is more effective across diverse sequence modeling tasks than recurrent architectures such as LSTMs.
- Convolutional networks should be regarded as a natural starting point and a powerful toolkit for sequence modeling.
* `[ A ]`


---

**Q: Which is NOT an advantage of TCN compared with RNN?**
- A long input sequence can be processed in parallel
- Avoid the exploding/vanishing of gradient
- Low memory requirement for training
- Low memory requirement for evaluation
* `[ D ]`


---

**Q: Which of the features are not a characteristic of Temporal Convolutional Networks: 1) "The convolutions are causal (meaning there is no information leaking from future to past". 2) A sequence of any length can be mapped to an output of the same length.**
- 1 - True. 2- False
- 1 - False. 2 - False
- 1 - True. 2 - True
- 1 - False. 2 - True
* `[ C ]`


---

**Q: Which of the following statements is true?**
- Basic RNN architectures are notoriously difficult to train.
- Basic RNN architectures are notoriously easy to train.
- Basic LSTM architectures are notoriously difficult to train.
- Basic RNN architectures are notoriously easy to train.
* `[ A ]`


---

**Q: Which of these statements is correct?**
- TCNs exhibit longer memory than recurrent architectures with the same capacity
- Dilations and residual connections can not be implemented in a TCN
- The TCN architecture has been known for years, therefore little advancements can still be made
- No parameter tuning is possible with a TCN
* `[ A ]`


---

**Q: What is not an advantage of TCN's over RNN's?**
- Parallelism 
- Flexible receptive field size
- Stable gradients
- Data storage during evaluation
* `[ D ]`


---

**Q: What is a distinctive feature of a temporal convolutional network (TCN)?**
- It maintains a vector of hidden activations that are propagated through time
- It only uses short history sizes to decrease memory usage
- It produces an output of the same length as the input
- None of the above
* `[ C ]`


---

**Q: Why are dilated convolutions used in Temporal Convolutional Networks (TCNs) for tasks such as sequence analysis. **
- They enable the network to handle dynamic input sizes.
- They prevent information leakage in the network.
- They enlarge the receptive field, providing the network with a longer memory. 
- By using them the number of filters is reduced and thus the network will be more performant.
* `[ C ]`


---

**Q: What is the goal of a dilution in an architecture for sequential modeling?**
- Increasing the spread of the values received in the input space for easier training
- Minimizing the effect of uncommon or unknown members of a sequence
- Progressive broadening of the input space of hidden layers
- Ignoring items which were input to the model long ago
* `[ C ]`


---

**Q: Can Temporal Convolutional Networks be drop-in replacement for RNNs?**
- Yes, by chopping up inputs to pre-defined lengths.
- Yes, by sliding the 1D convolutional kernels.
- No.
- Yes, by using multiple instances of the same network.
* `[ B ]`


---

**Q: Recurrent neural networks (RNN) are nowadays widely used for sequence modeling( such as audio systems) because of their theoretical infinite memory. Some researchers, comparing RNN to some other networks have discovered that:**
- RNN are very efficient even if errors occur that affect part of the memory they were working on
- Some kind of convolutional NNs are actually better for this kind of task
- Feed forward NNs can reach the same level of precision as  RNNs
- RNNs are also very good and useful in other tasks, especially in classification problems
* `[ B ]`


---

**Q: Which of the following statements is FALSE?**
- Dilation in TCN architectures can be used to increase the receptive field of the network, and thus the effective memory of the network
- Dilation and stride are the same for Convolutional Networks
- TCN architectures do not take future information into account
- TCNs can be used effectively for sequence modeling tasks
* `[ B ]`


---

**Q: Which of the following is/are true?
I. A long input sequence can be processed as a
whole in TCN, instead of sequentially as in RNN.
II. Gated RNNs use more memory compared to TCN.
III. TCN models outperform generic recurrent architectures such as LSTMs and GRUs**
- I-II
- II-III
- I-III
- I-II-III
* `[ D ]`


---

**Q: Which of these is not an advantage of using a temporal convolutional network (TCN)?**
- TCNs only need to maintain a hidden state and take in a current input in order to generate a prediction
- TCNs can take in inputs of arbitrary lengths by sliding the 1D convolutional kernels
- TCNs require a low amount of memory during training
- TCNs can perform convolutions in parallel
* `[ A ]`


---

**Q: Which of the following about Temporal Convolutional Network (TCN) is WRONG?**
- TCN is about to perform well on sequence modeling tasks.
- A TCN utilize causal convolutions to prevent information leak from the future to the past. 
- Dilated convolutions help to amplify the reception fields.
- All of the above.
* `[ D ]`


---

**Q: What is true about temporal neural networks**
- the convolutions in the architecture are causal, meaning that
there is no information “leakage” from future to past
- the architecture can take a sequence of any length and map it to
an output sequence of the same length.
- the architecture can take a sequence of any length and map it to
an output sequence of any length.
- temporal neural network have no distinguising features
* `[ B ]`


---

**Q: Which of the following statement about temporal convolutional network (TCN) is true?**
- TCN can take a sequence of any length and map it to an output sequence of the same length
- The convolutions in the architecture are causal
- TCN uses a 1D fully-convolutional network (FCN) architecture 
- All of the above
* `[ D ]`


---

**Q: hat is FALSE about TCN (Temporal Convolutional Network), developed in Bai et al. 2018 ?**
- in general TCN outperforms RNN in sequence data
- LSTM has longer memory than TCN
- TCN avoids the problem of exploding/vanishing gradients
- in general TCN uses less memory for training then CNN
* `[ B ]`


---

**Q: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" discusses an architecture called a Temporal Convolutional Network and states that "a simple causal convolution is only able to look back at a history with size linear in the depth of the network". Which additional component of a TCN allows for the exponential growth of the memory size of the network?**
- Dilated convolutional layers 
- Increasing the network depth
- Adding fully-connected layers
- Adding residual connections
* `[ A ]`


---

**Q: What is one of the advantages of Temporal Convolutional Networks in comparison to Recurrent Neural Networks?**
- TCNs have more flexibility with variable length inputs than RNNs
- TCNs have a more efficient way to store data during the evaluation than RNNs
- TCNs have fewer problems with vanishing/exploding gradients than RNNs
- TCNs are computational less expensive and therefore much faster than RNNs
* `[ C ]`


---

**Q: One of the advantages of the Temporal Convolutional Network (TCN) discussed in the paper “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling” is represented by the flexible receptive field size. What are possible ways to change the size of the receptive field, as discussed in the paper?**
- Stacking more dilated (causal) convolutional layers
- Using larger dilation factors
- Increasing the filter size
- All of the above
* `[ D ]`


---

**Q: Which of the following is true about the comparison between Convolutional Networks (CNN) and Recurrent Networks (RNN) for sequence modeling?**
- CNN's do not suffer from the exploding or vanishing gradients problems, as  is often the case for RNN's.
- RNN's have to store a lot of data in order to evaluate new data during testing.
- CNN's have to store a lot of data during training, because of the various filters used. RNN's, on the other hand, only have to store the past hidden state during training.
- All of the above.
* `[ A ]`


---

**Q: Which of the following statements is false?**
- Convolutional Networks using strided convolutions and residual connections are serious competitors for RNNs such as LSTMs across diverse sequence modeling tasks.
- Ceteris Paribus, training a Temporal Convolutional Networks is faster than training a RNN.
- Independent of the filter size and the number of layers, vanilla convolutions (no pooling) does not ‘see’ the entire history of the input sequence
- Recurrent Neural Networks (empirically) exhibit problems dealing with long-range (input-output) dependencies.
* `[ A ]`


---

**Q: Which of the following characteristics is a disadvantage in TCNs?**
- Parallelism
- Flexible receptive field size
- Stable gradients
- Data storage during evaluation
* `[ D ]`


---

**Q: Which of the following is a disadvantage of temporal convolutional networks?**
- Data storage during evaluation.
- Variable length inputs.
- High memory required for training.
- Unstable gradients
* `[ A ]`


---

**Q:  simple convolutional architecture is ..**
- More effictive across diverse sequence modeling tasks than recurrent architectures
- Less effective across diverse sequence modeling tasks than recurrent architectures
- None of the above
- All of the above
* `[ A ]`


---

**Q: Which of the following statements about is FALSE?**
- Temporal convolutional network (TCN) = 1D fully-convolutional network + causal convolutions. 
- A simple causal convolution is only able to look back at a history with size linear in the depth of the network.
- A major disadvantage of TCN is that in order to achieve a long effective history size, we need an extremely deep network or very large filters.
- Like in RNNs, with TCN the predictions for later timesteps must wait for their predecessors to complete, convolutions cannot be done in parallel.
* `[ D ]`


---

**Q: What of the following is true while comparing Temporal Networks and RNNs**
- CNNs need less memory since they do not need to store the hidden states, but rather only kernels, which are typically smaller in their size
- A TCN can take in variable length inputs, which is something RNNs cannot do
- A TCN cannot capture temporal dependencies since it is only a convolution, unlike RNNs which maintain hidden states and hence, can capture information of the past.
- None
* `[ A ]`


---

**Q: What is the nature of sequence modeling tasks?**
- Suppose for a given input sequence $x_{0}, ... , x_{T}$ and you want to predict some corresponding output $y_{0}, ... , y_{T}$ at each time. Then for a sequence modeling tasks a key constraint is that to predict the output $y_{T}$ for some time t, you need to use input $x_{T}$ as highest weight.
-  Suppose for a given input sequence $x_{0}, ... , x_{T}$ and you want to predict some corresponding output $y_{0}, ... , y_{T}$ at each time. Then for a sequence modeling tasks a key constraint is that to predict the output $y_{T}$ for some time t, you can only use the inputs that have been previously observed.
-  Suppose for a given input sequence $x_{0}, ... , x_{T}$ and you want to predict some corresponding output $y_{0}, ... , y_{T}$ at each time. Then for a sequence modeling tasks a key constraint is that to predict the output $y_{T}$ for some time t, you can only use the inputs $x_{T-1}, x_{T} and x_{T+1}$. So the previous current and next input.
-  None of the above answers is correct.
* `[ B ]`


---

**Q: What is not an advantage of the use of Temporal Convolutional Network?**
- In TCNs the convolutions can be applied in parallel
- TCNs only need to maintain a hidden state and take in a current input x_t in order to generate a prediction
- A TCN has a backpropagation path different from the temporal direction of the sequence
- A TCN shares its filters across a layer with the backpropagation path depending only on network depth
* `[ B ]`


---

**Q: Seeing as it is computationally infeasible, or at least ineficcient to use a very deep network or very large filters to get a large history size in TCNs, what techniques are used to still realize a large effective history size?**
- Dilated convolutions are used to exponentially increase the receptive field of the output nodes.
- Residual connections are used to enable the network to be as deep as it needs to be, while still making it possible to train within reasonable time.
- Both dilated convolutions and residual connections are used, and both benefit the network in the ways stated above.
- None of the listed techniques are used.
* `[ C ]`


---

**Q: What are the main advantages of Temporal convolutional networks for sequence modeling over RNN ?**
- stable gradients, Parallel predictions computation
- Parallel predictions computation,flexible receptive field size, stable gradients
- low memory for training, stable gradients, variable length inputs
- all of the above
* `[ D ]`


---

**Q: Why convolutional networks should be the first thing to try for sequence modelling?**
- This statement is wrong. Nowadays there are very powerful RNNs that are much better than convolutional networks
- Indeed, RNNs with advanced settings can outperform simple CNNs but a lot of work is required to achieve such powerful RNNs. If we update CNNs we can receive even better results - that is why a CNN is a nice first thing to try
- CNNs are always better that RNNs whatever the settings are. Even complex architectures of RNNs can not beat a simple CNN
- This statement is wrong. Training a CNN requires far more time than training an RNN. That is why RNNs is the best approach for sequence modelling
* `[ B ]`


---

**Q: Which type of network should be used according to the paper to sequence modeling?**
- Recurrent network
- Feed-Forward network
- Residual network
- Convolutional network
* `[ D ]`


---

**Q: Consider the following statements
a) Temporal Convolution Networks (TCNs) can be trained in parallel as opposed to RNNs
b) TCNs require less memory during evaluation than RNNs
c) By transfering a TCN model from one domain to another, no performance loss/gain can be observed.**
- Statement (a) is correct; Statement (b) is wrong; Statement (c) is wrong; 
- Statement (a) is wrong; Statement (b) is wrong; Statement (c) is correct; 
- All statements are wrong
- All statements are correct
* `[ A ]`


---

**Q: Which is not an advantage of a TCN over a RNN?**
- Low memory requirement for  training
- Stable gradients
- Being able to run in parallel instead of sequentially 
- Low memory requirement for evaluation
* `[ D ]`


---

**Q: what is a benefit of RNNS over traditional convolutional networks**
- they have theoretically unlimited memory
- they can be parallelized
- they can have more perceptrons
- they are easier to use
* `[ A ]`


---

**Q: One of the biggest advantages of using dilated convolution is:**
- effective receptive field of units grows exponentially with layer depth
-  better control of the model’s memory size
-  easy to adapt to different domains
- all of the above
* `[ D ]`


---

**Q: Which of the following statements is FALSE?**
- LSTM can be used on sequential data.
- Recurrent neural networks have the theoretical ability to retain information through sequences of unlimited length.
- Convolutional neural networks can not be used on sequential data.
- Temporal Convolutional Networks have longer effective memory than Recurrent Neural networks with the same capacity. 
* `[ C ]`


---

**Q: In what order is the maximum path length of a network with self attention? The size of the input sequence is indicated with $n$ and $r$ indicates the size of a local neighbourhood.**
- O(1)
- O(n)
- O(log n)
- O(n / r)
* `[ A ]`


---

**Q: Which of the following is true**
- In TCNs there can be no leakage from the future into the past
- TCNs receptive field can be increased by using a larger filter and increasing the dilation factor
- LSTMs and GRUs have more elaborate architectures than RNNs but are easier to train
- All of the above
* `[ D ]`


---

**Q: Which statements about the “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling” paper are true?

Statement1:  The distinguishing characteristics of TCNs is that  the convolutions in the architecture are causal, meaning that there is no information “leakage” from future to past;

Statement2: The distinguishing characteristics of TCNs is that the architecture can take a sequence of any length and map it to an output sequence of the same length, just as with an RNN. 

Statement3: A simple causal convolution is only able to look back at a history with size linear in the depth of the network.

Statement4: Whereas in standard ResNet the input is added directly to the output of the residual function, in TCN (and ConvNets in general) the input and output could have different widths. **
- 1
- 2 and 4
- 2 and 3
- All true
* `[ D ]`


---

**Q: What is not an advantage of a temporal convolutional network over RNNs?**
- TCN's can train and evaluate in parallel instead of sequentially
- TCN can change it's receptive field size and therefore afford better control of the models memory size
- TCN has lower memory requirements for training
- TCN has lower memory requirements during evaluation
* `[ D ]`


---

**Q: What is not a benefit of using CNNs for sequence modeling?**
- Low memory requirement for training.
- Variable length inputs are better supported.
- CNNs can be more easily parallelized. 
- Low memory requirement during evaluation.
* `[ D ]`


---

**Q: Which of the following charateristic is not an advantage or disadvantage of using TCNs for sequence modelling?**
- Parallelism
- Flexible recetive field size
- Stable gradients
- High memory requiremnet for training
* `[ D ]`


---

**Q: The temporal convolutional network combines the best practices of recurrent and convolutional architectures. Which of the following is not one of the best practices that are combined?**
- Dilations
- Residual connections
- Long memory
- Data storage during evaluation
* `[ D ]`


---

**Q: which of the following is not an advantage of using TCNs for sequence modelling?**
- Parallelism 
- Stable gradients 
- Data storage during evaluation
- Low memory requirement for training
* `[ C ]`


---

**Q: Which one are wrong?**
- It show that TCNs architectures are competitive with RNNs across a diverse set of discrete sequence tasks.
- They try to use the techniques: causal and Long history
- The advanced techniques introduced in this paper could help with training deep network and large respective fields, However, for causal: give the limitation of the receptive field(no future values), construct the structure which is more related to the previous input.
- None
* `[ D ]`


---

**Q: One of the disadvantages of Temporal Convolutional Networks (TCN) is:**
- Large memory requirement at training time
- Large memory requirement at testing time 
- Low parallelism degree
- Poor performance on tasks where long history is required
* `[ B ]`


---

**Q: Consider the following statements
a) Temporal Convolution Networks (TCNs) can be trained in parallel as opposed to RNNs
b) TCNs require less memory during evaluation than RNNs
c) By transfering a TCN model from one domain to another, no performance loss/gain can be observed.**
- Statement (a) is correct; Statement (b) is wrong; Statement (c) is wrong;
- Statement (a) is wrong; Statement (b) is wrong; Statement (c) is correct;
- All statements are wrong
- All statements are correct
* `[ A ]`


---

**Q: Assume for a sequence model, there are time steps from $1,2,…,T$, and there is a midway step t, for the $t$^{th} output, which description is correct?**
- It depends on input $t$
- It depends on $1,2,…,T$
- It depends on $1,2,…,t$
- It depends on $1,2,…,t-1$
* `[ C ]`


---

**Q: Conserning the dilation factor d of equation 2 of paper 12, which statement is correct?**
- If d > 1 the receptive field of a ConvNet is expanded because the output at the top level of the network represents a wider range of inputs than d = 1.
- If d = 1 the receptive field of a ConvNet is expanded because the output at the top level of the network represents a wider range of inputs than d > 1.
- If d = 1 the receptive field of a ConvNet is singular because the output at the top level of the network represents a wider range of inputs than d > 1.
- If d > 1 the receptive field of a ConvNet is singular because the output at the top level of the network represents a smaller range of inputs than d = 1.
* `[ A ]`


---

**Q: What is the advantage of multi-head attention over dot product attention**
- The network can focus on multiple areas of the input at once
- The network can generate multiple outputs for a given input
- The network is able to distribute attention over the hidden representation and the input
- The attention is a normalized distribution.
* `[ A ]`


---

**Q: What is not a valid motivation for the use of self-attention models over RNNs:**
- Self-attention models have lower computational complexity per layer
- Self-attention models achieve higher parallelization
- Self-attention models have a higher path length between long-range dependencies, which makes them easier to learn
- All are valid motivation
* `[ C ]`


---

**Q: which statement is wrong?**
- Recurrent networks are dedicated sequence models that maintain a vector of hidden activations that are propagated through time. 
-  The “infinite memory” advantage of RNNs is largely absent in practice.
-  In RNNs, the predictions for later timesteps doesn't have to wait for their predecessors to complete
- Non of them.
* `[ C ]`


---

**Q: The paper argues that convolutional networks can often work as well or even better than recurrent networks. What is one disadvantage of using a CNN rather than an RNN though?**
- It is not as flexible with input sizes
- Data can not be computed in parallel
- It may need more memory during evaluation
- It is more difficult to calculate the gradient during backpropagation
* `[ C ]`


---

**Q: In the paper by Bai et al. (2018), they talk about how generic convolutional networks (in their case: a Temporal Convolutional Network, or TCN) have some advantages over recurrent networks (such as LSTMs) doing sequence modelling. What can be said about the following characteristics of using a generic convolutional network for sequence modelling?
1)	When using RNNs, predictions for later time steps must wait for their predecessors to complete. When using TCNs, convolutions can be done in parallel since the same filter is used in each layer.
2)	In evaluation / testing, RNNs only need to maintain a hidden state and take in a current input x in order to generate a prediction. TCNs need to take in the whole sequence up to the effective history length.**
- a) TCNs have an advantage over RNNs in both statements
- b) TCNs have an advantage over RNNs only in the first statement
- c) TCNs have an advantage over RNNs only in the second statement
- d) TCNs have an advantage over RNNs in neither statements
* `[ B ]`


---

**Q: What is NOT the advantage of using Temporal Convolutional Networks instead of an RNN?**
- TCN can be done in parallel since the same filter is used in each layer.
- A TCN can change its receptive field size in multiple ways thus afford better control of the model's memory size, and are easy to adapt to different domains.
- TCN avoids the problem of exploding/vanishing gradients, which is a major issue for RNNs.
- TCNs does not need to take in the raw sequence up to the effective history length, thus possibly requiring less memory during evaluation.
* `[ D ]`


---

**Q: In the paper: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling", the authors propose a Temporal convolution neural network which outperforms RNN architectures on some sequence transduction tasks. Which of the following is not an advantage of TCN over RNNs**
- It has a higher level of parallelism as computation is not sequential
- It has lower memory requirements when training
- It has lower memory requirements when testing/evaluating
- It avoids the vanishing/exploding gradient problem by using different forward and backward pass paths
* `[ C ]`


---

**Q: Consider CNNs and RNNs. What holds true?**
-  In sequence modeling RNNs always outperforms CNNs
-  In sequence modeling CNNs always outperforms RNNs
- CNNs tends to perform better overal
- none of the above
* `[ A ]`


---

**Q: Temporal Convolutional Networks is based on which of the following?**
- The fact that the network produces an output of the same length as the input.
- The fact that there can be no leakage from the future into the past.
- Both a and b
- None of the above
* `[ C ]`


---

**Q: Which of the following are advantages of Temporal Convolution Networks (TCNs)?**
- Parallelism (used across layers)
- Stable Gradient present
- Requires taking raw input along the depths of history
- Arbitrary input lengths by sliding the 1D Conv Kernels.
* `[ C ]`


---

**Q: What is a disadvantage of a Temporal Convolutional Network (TCN)?**
- Fixed length inputs
- Unstable gradients
- High memory requirements for training
- Sequential processing
* `[ D ]`


---

**Q: TCN are ________ than CNN **
- less expensive
- more expensive
- equally expensive 
- none
* `[ B ]`


---

**Q: Which of the following sentences are true for the comparison of an RNN and a TCN according to the paper?**
- The reason that TCNs outperforms the RNNs lies to the fact that TCNs have infinity memory in comparison with the RNNs
- TCNs performs better for image classification tasks than the RNNs
- TCNs may even perform better than RNNs in LSTMs tasks even with the big advantage of the RNNs (infinity memory)
- RNNs always outperform the TCNs in sequential data tasks
* `[ C ]`


---

**Q: Which of the following statements is/are true about temporal convolutional networks (TCNs)?**
- Convolutions in the TCNs architecture are causal. 
- The TCNs architecture can take a sequence of any length and map it to an output sequence of the same length. 
- TCNs do not use gating mechanisms. 
- All of the above. 
* `[ D ]`


---

**Q: Which of the following is a distinguishing characteristic of Temporal convolutional network?**
- The convolutions in the architecture have no information “leakage” from future to past
- The Architecture can take a sequence of any length and map it to an output sequence of the same length
- Both of the above
- None of the above
* `[ C ]`


---

**Q: What is not an advantage of a TCN (temporal convolutional network)**
- parellelism
- stable gradients
- variable length inputs
- data storage during evaluation
* `[ D ]`


---

**Q:  What is a distinguishing characteristic of a TCN?**
- There is no information leakage from future to past
- A TCN can take any length sequence and map it to an output of any length
- Both of the above
- None of the above
* `[ C ]`


---

**Q: What is the advantage of RNNs over CNNs?**
- They are more accurate
- They suffer less from vanishing gradient problem due to long dependencies
- They can be used for flexible input/output sizes
- They require less data storage during evaluation
* `[ D ]`


---

**Q: What are the two principles of Temporal Convolutional Networks?**
- The network produces an output of the same length as the input.
- There can be no leakage from the future into the past.
- Both A and B.
- None of the above.
* `[ C ]`


---

**Q: What could be a point of critisism of the promising results obtained in the paper: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks
for Sequence Modeling".**
- The datasets chosen could be biased towards a better performance.
- The paper does not look at other types of architectures of RNNs.
- Both A and B.
- Neither A and B.
* `[ C ]`


---

**Q: After paper 12, sequence modelling...**
- ...is always better with recurrent networks.
- ...might be better with convolutional nets than recurrent nets.
- ...is always better with convolutional nets.
- ...cannot be done with neither recurrent nets nor convolutional nets.
* `[ B ]`


---

**Q: Which of the following approaches is NOT suited for processing sequential modelling problems, such as  language modelling and machine translation?**
- Convolutional Neural Networks
- Recurrent Neural Networks
- Long Short-Term Memory
- Principle Component Analysis
* `[ D ]`


---

**Q: What is one characteristic of a Temporary Convolutional Network (TCN)?**
- The architecture can take sequence of any length and output a sequence of the same length
- The neural network is built using convolutions of a long fixed size in 1d allowing to look in the past and the future.
- All of the above
- None of the above
* `[ A ]`


---

**Q: What is NOT an advantage of TCN's **
- Parralelism
- Stable gradients
- Low memory requirement for training
- Low memory requirement during evaluation
* `[ D ]`


---

**Q: Which of the following is false with respect to RNNs and CNNs in sequence modelling?**
-  LSTM always outperform CNNs in sequence modelling problems.
- CNNs should be regarded as a natural starting point for sequence modelling problems. 
- LSTMs are typically used for sequence modelling problems.
- In some cases using CNNs may be better than RNNs for sequence modelling problems.
* `[ A ]`


---

**Q: -**
- -
- -
- -
- -
* `[ A ]`


---

**Q: Which option below could contribute to a longer effective history size of TCN, compared with generic RNN?**
- Casual architecture
- Same-size imput and output
- Residual connections and dilations
- All of the three
* `[ C ]`


---

**Q: Temporal Convolutional Networks characteristics can be summed to:**
- 1) the convolutions are i.i.d 2) output layers transforms every input to a normal distribution
- 1) the convolutions in the architecture are causal 2) the architecture can take a sequence of any length and map it to an output sequence of the same length.
- 1) the first convolution to pooling layer ensures the architecture is size independent for inputs 2) the inbetween layers can be inspected at anytime (temporal) during training
- 1) the pooling of the architecture is causal 2) the convolutional layers are time invariant, so all information stays at it's time step
* `[ B ]`


---

**Q: The paper challenges the common assumption that RNN's should be used for processing sequential data and instead CNN's should be used. Which of these statements doesn't support this claim?**
- RNN's have unlimited memory
- TCN's require little memory for training
- TCN's have the ability to run different part of a sequence in parallel.
- TCN's can change their receptive field size  
* `[ A ]`


---

**Q: Is a long or short history suitable for TCNs and canonical recurrent architectures such as LSTMs and GRUs**
- long; short
- short; long
- long; long
- short; short
* `[ C ]`


---

**Q: Which network architectures (out of TCN – Temporal Convolutional Network and RNN – Recurrent Neural Networks) would you recommend for the following task descriptions. 

P : To process a sequence to get another sequence in which relations between elements which are far away from each other in the input sequence are imporatnt.
Q: A real-time classifier for elements of a long sequence that are dependent only on the past values. An additional requirement is that each element should be classified as soon as it is presented to the network (ie, before future elements are available.)**
- P: TCN , Q: RNN
- P: TCN , Q: TCN
- P: RNN , Q: TCN
- P: RNN , Q: RNN
* `[ A ]`


---

**Q: Apologies, did not manage to find a question on time**
- nan
- nan
- nan
- nan
* `[ A ]`


---

**Q: What is not an advantage of a Temporal Convolutional Network**
- Long sequence can be processed as a whole instead of sequential in an RNN
- The data stored during evaluating or testing in less in a TCN than in a RNN
- TCNs avoid the problem of vanishing and exploding gradients
- Using convolution the data seen or interpreted can be scaled
* `[ B ]`


---

**Q: Which of the following is not right**
- TCN models substantially outperform generic recurrent architectures 
- infinite memory advantage of RNNs is largely absent in practice
- For sequence modeling, convolutional networks should only be considered if RNN cannot solve the problem
-  TCNs
exhibit longer memory than recurrent architectures with the
same capacity
* `[ C ]`


---

