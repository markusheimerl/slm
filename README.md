# slm
A small language model implementation

Consider a character-level language model built on a state space model backbone with MLP projection, operating on character sequences of shape (seq_len × batch_size). The architecture combines learned character embeddings with temporal state dynamics through SSM, followed by MLP transformation and softmax normalization for next-character prediction. The forward propagation follows:

$$
\begin{align*}
E_t &= W_E[X_t] \\
H_t &= E_tB^T + H_{t-1}A^T \\
O_t &= H_t\sigma(H_t) \\
Y_t &= O_tC^T + E_tD^T \\
Z_t &= Y_tW_1 \\
A_t &= Z_t\sigma(Z_t) \\
L_t &= A_tW_2 \\
P_t &= \frac{\exp(L_t)}{\sum_c \exp(L_{t,c})}
\end{align*}
$$

The embedding matrix $W_E$ maps discrete character indices to dense vector representations via indexing $W_E[X_t]$. The SSM processes embedded inputs through state transition matrix $A$, input matrix $B$, output matrix $C$, and feedthrough matrix $D$. The MLP then transforms SSM outputs through weight matrices $W_1$ and $W_2$ with Swish activation $z\sigma(z)$. Finally, softmax normalization produces probability distributions over the character vocabulary.

For language modeling, the cross-entropy loss between predicted and actual next characters is minimized, where $\odot$ denotes elementwise multiplication:

$$
\begin{align*}
\mathcal{L} &= -\frac{1}{T \cdot B}\sum_{t=1}^{T}\sum_{b=1}^{B} \log P_{t,b,y_{t,b}} \\
\frac{\partial \mathcal{L}}{\partial L_t} &= P_t - \mathbf{1}_{y_t} \\
\frac{\partial \mathcal{L}}{\partial W_2} &= A_t^T(\frac{\partial \mathcal{L}}{\partial L_t}) \\
\frac{\partial \mathcal{L}}{\partial A_t} &= (\frac{\partial \mathcal{L}}{\partial L_t})(W_2)^T \\
\frac{\partial \mathcal{L}}{\partial Z_t} &= \frac{\partial \mathcal{L}}{\partial A_t} \odot [\sigma(Z_t) + Z_t\sigma(Z_t)(1-\sigma(Z_t))] \\
\frac{\partial \mathcal{L}}{\partial W_1} &= Y_t^T(\frac{\partial \mathcal{L}}{\partial Z_t}) \\
\frac{\partial \mathcal{L}}{\partial Y_t} &= (\frac{\partial \mathcal{L}}{\partial Z_t})(W_1)^T
\end{align*}
$$

The gradient then flows backward through the SSM following standard BPTT with Swish derivatives:

$$
\begin{align*}
\frac{\partial \mathcal{L}}{\partial C} &= \sum_t (\frac{\partial \mathcal{L}}{\partial Y_t})^T O_t \\
\frac{\partial \mathcal{L}}{\partial D} &= \sum_t (\frac{\partial \mathcal{L}}{\partial Y_t})^T E_t \\
\frac{\partial \mathcal{L}}{\partial O_t} &= (\frac{\partial \mathcal{L}}{\partial Y_t})C \\
\frac{\partial \mathcal{L}}{\partial H_t} &= \frac{\partial \mathcal{L}}{\partial O_t} \odot [\sigma(H_t) + H_t\sigma(H_t)(1-\sigma(H_t))] + (\frac{\partial \mathcal{L}}{\partial H_{t+1}})A \\
\frac{\partial \mathcal{L}}{\partial A} &= \sum_t (\frac{\partial \mathcal{L}}{\partial H_t})^T H_{t-1} \\
\frac{\partial \mathcal{L}}{\partial B} &= \sum_t (\frac{\partial \mathcal{L}}{\partial H_t})^T E_t \\
\frac{\partial \mathcal{L}}{\partial W_E[c]} &= \sum_{\substack{t,b \\ X_{t,b}=c}} \left(B^T\frac{\partial \mathcal{L}}{\partial H_t} + D^T\frac{\partial \mathcal{L}}{\partial Y_t}\right)
\end{align*}
$$

where $\mathbf{1}_{y_t}$ denotes the one-hot encoding of target characters. The embedding gradients accumulate contributions from all occurrences of each character across the sequence.

During generation, the model maintains hidden state across timesteps and samples from the predicted character distribution using temperature-controlled sampling:

$$
P_{\tau}(c) = \frac{\exp(L_c / \tau)}{\sum_{c'} \exp(L_{c'} / \tau)}
$$

where temperature $\tau$ controls sampling entropy - $\tau \rightarrow 0$ approaches argmax sampling while $\tau > 1$ increases randomness.

The AdamW optimizer maintains exponential moving averages for all parameters $\theta = \{A, B, C, D, W_E, W_1, W_2\}$ with momentum $\beta_1$, second moment $\beta_2$, and weight decay $\lambda$. The learning rate is denoted by $\eta$, $t$ is the current training iteration, and $\epsilon$ is a small constant for numerical stability. For each weight matrix $W$, the update rule is:

$$
\begin{align*}
m &= \beta_1m + (1-\beta_1)(\frac{\partial L}{\partial W}) \\
v &= \beta_2v + (1-\beta_2)(\frac{\partial L}{\partial W})^2 \\
W &= (1-\lambda\eta)W - \eta\cdot\frac{m}{1-\beta_1^t}/\sqrt{\frac{v}{1-\beta_2^t} + \epsilon}
\end{align*}
$$

The implementation leverages CUDA for parallel computation across batch and sequence dimensions, with efficient kernel implementations for embedding lookup, Swish activation, softmax normalization, and gradient accumulation. Character sequences are extracted from text corpora through random position sampling, providing diverse training contexts.

## How to run
```
sudo apt update
sudo apt install clang time curl
# Ensure CUDA toolkit is installed
make run
```

## Performance

At ~1 million parameters, 300 million tokens of corpus data and ~35 thousand TFLOPs of invested compute the model achieves ~1.5 cross entropy loss at the character level. The model can spell many words correctly and has some basic understanding of grammar but exhibits little semantic understanding overall.

<details>
<summary>Sample generation</summary><pre><code>
...
Batch [50000/100000], Loss: 1.510576

--- Sample Generation at Batch 50000 ---
Seed: "The quick brown fox"
Generated: ing the travail in lines contain with a few times to get their was sure in a picture. He was all repeated the demonas colland. See Scale, HECE BEhalam.  Heavy and took a difficulty of Roman, for access, made burned his hands of the person particular congunity of pinconsists described with in the stood for the same praced that the character a way the name waite the round up with ravide ardicate came he towards the combe and the flesh consideration.  The world. And the Ring; If. F. strong being place of the solitation of which the servants; striding the house beneath in conducted credition to keep and the emember. In arm older to impression round be soundation; and the are. Well, we have experient the exceeded by the season, or our company monking, the atten and quaintly speaking any of his trank and right was ejsert; and could away, to talk way, but the General, and day. In could not the most as the only to revel adory not the door of storie of the faith was that a more line; but it was
Seed: "Once upon a time"
Generated: l save or but thoughts, that had this possessed of a new for ne kissed and of the polittered to answered.  de loss unmost at the sailing. The distined Turned they were so purpose of the head of gentleman out of the utterly wonch you will be did not be paper, and itself in the happer measures and bulies back that he was doman at her not vasidly displaying his mouth and sound the monger to little as thou tale and Sardlock with the other me tall the first enother with anybody had places of thoughts upon his time. She was carts of the body shall, for was the first Rome. And I was she important of section to-more of the two phoround for her form of pour, and have been his languisdared the moon. Pather Cyship of civiolational. Phily Auntain to any present point him in the table in the though the world the demmences of the discouracious and gentlemance, and them from the subleman, and one hills.  To see rather; and the authority of the south his eyemed to a sole soft the to know of the was go
Seed: "In the beginning"
Generated: le to the under into, the faction: of the inclined to recorded the author horses of with a less comforted, and added that keep was to soon and softed to be she was he had seemed pick. Dam and for threw he speaked of the body water the shooks were and see head of an old spendity. I hatendly could have him of point. But if it was in God it was do office in his galls; and equalified, the engrip in has been since which is in press of ted pleased to the next and all the socinion. So I was as all rise precistinct on the differed, as yet Among them. She rest the quarters constered and the last down me to him door: the Chars of God might peness were the color for wheel eres. The offerster, I hat been believe of his hair and the attemptury of the smoothing the moistery regulation make any complying that work; I shamed a real to that he had despare of his named. The two well advances of the work of the look of the convesity, and country (whom the brumber of whose wheel contempt to week. The look
--- End Sample Generation ---
...
</code></pre></details>