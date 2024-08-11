# Neural_Networks_0-1_OCR
This was a workshop under IEEE SIES GST, focuses on core principles of Machine Learning. More we covered concepts from basic math used in computing a neuron to classifying 10 digit numbers using a fully connected Neural Network


## Lets start with computing output of a single neuron. the formula goes as:

$$ \text{neuron output} = \mathbf{input} \cdot \mathbf{weight} + bias $$

## Now, lets compute multiple neurons at once, also known as a layer of neurons.

$$ \text{output} = \sum_{i=0}^{n} (\text{input}_i \cdot \text{weight}_i) + \text{biases} $$

Here input and weight are arrays of float values of which dot product is calculated.

## ReLU Activation Function.
$$ \text{ReLU}(x) = \max(0, x) $$

ReLU is mostly used to activate hidden layers.

## SoftMax Activation

$$ \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}} $$

## Loss Calculation (Categorical Cross-Entropy)

$$ \text{Loss} = -\sum_{i} y_i \log(p_i) $$

## Adam Optimizer

### Compute the biased first moment estimate $m_t$
$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
$$

- where $g_t$ is the gradient at time step $t$.

### Compute the biased second raw moment estimate $v_t$:

$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
$$
### Compute bias-corrected first moment estimate $\hat{m}^t$:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

### Compute bias-corrected second raw moment estimate $\hat{v}^t$:
$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t $$

### Update the parameters $θ$:
$$\theta_t = \theta_{t-1} - \frac{\alpha \cdot \hat{v}_t}{\sqrt{\hat{m}_t} + \epsilon}$$

We have created this project under the guidance of [Adityadikonda10](https://github.com/Adityadikonda10) and thankful to the entire IEEE team who had organised the workshop and excited to attend more sessions like this.

