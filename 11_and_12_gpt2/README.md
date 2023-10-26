
# Multi-head attention

The goal of the next two exercices is to delve deeper into Neural Network archetecture.
We will be implementing the GPT2 model from scratch including multi head attention with masking and transformer blocks.

In this exercise we take a closer look at the attention layer and create a multi head attantion function.

In the second exercise we finish our basic functions and create the transformer blocks.

Note: to run the code you must install the requirements in `requirements.txt`. You can do this with pip by running something like:

```
# If you are using a virtual environment, make sure it is activated before you install the requirements
# Also, make sure you run this with the same python you run your code with

python3 -m pip install -r requirements.txt

# or

pip3 install -r requirements.txt

# or 

pip install -r requirements.txt
```

## Section 1

### Section 1.1

Let's start with creating our first activation function, Softmax:

$$ 
    \sigma(z_i) = \frac{e^{z_{i}-max(Z)}}{\sum_{j=1}^{K} e^{z_{j}-max(Z)}} \ \ \ for\ i=1,2,\dots,K
$$

where $Z = {z_0, z_1, ..., z_K}$

Create the Softmax function `softmax(x)`, our input `x` is a matrix where each row is a $Z$ vector.

Hint: Use `keepdims=True`

Example input and output: `softmax(np.array([[-1, 0,], [0.2, 1]])` -> `array([[0.26894142, 0.73105858], [0.31002552, 0.68997448]])`



### Section 1.2
Create a function `attention`, that computes the attention values given a set of queries, keys and values.

$$
	Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

inputs:
    - q: queries $Q$
    - k: keys $K$
    - v: values $V$

Hint: Use `@` for matrix multiplication.

Example input and output:
* `attention(...)`

### Section 1.3

Now we add a mask to the attention function.
Create a function `maked_attention` that has an additional input parameter `mask`.

$$
	Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}} + mask)V
$$

Example input and output:
* `masked_attention(...)`


## Section 2

### Section 2.1

Create a function `linear_projection` that performs a linear projection on input `x` given weights `w` and bias `b`:

$$
    x \times w + b
$$

Both inputs `x` and `w` are matrices.

### Section 2.2

Now lets create the multi head attention layer in our GPT2 Model.

Create a function `multi_head_attention`, the function should perform the following steps in order:

   1. First linear projection with `w_1` and `b_1`*
   2. Separate the $Q$, $K$ and $V$ matrices  
   3. Split each of $Q$, $K$ and $V$  into heads**
   4. Perform masked attention over each head
   5. Merge heads (horizontally)
   6. Second linear projection with `w_2` and `b_2`

*this produces a matrix $[Q K V]$ where $Q$, $K$ and $V$ are the queries, keys and value matrices 

**Similar to the previous step, split the three matrices into heads, the number of heads is given with the `number_of_heads` parameter. 

Use the `mask` given in the code.


## What to turn in to Gradescope
*Read this carefully before you submit your solution.*

Submit like regularly in **11 Sequential Data with Neural Networks**.

This is an individual project so you can of course help each other out but your report should be your own.

Files to turn in:

- `template.py`

Make sure the file names are exact.

# Transformer blocks and GPT2

## Section 1

We need a few more utility functions.

### Section 1.1

Our second activation function, Gelu:

$$
Gelu(x) = x\Phi(x) \approx 0.5 x(1+\tanh[\sqrt{2/\pi}(x+0.044715x^3] )
$$

Create The Gelu function `gelu`.

Example input and output: `gelu([-1, 0, 0.2, 1])` -> `array([[-0.15880801,  0.        ], [ 0.11585143,  0.84119199]])`


### Section 1.2

Create a function `layer_normalization` that normalizes and scales an input matrix `x` **row-wise** with `gamma` and `beta`.

$$
    h = g \odot N(x) + b, N(x) = \frac{x-\mu}{\sqrt{\sigma+\epsilon}}
$$
where
$$
    \mu = \frac{1}{H}\sum_{i=1}^{H}x_i, \sigma = \frac{1}{H}\sum_{i=1}^{H}(x_i - \mu)^2
$$

You can use `numpy.mean` and `numpy.var` to find the $\mu$ and $\sigma$.
Use $\epsilon$ = `1e-5`. 

Hint: Use `axis=1` and`keepdims=True`

Example input and output:
* `layer_normalization(x, gamma, beta)`

## Section 2

We have all our utility functions, lets create the model.

### Section 2.1

Our feed forward neural network layer is quite simple. Create a function `feed_forward_network` that does the following steps:
    1. First linear projection with `w_1` and `b_1`
    2. Activation function (Gelu)
    3. Second linear projection with `w_2` and `b_2`


### Section 2.2

Create a function `transformer_block` that does the following steps:

1. First layer normalization with `g_1` and `b_1`
2. Forward pass through multi head attention
3. Add input x and store
4. Second layer normalization with `g_2` and `b_2`
6. Forward pass through feed forward network
7. Add stored x


### Section 2.3

Create a function `gpt2` that does the following steps:

1. Get word and positional embedding (given)
2. Forward pass through all transformer blocks
3. Layer normalization with `g_final` and `b_final`
4. Map back from embedding to vocabulary (given)


## What to turn in to Gradescope
*Read this carefully before you submit your solution.*

Submit like regularly in **12 Transformers**.

This is an individual project so you can of course help each other out but your report should be your own.

Files to turn in:

- `template.py` (Same file as previously)

Make sure the file names are exact.