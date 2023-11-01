
# Multi-head attention

The goal of the next two exercices is to delve deeper into Neural Network architecture.
We will be implementing the GPT2 model [1] from scratch including multi head attention with masking and transformer blocks [2].
You can find (and use for support) the original code here: [gpt-2](https://github.com/openai/gpt-2/blob/master).

In the first part of the assignment we take a closer look at the attention layer and create a multi head attantion function.

In the second assignment we finish our utility functions and create model.

Note: to run the code you must install the requirements in `requirements.txt`. You can do this with pip by running something like:

[1]: ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

[2]: ["Attention is all you need"](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)


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

Example input and output: `softmax(np.array([[-1., 0.], [0.2, 1.]]))` -> `[[0.26894142 0.73105858], [0.31002552 0.68997448]]`



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
```
    np.random.seed(4321)
    q = np.random.rand(3,2)
    k = np.random.rand(3,2)
    v = np.random.rand(3,2)
    x = attention(q, k, v)
```
-> 

```
    [[0.37285946 0.73278279]
     [0.36712163 0.72522747]
     [0.36637032 0.72842298]]
```

### Section 1.3

Now we add a mask to the attention function.
Create a function `masked_attention` that has an additional input parameter `mask`.

$$
	Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}} + mask)V
$$

Example input and output:
```
np.random.seed(4321)
nf = 10
q = np.random.rand(nf,2)
k = np.random.rand(nf,2)
v = np.random.rand(nf,2)
mask = (1 - np.tri(nf)) * -1e10
x = masked_attention(q, k, v, mask)
```

->

```
[[0.37646796 0.24378126]
 [0.48299578 0.28439644]
 [0.46590072 0.41837738]
 [0.52991302 0.51314059]
 [0.49214214 0.55574465]
 [0.39568092 0.59955323]
 [0.38462954 0.61108759]
 [0.37248739 0.5645996 ]
 [0.35915127 0.57331419]
 [0.41913397 0.51187079]]
```


## Section 2

### Section 2.1

Create a function `linear_projection` that performs a linear projection on input `x` given weights `w` and bias `b`:

$$
    x \times w + b
$$

Both inputs `x` and `w` are matrices.


Example input and output:

```
np.random.seed(4321)
x = np.random.rand(3,2)
w = np.random.rand(2,3)
b = np.random.rand(3,1)
lp = linear_projection(x, w, b)
```

->

```
[[0.49964645 0.7764272  0.59947811]
 [1.0642018  1.42264665 0.86367775]
 [1.06047186 1.43087917 1.14610938]]
```
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

Example input and output:

```
np.random.seed(4321)
x = np.random.rand(3,4)
w_1 = np.random.rand(4,12)
b_1 = np.random.rand(3,1)
w_2 = np.random.rand(4,3)
b_2 = np.random.rand(3,1)
attn = {"c_attn": {"w": w_1, "b": b_1}, "c_proj": {"w": w_2, "b": b_2}}
x = multi_head_attention(x, attn, 2)

```

->

```
[[3.4897257  2.74884012 2.6448295 ]
 [3.15425828 2.46024887 2.34563449]
 [3.22513764 2.50993895 2.38375606]]
```

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

Example input and output: `gelu(np.array([[-1., 0.], [0.2,  1.]]))` -> `[[-0.15880801  0.] [ 0.11585143  0.84119199]]`


### Section 1.2

Create a function `layer_normalization` that normalizes and scales an input matrix `x` **row-wise** with `gamma` and `beta` similar to [this paper](https://arxiv.org/abs/1607.06450).

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

```
np.random.seed(4321)
x = np.random.rand(3,2)
g = np.random.rand(3,2)
b = np.random.rand(3,1)
ln = layer_normalization(x, g, b)
```

->

```
[[-0.18790462  0.97604994]
 [ 0.75266431  0.35366349]
 [ 0.05977512  1.13857828]]
```

## Section 2

We have all our utility functions, lets create the model.

### Section 2.1

Our feed forward neural network layer is quite simple. Create a function `feed_forward_network` that does the following steps:
   
1. First linear projection with `w_1` and `b_1`
2. Activation function (Gelu)
3. Second linear projection with `w_2` and `b_2`

Example input and output:

```
np.random.seed(4321)
x = np.random.rand(3,4)
w_1 = np.random.rand(4,5)
b_1 = np.random.rand(3,1)
w_2 = np.random.rand(5,4)
b_2 = np.random.rand(3,1)
mlp = {"c_fc": {"w": w_1, "b": b_1}, "c_proj": {"w": w_2, "b": b_2}}
x = feed_forward_network(x, mlp)
```

->

```
[[3.50980416 2.64636922 3.27141858 2.96212932]
 [4.45049282 2.74903161 3.7033384  3.07794882]
 [3.19782584 2.47054632 2.96733082 2.75125028]]
```
        

### Section 2.2

Create a function `transformer_block` that does the following steps:

1. First layer normalization with `g_1` and `b_1`
2. Forward pass through multi head attention
3. Add input x
4. Store x for later
5. Second layer normalization with `g_2` and `b_2`
6. Forward pass through feed forward network
7. Add stored x


### Section 2.3

Create a function `gpt2` that does the following steps:

1. Get word and positional embedding (given)
2. Forward pass through all transformer blocks in `blocks`
3. Layer normalization with `g_final` and `b_final`
4. Map back from embedding to vocabulary (given)

### Section 2.4

Use the `generate` function to run your model. Produce a few examples with your own input and submit as `2_4.txt`.
You can use a bigger model to produce better results.
To try different sized models change `model_size` to any of `["124M", "355M", "774M", "1558M"]`. keep in mind the bigger models will take more space on your computer and longer to run.

Example input and output with the smallest model:

1.

```
generate("Hello! How are you?")
``` 

-> 

```
I'm a little bit nervous. I'm not sure if I'm going to be able to do this, but I'm going to be able to do it. I'm going to be
```

2.

```
generate("What is the weather like tomorrow?")
``` 

-> 

```
The weather is pretty good today. The weather is pretty good today.
The weather is pretty good today. The weather is pretty good today.
The weather is pretty good today
```

3.

```
generate("Tell me a story")
```

 -> 
 
 ```
 about a guy who was a good friend of mine who was a good friend of mine who was a good friend of mine who was a good friend of mine who was a good friend of mine who was
 ```

## What to turn in to Gradescope
*Read this carefully before you submit your solution.*

Submit like regularly in **12 Transformers**.

This is an individual project so you can of course help each other out but your report should be your own.

Files to turn in:

- `template.py` (Same file as previously)

Make sure the file names are exact.
