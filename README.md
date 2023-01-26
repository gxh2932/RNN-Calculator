Recurrent neural networks (RNNs) are universal Turing machines. This means that we can perform basic arithmetic using their architecture. The following is a toy example of how RNNs can be constructed to perform addition with handcrafted parameters.

The input to the RNN will be two binary numbers, starting with the *least* significant bit. The longest binary number will be padded with an additional zero on the left side and the other number will also be padded with zeros such that they are both the same length. For instance, the equation

$$
100111+110010=1011001
$$

would be input to your RNN as:
- Input 1: $1, 1, 1,0,0,1,0$
- Input 2: $0,1,0,0,1,1,0$
- Correct output: $1,0,0,1,1,0,1$

The RNN has two input units and one output unit. In this example, the sequence of inputs and outputs would be:

![example](https://user-images.githubusercontent.com/29025503/214734148-7a9157df-d225-48b4-8080-8c5f0434c97c.png)


The RNN has three hidden units, and all of the units use the following non-differentiable hard-threshold activation function

```math
\sigma(a)=\left\{\begin{array}{cc}
1 & \text { if } a>0 \\
0 & \text { otherwise }
\end{array}\right.
```

The equations for the network are given by

$$
\begin{aligned}
& y_t=\sigma\left(\mathbf{v}^T \mathbf{h}_t+b_y\right) \\
& \mathbf{h}_t=\sigma\left(\mathbf{U x}_t+\mathbf{W} \mathbf{h}_{t-1}+\mathbf{b}_{\mathbf{h}}\right)
\end{aligned}
$$

where $\mathbf{x}\_t \in \mathbb{R}^2$, $\mathbf{U} \in \mathbb{R}^{3 \times 2}$, $\mathbf{W} \in \mathbb{R}^{3 \times 3}$, $\mathbf{b}_{\mathbf{h}} \in \mathbb{R}^3$, $\mathbf{v} \in \mathbb{R}^3$, and $b_y \in \mathbb{R}^2$.

We define our parameters as follows:

```math
U = \begin{bmatrix}1 & 1 \\ 1 & 1 \\ 1 & 1 \end{bmatrix}, W = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 1 & 0 \\ 0 & 1 & 0 \end{bmatrix}, b_h = \begin{bmatrix}0 \\ -1 \\ -2 \end{bmatrix}, v = \begin{bmatrix}1 \\ -1 \\ 1 \end{bmatrix}, b_y = 0.
```

The philosophy behind our parameter selection is to create a system in which the hidden state can take on four possible states. These states represent two pieces of information: the resulting bit at the current position and if we need to carry a bit to the next state. Since there are two possible values for the bit (0 or 1) and two possible answers to whether we carry a bit to the next state (True of False) then it's clear why we have four possible states for $h_t$ since $2\times2=4$.

By choosing 

```math
U = \begin{bmatrix}1 & 1 \\ 1 & 1 \\ 1 & 1 \end{bmatrix}
```

the product $Ux_t$ has three possible values:

```math
\begin{bmatrix}1 & 1 \\ 1 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix}0 \\ 0 \end{bmatrix} = \begin{bmatrix}0 \\ 0 \\ 0 \end{bmatrix},
```

```math
\begin{bmatrix}1 & 1 \\ 1 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix}1 \\ 0 \end{bmatrix} = \begin{bmatrix}1 \\ 1 \\ 1 \end{bmatrix},
```

```math
\begin{bmatrix}1 & 1 \\ 1 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix}1 \\ 1 \end{bmatrix} = \begin{bmatrix}2 \\ 2 \\ 2 \end{bmatrix}.
```

By choosing 

```math
W = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 1 & 0 \\ 0 & 1 & 0 \end{bmatrix}
```

the product $Wh_t$ has two possible values:

```math
\begin{bmatrix} 0 & 1 & 0 \\ 0 & 1 & 0 \\ 0 & 1 & 0 \end{bmatrix} \begin{bmatrix}0 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix}0 \\ 0 \\ 0 \end{bmatrix},
```

```math
\begin{bmatrix} 0 & 1 & 0 \\ 0 & 1 & 0 \\ 0 & 1 & 0 \end{bmatrix} \begin{bmatrix}1 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix}0 \\ 0 \\ 0 \end{bmatrix},
```

```math
\begin{bmatrix} 0 & 1 & 0 \\ 0 & 1 & 0 \\ 0 & 1 & 0 \end{bmatrix} \begin{bmatrix}1 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix}1 \\ 1 \\ 1 \end{bmatrix},
```

```math
\begin{bmatrix} 0 & 1 & 0 \\ 0 & 1 & 0 \\ 0 & 1 & 0 \end{bmatrix} \begin{bmatrix}1 \\ 1 \\ 1 \end{bmatrix} = \begin{bmatrix}1 \\ 1 \\ 1 \end{bmatrix}.
```

Following this, the sum $Ux_t + Wh_t$ can take on four possible values: 
```math
\begin{bmatrix}0 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix}1 \\ 1 \\ 1 \end{bmatrix}, \begin{bmatrix}2 \\ 2 \\ 2 \end{bmatrix}, \begin{bmatrix}3 \\ 3 \\ 3 \end{bmatrix}.
```

This sum when paired with the bias term $b_f$ gives us four possible values for the expression $Ux_t + Wh_t + b_f$ inside the activation function: 

```math
\begin{bmatrix}0 \\ -1 \\ -2 \end{bmatrix}, \begin{bmatrix}1 \\ 0 \\ -1 \end{bmatrix}, \begin{bmatrix}2 \\ 1 \\ 0 \end{bmatrix}, \begin{bmatrix}3 \\ 2 \\ 1 \end{bmatrix}.
```

Once passed through the activation function, the four possible states are as follows:

```math
\begin{bmatrix}0 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix}1 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix}1 \\ 1 \\ 0 \end{bmatrix}, \begin{bmatrix}1 \\ 1 \\ 1 \end{bmatrix}.
```

These four possible states represent the following instructions respectively: 0 at the current position and do not carry a 1 to the next state, 1 at the current position and do not carry a 1 to the next state, 0 at the current position and do carry a 1 to the next state, and 1 at the current position and do carry a 1 to the next state. The reason why we need the option to carry a bit to the next state is because of the case where the sum of the bits at the current position exceeds 1. This is necessary because in binary, the only two possible values for a bit are 0 and 1, so when two 1s are added together, the result is greater than 1 and must be represented by a carry bit.

Now that we have constructed a way for the states to represent what we want, now we want to create a function that can interpret the instructions from the hidden state such that we output the desired value at the given position. By defining $v$ as 

```math
v = \begin{bmatrix}1 \\ -1 \\ 1 \end{bmatrix}
```
and $b_y = 0$ then $y_t=\sigma\left(\mathbf{v}^T \mathbf{h}_t+b_y\right)$ will either output $0$ or $1$ based on the instructions from the hidden state as can be seen here:

```math
\sigma\left(\begin{bmatrix}1 ,-1, 1 \end{bmatrix} \begin{bmatrix}0 \\ 0 \\ 0 \end{bmatrix} + 0\right) = 0
```

```math
\sigma\left(\begin{bmatrix}1 ,-1, 1 \end{bmatrix} \begin{bmatrix}1 \\ 0 \\ 0 \end{bmatrix} + 0\right) = 1
```

```math
\sigma\left(\begin{bmatrix}1 ,-1, 1 \end{bmatrix} \begin{bmatrix}1 \\ 1 \\ 0 \end{bmatrix} + 0\right) = 0
```

```math
\sigma\left(\begin{bmatrix}1 ,-1, 1 \end{bmatrix} \begin{bmatrix}1 \\ 1 \\ 1 \end{bmatrix} + 0\right) = 1.
```

We now have a working calculator for binary addition using an RNN. Note that the exact values of our parameters are not what matter due to the existence of the activation function. What matters is that they are defined in such a way that the logic of the system is maintained. For example, if we multiplied $U$, $W$, and $b_h$ by the same **positive** scalar (multiplication by a non-positive scalar would disrupt the logic of the system) it would not change the output of the calculator since the output from the activation function would be the same.
