Recurrent neural networks (RNNs) can be purposed to act as calculators. The following is a toy example of how RNNs can be constructed to perform addition with handcrafted parameters.

The input to the RNN will be two binary numbers, starting with the *least* significant bit. The longest binary number will be padded with an additional zero on the left side and the other number will also be padded with zeros such that they are both the same length. For instance, the equation

$$
100111+110010=1011001
$$

would be input to your RNN as:
- Input 1: $1, 1, 1,0,0,1,0$
- Input 2: $0,1,0,0,1,1,0$
- Correct output: $1,0,0,1,1,0,1$

The RNN has two input units and one output unit. In this example, the sequence of inputs and outputs would be:

<p align="center">
  <img src="https://user-images.githubusercontent.com/29025503/214734148-7a9157df-d225-48b4-8080-8c5f0434c97c.png" alt="Sublime's custom image"/>
</p>

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
& \mathbf{h}_t=\sigma\left(\mathbf{U x}_t+\mathbf{W} \mathbf{h}_{t-1}+\mathbf{b}_{\mathbf{h}}\right) \\
& y_t=\sigma\left(\mathbf{v}^T \mathbf{h}_t+b_y\right)
\end{aligned}
$$

where $\mathbf{x}\_t \in \mathbb{R}^2$, $\mathbf{U} \in \mathbb{R}^{3 \times 2}$, $\mathbf{W} \in \mathbb{R}^{3 \times 3}$, $\mathbf{b}_{\mathbf{h}} \in \mathbb{R}^3$, $\mathbf{v} \in \mathbb{R}^3$, and $b_y \in \mathbb{R}^2$.

We define our parameters as follows:

```math
U = \begin{bmatrix}1 & 1 \\ 1 & 1 \\ 1 & 1 \end{bmatrix}, W = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 1 & 0 \\ 0 & 1 & 0 \end{bmatrix}, b_h = \begin{bmatrix}0 \\ -1 \\ -2 \end{bmatrix}, v = \begin{bmatrix}1 \\ -1 \\ 1 \end{bmatrix}, b_y = 0.
```

The philosophy behind our parameter selection is to create a system in which the hidden state can take on four possible states. These states represent two pieces of information: the resulting bit at the current position and if we need to carry a bit to the next state. Since there are two possible values for the bit (0 or 1) and two possible answers to whether we carry a bit to the next state (True of False) then it's clear why we have four possible states for $h_t$ since $2\times2=4$.

We found the values for our parameters by first defining our possible states and then building from that. We define the four possible states are as follows:

```math
\begin{bmatrix}0 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix}1 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix}1 \\ 1 \\ 0 \end{bmatrix}, \begin{bmatrix}1 \\ 1 \\ 1 \end{bmatrix}.
```

These four possible states represent the following instructions respectively: 0 at the current position and do not carry a 1 to the next state, 1 at the current position and do not carry a 1 to the next state, 0 at the current position and do carry a 1 to the next state, and 1 at the current position and do carry a 1 to the next state. The reason why we need the option to carry a bit to the next state is because of the case where the sum of the bits at the current position exceeds 1. This is necessary because in binary, the only two possible values for a bit are 0 and 1, so when two 1s are added together, the result is greater than 1 and must be represented by a carry bit.

Now that we have defined our states, we must now find values for $U$, $W$, and $b_h$ such that $h_t$ takes on the proper state given the previous state $h_{t-1}$. To do this, you construct a system of matrix equations with every combination of $x_t$ and $h_t$ as follows:

```math
\sigma \left(\begin{bmatrix}u_{11} & u_{12} \\ u_{21} & u_{22} \\ u_{31} & u_{32} \end{bmatrix} \begin{bmatrix}0 \\ 0 \end{bmatrix} + \begin{bmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \\ w_{31} & w_{32} & w_{33} \end{bmatrix} \begin{bmatrix}0 \\ 0 \\ 0 \end{bmatrix} + \begin{bmatrix}b_1 \\ b_2 \\ b_3 \end{bmatrix} \right) = \begin{bmatrix}0 \\ 0 \\ 0 \end{bmatrix},
```

```math
\sigma \left(\begin{bmatrix}u_{11} & u_{12} \\ u_{21} & u_{22} \\ u_{31} & u_{32} \end{bmatrix} \begin{bmatrix}0 \\ 0 \end{bmatrix} + \begin{bmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \\ w_{31} & w_{32} & w_{33} \end{bmatrix} \begin{bmatrix}1 \\ 0 \\ 0 \end{bmatrix} + \begin{bmatrix}b_1 \\ b_2 \\ b_3 \end{bmatrix} \right) = \begin{bmatrix}0 \\ 0 \\ 0 \end{bmatrix},
```

```math
\cdots
```

```math
\sigma \left(\begin{bmatrix}u_{11} & u_{12} \\ u_{21} & u_{22} \\ u_{31} & u_{32} \end{bmatrix} \begin{bmatrix}1 \\ 0 \end{bmatrix} + \begin{bmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \\ w_{31} & w_{32} & w_{33} \end{bmatrix} \begin{bmatrix}1 \\ 0 \\ 0 \end{bmatrix} + \begin{bmatrix}b_1 \\ b_2 \\ b_3 \end{bmatrix} \right) = \begin{bmatrix}1 \\ 0 \\ 0 \end{bmatrix},
```

```math
\sigma \left(\begin{bmatrix}u_{11} & u_{12} \\ u_{21} & u_{22} \\ u_{31} & u_{32} \end{bmatrix} \begin{bmatrix}1 \\ 0 \end{bmatrix} + \begin{bmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \\ w_{31} & w_{32} & w_{33} \end{bmatrix} \begin{bmatrix}1 \\ 1 \\ 0 \end{bmatrix} + \begin{bmatrix}b_1 \\ b_2 \\ b_3 \end{bmatrix}\right) = \begin{bmatrix}1 \\ 1 \\ 1 \end{bmatrix},
```

```math
\cdots
```

```math
\sigma \left(\begin{bmatrix}u_{11} & u_{12} \\ u_{21} & u_{22} \\ u_{31} & u_{32} \end{bmatrix} \begin{bmatrix}1 \\ 1 \end{bmatrix} + \begin{bmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \\ w_{31} & w_{32} & w_{33} \end{bmatrix} \begin{bmatrix}1 \\ 1 \\ 0 \end{bmatrix} + \begin{bmatrix}b_1 \\ b_2 \\ b_3 \end{bmatrix} \right) = \begin{bmatrix}1 \\ 1 \\ 1 \end{bmatrix},
```

```math
\sigma \left(\begin{bmatrix}u_{11} & u_{12} \\ u_{21} & u_{22} \\ u_{31} & u_{32} \end{bmatrix} \begin{bmatrix}1 \\ 1 \end{bmatrix} + \begin{bmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \\ w_{31} & w_{32} & w_{33} \end{bmatrix} \begin{bmatrix}1 \\ 1 \\ 1 \end{bmatrix} + \begin{bmatrix}b_1 \\ b_2 \\ b_3 \end{bmatrix} \right) = \begin{bmatrix}1 \\ 1 \\ 1 \end{bmatrix},
```

and then solve. Once we have found a solution for the system we then turn our attention from $h_t$ and construct a system of equations for $y_t$. Our system of equations will need to account for each state $h_t$ and so we construct the system as follows:

```math
\sigma\left(\begin{bmatrix}v_1 ,v_2, v_3 \end{bmatrix} \begin{bmatrix}0 \\ 0 \\ 0 \end{bmatrix} + b_y\right) = 0,
```

```math
\sigma\left(\begin{bmatrix}v_1 ,v_2, v_3 \end{bmatrix} \begin{bmatrix}1 \\ 0 \\ 0 \end{bmatrix} + b_y\right) = 1,
```

```math
\sigma\left(\begin{bmatrix}v_1 ,v_2, v_3 \end{bmatrix} \begin{bmatrix}1 \\ 1 \\ 0 \end{bmatrix} + b_y\right) = 0,
```

```math
\sigma\left(\begin{bmatrix}v_1 ,v_2, v_3 \end{bmatrix} \begin{bmatrix}1 \\ 1 \\ 1 \end{bmatrix} + b_y\right) = 1,
```

and solve. There exist an infinite number of possible solutions that satisfy our two systems of equations but we settled with the parameters we defined above:

```math
U = \begin{bmatrix}1 & 1 \\ 1 & 1 \\ 1 & 1 \end{bmatrix}, W = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 1 & 0 \\ 0 & 1 & 0 \end{bmatrix}, b_h = \begin{bmatrix}0 \\ -1 \\ -2 \end{bmatrix}, v = \begin{bmatrix}1 \\ -1 \\ 1 \end{bmatrix}, b_y = 0.
```
