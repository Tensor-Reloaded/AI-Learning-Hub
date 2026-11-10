# Internal Covariate Shift (ICS)

Consider layer $l$. During the forward pass, we compute:

```math
z_t^{(l)} = W_t^{(l)}a_t^{(l-1)} + b_t^{(l)}, \qquad a_t^{(l)} = f\left(z_t^{(l)}\right).
```
During backpropagation, the weight gradient is:
```math
\frac{\partial \mathcal{L}}{\partial W_t^{(l)}} = \delta_t^{(l)} \left(a_t^{(l-1)}\right)^\top,
```
and the weights are updated as:
```math
W_{t+1}^{(l)} = W_t^{(l)} - \eta\, \delta_t^{(l)} \left(a_t^{(l-1)}\right)^\top.
```
The update direction for layer $l$ depends on the activations produced by layer $l−1$. All gradients are computed from the same forward pass, after which the weights of all layers are updated. However, updating layer $l−1$ changes the activations it produces:
```math
W_t^{(l-1)} \rightarrow W_{t+1}^{(l-1)},
```
```math
a_t^{(l-1)} \rightarrow a_{t+1}^{(l-1)}.
```

The update for layer l was computed using activations from a previous layer, that no longer correspond to the updated network. 
The gradient is correct for the old network state, but it becomes stale relative to the new internal representation. 
As a consequence, we have ICS: each layer is optimized in parallel while the inputs produced by the preceding layers are simultaneously changing.

**Why does backpropagation still work?** Because we use small learning rates, and the resulting shift is limited. The stale gradients often remain approximately correct.
For deeper networks, Batch Normalization was introduced as a solution to ICS.

**Want to learn more about ICS?** A very good resource is https://www.geeksforgeeks.org/deep-learning/internal-covariant-shift-problem-in-deep-learning/. 

### Alternative solution: Backforward Propagation
Instead of computing every gradient from one forward pass, Backforward Propagation updates layers sequentially and recomputes the activations after each update. The next layer receives the latest activations, and its gradient is computed for the latest available network state.
Standard backpropagation follows a Jacobi-style update. Backforward follows a Gauss–Seidel-style update.

Check the following abstract for more info about Backforward Propagation: https://ojs.aaai.org/index.php/AAAI/article/view/27029/26801.
