# Logistic Regression

## Hypothesis Representation

We could approach the classification problem ignoring the fact that $y$ is
discrete-valued, and use our old linear regression algorithm to try to predict
$y$ given $x$. However, it is easy to construct examples where this method
performs very poorly. Intuitively, it also doesn't make sense for
$hθ(x)$ to take values larger than 1 or smaller than 0 when we know
that $y ∈ {0, 1}$. To fix this, let's change the form for our hypotheses
$h_θ(x)$ to satisfy $0≤h_θ(x)≤1$. This is accomplished by plugging
$θ^Tx$ into the Logistic Function.

Our new form uses the "Sigmoid Function," also called the "Logistic Function":

$$
h_θ(x)=g(θ^Tx)
$$

$$
z=θ^Tx
$$

$$
g(z)={1 \over {1+e^{−z}}}
$$
