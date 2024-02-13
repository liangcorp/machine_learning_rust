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

The following image shows us what the sigmoid function looks like:

![1WFqZHntEead-BJkoDOYOw_2413fbec8ff9fa1f19aaf78265b8a33b_Logistic_function](https://github.com/liangcorp/machine_learning_rust/assets/2737157/d9d35e7b-1cc1-42c1-9cea-9b58975bd892)

The function $g(z)$, shown here, maps any real number to the (0, 1) interval,
making it useful for transforming an arbitrary-valued function into a function
better suited for classification.

$h_θ(x)$ will give us the **probability** that our output is 1.
For example, $h_θ(x)=0.7$ gives us a probability of 70% that our output is 1.
Our probability that our prediction is 0 is just the complement of our
probability that it is 1 (e.g. if probability that it is 1 is 70%, then
the probability that it is 0 is 30%).

$$
h_θ(x)=P(y=1|x;θ)=1−P(y=0|x;θ)
$$

$$
P(y=0|x;θ)+P(y=1|x;θ)=1
$$

## Decision Boundary

In order to get our discrete 0 or 1 classification, we can translate the output
of the hypothesis function as follows:

$$
h_θ(x)≥0.5→y=1
$$

$$
h_θ(x)<0.5→y=0
$$

The way our logistic function g behaves is that when its input is greater than
or equal to zero, its output is greater than or equal to 0.5:

$$
g(z)≥0.5
$$

$$
\text{when }z≥0
$$

Remember.

$$
z=0,e^0=1⇒g(z)=1/2
$$

$$
z→∞,e^{−∞}→0⇒g(z)=1
$$

$$
z→−∞,e^∞→∞⇒g(z)=0
$$

So if our input to g is $θ^TX$, then that means:

$$
hθ(x)=g(θ^Tx)≥0.5
$$

$$
\text{when } θ^Tx≥0
$$

From these statements we can now say:

$$
θ^Tx≥0⇒y=1
$$

$$
θ^Tx<0⇒y=0
$$

The **decision boundary** is the line that separates the area where y = 0 and
where y = 1. It is created by our hypothesis function.

### Example

$$
5

θ=−10

y=1\text{if}5+(−1)x_1 + 0x_2≥0

5−x_1≥0

−x_1≥−5

x_1≤5
$$

In this case, our decision boundary is a straight vertical line placed on the
graph where $x_1=5$, and everything to the left of that denotes y = 1, while
everything to the right denotes y = 0.

Again, the input to the sigmoid function g(z) (e.g. $θ^TX$) doesn't need to
be linear, and could be a function that describes a
circle (e.g. $z=θ_0+θ_1x_1^2+θ_2x^2_2$) or any shape to fit our data.
