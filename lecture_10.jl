### A Pluto.jl notebook ###
# v0.19.29

#> [frontmatter]
#> chapter = 2
#> video = "https://www.youtube.com/watch?v=xKAO38UsXo0"
#> image = "https://user-images.githubusercontent.com/6933510/136196570-478bbb89-05fb-4799-99a0-0ede06354cb6.png"
#> section = 4
#> order = 4
#> title = "Random Variables as Types"
#> layout = "layout.jlhtml"
#> youtube_id = "xKAO38UsXo0"
#> description = ""
#> tags = ["lecture", "module2", "track_julia", "type", "programming", "probability", "interactive", "random", "track_math", "Symbolics"]

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 103cd2f4-903c-11eb-1116-a51dc540175c
using Statistics, Plots, PlutoUI, LinearAlgebra, Symbolics

# ╔═╡ 5d62e16c-8fd9-11eb-1c44-0b0232614011
TableOfContents(aside=true)

# ╔═╡ 6bbfa37e-8ffe-11eb-3031-19ea76a6a8d2
md"""
# Concepts for today

We won't be introducing any new Julia functions in this lecture, but we are going to show off a key way in which Julia really shines: its **type system**.

With a few carefully-chosen definitions, we can gain a great deal of power!

Concepts for today:

- Types for code organisation of abstract concepts
- Abstract types
- Subtypes
- Building up expressions "under the hood" using types


- `sum` over an iterator / generator expression
"""

# ╔═╡ 8a125ca0-8ff5-11eb-0607-45f993fb5ca7
md"""
# Random variables as types
"""

# ╔═╡ b2971770-8ff7-11eb-002c-f9dc9d6d0d70
md"""
This lecture might appear to be about random variables (and it is). But we would be thrilled if you see it rather as a more general discussion on software engineering and abstractions, since the principles involved extend to *many* different contexts.
"""

# ╔═╡ cdd4497c-903d-11eb-03be-abf6002e75e7
md"""
## Random variables
"""

# ╔═╡ ae1b3a26-8fd3-11eb-3746-ad48301ff96e
md"""
Recall that a **random variable** $X$ is an object which has different possible **outcomes** $x$, to which we assign **probabilities** $\mathbb{P}(X = x)$.

The correspondence of probabilities to outcomes is called the **probability distribution** of the random variable.
"""

# ╔═╡ ae0008ee-8fd3-11eb-38bd-f52598a97dce
md"""
## Gaussian distributions
"""

# ╔═╡ c6c3cf54-8fd4-11eb-3b4f-415f1a2da18e
md"""
Let's remind ourselves of a very important type of distribution, namely the **Gaussian distribution** with mean $\mu$ and standard deviation $\sigma$ (or variance $\sigma^2$).

We can sample from a Gaussian distribution with mean $0$ and variance $1$ with the `randn` function (short for "random normal"). We can then shift and scale to get the Gaussian distribution that we want:
"""

# ╔═╡ d8b74772-8fd4-11eb-3943-f98c29d02171
md"""
μ = $(@bind μ Slider(-3:0.01:3, show_value=true, default=0.0))
σ = $(@bind σ Slider(0.01:0.01:3, show_value=true, default=1.0))
"""

# ╔═╡ ad7b3bee-8fd5-11eb-06f6-b39738d4b1fd
bell_curve(x) = exp(-x^2 / 2) / √(2π)

# ╔═╡ c76cd760-8fd5-11eb-3500-5d15515c33f5
bell_curve(x, μ, σ) = bell_curve( (x - μ) / σ ) / σ

# ╔═╡ f31275fa-8fd5-11eb-0b76-7d0513705273
bell_curve(0, 3, 2)

# ╔═╡ 276a7c36-8fd8-11eb-25d8-3d4cfaa1f71c
md"""
### Sum of two Gaussians
"""

# ╔═╡ 11f3853c-903e-11eb-04cd-a125017ad5d8
md"""
Let's try to *add* two Gaussians. From the empirical (experimental) point of view the sum of two random variables is easy to compute: Just sample both of them and add the resulting values to get the value of the random variable that is their sum:
"""

# ╔═╡ 79fb368c-8fd9-11eb-1c9c-bd0ceb122b11
md"""
We see that we get back a Gaussian again!
In fact, the sum of two Gaussians with means $\mu_1$ and $\mu_2$, and variances $\sigma_1^2$ and $\sigma_2^2$, is a Gaussian again! -- with mean $\mu_1 + \mu_2$ and variance $\sigma_1^2 + \sigma_2^2$. Below we will see how to encode this fact computationally.

"""

# ╔═╡ a2481afa-8fd3-11eb-1769-bf97f42ea79e
md"""
## Theoretical random variables vs. sampling

What should a theoretical random variable be able to do? What should we be able to do with sampling?


- **Naming**, e.g. `Gaussian`
- **Parameters**, e.g. $\mu$ and $\sigma^2$



### Theoretical 

- Theoretical mean
- Theoretical variance




- Theoretical sum of  two random variables
- Theoretical product of two random variables


- Probability distribution


### Sampling

- **Sample** a  random variable
- Sample mean
- Sample variance

- Sample sum


- Histogram


"""

# ╔═╡ a9654334-8fd9-11eb-2ea8-8d308ea66963
md"""

# Why define a type at all?
"""

# ╔═╡ bb2132e0-8fd9-11eb-3bdd-594726c04859
md"""
How can we *represent a random variable* in software?

In some languages there are different names for the various functions associated to a random variable, but no name for the random variable itself!

For example, in R there is a standard naming convention in R, with `d` for density, etc., followed by the name like `norm` for normal. The indicators are:

- `d` for the density
- `p` for the distribution function
- `q` for the quantile function
- `r` for generating random variates

See e.g.
- [Normal distribution in R](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/Normal)

- [Chi-squared distribution in R](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/Chisquare)


What's wrong with this? All these functions are referring to an underlying random variable (or probability distribution), which you will find in any course in probability, and yet there's no way to refer to the underlying mathematical object!
"""

# ╔═╡ f307e3b8-8ff0-11eb-137e-4f9a03bb4d78
md"""
Instead, we would like to be able to refer to the random variable (or probability distribution) itself. We should be able to provide a type with the name and parameters of a random variable, but not yet specify how to generate random instances (variates)? It turns out that this is a very good example of thinking ahead by providing an **abstraction**.

We can *later* provide a means for random sampling -- and even, if a new algorithm comes along that is more efficient, we can replace it some day!
"""

# ╔═╡ e0ef47a6-903c-11eb-18aa-6ff06f0e28ac
md"""
## Defining abstract types for random variables
"""

# ╔═╡ 02051416-903d-11eb-0ade-3b20897989c5
md"""
We define an **abstract type** using `abstract type <Name> end`, where `<Name>` is replaced with the name of the type.

We can think of an abstract type as being a collection of types that share a particular property. In our case, we want to create a type to represent "any random variable", and also the sub-types "any continuous(-valued) random variable" and "any discrete(-valued) random variable".

This will allow us to specify later on whether a given concrete (i.e. particular) random variable is discrete or continuous.

We use `<:` to denote **sub-type**:
"""

# ╔═╡ 51ee3c3c-903d-11eb-1bfa-3bbcda98e977
begin
	abstract type RandomVariable end
	
	abstract type DiscreteRandomVariable <: RandomVariable end
	abstract type ContinuousRandomVariable <: RandomVariable end
end

# ╔═╡ 681429d8-8fff-11eb-0fa1-bbf9e05e6cea
md"""
## Defining a type for a Gaussian random variable
"""

# ╔═╡ 21236a86-8fda-11eb-1fcf-59d1de75470c
md"""
Let's start off by looking at **Gaussian** random variables, also called normal random variables. We reminded ourselves above what they look like.
"""

# ╔═╡ dd130aae-8ff2-11eb-2b15-2f5123b40d20
md"""
### Name and parameters
"""

# ╔═╡ 4771e8e6-8fd2-11eb-178c-419cbdb348f4
begin
	struct Gaussian <: ContinuousRandomVariable
		μ     # mean
		σ²    # variance
	end
	
	Gaussian() = Gaussian(0.0, 1.0)  # normalised Gaussian with mean 0 and variance 1
end

# ╔═╡ ece14f6c-8fda-11eb-279b-c18dc0fc46d7
G = Gaussian(1, 2)

# ╔═╡ 0be6c548-8fd3-11eb-07a2-5bb382614cab
md"""
Note that here we have created a Gaussian random variable with given parameter values, *without* sampling from it.
"""

# ╔═╡ 9c03814c-8ff2-11eb-189d-f3b0507507cb
md"""
### Theoretical mean and variance
"""

# ╔═╡ f24bcb1a-8fda-11eb-1828-0307c65b81cd
md"""
Now we can extend the `mean`, `var` (variance) and `std` (standard deviation) functions from the `Statistics` library to act on this object:
"""

# ╔═╡ 0c76f49e-8fdb-11eb-2c2c-59a4060e8e1d
begin
	Statistics.mean(X::Gaussian) = X.μ
	Statistics.var(X::Gaussian) = X.σ²
end

# ╔═╡ 43292298-8ff7-11eb-2b9d-017acc9ef185
md"""
### Planning ahead: Standard deviation for *any* random variable, not just Gaussians
"""

# ╔═╡ 1bdc6068-8ff7-11eb-069c-6d0f1a83f373
md"""
Once we have defined the variance, we know how to calculate the standard deviation: it's just the square root of the variance. But, thinking ahead, this is true for *any* random variable, so we can define it to act on any random variable that we will define later!:
"""

# ╔═╡ 592afd96-8ff7-11eb-1200-47eaf6b6a28a
md"""
This is an example of good software design.
"""

# ╔═╡ 76f4424e-8fdc-11eb-35ee-dd09752b947b
md"""
### Sum of two Gaussian random variables
"""

# ╔═╡ dede8022-8fd2-11eb-22f8-a5614d703c01
md"""
Gaussians have a special property: the sum of two Gaussians is always a Gaussian. (We say that Gaussians are **stable** distributions; [there are others](https://en.wikipedia.org/wiki/Stable_distribution).) Note that we don't need random samples for this theoretical observation, embodied in the following code:
"""

# ╔═╡ b4f18188-8fd2-11eb-1950-ef3bee3e4724
Base.:+(X::Gaussian, Y::Gaussian) = Gaussian(X.μ + Y.μ, X.σ² + Y.σ²)

# ╔═╡ 292f0e44-8fd3-11eb-07cf-5733a5327630
begin
	G1 = Gaussian(0, 1)
	G2 = Gaussian(5, 6)
end

# ╔═╡ ac3a7c18-8ff3-11eb-1c1f-17939cff40f9
md"""
The theoretical product of two Gaussians is not Gaussian; we will do the general case later.
"""

# ╔═╡ b8e59db2-8ff3-11eb-1fa1-595fdc6652bc
md"""
### Probability distribution of a Gaussian
"""

# ╔═╡ cb167a60-8ff3-11eb-3885-d1d8a70e2a5e
md"""
A Gaussian random variable is a **continuous** random variable, i.e. it has a continuous range of possible outcomes. The possible range of outcomes is called the **support** of the distribution. For a Gaussian it is the whole real line, $(-\infty, \infty)$.
"""

# ╔═╡ fc0acee6-8ff3-11eb-13d1-1350364f03a9
md"""
One way to specify a continous random variable $X$ is via its **probability density function**, or **PDF**, $f_X$. The probability that $X$ lies in the interval $[a, b]$ is given by an area under the curve $f_X(x)$ from $a$ to $b$:

$$\mathbb{P}(X \in [a, b]) = \int_{a}^b f_X(x) \, dx.$$
"""

# ╔═╡ 2c8b75de-8ff4-11eb-1bc6-cde7b67a2007
md"""
For a Gaussian distribution with mean $\mu$ and variance $\sigma^2$, the PDF is given by

$$f_X(X) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp \left[ -\frac{1}{2} \left( \frac{x - \mu}{\sigma} \right)^2 \right]$$.
"""

# ╔═╡ 639e517c-8ff4-11eb-3ea8-07b73b0bff78
pdf(X::Gaussian) = x -> exp(-0.5 * ( (x - X.μ)^2 / X.σ²) ) / √(2π * X.σ²)

# ╔═╡ 3d78116c-8ff5-11eb-0ee3-71bf9413f30e
pdf(G)

# ╔═╡ b63c02e4-8ff4-11eb-11f6-d760e8760118
pdf(Gaussian())(0.0)

# ╔═╡ cd9b10f0-8ff5-11eb-1671-99c6738b8074
md"""
μ = $(@bind μμ Slider(-3:0.01:3, show_value=true, default=0.0))
σ = $(@bind σσ Slider(0.01:0.01:3, show_value=true, default=1.0))
"""

# ╔═╡ a899fc08-8ff5-11eb-1d00-e95b55c3be4b
begin
	plot(pdf(Gaussian(μμ, σσ)), leg=false)
	xlims!(-6, 6)
	ylims!(0, 0.5)
end

# ╔═╡ 0b901c34-8ff6-11eb-225b-511718412309
md"""
### Sampling from a Gaussian distribution
"""

# ╔═╡ 11cbf48a-903f-11eb-1e77-81eeb358ec24
md"""
We can also specify how to sample from a Gaussian distribution. We can re-purpose `rand` for this!
"""

# ╔═╡ bfb62f1a-8ff5-11eb-0f7a-cf725f3269c5
md"""
# More general distributions
"""

# ╔═╡ 0a150880-8ff6-11eb-0046-45fa2d4b476e
md"""
Let's recall the Bernoulli distribution from last lecture. This represents a weighted coin with probability $p$ to come up "heads" (1), and probability $1-p$ to come up "tails" (0).

Note that this is a **discrete** random variable: the possible outcomes are the discrete values $0$ and $1$.
"""

# ╔═╡ 26569eb4-8fd1-11eb-2df9-098a0792a09e
struct Bernoulli <: DiscreteRandomVariable
	p::Float64
end

# ╔═╡ baf1fe40-8ff6-11eb-1da1-cd43880db334
B = Bernoulli(0.25)

# ╔═╡ 38b10d94-903f-11eb-3e26-890382342dc1
md"""
Again we can specify the theoretical mean and variance:
"""

# ╔═╡ be5f9900-8ff6-11eb-1816-03447cabd9a9
begin
	Statistics.mean(X::Bernoulli) = X.p
	Statistics.var(X::Bernoulli) = X.p * (1 - X.p)
end

# ╔═╡ 6a06b64e-903f-11eb-0d23-53f223ed1ed3
md"""
And again `std` just works!
"""

# ╔═╡ 6e9ad41a-903f-11eb-06ae-1b34a6674eaf
md"""
Finally we specify how to sample:
"""

# ╔═╡ 754b0a80-8ff8-11eb-364b-85fa49d6bb8e
Base.rand(X::Bernoulli) = Int(rand() < X.p)

# ╔═╡ bd484878-8ff8-11eb-0337-7730ab2b07d4
md"""
## Adding two random variables
"""

# ╔═╡ c3043eb6-8ff8-11eb-2cae-13d1c1613234
md"""
What happens if we add two Bernoulli random variables? There are two routes we could go: We could use the known theoretical sum, or we could write a general-purpose tool. Let's do the latter.
"""

# ╔═╡ 7e3cea0c-903f-11eb-0b41-0f381c1cce4b
md"""
When we add two Bernoulli random variables we do *not* get a Bernoulli back. To see this it's enough to observe that the sum can have the outcome 2, which is impossible for a Bernoulli. 
		
So the result is just the random variable "the sum of these two given random variables". In general it won't even have a common name. 
		
So we actually need to *define a new type* to represent the "sum of two given random variables", which itself will be a random variable!:
		
		
"""

# ╔═╡ eb555508-8ff8-11eb-1b70-e95290084742
struct SumOfTwoRandomVariables <: RandomVariable
	X1::RandomVariable
	X2::RandomVariable
end

# ╔═╡ 1e5047c4-8ff9-11eb-2628-c7824725678a
begin
	B1 = Bernoulli(0.25)
	B2 = Bernoulli(0.6)
end

# ╔═╡ cb2cd908-903f-11eb-09c0-99acde6d765c
md"""
Now we can define the sum of two random variables of *any* type:
"""

# ╔═╡ 44a5ef96-8ff9-11eb-06a0-d3a8dcf5c1aa
Base.:+(X1::RandomVariable, X2::RandomVariable) = SumOfTwoRandomVariables(X1, X2)

# ╔═╡ b11d964e-8fd4-11eb-3f6a-43e8d2fa462c
data = μ .+ σ .* randn(10^5)   # transform standard normal

# ╔═╡ 2be60570-8fd8-11eb-0bdf-951280dc6181
begin
	data1 = 4 .+ sqrt(0.3) .* randn(10^5)
	data2 = 6 .+ sqrt(0.7) .* randn(10^5)
	
	total = data1 + data2
end

# ╔═╡ 6b50e9e6-8fd3-11eb-0ab3-6fc3efea7e37
G1 + G2

# ╔═╡ 180a4746-8ff6-11eb-046f-ddf6bb938a35
Base.rand(X::Gaussian) = X.μ + √(X.σ²) * randn()

# ╔═╡ d8437e6c-903f-11eb-3ac2-5f7c380c0872
md"""
For example, let's sum two Bernoullis:
"""

# ╔═╡ 574744ec-8ff9-11eb-033c-a3dff07a292b
B1 + B2

# ╔═╡ e024377a-903f-11eb-316a-b5b7936e610f
md"""
However, for the special case of Gaussians we still get the correct result (we have *not* overwritten the previous definition):
"""

# ╔═╡ f37bb49c-903f-11eb-03fb-35d6ac35822d
G1 + G2

# ╔═╡ 318f5274-8ff9-11eb-1c88-5fde5b546099
md"""
Now we need to define the various functions on this type representing a sum
"""

# ╔═╡ 90ee7558-8ff9-11eb-3602-271890987ece
Statistics.mean(S::SumOfTwoRandomVariables) = mean(S.X1) + mean(S.X2)

# ╔═╡ bd083794-8fd8-11eb-0155-e59fe27d64f2
Statistics.mean(total)

# ╔═╡ 4596edec-8fdb-11eb-1369-f3e98cb831cd
mean(G)

# ╔═╡ 5300b082-8fdb-11eb-03fe-55f511364ad9
mean(G1 + G2) == mean(G1) + mean(G2)

# ╔═╡ a168ace6-8ff9-11eb-393d-45c34fdf577c
mean(B1 + B2)

# ╔═╡ a51b1538-8ff9-11eb-0635-81088e826bb3
md"""
To have a simple equation for the variance, we need to assume that the two random variables are **independent**. Perhaps the name should have been `SumOfTwoIndependentRandomVariables`, but it seems too long.
"""

# ╔═╡ d88c0830-8ff9-11eb-3e71-c1ac327f4e25
Statistics.var(S::SumOfTwoRandomVariables) = var(S.X1) + var(S.X2)

# ╔═╡ 255d51b6-8ff3-11eb-076c-31ba4c5ce10d
var(G)

# ╔═╡ 11ef1d18-8ff7-11eb-1645-113c1aae6e9b
Statistics.std(X::RandomVariable) = sqrt(var(X))

# ╔═╡ a32079ea-8fd8-11eb-01c4-81b603033b55
Statistics.std(total)

# ╔═╡ 4ef2153a-8fdb-11eb-1a23-8b1c28381e34
std(G)

# ╔═╡ 52a2cac8-8ff8-11eb-2da4-0b113618c64b
mean(B), var(B), std(B)

# ╔═╡ dd995120-8ff9-11eb-1a53-2d65f0b5585a
md"""
How can we sample from the sum? It's actually easy!
"""

# ╔═╡ f171c27c-8ff9-11eb-326c-6b2d8c38451d
Base.rand(S::SumOfTwoRandomVariables) = rand(S.X1) + rand(S.X2)

# ╔═╡ ab7c6b18-8ffa-11eb-2b6c-dde3dca1c6f7
md"""
Now it's easy to look at the sum of a Bernoulli and a Gaussian. This is an example of a [**mixture distribution**](https://en.wikipedia.org/wiki/Mixture_distribution).
"""

# ╔═╡ 79c55fc0-8ffb-11eb-229f-49198aee8245
md"""
Let's extend the `histogram` function to easily draw the histogram of a random variable:
"""

# ╔═╡ 1d4e236c-8ffb-11eb-209d-851e1af231d4
md"""
Now... What if we sum more random variables?
"""

# ╔═╡ 0c3cfb16-8ffb-11eb-3ef9-33ea9acbb8c0
mixture = Bernoulli(0.25) + Bernoulli(0.75) + Gaussian(0, 0.1)

# ╔═╡ 71cf1724-9040-11eb-25c3-69ccde4abf0d
md"""
## Generic programming: `sum`
"""

# ╔═╡ 646ac706-9040-11eb-297c-7b6d8fdf3c16
md"""
Now we have defined `+`, Julia's generic definition of `sum` can kick in to define the sum of many random variables!:
"""

# ╔═╡ 34bcab72-8ffb-11eb-1d0c-29bd83de638b
S = sum(Bernoulli(0.25) for i in 1:30)

# ╔═╡ 77b54a34-9040-11eb-00d1-57f712ff6732
md"""
Note that we do not need the `[...]` in the following expression. There is no need to actually create an array of random variables; instead we are using an **iterator** or **generator expression**:
"""

# ╔═╡ 611f770c-8ffc-11eb-2c23-0d5750afd7c8
mean(S)

# ╔═╡ 636b2ce0-8ffc-11eb-2411-571f78fb8a84
var(S)

# ╔═╡ 656999be-8ffc-11eb-0f51-51d5a162a004
md"""
This is a big deal! Everything just works.

By the way, the sum of $n$ Bernoulli random variables with the *same* probability $p$ is called a **binomial** random variable with parameters $(n, p)$.
"""

# ╔═╡ 8cb5373a-8ffc-11eb-175d-b11ec5fae1ab
md"""
If we were worried about performance, we would probably want to define a separate `Binomial` type, rather than using nested sums of Bernoullis. You will do this in the homework.
"""

# ╔═╡ 25b4ccca-8ffd-11eb-3fe0-2d14ce90d8b3
md"""
## χ₁² distribution
"""

# ╔═╡ 9913427a-8ffd-11eb-3c28-c12ee05ed9d3
md"""
Another well-known distribution is the chi-squared distribution, which is defined as the sum of squares of Gaussians. Chi-1 squared ($\chi_1^2$) is the square of a single Gaussian:
"""

# ╔═╡ aa9d94dc-8ffd-11eb-0537-c39344c224c2
struct ChiSquared1 <: ContinuousRandomVariable
end

# ╔═╡ b5300720-8ffd-11eb-2ca4-ed7a7c9b5516
Base.rand(X::ChiSquared1) = rand(Gaussian())^2

# ╔═╡ 27595716-8ff6-11eb-1d99-7bf13eedddf7
histogram!([rand(Gaussian(μμ, σσ)) for i in 1:10^4], alpha=0.5, norm=true)

# ╔═╡ 4493ac9e-8ffb-11eb-15c9-a146091d7696
Plots.histogram(X::RandomVariable; kw...) = histogram([rand(X) for i in 1:10^6], norm=true, leg=false, alpha=0.5, size=(500, 300), kw...)

# ╔═╡ b8c253f8-8fd4-11eb-304a-b1be962687e4
begin
	histogram(data, alpha=0.2, norm=true, bins=100, leg=false, title="μ=$(μ), σ=$(σ)", size=(500, 300))
	xlims!(-6, 6)
	ylims!(0, 0.7)
	
	
	xs = [μ - σ, μ, μ + σ]
	
	plot!(-6:0.01:6, x -> bell_curve(x, μ, σ), lw=2)
	
	plot!((μ - σ):0.01:(μ + σ), x -> bell_curve(x, μ, σ), fill=true, alpha=0.5, c=:purple)
	
	plot!([μ, μ], [0.05, bell_curve(μ, μ, σ)], ls=:dash, lw=2, c=:white)
	annotate!(μ, 0.03, text("μ", :white))
#	annotate!(μ + σ, 0.03, text("μ+σ", :yellow))
#	annotate!(μ, 0.03, text("μ", :white))

	
end

# ╔═╡ 5e094b52-8fd8-11eb-2ac4-5797599ab013
histogram(total, alpha=0.5, leg=false, norm=true, size=(500, 300))

# ╔═╡ a591ac12-8ffb-11eb-1734-09657f875b1e
histogram(Bernoulli(0.25) + Bernoulli(0.75))

# ╔═╡ 6c5ef6d4-8ffb-11eb-2fe9-9dd87d92abe4
histogram(Bernoulli(0.25) + Gaussian(0, 0.1))

# ╔═╡ 89138e02-8ffb-11eb-2ad2-c32e663f57b0
histogram( mixture )

# ╔═╡ bf9a4722-8ffb-11eb-1652-f1bfb4916d2a
histogram(S)

# ╔═╡ 325560f4-8ffb-11eb-0e9b-53f6869bdd97
rand( mixture )

# ╔═╡ c7613c44-8ffc-11eb-0a43-dbc8b1f62be9
rand(S)

# ╔═╡ eea0c788-8ffd-11eb-06e0-cfb65dbba7f1
histogram(ChiSquared1())

# ╔═╡ c57fb088-9040-11eb-3881-dfbd82e72b35
md"""
We can get the $\chi_n^2$ distribution by summing several $\chi_1^2$:
"""

# ╔═╡ 028695a2-8ffe-11eb-3cf4-bd07bfe4df3a
histogram(sum(ChiSquared1() for i in 1:4))

# ╔═╡ 820af6dc-8fdc-11eb-1792-ad9f32eb915e
md"""
## Using symbolics
"""

# ╔═╡ ce330b78-8fda-11eb-144d-756275542eea
md"""
We can even do some of this symbolically!, e.g.:
"""

# ╔═╡ 668ade6a-8fda-11eb-0a5e-cb791b245ec0
@variables μ₁, σ₁², μ₂, σ₂²   # introduce symbolic variables from Symbolics.jl

# ╔═╡ 4e94d1da-8fda-11eb-3572-398fb4a12c3c
Gaussian(μ₁, σ₁²) + Gaussian(μ₂, σ₂²)

# ╔═╡ 9ca993a6-8fda-11eb-089c-4d7f89c81b94
Gaussian(17, 3) + Gaussian(μ₂, σ₂²)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[compat]
Plots = "~1.29.1"
PlutoUI = "~0.7.48"
Symbolics = "~4.4.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "5721046c5c6c33bdc11b1c4d1ed76ee57504084e"

[[deps.AbstractAlgebra]]
deps = ["GroupsCore", "InteractiveUtils", "LinearAlgebra", "MacroTools", "Markdown", "Random", "RandomExtensions", "SparseArrays", "Test"]
git-tree-sha1 = "29e65c331f97db9189ef00a4c7aed8127c2fd2d4"
uuid = "c3fe647b-3220-5bb0-a1ea-a7954cac585d"
version = "0.27.10"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[deps.AbstractTrees]]
git-tree-sha1 = "faa260e4cb5aba097a73fab382dd4b5819d8ec8c"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "02f731463748db57cc2ebfbd9fbc9ce8280d3433"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "81f0cb60dc994ca17f68d9fb7c942a5ae70d9ee4"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "5.0.8"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e5f08b5689b1aad068e01751889f2f615c7db36d"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.29"

[[deps.ArrayInterfaceStaticArraysCore]]
deps = ["ArrayInterfaceCore", "LinearAlgebra", "StaticArraysCore"]
git-tree-sha1 = "01a9f8e6cfc2bfdd01d333f70b8014a04893103c"
uuid = "dd5226c6-a4d4-4bc7-8575-46859f9c95b9"
version = "0.1.4"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AutoHashEquals]]
git-tree-sha1 = "45bb6705d93be619b81451bb2006b7ee5d4e4453"
uuid = "15f4f7f2-30c1-5605-9d31-71845cf9641f"
version = "0.2.0"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "e28912ce94077686443433c2800104b061a827ed"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.39"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Bijections]]
git-tree-sha1 = "c9b163bd832e023571e86d0b90d9de92a9879088"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.1.6"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e0af648f0692ec1691b5d094b8724ba1346281cf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.18.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "6c0100a8cf4ed66f66e2039af7cde3357814bad2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.46.2"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.CompositeTypes]]
git-tree-sha1 = "02d2316b7ffceff992f3096ae48c7829a8aa0638"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.3"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

    [deps.CompositionsBase.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "3d5873f811f582873bb9871fc9c451784d5dc8c7"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.102"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "85cf537e38b7f34a84eaac22b534aa1b5bf01949"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.5.14"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.DynamicPolynomials]]
deps = ["DataStructures", "Future", "LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Pkg", "Reexport", "Test"]
git-tree-sha1 = "8b84876e31fa39479050e2d3395c4b3b210db8b0"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.4.6"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.Extents]]
git-tree-sha1 = "2140cd04483da90b2da7f99b2add0750504fc39c"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.2"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "7072f1e3e5a8be51d525d64f63d3ec1287ff2790"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.11"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c98aea696662d09e215ef7cda5296024a9646c75"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.4"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "bc9f7725571ddb4ab2c4bc74fa397c1c5ad08943"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.69.1+0"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "d53480c0793b13341c40199190f92c611aa2e93c"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.2"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "424a5a6ce7c5d97cca7bcc4eac551b97294c54af"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.9"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.Groebner]]
deps = ["AbstractAlgebra", "Combinatorics", "Logging", "MultivariatePolynomials", "Primes", "Random"]
git-tree-sha1 = "47f0f03eddecd7ad59c42b1dd46d5f42916aff63"
uuid = "0b43b601-686d-58a3-8a1c-6623616c7cd4"
version = "0.2.11"

[[deps.GroupsCore]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9e1a5e9f3b81ad6a5c613d181664a0efc6fe6dd7"
uuid = "d5909c97-4eac-4ecc-a3dc-fdd0858a4120"
version = "0.4.0"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "b8ffb903da9f7b8cf695a8bead8e01814aa24b30"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IntervalSets]]
deps = ["Dates", "Random"]
git-tree-sha1 = "3d8866c029dd6b16e69e0d4a939c4dfcb98fac47"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.8"
weakdeps = ["Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsStatisticsExt = "Statistics"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.LabelledArrays]]
deps = ["ArrayInterface", "ChainRulesCore", "LinearAlgebra", "MacroTools", "StaticArrays"]
git-tree-sha1 = "1cccf6d366e51fbaf80303158d49bb2171acfeee"
uuid = "2ee39098-c373-598a-b85f-a56591580800"
version = "1.9.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "8c57307b5d9bb3be1ff2da469063628631d4d51e"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.21"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    DiffEqBiologicalExt = "DiffEqBiological"
    ParameterizedFunctionsExt = "DiffEqBase"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    DiffEqBase = "2b5f629d-d688-5b77-993f-72d75c75574e"
    DiffEqBiological = "eb300fae-53e8-50a0-950c-e21f52c2b7e0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Metatheory]]
deps = ["AutoHashEquals", "DataStructures", "Dates", "DocStringExtensions", "Parameters", "Reexport", "TermInterface", "ThreadsX", "TimerOutputs"]
git-tree-sha1 = "0f39bc7f71abdff12ead4fc4a7d998fb2f3c171f"
uuid = "e9d8d322-4543-424a-9be4-0cc815abe26c"
version = "1.3.5"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "629afd7d10dbc6935ec59b32daeb33bc4460a42e"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.MultivariatePolynomials]]
deps = ["ChainRulesCore", "DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "eaa98afe2033ffc0629f9d0d83961d66a021dfcc"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.4.7"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "6985021d02ab8c509c841bb8b2becd3145a7b490"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.3.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a12e56c72edee3ce6b96667745e6cbbe5498f200"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.23+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "66b2fcd977db5329aa35cac121e5b94dd6472198"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.28"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "9e42de869561d6bdf8602c57ec557d43538a92f0"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.29.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e47cd150dbe0443c3a3651bc5b9cbd5576ab75b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.52"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "4c9f306e5d6603ae203c2000dd460d81a5251489"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.4"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9ebcd48c498668c7fa0e97a9cae873fbee7bfee1"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RandomExtensions]]
deps = ["Random", "SparseArrays"]
git-tree-sha1 = "b8a399e95663485820000f26b6a43c794e166a49"
uuid = "fb686558-2515-59ef-acaa-46db3789a887"
version = "0.4.4"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterfaceCore", "ArrayInterfaceStaticArraysCore", "ChainRulesCore", "DocStringExtensions", "FillArrays", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "Tables", "ZygoteRules"]
git-tree-sha1 = "a5ce741acddc02f0d4fc6505463ca89697d7fb23"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.32.3"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Referenceables]]
deps = ["Adapt"]
git-tree-sha1 = "e681d3bfa49cd46c3c161505caddf20f0e62aaa9"
uuid = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"
version = "0.1.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "6aacc5eefe8415f47b3e34214c1d79d2674a0ba2"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.12"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLBase]]
deps = ["ArrayInterfaceCore", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "Preferences", "RecipesBase", "RecursiveArrayTools", "RuntimeGeneratedFunctions", "StaticArraysCore", "Statistics", "Tables"]
git-tree-sha1 = "fe89a8113ea445bcff9ee570077830674babb534"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.81.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "38d88503f695eb0301479bc9b0d4320b378bafe5"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "5165dfb9fd131cf0c6957a3a7605dede376e7b63"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "5d2c08cef80c7a3a8ba9ca023031a85c263012c5"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.6.6"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore"]
git-tree-sha1 = "0adf069a2a490c47273727e029371b31d44b72b2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.5"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StructArrays]]
deps = ["Adapt", "ConstructionBase", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "0a3db38e4cce3c54fe7a71f831cd7b6194a54213"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.16"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.SymbolicUtils]]
deps = ["AbstractTrees", "Bijections", "ChainRulesCore", "Combinatorics", "ConstructionBase", "DataStructures", "DocStringExtensions", "DynamicPolynomials", "IfElse", "LabelledArrays", "LinearAlgebra", "Metatheory", "MultivariatePolynomials", "NaNMath", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "TermInterface", "TimerOutputs"]
git-tree-sha1 = "027b43d312f6d52187bb16c2d4f0588ddb8c4bb2"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "0.19.11"

[[deps.Symbolics]]
deps = ["ArrayInterface", "ConstructionBase", "DataStructures", "DiffRules", "Distributions", "DocStringExtensions", "DomainSets", "Groebner", "IfElse", "Latexify", "Libdl", "LinearAlgebra", "MacroTools", "Metatheory", "NaNMath", "RecipesBase", "Reexport", "Requires", "RuntimeGeneratedFunctions", "SciMLBase", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicUtils", "TermInterface", "TreeViews"]
git-tree-sha1 = "aab7c217abd7427e91004a2486bef9af42a1047a"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "4.4.3"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TermInterface]]
git-tree-sha1 = "7aa601f12708243987b88d1b453541a75e3d8c7a"
uuid = "8ea1fca8-c5ef-4a55-8b96-4e9afe9c9a3c"
version = "0.2.3"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadsX]]
deps = ["ArgCheck", "BangBang", "ConstructionBase", "InitialValues", "MicroCollections", "Referenceables", "Setfield", "SplittablesBase", "Transducers"]
git-tree-sha1 = "34e6bcf36b9ed5d56489600cf9f3c16843fa2aa2"
uuid = "ac1d9e8a-700a-412c-b207-f0111f4b6c0d"
version = "0.1.11"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "53bd5978b182fa7c57577bdb452c35e5b4fb73a5"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.78"

    [deps.Transducers.extensions]
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "24b81b59bd35b3c42ab84fa589086e19be919916"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.11.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "9d749cd449fb448aeca4feee9a2f4186dbb5d184"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.4"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─5d62e16c-8fd9-11eb-1c44-0b0232614011
# ╠═103cd2f4-903c-11eb-1116-a51dc540175c
# ╟─6bbfa37e-8ffe-11eb-3031-19ea76a6a8d2
# ╟─8a125ca0-8ff5-11eb-0607-45f993fb5ca7
# ╟─b2971770-8ff7-11eb-002c-f9dc9d6d0d70
# ╟─cdd4497c-903d-11eb-03be-abf6002e75e7
# ╟─ae1b3a26-8fd3-11eb-3746-ad48301ff96e
# ╟─ae0008ee-8fd3-11eb-38bd-f52598a97dce
# ╟─c6c3cf54-8fd4-11eb-3b4f-415f1a2da18e
# ╠═b11d964e-8fd4-11eb-3f6a-43e8d2fa462c
# ╟─d8b74772-8fd4-11eb-3943-f98c29d02171
# ╟─b8c253f8-8fd4-11eb-304a-b1be962687e4
# ╠═ad7b3bee-8fd5-11eb-06f6-b39738d4b1fd
# ╠═c76cd760-8fd5-11eb-3500-5d15515c33f5
# ╠═f31275fa-8fd5-11eb-0b76-7d0513705273
# ╟─276a7c36-8fd8-11eb-25d8-3d4cfaa1f71c
# ╟─11f3853c-903e-11eb-04cd-a125017ad5d8
# ╠═2be60570-8fd8-11eb-0bdf-951280dc6181
# ╠═5e094b52-8fd8-11eb-2ac4-5797599ab013
# ╠═bd083794-8fd8-11eb-0155-e59fe27d64f2
# ╠═a32079ea-8fd8-11eb-01c4-81b603033b55
# ╟─79fb368c-8fd9-11eb-1c9c-bd0ceb122b11
# ╟─a2481afa-8fd3-11eb-1769-bf97f42ea79e
# ╟─a9654334-8fd9-11eb-2ea8-8d308ea66963
# ╟─bb2132e0-8fd9-11eb-3bdd-594726c04859
# ╟─f307e3b8-8ff0-11eb-137e-4f9a03bb4d78
# ╟─e0ef47a6-903c-11eb-18aa-6ff06f0e28ac
# ╟─02051416-903d-11eb-0ade-3b20897989c5
# ╠═51ee3c3c-903d-11eb-1bfa-3bbcda98e977
# ╟─681429d8-8fff-11eb-0fa1-bbf9e05e6cea
# ╟─21236a86-8fda-11eb-1fcf-59d1de75470c
# ╟─dd130aae-8ff2-11eb-2b15-2f5123b40d20
# ╠═4771e8e6-8fd2-11eb-178c-419cbdb348f4
# ╟─ece14f6c-8fda-11eb-279b-c18dc0fc46d7
# ╟─0be6c548-8fd3-11eb-07a2-5bb382614cab
# ╟─9c03814c-8ff2-11eb-189d-f3b0507507cb
# ╟─f24bcb1a-8fda-11eb-1828-0307c65b81cd
# ╠═0c76f49e-8fdb-11eb-2c2c-59a4060e8e1d
# ╠═4596edec-8fdb-11eb-1369-f3e98cb831cd
# ╠═255d51b6-8ff3-11eb-076c-31ba4c5ce10d
# ╟─43292298-8ff7-11eb-2b9d-017acc9ef185
# ╟─1bdc6068-8ff7-11eb-069c-6d0f1a83f373
# ╠═11ef1d18-8ff7-11eb-1645-113c1aae6e9b
# ╟─592afd96-8ff7-11eb-1200-47eaf6b6a28a
# ╠═4ef2153a-8fdb-11eb-1a23-8b1c28381e34
# ╟─76f4424e-8fdc-11eb-35ee-dd09752b947b
# ╟─dede8022-8fd2-11eb-22f8-a5614d703c01
# ╠═b4f18188-8fd2-11eb-1950-ef3bee3e4724
# ╠═292f0e44-8fd3-11eb-07cf-5733a5327630
# ╠═6b50e9e6-8fd3-11eb-0ab3-6fc3efea7e37
# ╠═5300b082-8fdb-11eb-03fe-55f511364ad9
# ╟─ac3a7c18-8ff3-11eb-1c1f-17939cff40f9
# ╟─b8e59db2-8ff3-11eb-1fa1-595fdc6652bc
# ╟─cb167a60-8ff3-11eb-3885-d1d8a70e2a5e
# ╟─fc0acee6-8ff3-11eb-13d1-1350364f03a9
# ╟─2c8b75de-8ff4-11eb-1bc6-cde7b67a2007
# ╠═639e517c-8ff4-11eb-3ea8-07b73b0bff78
# ╠═3d78116c-8ff5-11eb-0ee3-71bf9413f30e
# ╠═b63c02e4-8ff4-11eb-11f6-d760e8760118
# ╠═a899fc08-8ff5-11eb-1d00-e95b55c3be4b
# ╟─cd9b10f0-8ff5-11eb-1671-99c6738b8074
# ╟─0b901c34-8ff6-11eb-225b-511718412309
# ╟─11cbf48a-903f-11eb-1e77-81eeb358ec24
# ╠═180a4746-8ff6-11eb-046f-ddf6bb938a35
# ╠═27595716-8ff6-11eb-1d99-7bf13eedddf7
# ╟─bfb62f1a-8ff5-11eb-0f7a-cf725f3269c5
# ╟─0a150880-8ff6-11eb-0046-45fa2d4b476e
# ╠═26569eb4-8fd1-11eb-2df9-098a0792a09e
# ╠═baf1fe40-8ff6-11eb-1da1-cd43880db334
# ╟─38b10d94-903f-11eb-3e26-890382342dc1
# ╠═be5f9900-8ff6-11eb-1816-03447cabd9a9
# ╟─52a2cac8-8ff8-11eb-2da4-0b113618c64b
# ╟─6a06b64e-903f-11eb-0d23-53f223ed1ed3
# ╟─6e9ad41a-903f-11eb-06ae-1b34a6674eaf
# ╠═754b0a80-8ff8-11eb-364b-85fa49d6bb8e
# ╟─bd484878-8ff8-11eb-0337-7730ab2b07d4
# ╟─c3043eb6-8ff8-11eb-2cae-13d1c1613234
# ╟─7e3cea0c-903f-11eb-0b41-0f381c1cce4b
# ╠═eb555508-8ff8-11eb-1b70-e95290084742
# ╠═1e5047c4-8ff9-11eb-2628-c7824725678a
# ╟─cb2cd908-903f-11eb-09c0-99acde6d765c
# ╠═44a5ef96-8ff9-11eb-06a0-d3a8dcf5c1aa
# ╟─d8437e6c-903f-11eb-3ac2-5f7c380c0872
# ╠═574744ec-8ff9-11eb-033c-a3dff07a292b
# ╟─e024377a-903f-11eb-316a-b5b7936e610f
# ╠═f37bb49c-903f-11eb-03fb-35d6ac35822d
# ╟─318f5274-8ff9-11eb-1c88-5fde5b546099
# ╠═90ee7558-8ff9-11eb-3602-271890987ece
# ╠═a168ace6-8ff9-11eb-393d-45c34fdf577c
# ╟─a51b1538-8ff9-11eb-0635-81088e826bb3
# ╠═d88c0830-8ff9-11eb-3e71-c1ac327f4e25
# ╟─dd995120-8ff9-11eb-1a53-2d65f0b5585a
# ╠═f171c27c-8ff9-11eb-326c-6b2d8c38451d
# ╟─ab7c6b18-8ffa-11eb-2b6c-dde3dca1c6f7
# ╟─79c55fc0-8ffb-11eb-229f-49198aee8245
# ╠═4493ac9e-8ffb-11eb-15c9-a146091d7696
# ╠═a591ac12-8ffb-11eb-1734-09657f875b1e
# ╠═6c5ef6d4-8ffb-11eb-2fe9-9dd87d92abe4
# ╟─1d4e236c-8ffb-11eb-209d-851e1af231d4
# ╠═0c3cfb16-8ffb-11eb-3ef9-33ea9acbb8c0
# ╠═325560f4-8ffb-11eb-0e9b-53f6869bdd97
# ╠═89138e02-8ffb-11eb-2ad2-c32e663f57b0
# ╟─71cf1724-9040-11eb-25c3-69ccde4abf0d
# ╟─646ac706-9040-11eb-297c-7b6d8fdf3c16
# ╠═34bcab72-8ffb-11eb-1d0c-29bd83de638b
# ╟─77b54a34-9040-11eb-00d1-57f712ff6732
# ╠═bf9a4722-8ffb-11eb-1652-f1bfb4916d2a
# ╠═611f770c-8ffc-11eb-2c23-0d5750afd7c8
# ╠═636b2ce0-8ffc-11eb-2411-571f78fb8a84
# ╠═c7613c44-8ffc-11eb-0a43-dbc8b1f62be9
# ╟─656999be-8ffc-11eb-0f51-51d5a162a004
# ╟─8cb5373a-8ffc-11eb-175d-b11ec5fae1ab
# ╟─25b4ccca-8ffd-11eb-3fe0-2d14ce90d8b3
# ╟─9913427a-8ffd-11eb-3c28-c12ee05ed9d3
# ╠═aa9d94dc-8ffd-11eb-0537-c39344c224c2
# ╠═b5300720-8ffd-11eb-2ca4-ed7a7c9b5516
# ╠═eea0c788-8ffd-11eb-06e0-cfb65dbba7f1
# ╟─c57fb088-9040-11eb-3881-dfbd82e72b35
# ╠═028695a2-8ffe-11eb-3cf4-bd07bfe4df3a
# ╟─820af6dc-8fdc-11eb-1792-ad9f32eb915e
# ╟─ce330b78-8fda-11eb-144d-756275542eea
# ╠═668ade6a-8fda-11eb-0a5e-cb791b245ec0
# ╠═4e94d1da-8fda-11eb-3572-398fb4a12c3c
# ╠═9ca993a6-8fda-11eb-089c-4d7f89c81b94
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
