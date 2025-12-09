# Getting Started

## Installation
* Clone this repo:
```
git clone https://github.com/loganbrenningmeyer/diffusion.git
cd diffusion
```

* Install dependencies or create/activate a new Conda environment with:
```
conda env create -f environment.yml
conda activate diffusion
```

## Training
* Login to wandb / set environment variables:
```
wandb login

export WANDB_ENTITY="<entity>"
export WANDB_PROJECT="<project>"
```

* Configure `configs/train.yml` and launch train script:
```
sh scripts/train.sh
```

## Testing
* Configure `configs/test.yml` and launch test script:
```
sh scripts/test.sh
```

# Diffusion Models

A **denoising diffusion probabilistic model (DDPM)** [(Ho et al., 2020)](https://arxiv.org/abs/2006.11239), is a generative model that produces samples by iteratively denoising Gaussian noise. 

<div align="center">
    <img src="figs/trajectory_800000.gif" width="100%">
    <p align="center">Diffusion model trained on CIFAR10 for 800k steps</p>
</div>

## What is the goal?

All generative models share the same goal: **learn how to produce samples from some true distribution**, e.g., "here's lots of photos of dogs, now make me a new one." Diffusion models are no different, but they don't simply pull a dog out of thin air, they begin with pure noise and repeatedly denoise it until, ideally, it is "dog-like."

Let's call the true distribution $p(\mathbf{x})$, e.g., a distribution of dog images where $\mathbf{x} \sim p(\mathbf{x})$ is a sample image of a dog. To generate new samples, we want to learn how to **sample a latent variable $\mathbf{z}$** from some random, easy-to-sample distribution $p(\mathbf{z})$—often a standard normal distribution $\mathbf{z} \sim \mathcal{N}(0,\mathbf{I})$—and **transform it into a sample from the true distribution**.

<div align="center">
    <img src="figs/mapping.png" width="40%">
    <p align="center">Mapping samples from p(z) to the true distribution p(x)</p>
</div>

However, actually knowing $p(\mathbf{x})$ is intractable—you can't know every possible dog image. So we want to create a model with parameters $\theta$ that approximates $p_ \theta(\mathbf{x}|\mathbf{z})$ given a random latent $\mathbf{z} \sim p(\mathbf{z})$, i.e., "given some noise, what clean images does it likely map to?". This allows us to produce new samples $\mathbf{x} \sim p_ \theta(\mathbf{x}|\mathbf{z})$ using our model.

## Simplifying the Problem

In a perfect world, our model could learn to produce a perfect sample from the true distribution in one shot from any given latent $\mathbf{z}$. In very simple cases, this perfect world actually exists. If $p(x)$ were simply a one-dimensional normal distribution $x \sim \mathcal{N}(\mu,\sigma^2)$, our model could easily predict the mean $\hat{\mu}$ and variance $\hat{\sigma}^2$ of $p(x)$. We could then simply create new samples by:

1. Sampling a latent $z \sim \mathcal{N}(0,1)$ 
2. Producing a sample $x = \hat{\mu} + \hat{\sigma}z$ 

<div align="center">
    <img src="figs/px_pz.png" width="100%">
    <p align="center">Generating new samples from a 1-D normal distribution using the learned mean and variance</p>
</div>

In practice, $p(\mathbf{x})$ is a highly complex, high-dimensional distribution ($256\times256\text{px}$ RGB images have 196,608 dimensions!), so it is impossible to perfectly map $\mathbf{z} \sim p(\mathbf{z}) \to \mathbf{x} \sim p(\mathbf{x})$. Therefore, we can try to simplify the problem by moving from $\mathbf{z} \to \mathbf{x}$ in *very small* steps.

### Incremental Steps

Let's try and break the problem into $T$ steps, incrementally transforming $\mathbf{z}$ many times until we reach $x$. In diffusion models, we sample $\mathbf{z} \sim \mathcal{N}(0,\mathbf{I})$, beginning with pure Gaussian noise with zero-mean and unit-variance. We'll call the target variable $\mathbf{x}_ 0$—the clean image after $T$ incremental steps of denoising $\mathbf{z}$. To keep things in terms of $T$, we'll refer to our starting latent $\mathbf{z}$ as $\mathbf{x}_ T$ from now on—the pure Gaussian noise we want to clean up. Therefore, the incremental process we hope to perform is:

$$\mathbf{x}_ T \to \mathbf{x}_ {T-1} \to \ldots \to \mathbf{x}_ 1 \to \mathbf{x}_ 0$$

This greatly simplifies our model's task, as it now simply needs to learn the conditional distribution

$$\boxed{{\mathbf{x}_ {t-1} \sim p_ \theta(\mathbf{x}_ {t-1}|\mathbf{x}_ t)}},$$

meaning "given the current noisy $\mathbf{x}_ t$, what is the distribution of the slightly less noisy $\mathbf{x}_ {t-1}$?" Rather than trying to guess the correct clean image from pure noise—an impossible task—the model only predicts a slightly denoised version of the current image.

<div align="center">
    <img src="figs/reverse_steps.png" width="80%">
    <p align="center">Denoising from pure noise to a clean image</p>
</div>

### Learning to Denoise

But how do we train a model to denoise images? Let's say we have a training dataset $\mathcal{D}$ of clean images $\mathbf{x}_ 0 \in \mathcal{D}$, and we need to provide the model with lots of examples of denoising from $\mathbf{x}_ {t} \to \mathbf{x}_ {t-1}$. However, since we don't have some true dataset of noisy $\mathbf{x}_ t$ samples, we'll have to make them ourselves. This is known as the **forward diffusion process**, the process of iteratively noising clean samples.

## Forward Process

So let's make some noise! We'll start with the clean image $\mathbf{x}_ 0$ at $t=0$ and add noise at each step until we reach $\mathbf{x}_ T$ at $t=T$, where we want $\mathbf{x}_ T$ to be pure Gaussian noise, $\mathbf{x}_ T \sim \mathcal{N}(0,\mathbf{I})$. Keep in mind, we will be feeding $\mathbf{x}_ t$ into a neural network, which learns best from normalized data. Therefore, we want to ensure $\mathbf{x}_ t$ is normalized at every step to have zero-mean and unit-variance, meaning we want:

$$E[\mathbf{x}_ t]=0,\quad \text{Var}[\mathbf{x}_ t]=1,\quad t=0,1,\ldots,T$$

### Adding Noise

Let's start out simple by adding some standard Gaussian noise at each step, scaled to have some variance $\sigma^2$:

$$\mathbf{x}_ t = \mathbf{x}_ {t-1} + \sigma\boldsymbol{\epsilon}_ t,\quad \boldsymbol{\epsilon}_ t \sim \mathcal{N}(0,\mathbf{I})$$

This may seem like a good idea at first glance, but naively adding the same Gaussian noise at every step means the variance of $\mathbf{x}_ t$ increases linearly over time:

$$
\begin{aligned}
\text{Var}[\mathbf{x}_ 0] &= 1 \\
\text{Var}[\mathbf{x}_ 1] &= \text{Var}[\mathbf{x}_ 0 + \sigma\boldsymbol{\epsilon}_ 1] = \text{Var}[\mathbf{x}_ 0] + \sigma^2 = 1 + \sigma^2 \\
\text{Var}[\mathbf{x}_ 2] &= \text{Var}[\mathbf{x}_ 1 + \sigma\boldsymbol{\epsilon}_ 2] = \text{Var}[\mathbf{x}_ 1] + \sigma^2 = 1 + 2\sigma^2 \\
\vdots \\
\text{Var}[\mathbf{x}_ t] &= \text{Var}[\mathbf{x}_ {t-1} + \sigma\boldsymbol{\epsilon}_ t] = \text{Var}[\mathbf{x}_ {t-1}] + \sigma^2 = \boxed{1 + t\sigma^2}
\end{aligned}
$$

This means that at timestep $t$, the noisy image's values are roughly in the range $[-1-t\sigma^2,\ 1+t\sigma^2]$, making the magnitudes of pixel-values *much* larger than they should be: 

<div align="center">
    <img src="figs/bad_noise.png" width="80%">
    <p align="center">Adding the same Gaussian noise at each step linearly increases variance</p>
</div>

### Preserving Variance

But remember, we want keep each $\mathbf{x}_ t$ properly normalized to have $E[\mathbf{x}_ t]=0$ and $\text{Var}[\mathbf{x}_ t]=1$. So let's keep $\mathbf{x}_ t$'s values in check by introducing a scaling factor $c<1$ to shrink $\mathbf{x}_ t$ towards 0:

$$\mathbf{x}_ t = c\mathbf{x}_ {t-1} + \sigma\boldsymbol{\epsilon}_ t,\quad \boldsymbol{\epsilon}_ t \sim \mathcal{N}(0,\mathbf{I})$$

We want $\text{Var}[\mathbf{x}_ t]=1$ at every timestep $t$, so we can solve for the $c$ that keeps it that way:

$$
\begin{aligned}
\text{Var}[\mathbf{x}_ t] = 1 &= \text{Var}[c\mathbf{x}_ {t-1} + \sigma\boldsymbol{\epsilon}_ t] \\
&= \text{Var}[c\mathbf{x}_ {t-1}] + \text{Var}[\sigma\boldsymbol{\epsilon}_ t] \\
&= c^2\underbrace{\text{Var}[\mathbf{x}_ {t-1}]}_ {1} + \sigma^2\underbrace{\text{Var}[\boldsymbol{\epsilon}_ t]}_ {1}
\end{aligned}
$$

$$
\begin{aligned}
1 &= c^2 + \sigma^2 \\
c &= \boxed{\sqrt{1 - \sigma^2}}
\end{aligned}
$$

Now if we substitute in $c = \sqrt{1-\sigma^2}$, we can double check that our math works out such that $\text{Var}[\mathbf{x}_ t]=1$:

* Substitute $c=\sqrt{1-\sigma^2}$:

$$\mathbf{x}_ t = \sqrt{1 - \sigma^2}\mathbf{x}_ {t-1} + \sigma\boldsymbol{\epsilon}_ t,\quad \boldsymbol{\epsilon}_ t \sim \mathcal{N}(0,\mathbf{I})$$

* Compute $\text{Var}[\mathbf{x}_ t]$:

$$
\begin{aligned}
\text{Var}[\mathbf{x}_ t] &= \text{Var}[\sqrt{1 - \sigma^2}\mathbf{x}_ {t-1} + \sigma\boldsymbol{\epsilon}_ t] \\
&= \text{Var}[\sqrt{1 - \sigma^2}\mathbf{x}_ {t-1}] + \text{Var}[\sigma\boldsymbol{\epsilon}_ t] \\
&= (1 - \sigma^2)\underbrace{\text{Var}[\mathbf{x}_ {t-1}]}_ {1} + \sigma^2\underbrace{\text{Var}[\boldsymbol{\epsilon}_ t]}_ {1} \\
&= 1 - \sigma^2 + \sigma^2 \\
&= \boxed{1}
\end{aligned}
$$

### Variance Schedule

*Great! So we can just pick some variance* $\sigma^2$ *for our noise, scale* $\mathbf{x}_ {t-1}$ *by* $\sqrt{1-\sigma^2}$ *at each step, and everything works out!* Unfortunately, there are a few issues with that:

1. If $\sigma=1$, then $\sqrt{1-\sigma^2} = 0$, and $\mathbf{x}_ {t-1}$ is completely removed at every step—we'd only have noise from $t=1$ onwards.
2. Even if $\sigma<1$, say $\sigma=0.01$, the image gets noisy way too quickly, but we want our model to see smooth, subtle steps of noise.

<div align="center">
    <img src="figs/fixed_noise_01.png" width="80%">
    <p align="center"></p>
</div>

3. If $\sigma\ll1$, say $\sigma=0.0001$, the image gets noisy way too slowly, and our model never sees pure noise. 

<div align="center">
    <img src="figs/fixed_noise_0001.png" width="80%">
    <p align="center"></p>
</div>

We need to find a compromise: we want the model to see timesteps with little noise and timesteps with lots of noise, and we want the noise to be added smoothly over time. Therefore, we need a **variance schedule** to control how much noise is added at each timestep. So instead of a fixed variance $\sigma^2$ at all timesteps, we'll introduce $\beta_ t$ as the variance at timestep $t$. For notation convenience, we'll also let $\alpha_ t = 1 - \beta_ t$. By scheduling the variance of the noise, our old equation,

$$\mathbf{x}_ t = \sqrt{1 - \sigma^2}\mathbf{x}_ {t-1} + \sigma\boldsymbol{\epsilon}_ t,\quad \boldsymbol{\epsilon}_ t \sim \mathcal{N}(0,\mathbf{I}),$$

now becomes,

$$\boxed{\mathbf{x}_ t = \sqrt{\alpha_ t}\mathbf{x}_ {t-1} + \sqrt{\beta_ t}\boldsymbol{\epsilon}_ t,\quad \boldsymbol{\epsilon}_ t \sim \mathcal{N}(0,\mathbf{I})}$$

A single step can also be written as the conditional distribution:

$$\boxed{q(\mathbf{x}_ t|\mathbf{x}_ {t-1}) \sim \mathcal{N}(\sqrt{\alpha_ t}\mathbf{x}_ {t-1},\ \beta_ t I)}$$

meaning that given the previous sample $\mathbf{x}_ {t-1}$, the next sample $\mathbf{x}_ t$ is sampled from a normal distribution with mean $\sqrt{\alpha_ t}\mathbf{x}_ {t-1}$ and variance $\beta_ t$.

### Closed-Form Forward Equation

At each step, $\mathbf{x}_ {t-1}$ is scaled by $\sqrt{\alpha_ t}$, which means we keep $\sqrt{\alpha_ t}$ of $\mathbf{x}_ {t-1}$ at each noising step, i.e., its retained signal is $\sqrt{\alpha_ t}\mathbf{x}_ {t-1}.$ If we start at $\mathbf{x}_ 0$ and add noise for $t=1,2,\ldots,T$, we can see how much of the *original signal* we retain:

$$
\begin{align*}
\mathbf{x}_ 1 &= \sqrt{\alpha_ 1}\mathbf{x}_ 0 + \sqrt{\beta_ 1}\boldsymbol{\epsilon}_ 1 \\
\mathbf{x}_ 2 &= \sqrt{\alpha_ 2}(\sqrt{\alpha_ 1}\mathbf{x}_ 0 + \sqrt{\beta_ 1}\boldsymbol{\epsilon}_ 1) + \sqrt{\beta_ 2}\boldsymbol{\epsilon}_ 2 \\
\mathbf{x}_ 3 &= \sqrt{\alpha_ 3}(\sqrt{\alpha_ 2}(\sqrt{\alpha_ 1}\mathbf{x}_ 0 + \sqrt{\beta_ 1}\boldsymbol{\epsilon}_ 1) + \sqrt{\beta_ 2}\boldsymbol{\epsilon}_ 2) + \sqrt{\beta_ 3}\boldsymbol{\epsilon}_ 3 \\
\vdots \\
\end{align*}
$$

As you can see, at each timestep $t$, $\mathbf{x}_ 0$ is scaled by the cumulative product $\sqrt{\alpha_ 1} \cdot \sqrt{\alpha_ 2} \cdot \sqrt{\alpha_ 3} \cdots \sqrt{\alpha_ t}.$ We can then define the retained signal of $\mathbf{x}_ 0$ at timestep $t$ as $\sqrt{\bar\alpha_ t}\mathbf{x}_ 0$, where

$$\bar\alpha_ t = \prod_ {s=1}^t \alpha_ s$$

The variance of the retained signal of $\mathbf{x}_ 0$ is then $\text{Var}[\sqrt{\bar\alpha_ t}\mathbf{x}_ 0]=\bar\alpha_ t$. Since we've defined the forward diffusion process so that the total variance of $\mathbf{x}_ t$ is always $1$, and because $\mathbf{x}_ t$ is simply the retained signal from $\mathbf{x}_ 0$ at timestep $t$ plus noise, we can say:

$$\text{Var}[\text{signal}_ t] + \text{Var}[\text{noise}_ t] = 1$$

We've just defined that $\text{Var}[\text{signal}_ t]=\bar\alpha_ t$, so we can then solve for the variance of the noise at timestep $t$:

$$
\begin{align*}
\bar\alpha_ t + \text{Var}[\text{noise}_ t] &= 1 \\
\text{Var}[\text{noise}_ t] &= 1 - \bar\alpha_ t
\end{align*}
$$

Now we can make a closed-form equation for $\mathbf{x}_ t$ from $\mathbf{x}_ 0$ without having to noise every step individually!

$$\boxed{\mathbf{x}_ t = \sqrt{\bar\alpha_ t}\mathbf{x}_ 0 + \sqrt{1 - \bar\alpha_ t}\boldsymbol{\epsilon},\quad \boldsymbol{\epsilon} \sim \mathcal{N}(0,\mathbf{I})}$$

### Planning the Variance Schedule

As the variance of the retained signal and the variance of the noise sum to $1$, the signal variance $\bar\alpha_ t$ provides a good measure of how much of the clean image the model can see compared to noise at timestep $t$. Therefore, we want $\bar\alpha_ t$ to decrease smoothly over time, allowing the model to see the clean image nicely transition into noise. So we must define a variance schedule $\{\beta_ t\}_ {t=1}^T$ that makes $\bar\alpha_ t$ smoothly decay. Recall that $\alpha_ t = 1 - \beta_ t$, so we can write $\bar\alpha_ t$ as:

$$\bar\alpha_ t = \prod_ {s=1}^t 1 - \beta_ t$$

Clearly, as $\beta_ t$ increases, $\bar\alpha_ t$ decreases, so how should we increase $\beta_ t$? In the original DDPM paper, they went with the simplest route: linearly increase $\beta_ t$. They empirically chose to start with $\beta_ 1=0.0001$ and end with $\beta_ T=0.02$ over $T=1000$ steps, simply picking these values to ensure that:

1. At the start, $\bar\alpha_ 1 = 0.9999$, meaning $\mathbf{x}_ 1$ is nearly a completely clean image.
2. By the end, $\bar\alpha_ T \approx 0.00004$, meaning $\mathbf{x}_ T$ is essentially pure noise.
3. $\bar\alpha_ t$ exponentially decays smoothly over time, which is helpful for our model to learn. 

<div align="center">
    <img src="figs/betas_alpha_bars.png" width="80%">
    <img src="figs/forward_steps.png" width="80%">
    <p align="center">
        Linear variance schedule with &beta;<sub>1</sub>=0.0001, &beta;<sub>T</sub>=0.02
    </p>
</div>

Now we have a way to generate training samples! We can just use our linear variance schedule and our closed-form forward equation to produce any $\mathbf{x}_ t$ from $\mathbf{x}_ 0$.

## Reverse Process

Recall that a single step of the forward noising process is defined as:

$$q(\mathbf{x}_ t|\mathbf{x}_ {t-1}) \sim \mathcal{N}(\sqrt{\alpha_ t}\mathbf{x}_ {t-1},\ \beta_ t I).$$

Our goal is to train our model to learn $p(\mathbf{x}_ {t-1}|\mathbf{x}_ t)$, i.e., the reverse of the forward process. According to Bayes' Rule, we can solve for $p(\mathbf{x}_ {t-1}|\mathbf{x}_ t)$ with:

$$p(\mathbf{x}_ {t-1}|\mathbf{x}_ t) = \frac{q(\mathbf{x}_ t|\mathbf{x}_ {t-1})p(\mathbf{x}_ {t-1})}{p(\mathbf{x}_ t)}.$$

Just as we can't know the distribution of all dogs $p(\mathbf{x})$, we can't know the distribution of all noisy dogs $p(\mathbf{x}_ t)$—it is intractable—so we'll ignore the normalizing term in the denominator and write:

$$p(\mathbf{x}_ {t-1}|\mathbf{x}_ t) \propto q(\mathbf{x}_ t|\mathbf{x}_ {t-1})p(\mathbf{x}_ {t-1}).$$

### Gaussian Assumption

Here we make a key assumption about the distribution $p(\mathbf{x}_ {t-1}|\mathbf{x}_ t)$ that makes all of this work—it is Gaussian. 

Technically, $q(\mathbf{x}_ t|\mathbf{x}_ {t-1})$ defines the forward distribution for $\mathbf{x}_ t$, asking: "*What is the slightly noisier version of* $\mathbf{x}_ {t-1}$*?*" However, in the reverse process we *only know* $\mathbf{x}_ t$ and want to *predict* $\mathbf{x}_ {t-1}$. In that case, $q(\mathbf{x}_ t|\mathbf{x}_ {t-1})$ acts as the **likelihood** of $\mathbf{x}_ {t-1}$, instead asking: "*What previous sample likely produced* $\mathbf{x}_ t$*?*" 

As we know from the forward process, $\mathbf{x}_ t$ is just a scaled-down version of the previous image $\sqrt{\alpha_ t}\mathbf{x}_ {t-1}$ plus a little bit of noise. Therefore, given $\mathbf{x}_ t$, it's reasonable to believe that the previous image was likely around $\frac{\mathbf{x}_ t}{\sqrt{\alpha_ t}}$—the current image scaled back up. Because we made the crucial design choice to split up the problem into many *very, very small* steps, this approximation is very accurate. As a result, $q(\mathbf{x}_ t|\mathbf{x}_ {t-1})$, the likelihood of $\mathbf{x}_ {t-1}$ given $\mathbf{x}_ t$, is a sharp Gaussian spike around $\frac{\mathbf{x}_ t}{\sqrt{\alpha_ t}}$ and nearly $0$ everywhere else.

The data distribution $p(\mathbf{x}_ {t-1})$ is itself intractable like $p(\mathbf{x}_ t)$—it is the complex, curved manifold of all valid $\mathbf{x}_ {t-1}$ samples. However, multiplying it with the narrow Gaussian spike $q(\mathbf{x}_ t|\mathbf{x}_ {t-1})$ effectively "zooms in" very closely around $\frac{\mathbf{x}_ t}{\sqrt{\alpha_ t}}$, where $p(\mathbf{x}_ {t-1})$ is essentially flat or constant. 

Multiplying the Gaussian spike $q(\mathbf{x}_ t|\mathbf{x}_ {t-1})$ with the locally constant region of $p(\mathbf{x}_ {t-1})$ is just a scaled Gaussian, meaning $p(\mathbf{x}_ {t-1}|\mathbf{x}_ t)$ is Gaussian as well! This is very convenient, as to train our model to represent $p_ \theta(\mathbf{x}_ {t-1}|\mathbf{x}_ t)$, we only need to know the mean and variance that define the Gaussian distribution!

$$p_ \theta(\mathbf{x}_ {t-1}|\mathbf{x}_ t) = \mathcal{N}(\boldsymbol{\mu}_ \theta(\mathbf{x}_ t,t),\ \Sigma_ \theta(\mathbf{x}_ t,t))$$

### Computing Variance

Luckily, we don't need to predict $\Sigma_ \theta(\mathbf{x}_ t,t)$ using the model; we can compute it analytically. In the original DDPM paper, they experimented with two formulations:

1. **Forward Variance**: The variance $\beta_ t$ defined by the variance schedule, assumes no knowledge of $\mathbf{x}_ 0$.

$$\boxed{\Sigma_ \theta(\mathbf{x}_ t,t) = \beta_ t}$$

2. **Posterior Variance**: The true posterior variance assuming that we know $\mathbf{x}_ 0$, reflecting how much uncertainty we have given $\mathbf{x}_ 0$ at each denoising step: $\tilde\beta_ T \approx \beta_ T$, $\tilde\beta_ 1 \approx 0$.

$$\boxed{\Sigma_ \theta(\mathbf{x}_ t,t) = \tilde\beta_ t =  \frac{1 - \bar\alpha_ {t-1}}{1 - \bar\alpha_ t}\beta_ t}$$

### Computing the Posterior Mean

Recall that in the forward process, we produce $\mathbf{x}_ t$ from $\mathbf{x}_ {t-1}$ with:

$$\mathbf{x}_ t = \sqrt{\alpha_ t}\mathbf{x}_ {t-1} + \sqrt{\beta_ t}\boldsymbol{\epsilon}_ t,\quad \boldsymbol{\epsilon}_ t \sim \mathcal{N}(0,\mathbf{I})$$

Our goal is to compute the mean of the previous image, so let's solve for $\mathbf{x}_ {t-1}$,

$$\mathbf{x}_ {t-1} = \frac{1}{\sqrt{\alpha_ t}}(\mathbf{x}_ t - \sqrt{\beta_ t}\boldsymbol{\epsilon}_ t)$$

then let's take the expectation given the current image $\mathbf{x}_ t$:

$$
\begin{aligned}
\mathbb{E}[\mathbf{x}_ {t-1}|\mathbf{x}_ t] &= \mathbb{E}\Bigl[\frac{1}{\sqrt{\alpha_ t}}(\mathbf{x}_ t - \sqrt{\beta_ t}\boldsymbol{\epsilon}_ t)|\mathbf{x}_ t\Bigr] \\
\mathbb{E}[\mathbf{x}_ {t-1}|\mathbf{x}_ t] &= \frac{1}{\sqrt{\alpha_ t}}(\mathbf{x}_ t - \sqrt{\beta_ t}\mathbb{E}[\boldsymbol{\epsilon}_ t|\mathbf{x}_ t])
\end{aligned}
$$

Now, our only unknown in the right-hand side of the equation is $\mathbb{E}[\boldsymbol{\epsilon}_ t|\mathbf{x}_ t]$, the expected value of the noise added at step $t$. Instead of training our model to predict the mean itself, we can train it to predict $\boldsymbol{\epsilon}_ \theta(\mathbf{x}_ t,t)$, the total noise added from $\mathbf{x}_ 0$ to $\mathbf{x}_ t$. As in the equations above, we know that the standard deviation of the noise added at step $t$ is $\sqrt{\beta_ t}.$ We also know that the standard deviation of the total noise added from $\mathbf{x}_ 0$ to $\mathbf{x}_ t$ is $\sqrt{1 - \bar\alpha_ t}.$ Therefore, if we want to scale our model's predicted total noise $\boldsymbol{\epsilon}_ \theta(\mathbf{x}_ t,t)$ down to the noise only at step $t$, we can multiply it by step $t$'s proportion of the total noise, $\frac{\sqrt{\beta_ t}}{\sqrt{1-\bar\alpha_ t}}:$

$$\mathbb{E}[\boldsymbol{\epsilon}_ t|\mathbf{x}_ t] \approx \frac{\sqrt{\beta_ t}}{\sqrt{1-\bar\alpha_ t}}\boldsymbol{\epsilon}_ \theta(\mathbf{x}_ t,t)$$

If we plug that into the mean equation $\mathbb{E}[\mathbf{x}_ {t-1}|\mathbf{x}_ t]$, we arrive at the equation for $\boldsymbol{\mu}_ \theta(\mathbf{x}_ t,t):$

$$
\begin{aligned}
\boldsymbol{\mu}_ \theta(\mathbf{x}_ t,t) &= \frac{1}{\sqrt{\alpha_ t}}\Bigl(\mathbf{x}_ t - \sqrt{\beta_ t}\frac{\sqrt{\beta_ t}}{\sqrt{1 - \bar\alpha_ t}}\boldsymbol{\epsilon}_ \theta(\mathbf{x}_ t,t)\Bigr) \\
&= \boxed{\frac{1}{\sqrt{\alpha_ t}}\Bigl(\mathbf{x}_ t - \frac{\beta_ t}{\sqrt{1 - \bar\alpha_ t}}\boldsymbol{\epsilon}_ \theta(\mathbf{x}_ t,t)\Bigr)}
\end{aligned}
$$

### Denoising Step

To then perform a DDPM denoising step and obtain $\mathbf{x}_ {t-1}$, you simply produce a sample with mean $\boldsymbol{\mu}_ \theta$ and added Gaussian noise with variance $\tilde\beta_ t:$

$$
\begin{aligned}
\mathbf{x}_ {t-1} &= \boldsymbol{\mu}_ \theta(\mathbf{x}_ t,t) + \sqrt{\tilde\beta_ t}\mathbf{z},\quad \mathbf{z} \sim \mathcal{N}(0,\mathbf{I}) \\
\mathbf{x}_ {t-1} &= \boxed{\frac{1}{\sqrt{\alpha_ t}}\Bigl(\mathbf{x}_ t - \frac{\beta_ t}{\sqrt{1-\bar\alpha_ t}}\boldsymbol{\epsilon}_ \theta(\mathbf{x}_ t,t)\Bigl) + \sqrt{\tilde\beta_ t}\mathbf{z}}
\end{aligned}
$$

## Training / Generation

To train a model to predict $\boldsymbol{\epsilon}_ \theta(\mathbf{x}_ t,t)$, we define the training objective simply as the mean squared error between the model's predicted noise and the true noise:

$$\boxed{L(\theta) = \mathbb{E}_ {\mathbf{x}_ 0,t,\boldsymbol{\epsilon}}\Bigl[\Vert\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_ \theta(\mathbf{x}_ t,t)\Vert^2\Bigr]}$$

With a trained noise prediction model, DDPM-style image generation is as simple as sampling Gaussian noise and repeatedly applying the denoising update rule until you reach a clean image!

<div align="center">
    <img src="figs/generation.png">
    <p align="center">DDPM image generation</p>
</div>

# Diffusion Slides

If you'd like to see diffusion models explained from the stochastic differential equation perspective, check out [these slides](https://docs.google.com/presentation/d/1OqrJJelGhm7FuKFeKAIcXTOgOvlkOGR9AtnScGCXeKI/edit?usp=sharing)!
