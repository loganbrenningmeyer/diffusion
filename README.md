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

Let's call the true distribution $p(x)$, e.g., a distribution of dog images where $x \sim p(x)$ is a sample image of a dog. To generate new samples, we want to learn how to **sample a latent variable $z$** from some random, easy-to-sample distribution $p(z)$—often a normal distribution $z \sim \mathcal{N}(\mu,\sigma^2)$—and **transform it into a sample from the true distribution**.

<div align="center">
    <img src="figs/mapping.png" width="40%">
    <p align="center">Mapping samples from p(z) to the true distribution p(x)</p>
</div>

However, actually knowing $p(x)$ is intractable—you can't know every possible dog image. So we want to create a model with parameters $\theta$ that approximates $p_\theta(x|z)$ given a random latent $z \sim p(z)$, i.e., "given some noise, what clean images does it likely map to?". This allows us to produce new samples $x \sim p_\theta(x|z)$ using our model.

## Simplifying the Problem

In a perfect world, our model could learn to produce a perfect sample from the true distribution in one shot from any given latent $z$. In very simple cases, this perfect world actually exists. If $p(x)$ were simply a one-dimensional normal distribution $x \sim \mathcal{N}(\mu,\sigma^2)$, our model could easily predict the mean $\hat{\mu}$ and variance $\hat{\sigma}^2$ of $p(x)$. We could then simply create new samples by:

1. Sampling a latent $z \sim \mathcal{N}(0,1)$ 
2. Producing a sample $x = \hat{\mu} + \hat{\sigma}z$ 

<div align="center">
    <img src="figs/px_pz.png" width="100%">
    <p align="center">Generating new samples from a 1-D normal distribution using the learned mean and variance</p>
</div>

In practice, $p(x)$ is a highly complex, high-dimensional distribution ($256\times256\text{px}$ RGB images have 196,608 dimensions!), so it is impossible to perfectly map $z \sim p(z) \to x \sim p(x)$. Therefore, we can try to simplify the problem by moving from $z \to x$ in *very small* steps.

### Incremental Steps

Let's try and break the problem into $T$ steps, incrementally transforming $z$ many times until we reach $x$. In diffusion models, we sample $z \sim \mathcal{N}(0,I)$, beginning with pure Gaussian noise with zero-mean and unit-variance. We'll call the target variable $x_0$—the clean image after $T$ incremental steps of denoising $z$. To keep things in terms of $T$, we'll refer to our starting latent $z$ as $x_T$ from now on—the pure Gaussian noise we want to clean up. Therefore, the incremental process we hope to perform is:

$$x_T \to x_{T-1} \to \ldots \to x_1 \to x_0$$

This greatly simplifies our model's task, as it now simply needs to learn the conditional distribution

$$\boxed{x_{t-1} \sim p_\theta(x_{t-1}|x_t)},$$

meaning "given the current noisy $x_t$, what is the distribution of the slightly less noisy $x_{t-1}$?" Rather than trying to guess the correct clean image from pure noise—an impossible task—the model only predicts a slightly denoised version of the current image.

<div align="center">
    <img src="figs/reverse_steps.png" width="80%">
    <p align="center">Denoising from pure noise to a clean image</p>
</div>

### Learning to Denoise

But how do we train a model to denoise images? Let's say we have a training dataset $\mathcal{D}$ of clean images $x_0 \in \mathcal{D}$, and we need to provide the model with lots of examples of denoising from $x_{t} \to x_{t-1}$. However, since we don't have some true dataset of noisy $x_t$ samples, we'll have to make them ourselves. This is known as the **forward diffusion process**, the process of iteratively noising clean samples.

## Forward Process

So let's make some noise! We'll start with the clean image $x_0$ at $t=0$ and add noise at each step until we reach $x_T$ at $t=T$, where we want $x_T$ to be pure Gaussian noise, $x_T \sim \mathcal{N}(0,I)$. Keep in mind, we will be feeding $x_t$ into a neural network, which learns best from normalized data. Therefore, we want to ensure $x_t$ is normalized at every step to have zero-mean and unit-variance, meaning we want:

$$E[x_t]=0,\quad \text{Var}[x_t]=1,\quad t=0,1,\ldots,T$$

### Adding Noise

Let's start out simple by adding some standard Gaussian noise at each step, scaled to have some variance $\sigma^2$:

$$x_t = x_{t-1} + \sigma\epsilon_t,\quad \epsilon_t \sim \mathcal{N}(0,I)$$

This may seem like a good idea at first glance, but naively adding the same Gaussian noise at every step means the variance of $x_t$ increases linearly over time:

$$
\begin{aligned}
\text{Var}[x_0] &= 1 \\
\text{Var}[x_1] &= \text{Var}[x_0 + \sigma\epsilon_1] = \text{Var}[x_0] + \sigma^2 = 1 + \sigma^2 \\
\text{Var}[x_2] &= \text{Var}[x_1 + \sigma\epsilon_2] = \text{Var}[x_1] + \sigma^2 = 1 + 2\sigma^2 \\
\vdots \\
\text{Var}[x_t] &= \text{Var}[x_{t-1} + \sigma\epsilon_t] = \text{Var}[x_{t-1}] + \sigma^2 = \boxed{1 + t\sigma^2}
\end{aligned}
$$

This means that at timestep $t$, the noisy image's values are roughly in the range $[-1-t\sigma^2,\ 1+t\sigma^2]$, making the magnitudes of pixel-values *much* larger than they should be: 

<div align="center">
    <img src="figs/bad_noise.png" width="80%">
    <p align="center">Adding the same Gaussian noise at each step linearly increases variance</p>
</div>

### Preserving Variance

But remember, we want keep each $x_t$ properly normalized to have $E[x_t]=0$ and $\text{Var}[x_t]=1$. So let's keep $x_t$'s values in check by introducing a scaling factor $c<1$ to shrink $x_t$ towards 0:

$$x_t = cx_{t-1} + \sigma\epsilon_t,\quad \epsilon_t \sim \mathcal{N}(0,I)$$

We want $\text{Var}[x_t]=1$ at every timestep $t$, so we can solve for the $c$ that keeps it that way:

$$
\begin{aligned}
\text{Var}[x_t] = 1 &= \text{Var}[cx_{t-1} + \sigma\epsilon_t] \\
&= \text{Var}[cx_{t-1}] + \text{Var}[\sigma\epsilon_t] \\
&= c^2\underbrace{\text{Var}[x_{t-1}]}_{1} + \sigma^2\underbrace{\text{Var}[\epsilon_t]}_{1}
\end{aligned}
$$

$$
\begin{aligned}
1 &= c^2 + \sigma^2 \\
c &= \boxed{\sqrt{1 - \sigma^2}}
\end{aligned}
$$

Now if we substitute in $c = \sqrt{1-\sigma^2}$, we can double check that our math works out such that $\text{Var}[x_t]=1$:

* Substitute $c=\sqrt{1-\sigma^2}$:

$$x_t = \sqrt{1 - \sigma^2}x_{t-1} + \sigma\epsilon_t,\quad \epsilon_t \sim \mathcal{N}(0,I)$$

* Compute $\text{Var}[x_t]$:

$$
\begin{aligned}
\text{Var}[x_t] &= \text{Var}[\sqrt{1 - \sigma^2}x_{t-1} + \sigma\epsilon_t] \\
&= \text{Var}[\sqrt{1 - \sigma^2}x_{t-1}] + \text{Var}[\sigma\epsilon_t] \\
&= (1 - \sigma^2)\underbrace{\text{Var}[x_{t-1}]}_{1} + \sigma^2\underbrace{\text{Var}[\epsilon_t]}_{1} \\
&= 1 - \sigma^2 + \sigma^2 \\
&= \boxed{1}
\end{aligned}
$$

### Variance Schedule

*Great! So we can just pick some variance* $\sigma^2$ *for our noise, scale* $x_{t-1}$ *by* $\sqrt{1-\sigma^2}$ *at each step, and everything works out!* Unfortunately, there are a few issues with that:

1. If $\sigma=1$, then $\sqrt{1-\sigma^2} = 0$, and $x_{t-1}$ is completely removed at every step—we'd only have noise from $t=1$ onwards.
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

We need to find a compromise: we want the model to see timesteps with little noise and timesteps with lots of noise, and we want the noise to be added smoothly over time. Therefore, we need a **variance schedule** to control how much noise is added at each timestep. So instead of a fixed variance $\sigma^2$ at all timesteps, we'll introduce $\beta_t$ as the variance at timestep $t$. For notation convenience, we'll also let $\alpha_t = 1 - \beta_t$. By scheduling the variance of the noise, our old equation,

$$x_t = \sqrt{1 - \sigma^2}x_{t-1} + \sigma\epsilon_t,\quad \epsilon_t \sim \mathcal{N}(0,I),$$

now becomes,

$$\boxed{x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{\beta_t}\epsilon_t,\quad \epsilon_t \sim \mathcal{N}(0,I)}$$

A single step can also be written as the conditional distribution:

$$\boxed{q(x_t|x_{t-1}) \sim \mathcal{N}(\sqrt{\alpha_t}x_{t-1},\ \beta_t I)}$$

meaning that given the previous sample $x_{t-1}$, the next sample $x_t$ is sampled from a normal distribution with mean $\sqrt{\alpha_t}x_{t-1}$ and variance $\beta_t$.

### Closed-Form Forward Equation

At each step, $x_{t-1}$ is scaled by $\sqrt{\alpha_t}$, which means we keep $\sqrt{\alpha_t}$ of $x_{t-1}$ at each noising step, i.e., its retained signal is $\sqrt{\alpha_t}x_{t-1}.$ If we start at $x_0$ and add noise for $t=1,2,\ldots,T$, we can see how much of the *original signal* we retain:

$$
\begin{align*}
x_1 &= \sqrt{\alpha_1}x_0 + \sqrt{\beta_1}\epsilon_1 \\
x_2 &= \sqrt{\alpha_2}(\sqrt{\alpha_1}x_0 + \sqrt{\beta_1}\epsilon_1) + \sqrt{\beta_2}\epsilon_2 \\
x_3 &= \sqrt{\alpha_3}(\sqrt{\alpha_2}(\sqrt{\alpha_1}x_0 + \sqrt{\beta_1}\epsilon_1) + \sqrt{\beta_2}\epsilon_2) + \sqrt{\beta_3}\epsilon_3 \\
\vdots \\
\end{align*}
$$

As you can see, at each timestep $t$, $x_0$ is scaled by the cumulative product $\sqrt{\alpha_1} \cdot \sqrt{\alpha_2} \cdot \sqrt{\alpha_3} \cdots \sqrt{\alpha_t}.$ We can then define the retained signal of $x_0$ at timestep $t$ as $\sqrt{\bar\alpha_t}x_0$, where

$$\bar\alpha_t = \prod_{s=1}^t \alpha_s$$

The variance of the retained signal of $x_0$ is then $\text{Var}[\sqrt{\bar\alpha_t}x_0]=\bar\alpha_t$. Since we've defined the forward diffusion process so that the total variance of $x_t$ is always $1$, and because $x_t$ is simply the retained signal from $x_0$ at timestep $t$ plus noise, we can say:

$$\text{Var}[\text{signal}_t] + \text{Var}[\text{noise}_t] = 1$$

We've just defined that $\text{Var}[\text{signal}_t]=\bar\alpha_t$, so we can then solve for the variance of the noise at timestep $t$:

$$
\begin{align*}
\bar\alpha_t + \text{Var}[\text{noise}_t] &= 1 \\
\text{Var}[\text{noise}_t] &= 1 - \bar\alpha_t
\end{align*}
$$

Now we can make a closed-form equation for $x_t$ from $x_0$ without having to noise every step individually!

$$\boxed{x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1 - \bar\alpha_t}\epsilon,\quad \epsilon \sim \mathcal{N}(0,I)}$$

### Planning the Variance Schedule

As the variance of the retained signal and the variance of the noise sum to $1$, the signal variance $\bar\alpha_t$ provides a good measure of how much of the clean image the model can see compared to noise at timestep $t$. Therefore, we want $\bar\alpha_t$ to decrease smoothly over time, allowing the model to see the clean image nicely transition into noise. So we must define a variance schedule $\{\beta_t\}_{t=1}^T$ that makes $\bar\alpha_t$ smoothly decay. Recall that $\alpha_t = 1 - \beta_t$, so we can write $\bar\alpha_t$ as:

$$\bar\alpha_t = \prod_{s=1}^t 1 - \beta_t$$

Clearly, as $\beta_t$ increases, $\bar\alpha_t$ decreases, so how should we increase $\beta_t$? In the original DDPM paper, they went with the simplest route: linearly increase $\beta_t$. They empirically chose to start with $\beta_1=0.0001$ and end with $\beta_T=0.02$ over $T=1000$ steps, simply picking these values to ensure that:

1. At the start, $\bar\alpha_1 = 0.9999$, meaning $x_1$ is nearly a completely clean image.
2. By the end, $\bar\alpha_T \approx 0.00004$, meaning $x_T$ is essentially pure noise.
3. $\bar\alpha_t$ exponentially decays smoothly over time, which is helpful for our model to learn. 

<div align="center">
    <img src="figs/betas_alpha_bars.png" width="80%">
    <img src="figs/forward_steps.png" width="80%">
    <p align="center">
        Linear variance schedule with &beta;<sub>1</sub>=0.0001, &beta;<sub>T</sub>=0.02
    </p>
</div>

Now we have a way to generate training samples! We can just use our linear variance schedule and our closed-form forward equation to produce any $x_t$ from $x_0$.

## Reverse Process

Recall that a single step of the forward noising process is defined as:

$$q(x_t|x_{t-1}) \sim \mathcal{N}(\sqrt{\alpha_t}x_{t-1},\ \beta_t I).$$

Our goal is to train our model to learn $p(x_{t-1}|x_t)$, i.e., the reverse of the forward process. According to Bayes' Rule, we can solve for $p(x_{t-1}|x_t)$ with:

$$p(x_{t-1}|x_t) = \frac{q(x_t|x_{t-1})p(x_{t-1})}{p(x_t)}.$$

Just as we can't know the distribution of all dogs $p(x)$, we can't know the distribution of all noisy dogs $p(x_t)$—it is intractable—so we'll ignore the normalizing term in the denominator and write:

$$p(x_{t-1}|x_t) \propto q(x_t|x_{t-1})p(x_{t-1}).$$

### Gaussian Assumption

Here we make a key assumption about the distribution $p(x_{t-1}|x_t)$ that makes all of this work—it is Gaussian. 

Technically, $q(x_t|x_{t-1})$ defines the forward distribution for $x_t$, asking: "*What is the slightly noisier version of* $x_{t-1}$*?*" However, in the reverse process we *only know* $x_t$ and want to *predict* $x_{t-1}$. In that case, $q(x_t|x_{t-1})$ acts as the **likelihood** of $x_{t-1}$, instead asking: "*What previous sample likely produced* $x_t$*?*" 

As we know from the forward process, $x_t$ is just a scaled-down version of the previous image $\sqrt{\alpha_t}x_{t-1}$ plus a little bit of noise. Therefore, given $x_t$, it's reasonable to believe that the previous image was likely around $\frac{x_t}{\sqrt{\alpha_t}}$—the current image scaled back up. Because we made the crucial design choice to split up the problem into many *very, very small* steps, this approximation is very accurate. As a result, $q(x_t|x_{t-1})$, the likelihood of $x_{t-1}$ given $x_t$, is a sharp Gaussian spike around $\frac{x_t}{\sqrt{\alpha_t}}$ and nearly $0$ everywhere else.

The data distribution $p(x_{t-1})$ is itself intractable like $p(x_t)$—it is the complex, curved manifold of all valid $x_{t-1}$ samples. However, multiplying it with the narrow Gaussian spike $q(x_t|x_{t-1})$ effectively "zooms in" very closely around $\frac{x_t}{\sqrt{\alpha_t}}$, where $p(x_{t-1})$ is essentially flat or constant. 

Multiplying the Gaussian spike $q(x_t|x_{t-1})$ with the locally constant region of $p(x_{t-1})$ is just a scaled Gaussian, meaning $p(x_{t-1}|x_t)$ is Gaussian as well! This is very convenient, as to train our model to represent $p_\theta(x_{t-1}|x_t)$, we only need to know the mean and variance that define the Gaussian distribution!

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t,t),\ \Sigma_\theta(x_t,t))$$

### Computing Variance

Luckily, we don't need to predict $\Sigma_\theta(x_t,t)$ using the model; we can compute it analytically. In the original DDPM paper, they experimented with two formulations:

1. **Forward Variance**: The variance $\beta_t$ defined by the variance schedule, assumes no knowledge of $x_0$.

$$\boxed{\Sigma_\theta(x_t,t) = \beta_t}$$

2. **Posterior Variance**: The true posterior variance assuming that we know $x_0$, reflecting how much uncertainty we have given $x_0$ at each denoising step: $\tilde\beta_T \approx \beta_T$, $\tilde\beta_1 \approx 0$.

$$\boxed{\Sigma_\theta(x_t,t) = \tilde\beta_t =  \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\beta_t}$$

### Computing the Posterior Mean

Recall that in the forward process, we produce $x_t$ from $x_{t-1}$ with:

$$x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{\beta_t}\epsilon_t,\quad \epsilon_t \sim \mathcal{N}(0,I)$$

Our goal is to compute the mean of the previous image, so let's solve for $x_{t-1}$,

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \sqrt{\beta_t}\epsilon_t)$$

then let's take the expectation given the current image $x_t$:

$$
\begin{aligned}
\mathbb{E}[x_{t-1}|x_t] &= \mathbb{E}\Bigl[\frac{1}{\sqrt{\alpha_t}}(x_t - \sqrt{\beta_t}\epsilon_t)|x_t\Bigr] \\
\mathbb{E}[x_{t-1}|x_t] &= \frac{1}{\sqrt{\alpha_t}}(x_t - \sqrt{\beta_t}\mathbb{E}[\epsilon_t|x_t])
\end{aligned}
$$

Now, our only unknown in the right-hand side of the equation is $\mathbb{E}[\epsilon_t|x_t]$, the expected value of the noise added at step $t$. Instead of training our model to predict the mean itself, we can train it to predict $\epsilon_\theta(x_t,t)$, the total noise added from $x_0$ to $x_t$. As in the equations above, we know that the standard deviation of the noise added at step $t$ is $\sqrt{\beta_t}.$ We also know that the standard deviation of the total noise added from $x_0$ to $x_t$ is $\sqrt{1 - \bar\alpha_t}.$ Therefore, if we want to scale our model's predicted total noise $\epsilon_\theta(x_t,t)$ down to the noise only at step $t$, we can multiply it by step $t$'s proportion of the total noise, $\frac{\sqrt{\beta_t}}{\sqrt{1-\bar\alpha_t}}:$

$$\mathbb{E}[\epsilon_t|x_t] \approx \frac{\sqrt{\beta_t}}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)$$

If we plug that into the mean equation $\mathbb{E}[x_{t-1}|x_t]$, we arrive at the equation for $\mu_\theta(x_t,t):$

$$
\begin{aligned}
\mu_\theta(x_t,t) &= \frac{1}{\sqrt{\alpha_t}}\Bigl(x_t - \sqrt{\beta_t}\frac{\sqrt{\beta_t}}{\sqrt{1 - \bar\alpha_t}}\epsilon_\theta(x_t,t)\Bigr) \\
&= \boxed{\frac{1}{\sqrt{\alpha_t}}\Bigl(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\epsilon_\theta(x_t,t)\Bigr)}
\end{aligned}
$$

### Denoising Step

To then perform a DDPM denoising step and obtain $x_{t-1}$, you simply produce a sample with mean $\mu_\theta$ and added Gaussian noise with variance $\tilde\beta_t:$

$$\boxed{x_{t-1} = \mu_\theta(x_t,t) + \sqrt{\tilde\beta_t}z,\quad z \sim \mathcal{N}(0,I)}$$

## Training / Generation

To train a model to predict $\epsilon_\theta(x_t,t)$, we define the training objective simply as the mean squared error between the model's predicted noise and the true noise:

$$\boxed{L(\theta) = \mathbb{E}_{x_0,t,\epsilon}\Bigl[\|\epsilon - \epsilon_\theta(x_t,t)\|_2^2\Bigr]}$$