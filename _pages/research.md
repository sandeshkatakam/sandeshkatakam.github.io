---
permalink: /research/
title: "Research Themes"
excerpt: "Page contains updates about my current research interests"
author_profile: true
redirect_from: 
  - "/nmp/"
  - "/nmp.html"
---

# A Unified View of Generative Models: From Diffusions to Flow Matching via Stochastic Control

## Overview

Modern generative models—diffusion models, flow matching, and Schrödinger bridges—are often presented as distinct frameworks. However, at a deeper level, they can all be understood as instances of a single underlying principle:

> **Learning dynamics that transport a simple prior distribution into a complex data distribution.**

This page presents a unified perspective on these models through the lens of **measure evolution**, **stochastic processes**, and **optimal control**.

---

## 1. The General Problem

Let:
- \( p_0 \): a simple prior distribution (e.g., Gaussian)
- \( p_{\text{data}} \): the target data distribution

We seek a time-indexed process \( x_t \), \( t \in [0, T] \), such that:
\[
x_0 \sim p_0, \quad x_T \sim p_{\text{data}}
\]

This induces a family of intermediate distributions \( p_t(x) \), and the core question becomes:

> **How should probability mass evolve over time to transform \( p_0 \) into \( p_{\text{data}} \)?**

---

## 2. Two Fundamental Views of Dynamics

### (A) Stochastic Dynamics (SDEs)

\[
dx_t = f(x_t, t)\,dt + g(t)\,dW_t
\]

- Induces evolution via the **Fokker–Planck equation**:
\[
\partial_t p_t = -\nabla \cdot (f p_t) + \frac{1}{2} g(t)^2 \Delta p_t
\]

---

### (B) Deterministic Dynamics (ODEs)

\[
\frac{dx_t}{dt} = v(x_t, t)
\]

- Induces evolution via the **continuity equation**:
\[
\partial_t p_t = -\nabla \cdot (v p_t)
\]

---

## 3. Diffusion Models (Score-Based Generative Models)

Diffusion models define a forward SDE that gradually transforms data into noise:

\[
dx_t = f(x_t, t)\,dt + g(t)\,dW_t
\]

Sampling requires the reverse-time SDE:

\[
dx_t = \left[f(x_t,t) - g(t)^2 \nabla \log p_t(x_t)\right] dt + g(t)\,d\bar{W}_t
\]

### Key object:
\[
\nabla \log p_t(x) \quad \text{(score function)}
\]

This is learned via **score matching**.

---

## 4. Probability Flow ODE

The same marginal distributions \( p_t \) can be obtained via a deterministic ODE:

\[
\frac{dx}{dt} = f(x,t) - \frac{1}{2} g(t)^2 \nabla \log p_t(x)
\]

This removes stochasticity and reveals:

> **Diffusion models implicitly define a velocity field via the score.**

---

## 5. Flow Matching

Flow matching directly learns a velocity field:

\[
\frac{dx}{dt} = v_\theta(x,t)
\]

Instead of simulating an SDE, it constructs interpolation paths \( x_t \) between noise and data and trains:

\[
v_\theta(x_t,t) \approx \dot{x}_t
\]

### Key identity:
\[
v^*(x,t) = \mathbb{E}[\dot{x}_t \mid x_t = x]
\]

This avoids:
- forward SDE simulation
- score estimation

---

## 6. Schrödinger Bridge (Entropy-Regularized Transport)

Schrödinger bridge solves:

> Find the most likely stochastic process connecting two distributions under a reference diffusion.

Formulation:
\[
\min_{P} \; \mathrm{KL}(P \,\|\, P_{\text{ref}})
\quad \text{s.t.} \quad
P(x_0)=p_0,\; P(x_T)=p_{\text{data}}
\]

Resulting dynamics:
\[
dx_t = \nabla \log \psi(x_t,t)\,dt + dW_t
\]

This yields:
- a **controlled diffusion**
- optimal drift derived from a value function

---

## 7. Unifying View via Velocity / Drift

All frameworks define a velocity (or drift):

| Framework | Object Learned | Dynamics |
|----------|-------------|---------|
| Diffusion | \( \nabla \log p_t \) | SDE |
| Prob. Flow ODE | Score → velocity | ODE |
| Flow Matching | \( v(x,t) \) | ODE |
| Schrödinger Bridge | \( \nabla \log \psi \) | SDE |

---

## 8. Control-Theoretic Formulation

We can unify all methods via stochastic control:

\[
dx_t = u(x_t,t)\,dt + \sqrt{2}\,dW_t
\]

with objective:
\[
\min_u \; \mathbb{E} \left[
\int_0^T \frac{1}{2}\|u(x_t,t)\|^2 dt + \Phi(x_T)
\right]
\]

Optimal control satisfies:
\[
u^*(x,t) = \nabla \log \psi(x,t)
\]

---

## 9. Key Insight

> All modern generative models can be viewed as learning a **transport field** that evolves probability distributions over time.

- Diffusion → learns **score**
- Flow matching → learns **velocity**
- Schrödinger bridge → learns **optimal control**

---

## 10. Conceptual Diagram

\[
\text{Transport of probability mass}
\Rightarrow
\begin{cases}
\text{Stochastic (SDE)} \\
\text{Deterministic (ODE)} \\
\text{Controlled dynamics}
\end{cases}
\]

---

## 11. Important References

### Diffusion Models
- Ho et al. (2020) — *Denoising Diffusion Probabilistic Models*
- Song et al. (2021) — *Score-Based Generative Modeling via SDEs*

### Probability Flow ODE
- Song et al. (2021) — same as above

### Flow Matching
- Lipman et al. (2023) — *Flow Matching for Generative Modeling*

### Stochastic Interpolants
- Albergo et al. (2023) — *Stochastic Interpolants*

### Schrödinger Bridge
- Chen et al. (2021) — *Schrödinger Bridge for Generative Modeling*
- De Bortoli et al. (2021) — *Diffusion Schrödinger Bridge*

### Optimal Transport / Control
- Benamou & Brenier (2000)
- Todorov (2009) — *Linearly Solvable MDPs*

---

## 12. My Research Direction

The goal of my work is to:

> **Develop a unified framework that explicitly leverages stochastic control to design generative models that are efficient, stable, and interpretable.**

This includes:
- Bridging flow matching and Schrödinger bridges
- Designing velocity fields robust to discretization
- Understanding generative modeling as controlled measure dynamics

---

## 13. Summary

\[
\boxed{
\text{Generative modeling = learning dynamics that transport } p_0 \rightarrow p_{\text{data}}
}
\]

The differences between frameworks lie not in *what* they do, but in *how* they parameterize and learn these dynamics.

---
  


















