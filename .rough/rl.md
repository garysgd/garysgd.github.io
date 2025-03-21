---
layout: post
title: "Reasoning and RL for Large Language Models"
subtitle: "From PPO to GRPO"
date: 2025-02-12
categories: [machine-learning, reinforcement-learning]
tags: [python, llms, generative ai]
---


The introduction of reasoning methods marks a paradigm shift in the way computation is performed using large language models. 
Reasoning was first introduced into the mainstream with ChatGPT o1, where reasoning tokens were generated as part of the output.
The more tokens were generated, the higher quality the model would perform in benchmarks, in particular those related to math and coding.
This also has a nice analogy/mapping back to the classic space-time tradeoff for computing.
With a high amount of storage, less compute is needed to solve a problem.
If computing can be done on the fly, then less storage is needed.
This is also apt in another sense, since pretraining neural networks has also been shown as a compression/storage technique.
The prompts/inputs to the LLM can be thought of as a key or unique identifier, and when paired with the compressed LLM it is able to decompress a lossy version of the training data.

While previous LLMs could reason to some extent using basic prompt engineering techniques like 'think step by step', o1 was one of the first models where reasoning was part of training the model. A further breakthrough came with the introduction of deepseek r1, a reasoning model with open sourced code. The launch of r1 caused a huge waves in both the tech sector and the economy, as training costs were reported to be far lower than that of previous LLMs. This resulted in a large drop in market cap for GPU manufacturers that provide hardware for training models.

One of the key innovations of the deepseek model is the Group Relative Policy Optimization (GRPO) algorithm, which removes the need for training the critic network that is typically a part of the Proximal Policy Optimization (PPO) used to train previous LLMs. We will first introduce how PPO works in the context of LLMs before highlighting the innovations of GRPO.

The PPO surrogate objective is given by:
$$
J_{\text{PPO}}(\theta) = \mathbb{E}_{q \sim P(Q),\, o \sim \pi_{\theta}^{\text{old}}(\cdot \mid q)}
\left[
\frac{1}{|o|} \sum_{t=1}^{|o|}
\min\!\left\{
\frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{\theta}^{\text{old}}(o_t \mid q, o_{<t})} A_t,\,
\text{clip}\!\left(
\frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{\theta}^{\text{old}}(o_t \mid q, o_{<t})},\, 1-\epsilon,\, 1+\epsilon
\right) A_t
\right\}
\right].
$$

We can first go through this equation term by term in the context of natural language processing to gain better insight on how PPO works. We can start with the policy:

$$\pi_\theta(o_t \mid q, o_{<t})$$

For nlp the policy $$\pi_\theta$$ is determined by the neural network, typically based on the transformer architecture. The neural network takes in $$t-1$$ input tokens of a sequence $$o$$ along with the prompt sequence $$q$$ to predict the $$t$$ token of the sequence denoted by $$o_t$$. We can illustrate this with a simple example. While tokens typically denote subwords akin to phonemes rather than actual words, we will assume each word is a token for the sake of simplification.

Given a prompt $$q$$ such as "What is the capital of France?" the policy model $$\pi_\theta$$ starts to predict starting with the first token $$o_1$$ which could be "The". The policy model would then output subsequent tokens dependent based on the prompt and the previous output tokens. The ratio $$\frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{\theta}^{\text{old}}(o_t \mid q, o_{<t})}$$ then measures how much more likely the new policy is to predict the subsequent token $$o_t$$ relative to the old policy. It acts as a weight for the advantage $$A_t$$ which we will now elaborate now.

For PPO, the advantage $$A_t$$ can be expressed as 

$$A_t = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \left[ Q(s_{t+l}, a_{t+l}) - V(s_{t+l}) \right]$$

$$Q(s_t, a_t) represents the cumulative reward after outputing 
$$Q(s_t, a_t) \approx r_t + \gamma\, V(s_{t+1})$$

$$\delta_t = Q(s_t, a_t) - V(s_t) = r_t + \gamma\, V(s_{t+1}) - V(s_t)$$

$$A_t = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l\, \delta_{t+l}$$
