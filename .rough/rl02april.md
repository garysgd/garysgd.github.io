---
layout: post
title: "GRPO: How DeepSeek created value without value"
subtitle: "From PPO to GRPO"
date: 2025-02-12
categories: [machine-learning, reinforcement-learning]
tags: [python, llms, generative ai]
---

<p align="center">
  <img src="/images/grpo/grpo1.png" alt="Grid World Image" width="300">
</p>

When we first think of Reinforcement Learning, we usually imagine a grid with a start and end point. This is the well known problem of teaching an agent to reach its goal using a clearly defined reward function. While Reinforcement Learning techniques have existed since the 1970s (ref) , their application in Natural Language Processing (NLP) is fairly more recent, with arguably the most successful use case being to train ChatGPT by OpenAI. What is the reward for NLP?

Every technological leap in human history required a successful marriage between theory and experiment, software and hardware. In the case of machine learning, it was the backpropagation algorithm along with the GPU hardware necessary to perform their computations in a parallel and reasonable timeframe. To combine NLP and Reinforcement Learning there is a third party: Data. 

With trillions of words written and available on the internet, it would be reasonable to assume that data would not be a bottleneck when training language models. However, despite the large amount of text available, it is the amount of useful text that determines the quality of the trained models.  Before OpenAI, NLP researchers had devised of an ingenious way to train these models. They used an autoregressive technique known as next token prediction. Researchers worked around the lack of well defined rewards in language, by deciding to reward the model to predict the next token/word instead. Doing so allowed the researchers to create AI models that sounded like Shakespeare (ref) or that could seemingly write code (ref). Even though these models may be seen as simple next token predictors, training them in such a way did allow them to learn the semantics of language. 

<p align="center">
  <img src="/images/grpo/grpo2.png" alt="Murder Mystery" width="350">
</p>


Ilya Sutskever has proposed thought experiment to illustrate the usefulness of next token prediction. In the experiment he imagines someone reading a murder mystery book where the person has read the entire book barring the last sentence, of which the identity of the murderer is revealed. If one where to replace the reader with a well trained next token predictor, it would be able to predict who the murderer is. We can say that the model likely has learned some meaning of the language. However, while next token predictors are able to learn language semantics, their usefulness still pales into comparison compared to modern day LLMs like ChatGPT. There are two problems that current models are able to solve that set them apart. The first is the ability to respond in the form of an answer whenever the model receives a question as an input. The second is to be able to answer in a way that is preferred or aligned to humans.

OpenAI addressed these challenges partly through the power of belief and capitalism. The belief that larger models perform better, combined with significant investment capital (at a scale largely unavailable to academics), allowed OpenAI to hire human evaluators to label data. This labeled data was then used in the well-known Reinforcement Learning with Human Feedback (RLHF) training step. With a large amount of high-quality labeled data, they were able to fine-tune a pretrained transformer in two stages.

The first stage, instruction fine-tuning, involved training the model on a dataset of instruction-response pairs. During this phase, the model learns to follow human instructions by mapping prompts to desired outputs, essentially learning to mimic the behavior reflected in the labeled data. This stage aligns the model’s outputs more closely with human expectations and provides a strong initialization for the next stage.

The second stage is RLHF, where reinforcement learning is applied to further refine the model’s behavior. In RLHF, a reward model—trained from human feedback—evaluates the outputs of the model, and the model’s policy is updated to maximize the expected cumulative reward. This step ensures that the model’s responses are not only correct but also aligned with human preferences, balancing correctness with aspects like clarity, helpfulness, and safety.

Together, these stages enable modern LLMs like ChatGPT to generate high-quality, human-aligned outputs. The RLHF stage, in particular, leverages reinforcement learning to fine-tune the model’s policy, using sophisticated techniques such as Proximal Policy Optimization (PPO) and, more recently, Group Relative Policy Optimization (GRPO). GRPO removes the need for a separate critic network by computing advantages based on group comparisons among multiple outputs, along with an explicit KL divergence penalty to keep the policy updates stable.

In this post, I will first introduce the various Reinforcement Learning notations that are used in the context of NLP, followed by going through the classic PPO objective followed by GRPO introduced by DeepSeek.
## RL Definitions in an NLP Context

Before diving into PPO and GRPO, we first defined the various notations used in the context of reinforcement learning for LLMs:

- **Prompt ($$q$$)**: The initial user input providing context for generation.

- **State ($$s_t$$)**: The combination of the prompt and all tokens generated so far, i.e., 
  $$ s_t = (q, o_{<t}) $$

- **Action ($$o_t$$)**: The next token generated by the model at time $$t$$.

- **Reward ($$r_t$$)**: A numerical score indicating how “good” the chosen token or sequence is (can be step-level or sequence-level).

- **Policy ($$\pi_\theta$$)**: The mapping from state $$s_t$$ to a probability distribution over the next token; effectively, the LLM parameterized by $$\theta$$.

- **Discount Factor ($$\gamma$$)**: Weights future rewards relative to immediate ones. For example, with $$\gamma=0.9$$, rewards are discounted by 10% per time step.

- **GAE Parameter ($$\lambda$$)**: Balances bias and variance in advantage estimation by controlling how much future rewards influence the advantage calculation (higher $$\lambda$$ includes more future rewards).


## Proximal Policy Optimization

The PPO surrogate objective is given by:

$$
\begin{aligned}
J_{\text{PPO}}(\theta)
&= \mathbb{E}_{q \sim P(Q),\, o \sim \pi_{\theta}^{\text{old}}(\cdot \mid q)}
\Biggl[
  \frac{1}{|o|} \sum_{t=1}^{|o|}
  \min\Bigl\{
    \frac{\pi_\theta(o_t \mid q,\,o_{<t})}{\pi_{\theta}^{\text{old}}(o_t \mid q,\,o_{<t})} \, A_t,\; \\[1mm]
&\quad
    \text{clip}\Bigl(
      \frac{\pi_\theta(o_t \mid q,\,o_{<t})}{\pi_{\theta}^{\text{old}}(o_t \mid q,\,o_{<t})},\; 1-\epsilon,\; 1+\epsilon
    \Bigr)\,A_t
  \Bigr\}
\Biggr]
\end{aligned}
$$


We can first go through this equation term by term in the context of NLP to gain better insight into how PPO works. We start with the policy:

$$
\pi_\theta(o_t \mid q, o_{<t})
$$

For NLP, the policy $$\pi_\theta$$ is determined by a neural network (typically based on the transformer architecture). The network takes in $$t-1$$ input tokens of the sequence $$o$$ along with the prompt $$q$$ to predict the $$t^{th}$$ token, denoted by $$o_t$$. For simplicity, we assume each word is a token.

Given a prompt $$q$$ such as "What is the capital of France?", the policy model $$\pi_\theta$$ begins by predicting the first token $$o_1$$, which could be "The". The model then generates subsequent tokens based on the prompt and the previously generated tokens. The ratio

$$
\frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{\theta}^{\text{old}}(o_t \mid q, o_{<t})}
$$

measures how much more likely the new policy is to predict the token $$o_t$$ compared to the old policy. This ratio acts as a weight for the advantage $$A_t$$, which we elaborate on next.

For PPO, the advantage $$A_t$$ can be expressed as

$$
A_t = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \left[ Q(s_{t+l}, o_{t+l}) - V(s_{t+l}) \right]
$$

Here, $$Q(s_t, o_t)$$ represents the cumulative reward after outputting $$o_t$$. It is the sum of the immediate reward for predicting that token, denoted by $$r_t$$, and the discounted sum of future rewards determined by the value network. In practice, we approximate

$$
Q(s_t, o_t) \approx r_t + \gamma\, V(s_{t+1})
$$

Thus, the temporal-difference error is given by

$$
\delta_t = Q(s_t, o_t) - V(s_t) = r_t + \gamma\, V(s_{t+1}) - V(s_t)
$$

The value network itself is trained using the loss function:

$$
L_V = \frac{1}{2}\,\mathbb{E}_t\left[\left(V_\psi(s_t)-R_t\right)^2\right],
$$

where $$R_t$$ is the true expected sum of the discounted future rewards:

$$
R_t = r_t + \gamma\, r_{t+1} + \gamma^2\, r_{t+2} + \cdots + \gamma^{T-t}\, r_T.
$$

From these definitions, the advantage quantifies the difference between the cumulative reward received for the current action (which in the case of NLP is the token generated) and the reward that was expected according to the value network. A positive advantage means that the token performed better than expected, while a negative advantage indicates it performed worse.

Intuitively, having a positive advantage implies that the policy is outputting a better token than expected, which is what we want to optimize. Returning to our PPO equation, the advantage is used in two terms:

- The first term,
  
  $$
  \frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{\theta}^{\text{old}}(o_t \mid q, o_{<t})} A_t,
  $$

  ensures that the new policy does not deviate too much from the old policy, preserving the beneficial properties of the old policy.

- The second term,

  $$
  \text{clip}\!\left(\frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{\theta}^{\text{old}}(o_t \mid q, o_{<t})},\, 1-\epsilon,\, 1+\epsilon\right) A_t,
  $$

  prevents any single token from being overly rewarded by capping the ratio, ensuring that updates remain stable and do not lead to drastic changes.

Together, these terms form the PPO surrogate objective, which trains the network to optimize the cumulative reward for the next token while enforcing constraints that prevent large deviations from the previous policy. We can next move on to describe Group Relative Policy Optimization (GRPO) which was introduced by DeepSeek in their DeepSeekMath paper and how it improves upon PPO.

## Group Relative Policy Optimization

The GRPO surrogate objective is given by:

$$
\begin{aligned}
\mathcal{J}_{\text{GRPO}}(\theta)
&= \mathbb{E}_{\substack{q \sim P(Q), \\ \{o_i\}_{i=1}^{G} \sim \pi_{\theta}^{\text{old}}(\cdot \mid q)}}
\Biggl[
  \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}
  \Biggl\{
    \min \Biggl(
      \frac{\pi_\theta\bigl(o_{i,t} \mid q,\,o_{i,<t}\bigr)}{\pi_{\theta}^{\text{old}}\bigl(o_{i,t} \mid q,\,o_{i,<t}\bigr)}
      \,\hat{A}_{i,t}, \\
&\quad
      \text{clip}\Biggl(
        \frac{\pi_\theta\bigl(o_{i,t} \mid q,\,o_{i,<t}\bigr)}{\pi_{\theta}^{\text{old}}\bigl(o_{i,t} \mid q,\,o_{i,<t}\bigr)},
        1-\epsilon,\,1+\epsilon
      \Biggr)\,\hat{A}_{i,t}
    \Biggr)
    \;-\;
    \beta\, D_{\mathrm{KL}}\bigl[\pi_\theta \,\|\, \pi_{\mathrm{ref}}\bigr]
  \Biggr\}
\Biggr]
\end{aligned}
$$

From the equation above we can see some differences compared to the PPO objective.  There is an additional sum across $$G$$ actions $$o_i$$ drawn from the old policy denoted by 

$$
\{ o_i \}_{i=1}^{G} \sim \pi_{\theta}^{\text{old}}(\cdot \mid q)
$$

which is also expressed as a summation over $$G$$ in the equation. There is also an additional KL divergence term $$D_{KL}$$ which is measures the difference in policy distribution between the new policy and a reference policy, where 

$$
D_{\mathrm{KL}}\!\bigl[\pi_\theta \,\|\, \pi_{\mathrm{ref}}\bigr]
\;=\;
\frac{\pi_{\mathrm{ref}}\bigl(o_{i,t} \mid q,\,o_{i,<t}\bigr)}{\pi_{\theta}\bigl(o_{i,t} \mid q,\,o_{i,<t}\bigr)}
\;-\;
\log \frac{\pi_{\mathrm{ref}}\bigl(o_{i,t} \mid q,\,o_{i,<t}\bigr)}{\pi_{\theta}\bigl(o_{i,t} \mid q,\,o_{i,<t}\bigr)}
\;-\;1.
$$

with $$\beta$$ being the hyperparameter controlling $$D_{\mathrm{KL}}$$. This term prevents the new policy from deviating too far from a baseline policy. Unlike PPO, GRPO no longer uses a separate value network to derive its advantage or loss objective. In standard PPO, the value network $$V(s_t)$$ calculates the expected cumulative reward across a sequence, helping to stabilize training updates. Without a value network, the advantage estimates in GRPO would be less stable. To address this, GRPO computes advantages based on group comparisons among multiple outputs generated for the same prompt.

For example, suppose for a given prompt $$q$$ we sample a group of $$G$$ outputs $$\{o_i\}_{i=1}^{G}$$ using the old policy. Let $$r_i$$ be the reward for output $$o_i$$. Then the group-based advantage for each output (or at each token) can be defined as a normalized score:

$$
\hat{A}_{i,t} = \frac{r_i - \mu}{\sigma}
$$

where $$\mu$$ is the mean reward and $$\sigma$$ is the standard deviation of rewards across the group. This relative measure of advantage helps guide the policy update by favoring outputs that perform better than average, compensating for the absence of a value network. There is also an added benefit where in GRPO the advantage is $$\hat{A}_{i,t}$$ is computed for every $$\text{$i$th}$$ output, whereas for PPO it is computed after iterating throughout the sequence. This would lead to a more noisy signal due to the sparse reward.

### Technical Differences Between PPO and GRPO

1. **Value Network vs. Group-Based Advantage:**
   - **PPO:** Uses a separate value network $$V(s_t)$$ to compute a baseline via TD errors, which helps stabilize advantage estimates.
   - **GRPO:** Eliminates the value network and computes advantages by comparing multiple outputs generated for the same prompt. The relative advantage is normalized using group statistics:
     
     $$
     \hat{A}_{i,t} = \frac{r_i - \mu}{\sigma}.
     $$

2. **Reward Assignment:**
   In many LLM applications, rewards are sparse and are typically assigned only to the final token or overall output. This sparsity complicates token-level value estimation in PPO. GRPO addresses this by leveraging group comparisons to compute a robust, relative advantage.

3. **Explicit KL Divergence Penalty:**
   - **PPO:** Uses a clipping mechanism on the probability ratio to limit policy updates.
   - **GRPO:** In addition to clipping, GRPO includes an explicit KL divergence penalty term:
     
     $$
     -\beta\,D_{\mathrm{KL}}\!\bigl[\pi_\theta \,\|\, \pi_{\mathrm{ref}}\bigr],
     $$
     
     which directly penalizes the new policy for deviating too far from a fixed reference policy. This explicit regularization is crucial when no critic is available to provide a stable baseline.

4. **Stability Considerations:**
   Without a value network, the advantage estimates in GRPO can be more unstable. The group-based normalization (using the mean $$\mu$$ and standard deviation $$\sigma$$) and the KL penalty work together to ensure that the policy updates remain stable despite the absence of a traditional critic.

## Summary

- **PPO** relies on a value network to estimate cumulative rewards and computes the advantage using temporal-difference errors and a clipping mechanism.
- **GRPO** removes the value network by computing advantages based on group comparisons of multiple outputs for the same prompt. The group-based advantage is normalized as:

  $$
  \hat{A}_{i,t} = \frac{r_i - \mu}{\sigma},
  $$

  and an explicit KL divergence penalty
  
  $$
  -\beta\,D_{\mathrm{KL}}\!\bigl[\pi_\theta \,\|\, \pi_{\mathrm{ref}}\bigr]
  $$
  
  is used to ensure that the new policy remains close to a frozen reference policy.
- These modifications in GRPO address challenges in NLP such as sparse rewards and instability in token-level value estimation, resulting in more stable policy updates without requiring a separate critic.

This detailed explanation highlights the technical differences between PPO and GRPO and clarifies how GRPO adapts reinforcement learning to the unique challenges of training large language models.



{::comment}

- **PPO** relies on a value network to estimate cumulative rewards and computes the advantage using temporal-difference errors and a clipping mechanism.
- **GRPO** removes the value network by computing advantages based on group comparisons of multiple outputs for the same prompt. The group-based advantage is normalized as:
  
  $$
  \hat{A}_{i,t} = \frac{r_i - \mu}{\sigma},
  $$
  
  and an explicit KL divergence penalty
  
  $$
  -\beta\,D_{\mathrm{KL}}\!\bigl[\pi_\theta \,\|\, \pi_{\mathrm{ref}}\bigr]
  $$
  
  is used to ensure that the new policy remains close to a frozen reference policy.

These modifications in GRPO address challenges in NLP—such as sparse rewards and instability in token-level value estimation—resulting in more stable policy updates without the need for a separate critic.

This detailed explanation highlights the technical differences between PPO and GRPO and clarifies how GRPO adapts reinforcement learning to the unique challenges of training large language models.
{:/comment}
