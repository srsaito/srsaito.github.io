---
title: "Ch 2 Notes and Exercises"
---

Bandit problem is like life, each day. You pull the lever, what are you going to do today? Do what worked well before, or try something new?

## Exercise 2.1
In \\(\epsilon\\)-greedy action selection, for the case of two actions and \\(\epsilon\\) = 0.5, what is the probability that the greedy action is selected?

The probability of selecting the greedy action \\(P(Ag)\\) is:

$$P(Ag) = P(\text{Exploit}) + P(\text{Explore and pick greedy})$$

$$P(Ag) = (1 - \epsilon) + \left(\epsilon \times \frac{1}{k}\right)$$

Plugging in the numbers:

$$P(Ag) = (1 - 0.5) + \left(0.5 \times \frac{1}{2}\right)$$

$$P(Ag) = 0.5 + 0.25 = \mathbf{0.75}$$

\\(P(Ag)\\) always has to be slightly higher than \\(1-\epsilon\\) because the greedy action is sometimes selected as part of exploration. This is moreso when the number of actions, \\(k\\), is very small.


## Exercise 2.2: Bandit Example
Bandit example Consider a k -armed bandit problem with k = 4 actions, denoted 1, 2, 3, and 4. Consider applying to this problem a bandit algorithm using "-greedy action selection, sample-average action-value estimates, and initial estimates of Q1(a) = 0, for all a. Suppose the initial sequence of actions and rewards is A_1 = 1, R_1 = -1, A_2 = 2, R_2 = 1, A_3 = 2, R_3 = -2, A_4 = 2, R_4 = 2, A_5 = 3, R_5 = 0. On some of these time steps the " case may have occurred, causing an action to be selected at random. On which time steps did this definitely occur? On which time steps could this possibly have occurred?
#### Step-by-Step Trace

| **Time (t)** | **Action taken (At​)** | **Reward (Rt​)** | **Estimates Qt​ (before At​)** | **Greedy Action(s)** | **Analysis**                                                                                                            |
| ------------ | ---------------------- | ---------------- | ------------------------------ | -------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **1**        | **1**                  | -1               | \\([0, 0, 0, 0]\\)                 | 1, 2, 3, 4           | **Possibly Explored.** Since all are tied at 0, action 1 is greedy, but it could also have been picked via exploration. |
| **2**        | **2**                  | 1                | \\([-1, 0, 0, 0]\\)                | 2, 3, 4              | **Possibly Explored.** Action 2 is one of the greedy actions (0 > -1).                                                  |
| **3**        | **2**                  | -2               | \\([-1, 1, 0, 0]\\)                | 2                    | **Possibly Explored.** Action 2 is the unique maximum (1). It's likely greedy.                                          |
| **4**        | **2**                  | 2                | \\([-1, -0.5, 0, 0]\\)             | 3, 4                 | **Definitely Explored.** Action 2's value is \\(-0.5\\), which is lower than actions 3 and 4 (both at 0).                   |
| **5**        | **3**                  | 0                | \\([-1, 0.33, 0, 0]\\)             | 2                    | **Definitely Explored.** Action 2 is the unique maximum (\\(0.33\\)), but the agent chose Action 3.                         |


## Exercise 2.3 
In the comparison shown in Figure 2.2 (below), which method will perform best in the long run in terms of cumulative reward and probability of selecting the best action? How much better will it be? Express your answer quantitatively. (note earlier in the text they said "We call this suite of test tasks the _10-armed testbed_.")
![](_resources/Pasted image 20260305092922.png)
**Figure 2.2:** Average performance of "\\(\epsilon\\)-greedy action-value methods on the 10-armed testbed. These data are averages over 2000 runs with different bandit problems. All methods used sample averages as their action-value estimates.

In the long run, \\(\epsilon=0.01\\) will perform best. It will select the optimal action \\(99\%+1\%\ast1/10=99.1\%\\).  For   \\(\epsilon=0.1\\) it will select the optimal action only \\(90\%+10\%\ast1/10=91\%\\). For \\(\epsilon = 0\\) (Greedy) this one is the "wildcard." While it has a 1/10 chance of picking the right action _initially_, it often gets stuck on a sub-optimal action forever. As shown in your uploaded image, it plateaus early at around **35%** in this specific testbed because it occasionally finds the best action by luck and stays there, but usually doesn't.

So we expect the \\(\epsilon=0.01\\) case to be \\(99.1/91-1=8.9\%\\) better in terms of selecting the optimal action.

The cumulative reward takes a little more insight, starting with this sentence from the paragraph before Figure 2.2:
>"The greedy method improved slightly faster than the other methods at the very beginning, but then leveled o↵ at a lower level. It achieved a reward-per-step of only about 1, compared with the best possible of about 1.54 on this testbed." (Sutton と Barto, 2018, p. 29) (pdf) 

The bandit problem though simple is interesting because it is sampling from a set of random variables. To answer the question of what the cumulative reward is for the different \\(\epsilon\\)-greedy cases, we need to know what the reward would be if we just knew the optimum action. Then the \\(\epsilon\\)-greedy cases are just discounted versions of that due to the cost of exploration.

Going back to the problem definition, in Figure 2 below, the **value** of an action is the expected (average) reward you would receive if you chose that action an infinite number of times. For this particular test bed, these values are sampled once from a normal distribution (\\(\mu=0, \sigma=1\\)) at the start of each of the 2000 runs.

The **actual reward** is the single, noisy data point the agent receives at each time step \\(t\\) after taking an action, sampled from a distribution centered at \\(q_\ast(a)\\). Each run has 1000 time steps. 

![](_resources/Pasted image 20260305104010.png)
**Figure 2.1:** An example bandit problem from the 10-armed testbed. The true value \\(q_\ast(a)\\) of each of the ten actions was selected according to a normal distribution with mean zero and unit variance, and then the actual rewards were selected according to a mean \\(q_\ast(a)\\), unit-variance normal distribution, as suggested by these gray distributions.

Then the reward-per-step would be the expected value of the reward for each action selected from a normal distribution with a mean \\(q_\ast(a)\\) and variance \\(1\\), and the _optimum_ reward is the maximum of that. In formal statistics this is called the **expected value of the first order statistic** (the maximum). So in summary, we are looking for the the **expected value of the first order statistic** of 10 samples from a standard normal distribution.

There is no simple closed form solution for this, so we resort to numerical estimation.

```python
import numpy as np

# Simulate the 10-armed testbed setup
n_arms = 10
n_simulations = 1000000

# 1. Sample 10 true values (q*) from N(0,1) for many different "worlds"
q_star_samples = np.random.normal(0, 1, (n_simulations, n_arms))

# 2. Find the maximum q* in each world (the best possible action)
max_q_stars = np.max(q_star_samples, axis=1)

# 3. Average them
expected_max = np.mean(max_q_stars)
print(f"Expected maximum reward: {expected_max:.3f}")
# Result: ~1.538
```

Applying the optimum reward-per-step \\(q_\ast(a^\ast)\approx1.54\\) to the \\(\epsilon\\)-greedy cases 
- **Reward for \\(\epsilon = 0.1\\):** It picks the best action 91% of the time and a random action (average reward 0) 9% of the time.

$$0.91 \times 1.55 + 0.09 \times 0 \approx \mathbf{1.40}$$

- **Reward for \\(\epsilon = 0.01\\):** It picks the best action 99.1% of the time.

$$0.991 \times 1.55 + 0.009 \times 0 \approx \mathbf{1.54}$$

(Note in Sutton&Barto and in the RL field the `*` added to a variable denotes optimality or ground truth value. So \\(q_\ast(a)\\) is the "God's eye view" of the reward distribution for \\(a\\), and \\(q_\ast(a^\ast)\\) is the optimal action, defined as \\(a^\ast = \arg\max_a q_\ast(a)\\).)


>
Researchers often use these "perfect" theoretical ceilings to measure **Regret**. **Regret** is the difference between the reward you _actually_ got and the reward you _could_ have gotten if you had known the truth from the start (\\(q_\ast\\)). By knowing that the ceiling is **1.54**, they can quantify exactly how much "performance" the \\(\epsilon=0.1\\) method is leaving on the table (the "exploration tax" we discussed).


## Exercise 2.4
Spent some time starting with Equation 2.6 incremental update rule

$$\begin{aligned}
Q_{n+1} &= Q_n + \alpha \left[ R_n - Q_n \right] \\
        &= \alpha R_n + (1-\alpha) Q_n \\
        &= \alpha R_n + (1-\alpha)\left[\alpha R_{n-1} + (1-\alpha)Q_{n-1}\right]
\end{aligned}$$

and substituting in \\(\alpha_n\\), and I nearly got it but got sloppy with the second term. Anyway, Gemini had a good way to think about why this equation is simple and makes sense.

$$Q_{n+1} = \left[ \prod_{i=1}^{n} (1 - \alpha_i) \right] Q_1 + \sum_{i=1}^{n} \left[ \alpha_i \prod_{j=i+1}^{n} (1 - \alpha_j) \right] R_i$$

>
The reward \\(R_i\\) is weighted by the step-size it was received with (\\(\alpha_i\\)), and then it is progressively "shrunk" by every _succeeding_ step-size (\\(\alpha_{i+1}, \alpha_{i+2}, \dots, \alpha_n\\)).

## Exercise 2.5
Worked on this by myself, but got directional guidance from Gemini [link](https://gemini.google.com/app/5bfea84c35e904e9). 

Non-stationary bandit problems are those for which the reward probabilities change over time. We need to calculate two cases, \\(\alpha=n\\) and \\(\alpha=0.1\\), the incremental up. date rules are Equation 2.3, \\(Q_{n+1}=Q_n+\frac{1}{n}\left[R_n-Q_n\right]\\) , and Equation 2.5, \\(Q_{n+1}=Q_n+\alpha\left[R_n-Q_n\right]\\), respectively. 

The outputs we want are the average reward per step, as a function of time, and the % of time the optimal action is selected. 

![](_resources/Pasted image 20260310171633.png)

![](_resources/Pasted image 20260310171651.png)

Notes:
1. The average rewards per step for fixed step size plot approaches 1.4. Recall we estimated optimal is 1.54 for \\(\epsilon=0.1\\) in the discussion for [Exercise 2.3](#exercise-23). 

2. "Optimal" Spike at Step 0: At \\(t=0\\), all \\(q_\ast(a) = 0\\) per the problem instructions. Since every action is tied for the maximum, any action the agent picks is technically "optimal." This is why your graph starts at 100% (or 1.0) and then immediately plummeted as the random walk pulled the arms apart.

3. The Sample-Average "Anchor": The Sample-Average method (\\(1/n\\)) is essentially a **long-term memory**.

	- **Early on:** When \\(n\\) is small (e.g., \\(n=2\\)), \\(1/n\\) is large (0.5). It adapts quickly because it doesn't have much history to weigh down its estimates.
	    
	- **Later on:** As \\(n\\) grows to 1,000 or 5,000, \\(1/n\\) becomes tiny. The agent becomes "stubborn." It remembers what an arm was doing 4,000 steps ago almost as clearly as what it did 10 steps ago. In a nonstationary world, those 4,000-step-old rewards are now "garbage" data, but Sample-Average can't let go of them.
    
4. The Fixed Step-Size "Recency Bias": The Constant Step-Size method (\\(\alpha=0.1\\)) is a **short-term memory**.

	- It purposefully forgets the past. Because the weight of older rewards decays exponentially, it only "cares" about what has happened recently.
	    
	- This is why it eventually crushes the Sample-Average method. It stays "limber" and can track the \\(q_\ast\\) values as they drift, while the Sample-Average agent gets stuck trying to average out a history that no longer exists.

```python
import numpy as np

def run_simulation():
        
    num_runs = 2000
    num_steps = 10000
    num_bandits = 10
    epsilon = 0.1
    alpha_f = 0.1

    # Initialize matrices to store average rewards and optimal action counts
    average_rewards_f = np.zeros((num_runs, num_steps))
    average_rewards_sa = np.zeros((num_runs, num_steps))
    optimal_action_counts_f = np.zeros((num_runs, num_steps))
    optimal_action_counts_sa = np.zeros((num_runs, num_steps))

    for run in range(num_runs):

        # For each run, initialize state-action value estimates and counts
        Q_f = np.zeros(num_bandits)      # Action-value estimates for fixed step-size
        Q_sa = np.zeros(num_bandits)     # Action-value estimates for sample-average method
        N_sa : int = np.zeros(num_bandits)     # step count for each bandit
        q_star = np.zeros(num_bandits)   # True action values, drawn from a normal distribution

        for step in range(num_steps):
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:  # Explore
                action_sa = action_f = np.random.randint(num_bandits)
            else:  # Exploit
                action_sa = np.random.choice(max_actions(Q_sa))
                action_f = np.random.choice(max_actions(Q_f))

            # Update average rewards
            N_sa[action_sa] += 1
            Q_sa[action_sa] += (q_star[action_sa] - Q_sa[action_sa]) / N_sa[action_sa]
            Q_f[action_f] += alpha_f * (q_star[action_f] - Q_f[action_f])

            # Store optimal action counts
            if action_sa in max_actions(q_star):
                optimal_action_counts_sa[run, step] = 1
            if action_f in max_actions(q_star):
                optimal_action_counts_f[run, step] = 1

            # Store average rewards
            average_rewards_f[run, step] = Q_f[action_f]
            average_rewards_sa[run, step] = Q_sa[action_sa]

            # Update environment state for next step
            q_star += np.random.normal(0, 0.01, num_bandits)  # Random walk for true action values

            print(f"Run: {run+1:>4}/{num_runs:>4}, Step: {step+1:>5}/{num_steps:>5}", end='\r')

    return average_rewards_f, average_rewards_sa, optimal_action_counts_f, optimal_action_counts_sa

def max_actions(q_values):
    """
    Returns the indices of the actions with the maximum value. e.g., if q_values = [1, 2, 5, 4, 5], 
    it should return [2, 4] since those are the indices of the maximum value (5).
    
    Args:
        q_values (np.array): An array of action-value estimates.
        
    Returns:
        np.array: An array of indices of the actions with the maximum value.
    """
    max_value = np.max(q_values)
    max_indices = np.where(q_values == max_value)[0]
    return max_indices

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    average_rewards_f, average_rewards_sa, optimal_action_counts_f, optimal_action_counts_sa = run_simulation()

    # Plot average rewards
    plt.figure(figsize=(12, 6))
    plt.plot(average_rewards_f.mean(axis=0), label='Fixed Step-Size')
    plt.plot(average_rewards_sa.mean(axis=0), label='Sample-Average')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Rewards over Time')
    plt.legend()

    # Plot optimal action counts
    plt.figure(figsize=(12, 6))
    plt.plot(optimal_action_counts_f.mean(axis=0), label='Fixed Step-Size')
    plt.plot(optimal_action_counts_sa.mean(axis=0), label='Sample-Average')
    plt.xlabel('Steps')
    plt.ylabel('Optimal Action Percentage')
    plt.title('Optimal Action Percentage over Time')
    plt.legend()

    plt.show()
```


## Exercise 2.6: Mysterious Spikes
https://gemini.google.com/app/363ee02a262b913f

![](_resources/Pasted image 20260312100650.png)
**Figure 2.3:** The effect of optimistic initial action-value estimates on the 10-armed testbed. Both methods used a constant step-size parameter, \\(\alpha= 0.1\\).

Optimistic initial action-value forces the agent to sample *all* actions, since \\(Q_1=5\\) for all actions, each will be sampled then averaged down with the actual reward until the eleventh action, after all 10 actions have been samples, the averaged values of \\(Q_2\\) would be more likely to have the optimum action have the higher action-value estimate, which is the first spike. But once that next actual reward is averaged in, the agent will again start sampling the other actions, resulting in the drop after the spike, until the twenty-first step where there will be another spike.

## Exercise 2.7: Unbiased Constant-Step-Size Trick

The first sentence in the problem statement was not quite clear to me:

>In most of this chapter we have used sample averages to estimate action values because sample averages do not produce the initial bias that constant step sizes do (see the analysis leading to (2.6)).

But this is indeed true. Looking at equation 2.3, \\(Q_{n+1}=Q_n+\frac{1}{n}[R_n-Q_n]\\) and working through \\(n=2, 3\\):

$$\begin{aligned}Q_2&=Q_1+\frac{1}{1}[R_1-Q_1]=R_1\\Q_3&=Q_2+\frac{1}{2}[R_2-Q_2]=\frac{R_1+R_2}{2}\end{aligned}\tag{2.3a}$$

So *sample average method* is a pure average of the rewards because the bias from the initial estimate \\(Q_1\\) drops out on the first step. The *constant step-size method*, on the other hand, always retains a residual \\((1-\alpha)^nQ_1\\) from the \\(Q_1\\) initial estimate, which can be seen from equation 2.6, as the authors note:

$$Q_{n+1}=(1-\alpha)^nQ_1+\sum_{i=1}^n\alpha(1-\alpha)^{n-i}R_i\tag{2.6}$$

>
The *sample average method* does not produce the initial bias that constant step size methods do.


https://gemini.google.com/app/87a14f6374e194d6
Wording of problem tricky, so needed some help from Gemini explaining the problem statement:

<!-- REVIEW: unclosed $$ mid-line -->
>...\\(\bar{o}\_n\\) is a trace of \[the number] *one* that starts at 0:$$\begin{equation}
\bar{o}_n = \bar{o}_{n-1} + \alpha \left(1 - \bar{o}_{n-1}\right), \quad \text{for } n > 0,\ \text{with } \bar{o}_0 = 0
\tag{2.9}
\end{equation}$$

In this specific sentence, "one" does **not** refer to the action. Instead, "one" refers to the **number 1**, i.e., phrase "trace of one" describes the recurrence relation in Equation 2.9. Essentially, \\(\bar{o}\_n\\) is tracking (or "tracing") the value of a constant signal—the number 1—using the same exponential moving average logic you use for rewards, i.e., "What would my estimate be if every single reward I ever received was exactly 1.0?"

In reinforcement learning, a **trace** is a running statistic that "remembers" past events but gives more weight to recent ones.

- It starts at a initial value (here, \\(\bar{o}\_0 = 0\\)).
    
- It is updated at each step by some fraction of the difference between the "current target" (which is **1** here) and the current trace value.
    
Gemini notes that later in the book we’ll encounter "eligibility traces" (the \\(\lambda\\) in \\(TD(\lambda)\\)), which track which states or actions were recently visited. Its importance is that it is methematically how RL handles the *credit assignment problem*. Here, the "trace of one" is a specific mathematical trick used to calculate the cumulative weight applied to all past rewards so far.

Also, the authors are using these exercises to get the reader comfortable with recursive definitions. The form \\(NewValue = OldValue + \alpha(Target - OldValue)\\) will appear over and over, so Exercise 2.7 is an exercise to help us understand more complex proofs later in the book. 

I found this problem required two insights:
1. reducing \\(\bar{o}\_n\\) to a closed series form so then applying it to the analysis leading equation 2.6 is tractable, 
2. focus on demonstrating that the coefficient of the \\(Q_1\\) term goes to zero, and
3. finding the closed form sum of the weights in the summation in equation 2.6.

\\(\bar{o}\_1=\bar{o}\_0+\alpha(1-\bar{o}\_0)=0+\alpha-0=\alpha\\)

\\(\begin{aligned}\bar{o}\_2&=\bar{o}\_1+\alpha(1-\bar{o}\_1)\\&=\alpha+\alpha(1-\alpha)\\&=2\alpha-\alpha^2\\&=1-\alpha^2+2\alpha-1\\&=1-(\alpha^2-2\alpha+1)\\&=1-(1-\alpha)^2\end{aligned}\\)

\\(\begin{aligned}\bar{o}\_3&=\bar{o}\_2+\alpha(1-\bar{o}\_2)\\&=1-(1-\alpha)^2+\alpha(1-(1-(1-\alpha)^2))\\&=1-(1-\alpha)^2+\alpha(1-\alpha)^2\\&=1-(1-\alpha)(1-\alpha)^2\\&=1-(1-\alpha)^3\end{aligned}\\)

\\(\bar{o}\_n=1-(1-\alpha)^n\\)

Substituting step size  \\(\beta_n=\frac{\alpha}{\bar{o}\_n}=\frac{\alpha}{1-(1-\alpha)^n}\\) for \\(\alpha\\) in the update equation, \\(Q_{n+1} = Q_n + \beta_n[R_n - Q_n]\\). Just as for the sample average case, when \\(n=1\\), \\(\beta_1=1\\) and \\(Q_1\\) drops out, \\(Q_2=Q_1+R_1-Q_1=R_1\\).

Note also that as \\(n \to \infty\\), \\(\beta_n \to \frac{\alpha}{1 - 0} = \alpha\\) since \\((1-\alpha)<1\\), so we can think of the \\(\beta\\) term as a dynamic correction factor.

>
>For eliminating the initial \\(Q_1\\) bias, the most important characteristic of the \\(\beta^n\\) weights is that \\(\beta_1=1\\) or equivalently, \\(\bar{o}\_1=\alpha\\).

The problem asks "Carry out an analysis like that in (2.6) to show that \\(Q_n\\) is an exponential recency-weighted average *without initial bias*." We showed that initial bias is eliminated, but we still have to show it is an exponential recency-weighted average. 

Continuing the update equation series above,

$$\begin{align}Q_3 &= (1 - \beta_2)Q_2 + \beta_2 R_2\\&=(1 - \beta_2)R_1 + \beta_2 R_2\end{align}$$

which begins to look like a weighted average. 

To prove it systematically, we need to look at all the weights in equation 2.6. A key insight here is that a geometric series like that of the weights for the elements in the second half of the equation has a closed form solution \\(\sum_{k=0}^{n-1} ar^k = a\frac{1-r^n}{1-r}\\), where \\(a=\alpha\\) and \\(r=(1-\alpha)\\). Substituting:


$$\sum_{i=1}^{n} \alpha(1-\alpha)^{n-i}= \alpha \frac{1 - (1-\alpha)^n}{1 - (1-\alpha)} = \alpha \frac{1 - (1-\alpha)^n}{\alpha} = 1 - (1-\alpha)^n\tag{2.9a}$$

Since Equation 2.6's first term weight is \\((1-\alpha)^n\\), the sum of that with the weights from the second half of the equation in equation 2.9a above is therefore \\(1\\).

For the rest of the discussion, keep in mind that the weights must always sum to \\(1\\) because the update equation \\(Q_{n+1} = Q_n + \beta_n[R_n - Q_n]=(1 - \beta_n)Q_n + \beta_n R_n\\) enforces that constraint at each step.   

How was \\(\beta_n=\alpha/\bar{o}\_n\\) selected? Notice that the problem with the value of equation 2.6's weight for \\(Q_1\\) is that rather than \\(1\\), which would cause \\(Q_1\\) to drop out of the series as we showed above, it is instead \\((1-\alpha)^n\\) which causes a residual to hang around no matter how many steps there are. 

Note that the weights of the second term of equation 2.6 don't sum to \\(1\\), which is why there is a nonzero weight for \\(Q_1\\). But if we forced them to sum to \\(1\\) then the weight for \\(Q_1\\) would be forced to go to zero to maintain the constraint that total weights sum to \\(1\\) from the update equation. The normalization factor to force any sum to be \\(1\\) is to divide by that sum (e.g., to normalize a series that sums to 0.8 to 1, you need to divide it by 0.8), we normalize by equation 2.9a above, that is, \\(1/\bar{o}\_n\\) is the normalization to ensure the weights sum to one.

This techniques used in this problem will be useful for future RL stuff, in particular TD Learning, which uses the same geometric series for propagating TD Error back in time, and Q-Learning, where the fact that \\((1-\alpha)^n\\) shrinks toward zero is the mechanism ensuring old, incorrect Q-values eventually "contract" to zero, just like the \\(Q_1\\) bias did.

## Exercise 2.8: UCB Spikes

Double check: https://gemini.google.com/app/74f0b85003dc7844

At \\(t=1\\), \\(N_{1-10}(a)=0\\) by definition so all ten actions need to each be selected before all actions have \\(N_{11}(a)=1\\). For \\(t=11\\), since the UCB term is the same for all ten actions, and since per the simple bandit algorithm \\(Q_t(a)=R_t\\) that was discovered, the action with the highest \\(Q_t(a)\\) will be selected, which is the spike we see in the graph. \[n.b., while selecting the maximum value out of 10 independent random samples should result in a value higher than the mean seems intuitively obvious (or at least plausible), Gemini notes "the expected value of the maximum of 10 samples is significantly higher than the expected value of a single random sample, which is why the average reward jumps up so sharply" and created an animated graphic illustrating this conclusion that comes from a well-established branch of probability theory known as **Order Statistics**.]

This action's UCB will immediately drop because its denominator becomes \\(\sqrt{N_{11}(a)}=\sqrt{2}\\). Since its UCB is reduced, it is much less likely to be the highest action, so the next highest action will be selected for \\(t=12\\). This will happen for the subsequent actions, until enough actions have the halved UCB such that the action with the highest \\(Q_t(a)\\) is selected. This explains why after \\(t=11\\) the reward dips temporarily. If \\(c=1\\), the boost from UCB which encourages costly exploration is reduced, so the spike is less prominent - i.e., the reward line doesn't dip as much.

## Exercise 2.9

Starting with the softmax equation for two actions:

$$Pr\{A_t = a_1\} = \frac{e^{H_t(a_1)}}{e^{H_t(a_1)} + e^{H_t(a_2)}}$$

And dividing the numerator and the denominator by \\(e^{H_t(a_1)}\\) , we get:

$$Pr\{A_t = a_1\} = \frac{1}{1 + e^{H_t(a_2) - H_t(a_1)}}$$

Substituting \\(x=H_t(a_1)\\) and assuming \\(H_t(a_2)=0\\) because we are interested in the distribution of the first action, the equation becomes the sigmoid or logistic distribution function, which is the cumulative distribution function for the standard logistic distribution, which is, if you have a random variable \\(X\\) that follows a standard logistic distribution, the probability that \\(X\\) is less than or equal to a specific value \\(x\\).

$$Pr\{A_t = a_1\} = \frac{1}{1 + e^{-x}}$$

## Section 2.8 Gradient Bandit Algorithms

The "Bandit Gradient Algorithm as Stochastic Gradient Descent" proof had a few tricks that weren't completely obvious to me, so documenting here.

>We can include a baseline here without changing the equality because the gradient sums to zero over all the actions, \\(\sum_x \frac{\partial \pi_t(x)}{\partial H_t(a)} = 0\\). As \\(H_t(a)\\) is changed, some actions’ probabilities go up and some go down, but ==the sum of the changes must be zero== because the sum of the probabilities is always one.

Why must the sum of changes be zero? If the sum of the changes *to the probabilities, \\(\pi_t(x)\\)*, was anything other than zero, they wouldn't sum to \\(1\\) so they would no longer be probabilities.

>...and substituted \\(R_t\\) for \\(q_\ast(A_t)\\), which is permitted because \\(E[R_t\mid A_t] = q_\ast(A_t)\\).

How can we do this?

$$\frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)} = \mathbb{E}\left[(q_*(A_t) - B_t)\frac{\partial \pi_t(A_t)}{\partial H_t(a)}/\pi_t(A_t)\right]$$

We already know that the expected value (sum) of the second term, with \\(B_t\\) as its coefficient, sum to zero, so we focus on the first term, \\(q_\ast(A_t)\frac{\partial \pi_t(A_t)}{\partial H_t(a)}/\pi_t(A_t)\\), and group everything that depends on the chosen action \\(A_t\\) into a single function, \\(f(A_t) = \frac{\partial \pi_t(A_t)}{\partial H_t(a)} / \pi_t(A_t)\\), to simplify:

$$\frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)} = \mathbb{E}\left[q_*(A_t) f(A_t)\right]$$

By definition \\(q_\ast(a)=\mathbb{E}\left[R_t\mid A_t=a\right]\\), so

$$= \mathbb{E}\left[\mathbb{E}\left[R_t|A_t=a\right] f(A_t)\right]$$

For the inner expectation: \\(\mathbb{E}\left[R_t\mid A_t=a\right]\\), the vertical bar \\(\mid \\) means "assume \\(A_t\\) is fixed and known." In the hypothetical universe of that inner expectation, \\(A_t\\) is no longer random - it has already happened. And because \\(f(A_t)\\) is just a function of \\(A_t\\), **\\(f(A_t)\\) is also just a constant number in that universe**, so since expectation is a linear operator  **\\(f(A_t)\\) go inside.

$$= \mathbb{E} \left[ \mathbb{E}[R_t f(A_t) | A_t] \right]$$

---

### Aside: Law of Total Expectations

The Law of Total Expectations states that \\(\mathbb{E}[X] = \mathbb{E}\_Y[\mathbb{E}[X\mid Y]]\\), where the subscript \\(Y\\) on the outer expectation means "expected value calculated across all values of \\(Y\\)" (the variable being "averaged out" or "marginalized"). The inside expectation implicitly is over \\(X\\), but with the condition \\(Y\\). The Law of Total Expectation is often called the **Tower Rule** (or the Tower Property) because of how nested levels of information "collapse" down to their most basic foundation.

Imagine you have two sets of information, \\(\mathcal{G}\\) and \\(\mathcal{F}\\), where \\(\mathcal{G}\\) is a smaller subset of information contained entirely within \\(\mathcal{F}\\) (written as \\(\mathcal{G} \subset \mathcal{F}\\)).

- **\\(\mathcal{F}\\)** represents knowing a lot of details (e.g., the exact temperature, humidity, and wind speed today).
    
- **\\(\mathcal{G}\\)** represents knowing fewer details (e.g., just knowing that it's raining today).
    
The Tower Property states that if you take an expectation based on the detailed information \\(\mathcal{F}\\), and then take the expectation of _that_ result based on the vague information \\(\mathcal{G}\\), the detailed information gets entirely washed out. You are left only with the expectation based on the vague information:

$$E[E[X | \mathcal{F}] | \mathcal{G}] = E[X | \mathcal{G}]$$

A more concrete way to think of this is taking averages of averages in a corporate hierarchy. 

You want to know the average sales per employee in a company (\\(E[X]\\)). 
- **The First Level:** You ask each department manager to calculate the average sales for their specific team (\\(E[X\mid Y]\\), where \\(Y\\) is the department). 
- **The Tower:** As the CEO, you then take the average of all the managers' averages (\\(E[E[X\mid Y]]\\)).

The Tower Rule simply proves that taking the average of the nested averages yields the exact same number as if you had just calculated the overall company average from the start. You built a "tower" of management to get the numbers, but it all collapses down to the same foundational expected value.

>
>Note that in the example above we use expected value for the formulas. That is, we are taking the expected value of expected values, not averages of averages. Average of averages don't work because they don't take into account the weights (probabilities) of the respective random variables. See [Simpson's Paradox](https://en.wikipedia.org/wiki/Simpson%27s_paradox).

---

We now have an expression of the form \\(\mathbb{E}[ \mathbb{E}[X \mid  Y] ]\\), where our \\(X\\) is the entire term \\(R_t f(A_t)\\) and our \\(Y\\) is \\(A_t\\). The Law of Total Expectation states that the expected value of a conditional expected value is simply the unconditional expected value:

$$\mathbb{E} \left[ q_*(A_t) f(A_t) \right] = \mathbb{E} \left[ R_t f(A_t) \right]$$

If we wanted to perform **exact** gradient ascent, we would need to know \\(q_\ast(A_t)\\) to compute the exact gradient. But the agent doesn't know the true action values—that's the whole reason it is learning!

Instead, the agent uses \\(R_t\\). Because the expected value of the update using \\(R_t\\) is exactly the same as the update using \\(q_\ast(A_t)\\), \\(R_t\\) serves as an **unbiased sample** of the gradient. Taking small steps based on unbiased samples of a gradient is exactly what **stochastic gradient ascent** is.
