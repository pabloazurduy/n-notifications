# Optimizing number of push notifications
A small example on optimizing the number of optimization using a [CATE estimator](https://matheusfacure.github.io/python-causality-handbook/18-Heterogeneous-Treatment-Effects-and-Personalization.html).  

### Problem statement
We would like to estimate the number of notifications ($D = d$) that you would like to send to a user in order to maximize the expected conversion. For example, you can think on notifications of a CRM campaign to promote a membership program / premium subscription.

At the same time, overusing the notifications can make users block the notifications of the app, resulting in an elevated cost for future CRM campaigns.

### Model proposal (ATE) non-heterogenity 

In order to solve this problem, we want to estimate  $\hat{\tau}(D=d)$ that represents the effect of the treatment $D=d$ on the population. For the sake of the example let's think on CRM messages to entice our user to purchase a premium subscription. $d$ will be the number of notifications that I will send to each user in a period of a month (for example). 

Thinking on a basic Inference model we could think on running multiples experiments with different volumes of notifications $d \in [1,2,5,10]$ and then test the group that maximizes the expected value. 

$$\hat{ATE}(D=d) \quad d \in [1,2,5,10]$$

In our example we can evaluate the expected profit $\pi$ as the sum of total subscriptions purchased ($\pi_s*s$) minus the number of unsubscriptions ($u$) with their expected future value discounted

$$\hat{\tau}(d) = \pi(d) = \pi_s*s - \text{cost}_{unsubscribe}*u$$


We could run a simple experiment with all treatments $d \in [1,2,5,10]$ and then evaluate what is the treatment with higher expected value $\pi(d)$. However, what if I could choose the amount of notifications per user $i$ that maximizes the user expected value ?. 

Let's introduce the [CATE estimator](https://matheusfacure.github.io/python-causality-handbook/18-Heterogeneous-Treatment-Effects-and-Personalization.html)

### CATE estimator and the heterogeneity effect

Let's assume that I have a model that can infer the effect of the treatment $D=d$ on the outcome $Y^i$ in each one of our clients ($X_i$). For example, the probability of a user to purchase the premium subscription $Y_{conversion}$. Our model will look like this:

$$P(Y_{conversion}^{i}) = f(D=d|X_i)$$

Where $X_i$ is a vector of features that describe the user $i$. Now, we are interested on estimating the "sensitivity" of the user $i$ to the "treatment" (notifications) $d$. Similarly to the case of the ATE, we will call this effect estimator $\hat{\tau_i}(d)$. This model is known as CATE estimator (Conditional Average Treatment Effect). Its called "Conditional" estimator because we are "conditioning" the model to the user characteristics $X_i$. 

$${\tau_i}(d) = E[Y_{conversion}=1|d]- E[Y_{conversion}=0|d]$$

The most natural way to think on solving this consist on fitting a ML algorithm to predict $Y(d)$ and then estimate the effect of different treatments. However, fitting a ML model out of the box can bring [undesired consequences](https://matheusfacure.github.io/python-causality-handbook/When-Prediction-Fails.html). For the sake of simplicity on this example we will fit the easiest Inference model, the unique, the legend, the linear regression (logit in this probabilistic case ).  

$$Y_{conversion}(d) = f(D=d|X_i) =  logit(\hat{\beta}_0+ \hat{\beta}_1d + \hat{\beta}_2x_i + \hat{\beta}_3 dx_i)$$

Therefore, our sensitivity estimator will be given by:

$$\hat{\tau_i}(d) = \frac{\delta Y_{conversion}(d)}{\delta d} = \frac{\delta (sigmoid(\hat{\beta}_0^{c}+ \hat{\beta}_1^{c}d + \hat{\beta}_2^{c}x_i + \hat{\beta}_3^{c} dx_i))}{\delta d}$$

Where the sigmoid function is defined as:

$$Sigmoid(x) = \frac{1}{1+\exp(-x)}$$

Now we have a way to use past data to estimate the sensitivity to buy the premium subscription given certain number of notifications. 

### Choosing the optimum number of notifications

how can we use the CATE estimator to choose the optimum number of notifications ?. 
First, let's fit two CATE estimators to estimate the probability of conversion $\mathbb{P}(Y_{conversion}=1)$ and the probability of unsubscribe $P(Y_{unsubscribe}=1)$. Once we fit both models we can estimate the CATE for both outcomes:


$$\hat{P}(Y^{conversion}|d) = sigmoid(\hat{\beta}_0^{c}+ \hat{\beta}_1^{c}d + \hat{\beta}_2^{c}x_i + \hat{\beta}_3^{c} dx_i)$$
$$\hat{P}(Y^{unsubscribe}|d) = sigmoid(\hat{\beta}_0^{u}+ \hat{\beta}_1^{u}d + \hat{\beta}_2^{u}x_i + \hat{\beta}_3^{u} dx_i)$$

Thinking on the expected outcome of sending $d$ notifications to the user $i$ we have the following formula: 

$$E[\pi_i(d)|d] = \pi_s *\hat{P}(Y^{conversion}=1|d, X_i) - \text{cost}_{unsubscribe}  *\hat{P}(Y^{unsubscribe}=1|d, X_i)$$

replacing with the beta values we have: 

$$E[\pi_i(d)|d] = \pi_s *sigm(\hat{\beta}_0^{c}+ \hat{\beta}_1^{c}d + \hat{\beta}_2^{c}x_i + \hat{\beta}_3^{c} dx_i) - \text{cost}_{unsubscribe} *sigm(\hat{\beta}_0^{u}+ \hat{\beta}_1^{u}d + \hat{\beta}_2^{u}x_i + \hat{\beta}_3^{u} dx_i)$$

we can then optimize this function to gather the optimum N for each client:

$$\max_{d}{E[\pi_i(d)|d]}$$

## Simulated Example

In this example we will simulate a dataset with 1000 users and 4 features. We will simulate the probability of conversion and un-subscription for each user and then we will estimate the CATE for each user. 

```python
import pandas as pd
import numpy as np

np.random.seed(42)
num_contractors = 1000

# Generate random data
data = {
    'contractor_id': np.arange(1, num_contractors + 1),
    'tenure_on_app': np.random.randint(1, 40, size=num_contractors),  # Tenure in months
    'projects_completed': np.random.randint(0, 101, size=num_contractors),  # Number of projects completed
    'average_rating': np.random.uniform(1, 5, size=num_contractors),  # Average rating out of 5
    'total_revenue': np.random.uniform(10000, 100000, size=num_contractors),  # Total revenue in dollars
    'years_of_experience': np.random.randint(1, 21, size=num_contractors)  # Years of experience in construction
}
contractors_df = pd.DataFrame(data)
print(contractors_df.head())
contractors_df.to_csv('dataset.csv', index=False)
```

The dataset will look like this:
| contractor_id | tenure_on_app | projects_completed | average_rating | total_revenue | years_of_experience |
|---------------|----------------|--------------------|----------------|---------------|---------------------|
| 1             | 12             | 45                 | 3.5            | 50000         | 10                  |
| 2             | 24             | 30                 | 4.2            | 75000         | 15                  |
| 3             | 36             | 60                 | 4.8            | 90000         | 20                  |
| 4             | 18             | 20                 | 2.9            | 30000         | 5                   |
| 5             | 6              | 10                 | 3.0            | 20000         | 2                   |



we will now simulate the probability of conversion and un-subscription for each user. 

```python
# Simulate hidden variables 
# conversion betas
beta_c_x = np.array([1/40, -1/200, 1/10, 1/500000, -1/50])
beta_c_d = np.array([1/20])
beta_c_dx = beta_c_x * beta_c_d * (-0.5)
beta_c_0 = np.array([-2.5])

# unsubscribe betas 
beta_u_x = np.array([1/35, -1/210, 1/15, 1/400000, -1/40])
beta_u_d = np.array([1/24])
beta_u_dx = beta_c_x * beta_c_d * (-0.6)
beta_u_0 = np.array([-4])
```

Simulating the expected value of conversion and un-subscription for each user ($\pi_i$) will give us an understanding of each user maximum expected value. 

![alt text](profit_curves_plot.png)


Finally, we can optimize the expected value of each user to find the optimum number of notifications to send to each user. 

```python 
# optimize 
from scipy.optimize import minimize

p0 = np.ones(num_contractors)
bounds = [(0, None)] * num_contractors  # Non-negative d for each contractor
def objective(d: np.ndarray, x: np.ndarray) -> float:
    return -np.sum(pi(d, x))  # We negate to maximize

result = minimize(objective, x0=p0, args=(x,), bounds=bounds)
# Optimal d for each contractor
optimal_d = result.x
print("The optimal values of d that maximize π are:")
print(optimal_d)
```

which outputs the optimum notification values:

    The optimal values of d that maximize π are:
    [1.14539061e+02 
    0.00000000e+00 
    4.01680530e+01 
    2.27159468e+01
    4.83272758e+01 
    5.38936299e+01 
    3.95627651e+01 
    9.88400349e+01
    ...
    ]