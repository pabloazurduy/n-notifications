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

# Simulate hidden variables 
# conversion betas
from scipy.special import expit

beta_c_x = np.array([1/40, -1/200, 1/10, 1/500000, -1/50])
beta_c_d = np.array([1/20])
beta_c_dx = beta_c_x * beta_c_d * (-0.5)
beta_c_0 = np.array([-2.5])

# unsubscribe betas 
beta_u_x = np.array([1/35, -1/210, 1/15, 1/400000, -1/40])
beta_u_d = np.array([1/24])
beta_u_dx = beta_c_x * beta_c_d * (-0.6)
beta_u_0 = np.array([-4])


def conv_prob(d:np.ndarray, x:np.ndarray )-> np.ndarray:
    return expit(beta_c_0+ np.sum((beta_c_dx* np.multiply(x,np.transpose([d]))), axis=1) + 
                 beta_c_d*d +np.sum(beta_c_x*x, axis=1) )

def unsub_prob(d:np.ndarray, x:np.ndarray )-> np.ndarray:
    return expit(beta_u_0+ np.sum((beta_u_dx* np.multiply(x,np.transpose([d]))), axis=1) + 
                 beta_u_d*d +np.sum(beta_u_x*x, axis=1) )

d = np.ones(num_contractors)*3
#features
x = contractors_df[['tenure_on_app',
                    'projects_completed',
                    'average_rating',
                    'total_revenue',
                    'years_of_experience']].values
#print(conv_prob(d,x))
#print(unsub_prob(d,x))


# define expected value function 
inc = 2
cost = 10

def pi(d:np.ndarray, x:np.ndarray )-> np.ndarray:
    return (inc*expit(beta_c_0+ np.sum((beta_c_dx* np.multiply(x,np.transpose([d]))), axis=1) + 
                 beta_c_d*d +np.sum(beta_c_x*x, axis=1) ) -
            cost*expit(beta_u_0+ np.sum((beta_u_dx* np.multiply(x,np.transpose([d]))), axis=1) + 
                 beta_u_d*d +np.sum(beta_u_x*x, axis=1) ))


# plot expected utility with different levels of notifications 

import matplotlib.pyplot as plt
import numpy as np

# Define a plotting function
def plot_pi_for_contractors(x, d_range=np.linspace(0, 50, 100)):
    plt.figure(figsize=(10, 6))
    for idx in [1,2,3,4,10]: # contractors ids
        x_contractor = x[idx].reshape(1, -1) 
        pi_values = [pi(d, x_contractor)[0] for d in d_range]  # Calculate pi for each d

        plt.plot(d_range, pi_values, label=f'Contractor {idx + 1}')

    plt.xlabel('d')
    plt.ylabel('π (Expected Profit)')
    plt.title('Profit Function π for Different Values of d')
    plt.legend()
    plt.grid(True)
    plt.show()
# Plot π for the selected contractors over a range of d values
plot_pi_for_contractors(x)

# optimize 
from scipy.optimize import minimize

initial_guess = np.ones(num_contractors)
bounds = [(0, None)] * num_contractors  # Non-negative d for each contractor

# Optimize using scipy minimize

def objective(d: np.ndarray, x: np.ndarray) -> float:
    return -np.sum(pi(d, x))  # We negate to maximize

result = minimize(objective, x0=initial_guess, args=(x,), bounds=bounds)

# Optimal d for each contractor
optimal_d = result.x
print("The optimal values of d that maximize π are:")
print(optimal_d)
