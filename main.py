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
    'years_of_experience': np.random.randint(1, 31, size=num_contractors)  # Years of experience in construction
}
contractors_df = pd.DataFrame(data)
# Simulate hidden state variables (mean number of notifications to accept, mean number of notifications to unsuscribe)
contractors_df['tau_accept']      = np.random.uniform(0.05, 0.20, size=num_contractors)
contractors_df['tau_unsubscribe'] = np.random.uniform(0.01, 0.02, size=num_contractors)



print(contractors_df.head())
contractors_df.to_csv('dataset.csv', index=False)
