import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import chi2_contingency

# ---------------------------------------------------------
# Loading the dataset
# ---------------------------------------------------------
# Reading real (original) microdata file that includes key attributes + target
source_file = "/Users/idilalp/Desktop/canada2011_tcap_real_citizen.csv"
df = pd.read_csv(source_file, encoding='latin1')

# ---------------------------------------------------------
# Defining key variables and target
# ---------------------------------------------------------
# Setting the disclosure target variable
target = 'citizen'

# Automatically treating all other variables as candidate key variables
keys = [col for col in df.columns if col != target]

# ---------------------------------------------------------
# Binning continuous variables if needed (e.g., 'age')
# ---------------------------------------------------------
# If age is present, converting it into 5 equal-width bins using sklearn's KBinsDiscretizer
# This ensures that categorical measures like Cramér’s V are meaningful
if 'age' in keys:
    est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    df['age_binned'] = est.fit_transform(df[['age']])
    # Replacing 'age' in the key list with the binned version
    keys = [k if k != 'age' else 'age_binned' for k in keys]

# ---------------------------------------------------------
# Converting all keys and target column to string type
# ---------------------------------------------------------
# This is necessary for crosstabs to work properly and ensure compatibility with chi-square
for col in keys + [target]:
    df[col] = df[col].astype(str)

# ---------------------------------------------------------
# Function to compute  Cramér’s V between two variables
# ---------------------------------------------------------
def cramers_v(x, y):
    """
    Calculates Cramér’s V statistic between two categorical variables.
    This version adjusts for sample size and degrees of freedom 
    """
    confusion_matrix = pd.crosstab(x, y)  # Creating contingency table
    chi2 = chi2_contingency(confusion_matrix)[0]  # Extracting chi-square statistic
    n = confusion_matrix.sum().sum()  # Total number of observations
    phi2 = chi2 / n  # Raw effect size
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    # Final Cramér's V score 
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# ---------------------------------------------------------
# Computing pairwise Cramér’s V across key variables (optional and exploratory)
# ---------------------------------------------------------
pairwise_rows = []
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        v = cramers_v(df[keys[i]], df[keys[j]])
        pairwise_rows.append((keys[i], keys[j], v))

# Creating dataframe of key–key associations
cramers_df = pd.DataFrame(pairwise_rows, columns=['Var1', 'Var2', 'CramersV'])
cramers_df = cramers_df.sort_values(by='CramersV', ascending=False).reset_index(drop=True)

print("\n=== Pairwise Cramér's V among keys ===")
print(cramers_df.head(20))  # Print top associations between key variables

# ---------------------------------------------------------
# Computing Cramér’s V between each key and the target
# ---------------------------------------------------------
target_rows = []
for k in keys:
    v = cramers_v(df[k], df[target])
    target_rows.append((k, target, v))

# Creating dataframe showing predictive power of each key on the target
target_cramers = pd.DataFrame(target_rows, columns=['Var', 'Target', 'CramersV'])
target_cramers = target_cramers.sort_values(by='CramersV', ascending=False).reset_index(drop=True)

print("\n=== Cramér's V between each key and target ===")
print(target_cramers)

import pandas as pd

# Manually entered Cramér’s V scores between each key and the target (or from previous calculation)
cramers_df = pd.DataFrame({
    'Var': ['minority', 'birthplace', 'age_binned', 'empstat', 'marstat', 'sex'],
    'CramersV': [0.448024, 0.734582, 0.146277, 0.049825, 0.132051, 0.011891]
})

# ---------------------------------------------------------
# Normalising the Cramér’s V scores to sum to 1 to use as weights
# ---------------------------------------------------------
# Converting Cramér's V scores into proportions (weights) by dividing by their total sum
# Then rounding to 3 decimal places for readability
cramers_df['Weight'] = cramers_df['CramersV'] / cramers_df['CramersV'].sum()
cramers_df['Weight'] = cramers_df['Weight'].round(3)

# Displaying the weight table
print(cramers_df)

# ---------------------------------------------------------
# Creating a key:weight dictionary for TCAP weight input
# ---------------------------------------------------------
key_weights = dict(zip(cramers_df['Var'], cramers_df['Weight']))
print("\nkey_weights =", key_weights)





