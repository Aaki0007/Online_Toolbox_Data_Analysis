import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import shapiro
import numpy as np
from scipy.stats import levene
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu

# Loading the dataset
df = pd.read_csv(r'C:\Users\ABHISHEK\Desktop\Tools_Methods_Data_Analysis\Final Project\API_SL.TLF.CACT.FE.ZS_DS2_en_csv_v2_19415.csv', skiprows=4)

# Filter for India and Germany
countries = ['India', 'Germany']
df_filtered = df[df['Country Name'].isin(countries)]

# Select relevant columns
years = [str(year) for year in range(1960, 2023)]
df_years = df_filtered[['Country Name'] + years]

# Transpose the data
df_transposed = df_years.set_index('Country Name').T
df_transposed.index.name = 'Year'
df_transposed.reset_index(inplace=True)

# Rename columns
df_transposed.rename(columns={'India': 'India_FLFPR', 'Germany': 'Germany_FLFPR'}, inplace=True)

# Handle missing values
df_cleaned = df_transposed.dropna()

#Convert data types
df_cleaned['India_FLFPR'] = pd.to_numeric(df_cleaned['India_FLFPR'])
df_cleaned['Germany_FLFPR'] = pd.to_numeric(df_cleaned['Germany_FLFPR'])

# print(df_cleaned.head)

# Descriptive statistics
print("India Summary:")
print(df_cleaned['India_FLFPR'].describe())
print("Skewness:", skew(df_cleaned['India_FLFPR']))
print("Kurtosis:", kurtosis(df_cleaned['India_FLFPR']))

print("\nGermany Summary:")
print(df_cleaned['Germany_FLFPR'].describe())
print("Skewness:", skew(df_cleaned['Germany_FLFPR']))
print("Kurtosis:", kurtosis(df_cleaned['Germany_FLFPR']))

# Trend Visualization --> Line plot for FLFPR over time
plt.figure(figsize=(10,6))
plt.plot(df_cleaned['Year'], df_cleaned['India_FLFPR'], marker='o', label='India')
plt.plot(df_cleaned['Year'], df_cleaned['Germany_FLFPR'], marker='o', label='Germany')
plt.title('Female Labor Force Participation Rate (1990–2022)')
plt.xlabel('Year')
plt.ylabel('FLFPR (%)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Distribution visualization --> Histogram and KDE(Kernel Density Estimate)
plt.figure(figsize=(10,5))
sns.kdeplot(df_cleaned['India_FLFPR'], shade=True, label='India')
sns.kdeplot(df_cleaned['Germany_FLFPR'], shade=True, label='Germany')
plt.title('Distribution of FLFPR: India vs Germany')
plt.xlabel('FLFPR (%)')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

# Year-on-Year difference --> Calculate yearly change
df_cleaned['India_Change'] = df_cleaned['India_FLFPR'].diff()
df_cleaned['Germany_Change'] = df_cleaned['Germany_FLFPR'].diff()

plt.figure(figsize=(10,5))
plt.plot(df_cleaned['Year'], df_cleaned['India_Change'], label='India Change', linestyle='--')
plt.plot(df_cleaned['Year'], df_cleaned['Germany_Change'], label='Germany Change', linestyle='--')
plt.axhline(0, color='gray', linestyle='dotted')
plt.title('Year-on-Year Change in FLFPR')
plt.xlabel('Year')
plt.ylabel('Change (%)')
plt.legend()
plt.show()


# Plot histograms and KDE 
plt.figure(figsize=(12,5))

# India
plt.subplot(1,2,1)
sns.histplot(df_cleaned['India_FLFPR'], kde=True, bins=10, color='#1f77b4')
plt.title('India FLFPR Distribution')
plt.ylabel('Frequency')

# Germany
plt.subplot(1,2,2)
sns.histplot(df_cleaned['Germany_FLFPR'], kde=True, bins=10, color='#ff7f0e')
plt.title('Germany FLFPR Distribution')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Q-Q plot
plt.figure(figsize=(12,5))

# India
plt.subplot(1,2,1)
stats.probplot(df_cleaned['India_FLFPR'], dist="norm", plot=plt)
plt.title("Q-Q Plot: India (Blue)")
plt.gca().get_lines()[0].set_color('#1f77b4')  # Blue

# Germany
plt.subplot(1,2,2)
stats.probplot(df_cleaned['Germany_FLFPR'], dist="norm", plot=plt)
plt.title("Q-Q Plot: Germany (Orange)")
plt.gca().get_lines()[0].set_color('#ff7f0e')  # Orange

plt.tight_layout()
plt.show()

# Shapiro-Willks test
stat_india, p_india = shapiro(df_cleaned['India_FLFPR'])
stat_germany, p_germany = shapiro(df_cleaned['Germany_FLFPR'])

print(f"India Shapiro Test p-value: {p_india:.4f}")
print(f"Germany Shapiro Test p-value: {p_germany:.4f}")


# Central Limit Theorem Assessment
def simulate_clt(data, sample_size=30, num_samples=1000):
    return [np.mean(np.random.choice(data, size=sample_size, replace=True)) for _ in range(num_samples)]

# Define color shades for each sample size
blue_shades = ['#a6c8ff', '#4f94d4', '#1f77b4']     # Light to dark blue
orange_shades = ['#ffd6a5', '#ffae42', '#ff7f0e']   # Light to dark orange

sample_sizes = [5, 10, 30]

# ---- India ----
plt.figure(figsize=(12,8))
for i, size in enumerate(sample_sizes):
    sample_means = simulate_clt(df_cleaned['India_FLFPR'], sample_size=size)
    sns.histplot(sample_means, kde=True, stat='density',
                 label=f'n={size}', color=blue_shades[i], alpha=0.7)

plt.title('India FLFPR - CLT (Sample Means)')
plt.xlabel('Sample Mean')
plt.ylabel('Probability Density')
plt.legend(title='Sample Size')
plt.show()


# ---- Germany ----
plt.figure(figsize=(12,8))
for i, size in enumerate(sample_sizes):
    sample_means = simulate_clt(df_cleaned['Germany_FLFPR'], sample_size=size)
    sns.histplot(sample_means, kde=True, stat='density',
                 label=f'n={size}', color=orange_shades[i], alpha=0.7)

plt.title('Germany FLFPR - CLT (Sample Means)')
plt.xlabel('Sample Mean')
plt.ylabel('Probability Density')
plt.legend(title='Sample Size')
plt.show()


# Comparing normality: Original data vs the resampled means
def simulate_sample_means(data, sample_size=30, num_samples=1000):
    return [np.mean(np.random.choice(data, size=sample_size, replace=True)) for _ in range(num_samples)]

def test_shapiro_on_sample_means(data, label):
    for n in [5, 10, 30]:
        sample_means = simulate_sample_means(data, sample_size=n)
        stat, p = shapiro(sample_means)
        print(f"{label} (Sample size n={n}) → Shapiro p-value: {p:.4f} {'(Normal)' if p >= 0.05 else '(Not Normal)'}")

print("🔵 India FLFPR Sample Means")
test_shapiro_on_sample_means(df_cleaned['India_FLFPR'], label="India")

print("\n🟠 Germany FLFPR Sample Means")
test_shapiro_on_sample_means(df_cleaned['Germany_FLFPR'], label="Germany")

# Test for equal variance to choose t-test vs Welch
stat, p = levene(df_cleaned['India_FLFPR'], df_cleaned['Germany_FLFPR'])
print(f"Levene’s Test p-value: {p:.4f}")


# Running t-test (equal variance assumed)
stat, p = ttest_ind(df_cleaned['India_FLFPR'], df_cleaned['Germany_FLFPR'], equal_var=True)
print(f"t-test p-value: {p:.4f}")

# Welch's t-test (umequal variance assumed)
stat, p = ttest_ind(df_cleaned['India_FLFPR'], df_cleaned['Germany_FLFPR'], equal_var=False)
print(f"Welch’s t-test p-value: {p:.4f}")


# Mann-Whitney U test
stat, p = mannwhitneyu(df_cleaned['India_FLFPR'], df_cleaned['Germany_FLFPR'])
print(f"Mann-Whitney U test p-value: {p:.4f}")


# Hypothesis results visualization

# Prepare the data
means = [df_cleaned['India_FLFPR'].mean(), df_cleaned['Germany_FLFPR'].mean()]
stds = [df_cleaned['India_FLFPR'].std(), df_cleaned['Germany_FLFPR'].std()]
labels = ['India', 'Germany']
colors = ['blue', 'orange']

# Run t-test
stat, p = ttest_ind(df_cleaned['India_FLFPR'], df_cleaned['Germany_FLFPR'])

# Bar plot
plt.figure(figsize=(8,6))
bars = plt.bar(labels, means, yerr=stds, capsize=10, color=colors)

# Annotate means
for bar, mean in zip(bars, means):
    plt.text(bar.get_x() + bar.get_width()/2, mean + 1, f'{mean:.1f}', 
             ha='center', va='bottom', fontsize=12)

# Add significance stars
if p < 0.001:
    significance = '***'
elif p < 0.01:
    significance = '**'
elif p < 0.05:
    significance = '*'
else:
    significance = 'ns'

# Draw a significance line
y_max = max(means) + max(stds) + 2
plt.plot([0, 1], [y_max, y_max], color='black')
plt.text(0.5, y_max + 0.5, significance, ha='center', fontsize=14)

# Labels
plt.title('Mean Female Labor Force Participation Rate (1990–2022)')
plt.ylabel('FLFPR (%)')
plt.ylim(0, y_max + 5)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

