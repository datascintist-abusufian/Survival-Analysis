Proper dataset documentation link,

https://data.mendeley.com/datasets/x5nmsprjc9/1





import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_file = '/hpf/projects/jvorstman/Survival_Analysis/echocardiogram.data'
original_df = pd.read_csv(data_file, sep=',', on_bad_lines='skip')
print(original_df.head())

csv_file = '/hpf/projects/jvorstman/Survival_Analysis/original.csv'
original_df.to_csv(csv_file, index=False)
print(f"Conversion completed. CSV file saved at: {csv_file}")

extended_df = pd.read_csv('/hpf/projects/jvorstman/Survival_Analysis/updated_extended_echocardiogram (1).csv')
print(extended_df.head())

original_df.rename(columns={original_df.columns[2]: 'age'}, inplace=True)
original_df['age'] = pd.to_numeric(original_df['age'], errors='coerce')
print(original_df['age'])

extended_df.rename(columns={extended_df.columns[1]: 'age'}, inplace=True)
extended_df['age'] = pd.to_numeric(extended_df['age'], errors='coerce')
print(extended_df['age'])

mean_Original = original_df['age'].mean()
mean_Synthetic = extended_df['age'].mean()

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.kdeplot(data=original_df, x='age', label='Original', shade=True)
sns.kdeplot(data=extended_df, x='age', label='Synthetic', shade=True)

plt.axvline(mean_Original, color='blue', linestyle='--', label=f'Mean Original: {mean_Original:.2f}')
plt.axvline(mean_Synthetic, color='orange', linestyle='--', label=f'Mean Synthetic: {mean_Synthetic:.2f}')

plt.title("KDE Plot for Age at Heart Attack Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.savefig("/hpf/projects/jvorstman/Survival_Analysis/kde_age_plot.png", dpi=300, bbox_inches='tight')
