import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from prettytable import PrettyTable
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly as px
import plotly.graph_objects as go
from scipy.stats import probplot, normaltest, shapiro, kstest, boxcox, boxcox_normplot

pd.set_option('display.max_columns', None)  # Show all columns


df = pd.read_csv('adult.data')
print(df)
print(df.isna().sum().sum())
print(df.columns)
print(df.info)
print(df.describe())


col_change = {
    '39':'Age',
    ' State-gov':'workclass',
    ' 77516':'fnlwgt',
    ' Bachelors':'Education',
    ' 13':'Education_num',
    ' Never-married':'marital-status',
    ' Adm-clerical':'occupation',
    ' Not-in-family':'relationship',
    ' White':'race',
    ' Male':'sex',
    ' 2174':'capital_gain',
    ' 0':'capital_loss',
    ' 40':'hours_per_week',
    ' United-States':'country',
    ' <=50K':'income'
}

df.rename(columns = col_change, inplace=True)
print(df.columns)
df['income'] = df['income'].apply(lambda x: 1 if x == ' >50K' else 0)

print(df.describe())

# ========================================
# Pair Plot
# ========================================
plt.figure()
sns.pairplot(df, hue="income")
plt.title('Pair plot ')
plt.tight_layout()
plt.show()

# ========================================
# Dis Plot
# ========================================
plt.figure()
sns.displot(df, x='Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('histogram plot ')
plt.tight_layout()
plt.show()

sns.displot(df, x='Age', kde=True, fill=True, alpha=0.6, hue='sex')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('histogram plot with kernel density')
plt.tight_layout()
plt.show()

# rug plot
sns.displot(df['Age'], rug=True, kde=True, color='red')
plt.xlabel('X-axis')
plt.ylabel('Density')
plt.title('Rug Plot with KDE')
plt.show()


# ========================================
# Hist Plot
# ========================================
sns.histplot(df, x='hours_per_week', kde=True, binwidth=3)
plt.xlabel("Hours per Week")
plt.ylabel("Count")
plt.title("Histogram with KDE of Hours per Week")
plt.tight_layout()
plt.show()

sns.histplot(df, x='hours_per_week', hue='Education', multiple='dodge')
plt.xlabel("Hours per Week")
plt.ylabel("Count")
plt.title("Histogram of Hours per Week by Education Level")
plt.tight_layout()
plt.show()

sns.histplot(df, x='hours_per_week', hue='occupation')
plt.xlabel("Hours per Week")
plt.ylabel("Count")
plt.title("Histogram of Hours per Week by Occupation")
plt.tight_layout()
plt.show()

sns.histplot(df, x='hours_per_week', hue='race')
plt.xlabel("Hours per Week")
plt.ylabel("Count")
plt.title("Histogram of Hours per Week by Race")
plt.tight_layout()
plt.show()

df_filtered = df[df['hours_per_week'] != 40]

sns.histplot(df_filtered, x='hours_per_week', kde=True, binwidth=3)
plt.xlabel("Hours per Week")
plt.ylabel("Count")
plt.title("Histogram with KDE of filtered Hours per Week")
plt.tight_layout()
plt.show()

sns.histplot(df_filtered, x='hours_per_week', hue='Education', multiple='dodge')
plt.xlabel("Hours per Week")
plt.ylabel("Count")
plt.title("Histogram of filtered Hours per Week by Education Level")
plt.tight_layout()
plt.show()

sns.histplot(df_filtered, x='hours_per_week', hue='occupation')
plt.xlabel("Hours per Week")
plt.ylabel("Count")
plt.title("Histogram of filtered Hours per Week by Occupation")
plt.tight_layout()
plt.show()

sns.histplot(df_filtered, x='hours_per_week', hue='race')
plt.xlabel("Hours per Week")
plt.ylabel("Count")
plt.title("Histogram of filtered Hours per Week by Race")
plt.tight_layout()
plt.show()


sns.histplot(df, x='Age', hue='occupation')
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Histogram of Age by Occupation")
plt.tight_layout()
plt.show()

sns.histplot(df, x='Age', hue='Education', multiple='stack')
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Histogram of Age by Education")
plt.tight_layout()
plt.show()

sns.histplot(df, x='Age', hue='relationship', multiple='dodge')
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Histogram of Age by Relationship")
plt.tight_layout()
plt.show()

sns.histplot(df, x='Age', hue='marital-status', multiple='dodge')
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Histogram of Age by Marital Status")
plt.tight_layout()
plt.show()

# ========================================
# PiePlot
# ========================================

plt.figure(figsize=(6, 6))
sns.set(style="whitegrid")
plt.pie(df['race'].value_counts(), labels=df['race'].value_counts().index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Pie Plot for race Data')
plt.tight_layout()
plt.legend()
plt.show()

plt.figure(figsize=(6, 6))
sns.set(style="whitegrid")
plt.pie(df['occupation'].value_counts(), labels=df['occupation'].value_counts().index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Pie Plot for occupation Data')
plt.tight_layout()
plt.legend()
plt.show()

plt.figure(figsize=(6, 6))
sns.set(style="whitegrid")
plt.pie(df['relationship'].value_counts(), labels=df['relationship'].value_counts().index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Pie Plot for relationship Data')
plt.tight_layout()
plt.legend()
plt.show()

plt.figure(figsize=(6, 6))
sns.set(style="whitegrid")
plt.pie(df['Education'].value_counts(), labels=df['Education'].value_counts().index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Pie Plot for education Data')
plt.tight_layout()
plt.legend()
plt.show()

plt.figure(figsize=(6, 6))
sns.set(style="whitegrid")
plt.pie(df['sex'].value_counts(), labels=df['sex'].value_counts().index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Pie Plot for sex Data')
plt.tight_layout()
plt.legend()
plt.show()

# ========================================
# Count Plot
# ========================================
plt.figure()
sns.countplot(x='income', data=df)
plt.xlabel("income")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.title("Count Plot of income")
plt.tight_layout()
plt.show()

plt.figure()
sns.countplot(x='sex', data=df)
plt.xlabel("sex")
plt.ylabel("Count")
plt.title("Count Plot of sex")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
sns.countplot(x='Education', data=df)
plt.xlabel("Education")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.title("Count Plot of Education")
plt.tight_layout()
plt.show()

plt.figure()
sns.countplot(x='occupation', data=df)
plt.xlabel("occupation")
plt.ylabel("Count")
plt.title("Count Plot of occupation")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
sns.countplot(x='marital-status', data=df)
plt.xlabel("marital-status")
plt.ylabel("Count")
plt.title("Count Plot of marital-status")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
sns.countplot(x='relationship', data=df)
plt.xlabel("relationship")
plt.ylabel("Count")
plt.title("Count Plot of relationship")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
sns.countplot(x='race', data=df)
plt.xlabel("Race")
plt.ylabel("Count")
plt.title("Count Plot of Race")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#


plt.figure()
sns.countplot(x='sex', data=df, hue='income')
plt.xlabel("sex")
plt.ylabel("Count")
plt.title("Count Plot of sex")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
sns.countplot(x='Education', data=df, hue='income')
plt.xlabel("Education")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.title("Count Plot of Education")
plt.tight_layout()
plt.show()

plt.figure()
sns.countplot(x='occupation', data=df, hue='income')
plt.xlabel("occupation")
plt.ylabel("Count")
plt.title("Count Plot of occupation")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
sns.countplot(x='marital-status', data=df, hue='income')
plt.xlabel("marital-status")
plt.ylabel("Count")
plt.title("Count Plot of marital-status")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
sns.countplot(x='relationship', data=df, hue='income')
plt.xlabel("relationship")
plt.ylabel("Count")
plt.title("Count Plot of relationship")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
sns.countplot(x='race', data=df, hue='income')
plt.xlabel("Race")
plt.ylabel("Count")
plt.title("Count Plot of Race")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#
def box_plot(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df_temp = df[(df > lower) & (df < upper)]
    return df_temp


# ========================================
# Box Plot
# ========================================
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

# Boxplot on the age
axes[0].boxplot(df['Age'], vert=False)
axes[0].set_title('Boxplot of Age')

# Boxplot on the second subplot
axes[1].boxplot(box_plot(df['Age']), vert=False)
axes[1].set_title('Boxplot of Age after removing outliers')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

# Boxplot on the hours
axes[0].boxplot(df['hours_per_week'], vert=False)
axes[0].set_title('Boxplot of hours')

# Boxplot on the second subplot
axes[1].boxplot(box_plot(df['hours_per_week']), vert=False)
axes[1].set_title('Boxplot of hours after removing outliers')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

# Boxplot on the fnlwgt
axes[0].boxplot(df['fnlwgt'], vert=False)
axes[0].set_title('Boxplot of fnlwgt')

# Boxplot on the second subplot
axes[1].boxplot(box_plot(df['fnlwgt']), vert=False)
axes[1].set_title('Boxplot of fnlwgt after removing outliers')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

# Boxplot on the capital_loss
axes[0].boxplot(df['capital_loss'], vert=False)
axes[0].set_title('Boxplot of capital_loss')

# Boxplot on the second subplot
axes[1].boxplot(box_plot(df['capital_loss']), vert=False)
axes[1].set_title('Boxplot of capital_loss after removing outliers')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

# Boxplot on the capital_gain
axes[0].boxplot(df['capital_gain'], vert=False)
axes[0].set_title('Boxplot of capital_gain')

# Boxplot on the second subplot
axes[1].boxplot(box_plot(df['capital_gain']), vert=False)
axes[1].set_title('Boxplot of capital_gain after removing outliers')
plt.tight_layout()
plt.show()

# ========================================
# Point Plot
# ========================================
plt.figure()
sns.pointplot(x="Education", y="income", data=df, hue="Education")
plt.xlabel("Education")
plt.ylabel("Income")
plt.title("Point Plot of Income by Education Level")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
sns.pointplot(x="occupation", y="income", data=df, hue="occupation")
plt.xlabel("Occupation")
plt.ylabel("Income")
plt.title("Point Plot of Income by Occupation")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
sns.pointplot(x="relationship", y="income", data=df, hue="relationship")
plt.xlabel("Relationship")
plt.ylabel("Income")
plt.title("Point Plot of Income by Relationship")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ========================================
# Violin Plot
# ========================================
plt.figure()
sns.violinplot(x="relationship", y="income", data=df, hue="relationship")
plt.xlabel("Relationship")
plt.ylabel("Income")
plt.title("Violin Plot of Income by Relationship")
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()

plt.figure()
sns.violinplot(x="occupation", y="income", data=df, hue="occupation")
plt.xlabel("occupation")
plt.ylabel("Income")
plt.title("Violin Plot of Income by occupation")
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()

plt.figure()
sns.violinplot(x="occupation", y="income", data=df, hue="race")
plt.xlabel("occupation")
plt.ylabel("Income")
plt.title("Violin Plot of Income by occupation and race")
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()

plt.figure()
sns.violinplot(x="occupation", y="income", data=df, hue="sex")
plt.xlabel("occupation")
plt.ylabel("Income")
plt.title("Violin Plot of Income by occupation and sex")
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()

# ========================================
# Scatter Plot
# ========================================
sns.set(style="whitegrid")
sns.regplot(x='Age', y='hours_per_week', data=df, scatter_kws={'s': 100}, line_kws={'color': 'red'})
plt.xlabel('Age')
plt.ylabel('Hours')
plt.title('Scatter Plot with Regression Line')
plt.show()

plt.figure(figsize=(8, 6))
sns.lineplot(x='Age', y='Education_num', data=df, color='blue', label='Line Plot')
plt.fill_between(df['Age'], df['Education_num'], color='blue', alpha=0.2)  # Fills the area under the curve
plt.xlabel('Age')
plt.ylabel('Education')
plt.title('Area Plot')
plt.show()


# ========================================
# Joint Plot
# ========================================
sns.set(style="whitegrid")
sns.jointplot(x='Age', y='hours_per_week', data=df, kind='scatter', marginal_kws={'color': 'blue'}, color='blue')
plt.xlabel('Age')
plt.ylabel('hours_per_week')
plt.suptitle('Joint Plot with KDE and Scatter')
plt.show()

sns.jointplot(x=df['Age'], y=df['hours_per_week'], kind='hex', color='blue')
plt.xlabel("Age")
plt.ylabel("Hours per Week")
plt.title("Hexbin Joint Plot of Age and Hours per Week")
plt.tight_layout()
plt.show()

sns.stripplot(x=df['Education_num'], y=df['race'], palette='pastel')
plt.xlabel("Education Number")
plt.ylabel("Race")
plt.title("Strip Plot of Education Number by Race")
plt.tight_layout()
plt.show()

sns.stripplot(x=df['occupation'], y=df['race'], palette='pastel')
plt.xlabel("Occupation")
plt.ylabel("Race")
plt.title("Strip Plot of Occupation Number by Race")
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()

# ========================================
# Q-Q plot
# ========================================
probplot(df['Age'], dist='norm', plot=sns.mpl.pyplot)
plt.title('Q-Q Plot')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()
#
probplot(df['hours_per_week'], dist='norm', plot=sns.mpl.pyplot)
plt.title('Q-Q Plot')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()
#
probplot(df['Education_num'], dist='norm', plot=sns.mpl.pyplot)
plt.title('Q-Q Plot')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()

df['Country_USA'] = df['country'].apply(lambda x: 'USA' if x == 'United-States' else 'Other')
df.drop('country', axis=1, inplace=True)
df['income >50K'] = df['income'].apply(lambda x: 1 if x == ' >50K' else 0)
df.drop('income', axis=1, inplace=True)

y = df['income >50K']
df.drop('income >50K', axis=1, inplace=True)
numerical_columns = df.select_dtypes(include=['number']).columns

# Initialize StandardScaler
scaler = StandardScaler()

# Standardize numerical columns
df_stan = scaler.fit_transform(df[numerical_columns])
df_stan = pd.DataFrame(df_stan, columns=numerical_columns)

# Display the standardized DataFrame
print(f'standardised df:\n{df_stan}')

# The standardized_df now contains the standardized values
print(df_stan)

correlation_matrix = df[numerical_columns].corr()
covariance_matrix = df[numerical_columns].cov()
# ========================================
# Heatmaps
# ========================================
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# Set the title
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# Set the title
plt.title('Covariance Heatmap')
plt.show()
#
# ========================================
# Normality
# ========================================
def ks_test(x, title):
    mean = np.mean(x)
    std = np.std(x)
    dist = np.random.normal(mean, std, len(x))
    stats, p = kstest(x, dist)
    print('='*50)
    print(f'K-S test: {title} dataset: statistics= {stats:.2f} p-value = {p:.2f}' )
    if p > 0.05:
        print(f'{title} dataset is normal')
    else:
        print(f'{title} dataset is not normal')

def shapiro_test(x, title):
    stats, p = shapiro(x)
    print('=' * 50)
    print(f'Shapiro test : {title} dataset : statistics = {stats:.2f} p-value of ={p:.2f}' )
    if p > 0.05:
        print(f'{title} dataset is normal')
    else:
        print(f'{title} dataset is not normal')

def da_k_squared_test(x, title):
    stats, p = normaltest(x)
    print('='*50)
    print(f'da_k_squared test: {title} dataset: statistics= {stats:.2f} p-value = {p:.2f}' )
    if p > 0.05:
        print(f'{title} dataset is normal')
    else:
        print(f'{title} dataset is not normal')

# ========================================
# Age
# ========================================
ks_test(df['Age'], 'Age')
shapiro_test(df['Age'], 'Age')
da_k_squared_test(df['Age'], 'Age')

# Transforming the data from exponential to normal

transformed_data, best_lambda = boxcox(df['Age'])
print(f'Best lambda value: {best_lambda:.2f}')

sns.distplot(transformed_data, kde=True, hist=True)
plt.xlabel("Transformed Data")
plt.ylabel("Density")
plt.title("Distribution Plot of Transformed Age Data")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
prob = boxcox_normplot(df['Age'], -10, 10, plot=ax)
ax.axvline(best_lambda, color='r')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Ordered Values")
plt.title("Probability Plot for Box-Cox Transformation for age")
plt.tight_layout()
plt.show()

ks_test(transformed_data, 'Age transformed')
shapiro_test(transformed_data, 'Age transformed')
da_k_squared_test(transformed_data,'Age transformed')

# ========================================
# Hours
# ========================================
ks_test(df['hours_per_week'], 'hours')
shapiro_test(df['hours_per_week'], 'hours')
da_k_squared_test(df['hours_per_week'], 'hours')

# Transforming the data from exponential to normal

transformed_data, best_lambda = boxcox(df['hours_per_week'])
print(f'Best lambda value: {best_lambda:.2f}')

sns.distplot(transformed_data, kde=True, hist=True)
plt.xlabel("Transformed Data")
plt.ylabel("Density")
plt.title("Distribution Plot of Transformed hours Data")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
prob = boxcox_normplot(df['hours_per_week'], -10, 10, plot=ax)
ax.axvline(best_lambda, color='r')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Ordered Values")
plt.title("Probability Plot for Box-Cox Transformation for hours")
plt.tight_layout()
plt.show()

ks_test(transformed_data, 'hours transformed')
shapiro_test(transformed_data, 'hours transformed')
da_k_squared_test(transformed_data,'hours transformed')






