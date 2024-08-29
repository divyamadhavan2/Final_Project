#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[84]:


df= pd.read_csv("E:\Divya\SCCT\code\Data\Student_performance_data.csv")


# In[85]:


df.head()


# In[86]:


# checking for null values
df.isnull().sum()


# In[87]:


print(df.info())


# There is no null values in the data set 

# In[88]:


df.nunique() # unique values in each column


# In[89]:


print(df.describe())


# # Correlation Matrix

# In[90]:


# Drop the Student ID column
df = df.drop('StudentID', axis=1)


# In[91]:


# Compute the correlation matrix
corr_matrix = df.corr()
corr_matrix


# In[92]:


plt.figure(figsize=(7, 7))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[93]:


# Mapping dictionaries for categorical columns
gender_mapping = {0: 'Male', 1: 'Female'}
ethnicity_mapping = {0: 'Caucasian', 1: 'African American', 2: 'Asian', 3: 'Other'}
parental_education_mapping = {0: 'None', 1: 'High School', 2: 'Some College', 3: 'Bachelor\'s', 4: 'Higher'}
parental_support_mapping = {0: 'None', 1: 'Low', 2: 'Moderate', 3: 'High', 4: 'Very High'}
extracurricular_mapping = {0: 'No', 1: 'Yes'}
sports_mapping = {0: 'No', 1: 'Yes'}
music_mapping = {0: 'No', 1: 'Yes'}
volunteering_mapping = {0: 'No', 1: 'Yes'}
grade_class_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}

# Apply mappings to the dataset
df['Gender'] = df['Gender'].map(gender_mapping)
df['Ethnicity'] = df['Ethnicity'].map(ethnicity_mapping)
df['ParentalEducation'] = df['ParentalEducation'].map(parental_education_mapping)
df['ParentalSupport'] = df['ParentalSupport'].map(parental_support_mapping)
df['Extracurricular'] = df['Extracurricular'].map(extracurricular_mapping)
df['Sports'] = df['Sports'].map(sports_mapping)
df['Music'] = df['Music'].map(music_mapping)
df['Volunteering'] = df['Volunteering'].map(volunteering_mapping)
df['GradeClass'] = df['GradeClass'].map(grade_class_mapping)


# In[94]:


# Display unique values in categorical columns
categorical_columns = ['Gender', 'Ethnicity', 'ParentalEducation', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']
for col in categorical_columns:
    print(f'{col} unique values: {df[col].unique()}')

# Plot the distribution of categorical variables
for col in categorical_columns:
    plt.figure(figsize=(6, 3))  #  figure size (width, height) adjustment
    sns.countplot(x=col, data=df, width=0.3)  # Adjust the width of the bars (0.6 is narrower)
    plt.title(f'Distribution of {col}')
    plt.show()


# In[95]:


# Plot the distribution of categorical variables against GradeClass
for col in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, hue='GradeClass', data=df)
    plt.title(f'Relationship between {col} and GradeClass')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.legend(title='GradeClass')
    plt.show()


# In[96]:


# Plot the distribution of numerical variables
numerical_columns = ['GPA', 'StudyTimeWeekly', 'Absences', 'Age']
for col in numerical_columns:
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


# In[97]:


from scipy.stats import chi2_contingency

# Function to calculate Cramér's V to find the correlation between categorical variables and target variable
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = sum(confusion_matrix.sum())
    return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))

# List of categorical columns to check against GradeCategory
categorical_columns = ['Gender', 'Ethnicity', 'ParentalEducation', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']

# Calculate Cramér's V for each categorical feature
for feature in categorical_columns:
    cramers_v_value = cramers_v(df[feature], df['GradeClass'])
    print(f"Cramér's V for {feature} vs GradeClass: {cramers_v_value:.4f}")


# # Target Varibale Distribution

# In[98]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x='GradeClass', data=df)
plt.title('Distribution of Target Variable: GradeClass')
plt.xlabel('GradeClass')
plt.ylabel('Count')
plt.show()


# In[99]:


# Plots to show the distribution of target variable by GPA and Absences (highly correlated varibales with gradeclass)
# Boxplot for GPA and GradeClass
plt.figure(figsize=(6, 4))
sns.scatterplot(x='GradeClass', y='GPA', data=df)
plt.title('Distribution of GPA by GradeClass')
plt.xlabel('GradeClass')
plt.ylabel('GPA')
plt.show()

# Boxplot for StudyTimeWeekly and GradeClass
plt.figure(figsize=(6, 4))
sns.boxplot(x='GradeClass', y='Absences', data=df, palette='coolwarm')
plt.title('Distribution of Absences by GradeClass')
plt.xlabel('GradeClass')
plt.ylabel('Absences')
plt.show()


# # Outlier Detection 

# In[100]:


# Box plots to check for outliers in numerical features
numerical_columns = ['GPA', 'StudyTimeWeekly', 'Absences', 'Age']
for col in numerical_columns:
    sns.boxplot(x=df[col])
    plt.title(f'Box plot of {col}')
    plt.show()


# # Pairplot

# In[101]:


import seaborn as sns
import matplotlib.pyplot as plt

# Plot pair plot for selected variables
sns.pairplot(df[['GPA', 'StudyTimeWeekly', 'Absences', 'Age', 'GradeClass']], hue='GradeClass', height=2.5, palette='Set1')
plt.show()


# In[102]:


#saving data
df.to_csv('E:\Masters Project\Final_Final\Data.csv', index=False)


# In[ ]:





# In[ ]:




