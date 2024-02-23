#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


eda_df=pd.read_excel("EDA DATA.xlsx")
eda_df


# In[3]:


eda_df.head()


# In[4]:


eda_df.shape


# In[5]:


eda_df.describe()


# In[ ]:





# # Data Manipulation 

# ### Removing unwanted columns 

# In[6]:


eda_df.drop(columns='Unnamed: 0', inplace=True)
eda_df


# In[7]:


eda_df.drop(["DOJ","DOL","CollegeCityID","CollegeCityTier","Domain","ComputerProgramming","ElectronicsAndSemicon","ComputerScience","MechanicalEngg"
        ,"ElectricalEngg","TelecomEngg","CivilEngg"],axis=1,inplace=True)


# In[8]:


eda_df


# In[ ]:





# In[9]:


eda_df.info()


# In[10]:


eda_df["JobCity"].value_counts()


# In[11]:


eda_df[eda_df.Designation=="get"]


# In[11]:


eda_df


# In[12]:


eda_df.drop(eda_df[eda_df["Designation"]=="get"].index,inplace=True)
eda_df


# In[13]:


eda_df.drop(eda_df[eda_df["GraduationYear"]==0].index,inplace=True)
eda_df


# In[14]:


eda_df['GraduationYear'].value_counts()


# In[15]:


eda_df


# In[16]:


eda_df[eda_df.Designation=="associate software engineer"]


# In[17]:


eda_df["Designation"].value_counts()


# In[18]:


eda_df.Designation=eda_df.Designation.apply(lambda x : x.replace('ase', 'application support engineer') )
eda_df.Designation


# In[19]:


eda_df[eda_df.Designation=="ase"]


# In[20]:


eda_df['JobCity'] = eda_df['JobCity'].apply(lambda x : str(x))


# In[21]:


eda_df['JobCity'].dtype


# In[22]:


eda_df['Salary'].dtype


# In[23]:


eda_df.JobCity=eda_df["JobCity"].apply(lambda x : x.replace('-1','India') if '-1' in x else x )

eda_df.JobCity = eda_df['JobCity']
eda_df['JobCity']


# In[24]:


eda_df['JobCity'].head(50)


# In[25]:


eda_df['JobCity'].value_counts()


# In[28]:


y=eda_df["JobCity"].unique()
y.sort()


# In[30]:


len(eda_df["JobCity"].unique())


# In[34]:


eda_df['Specialization']=eda_df['Specialization'].replace(to_replace='electronics & instrumentation eng',value='electronics and instrumentation engineering',regex=False)


# In[35]:


len(eda_df["Designation"].unique())


# In[ ]:





# In[26]:


eda_df["Specialization"].value_counts()


# In[28]:


eda_df["10board"].value_counts()


# In[29]:


eda_df["10board"]=eda_df["10board"].apply(lambda x:str(x))
eda_df["10board"]=eda_df["10board"].apply(lambda x:x.replace("0","Indian Board of Secondary Education") if "0"in x else x)


# In[30]:


eda_df["10board"].value_counts()


# In[31]:


eda_df["12board"].value_counts()


# In[32]:


eda_df["12board"]=eda_df["12board"].apply(lambda x:str(x))
eda_df["12board"]=eda_df["12board"].apply(lambda x:x.replace("0","Indian Board of Secondary Education") if "0"in x else x)


# In[33]:


eda_df["12board"].value_counts()


# In[34]:


eda_df.head()


# In[35]:


eda_df.head()


# # Data Visualization

# In[36]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[37]:


import warnings
warnings.filterwarnings('ignore')


# In[38]:


eda_df


# In[48]:


eda_df.describe()


# In[39]:


eda_df["Gender"].value_counts()


# # Univariate Analysis 
# 
# 

# ##  Numerical columns
# ## Peda_df
# ## Histograms
# ## Boxplots
# ## Countplots etc..
# 

# ## Applying kde plot in all educational percentages

# In[40]:



fig = plt.subplots(figsize=(7,4))
sns.kdeplot(data=eda_df, x="10percentage",color="k")


# ###  Here we can see that this kdeplot shows the dencity of the 10percentage of each employee its shows like a left skewed  and the most of the employee's are 10percentage is approximatly 80 percentage

# In[41]:



fig = plt.subplots(figsize=(7,4))
sns.kdeplot(data=eda_df, x="12percentage",color="k")


# ###  This kdeplot shows the dencity of the 12percentage of each employee its shows like a normal distribution and the most of the employee's are 12percentage is approximatly 70-80 percentage

# In[42]:



fig = plt.subplots(figsize=(7,4))
sns.kdeplot(data=eda_df, x="collegeGPA",color="k")


# ### Here, this kdeplot shows the dencity of the collegeGPA of each employee it is a normal distribution and the most of the employee's are collegeGPA is approximatly 65-75

# # Histogram-Passed out year
# 
# 

# In[43]:


fig=plt.subplots(figsize=(10,5))
plt.xticks(rotation=90)
sns.histplot(data=eda_df,x="12graduation",color="r")


# ### Here we can see that most of the employee's are passed out in year 2009 in 12graduation .  

# In[44]:


fig=plt.subplots(figsize=(10,5))
plt.xticks(rotation=90)
sns.histplot(data=eda_df,x="GraduationYear",color="r")


# ### Using the GraduationYear data ,we can see that most of the employee's are passed out in year 2013.

# # Boxplot in General intelligence
# 

# In[45]:


fig=plt.subplots(figsize=(7,4))
plt.boxplot(eda_df['English'])
plt.show()


# ### This Box plot shows the marks of each employee and this column have many high extream outliers and low extream outliers.

# In[46]:


fig=plt.subplots(figsize=(7,4))
plt.boxplot(eda_df['Logical'])
plt.show()


# ### This Box plot shows the marks of each employee and this coloumn  have high extream outliers and low extream outliers.

# In[47]:


fig=plt.subplots(figsize=(7,4))
plt.boxplot(eda_df['Quant'])
plt.show()


# ### This Box plot tells abouts the Quant coloumn it shows the marks of each employee and this coloumn high extream outliers and low extream outliers 

# # countplot in categorical coloumns

# In[48]:


fig, ax = plt.subplots(figsize=(50,20))
plt.xticks(rotation=90)
sns.countplot(data=eda_df, x='JobCity', ax=ax)


# ### This countplot shows the most of the employees are working in Bangalore and Less employees are working in Bhopal, Dhanbad.

# In[49]:


fig, ax = plt.subplots(figsize=(10, 7))
sns.countplot(x='Degree', data=eda_df, ax=ax)


# ### This countplot tells about the all the employee's are which stream in the Degree coloumn  

# In[60]:


fig=plt.subplots(figsize=(20,10))
plt.xticks(rotation=90)
sns.countplot(x = eda_df['CollegeState']) 


# ### This countplot shows most of the employees are from Uttar pradesh. 

# # Bivariate analysis on Numeric columns
# # Scatter plots,
# # hexbin plots, 
# # pair plots, etc..
# 

# ### Applying Scatterplot on  Specialization, Salary comparing the Gender.

# In[50]:



fig=plt.subplots(figsize=(15,7))
plt.xticks(rotation=90)
sns.scatterplot(x = "Specialization", y = "Salary", data = eda_df, hue = "Gender",hue_order= ['m', 'f'])


# ### This scatter plot tells about the specialization and salary of the employee's and Gender how many employee's are male and female which is marked as Blue and Yellow respectively.

# ## Hexbinplot on Salary,collegeGPA 

# In[51]:


fig=plt.subplots(figsize=(10,7))
plt.hexbin(x='Salary', y='collegeGPA', data=eda_df, cmap ='icefire_r')


# 

# In[52]:



sns.pairplot( eda_df, x_vars=["10percentage", "12percentage", "collegeGPA"],
                  y_vars=["10percentage", "12percentage","collegeGPA"],
                  hue="Gender")


# ### This pair plot tells about all the employee's percentages and it shows the difference of male and female. 

# # Categorical and numerical coloumns

# In[53]:


fig=plt.subplots(figsize=(10,7))
sns.swarmplot(data = eda_df, x='Degree', y='collegeGPA')


# ###  This swarmplot shows that most of the employee's are from B.E/B.Tech.

# In[54]:



fig = plt.subplots(figsize=(10,6))
sns.barplot(x = "Degree", y = "Salary",  data=eda_df)


# ### This barplot shows that most of the salary got are  M.Tech/M.E employee's.

# In[55]:


fig=plt.subplots(figsize=(12,6))
sns.boxplot(x="Degree", y="collegeGPA", data=eda_df)


# ### This Boxplot tells about the Degree and collegeGPA M.sc(Tech) employee's having there is  no outliers all the remaining streams are having extream outliers.

# # Research Questions
# 

# Times of India article dated Jan 18, 2019 states that “After doing your Computer Science Engineering if you take up jobs as a Programming Analyst, Software Engineer, Hardware Engineer and Associate Engineer you can earn up to 2.5-3 lakhs as a fresh graduate.” Test this claim with the data given to you.

# In[56]:


eda_df_R=eda_df[(eda_df["Designation"]=="programmer analyst")|(eda_df["Designation"]=="software engineer")|(eda_df["Designation"]=="hardware engineer")
       |(eda_df["Designation"]=="associate engineer")]
eda_df_R


# In[57]:


eda_df_R["Salary"]


# In[58]:


ab=eda_df_R["Salary"]
bc=[]
for i in ab:
    bc.append(i)
print(bc)
    


# In[59]:


import random
n=40
cd=random.sample(bc,n)
print(cd)


# In[71]:



def t_score(sample_size, sample_mean, pop_mean, sample_std):
    numerator = sample_mean - pop_mean
    denomenator = sample_std / sample_size**0.5
    return numerator / denomenator


# In[72]:


import numpy as np
import statistics
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t,norm


# In[73]:


sum(cd)/len(cd)


# In[74]:


statistics.stdev(cd)


# In[75]:


sample_size = 100
sample_mean =332250.0
pop_mean = 300000
sample_std=89621.3


# In[76]:


t_value = t_score(sample_size, sample_mean, pop_mean, sample_std)

print(t_value)


# In[82]:


x_min = -200000
x_max = 800000

mean = pop_mean
std = sample_std 

x = np.linspace(x_min, x_max, 100)
y = norm.peda_df(x, mean, std)
plt.xlim(x_min, x_max)
plt.plot(x, y)

t_critical_left = pop_mean + (-t_critical * std)
t_critical_right = pop_mean + (t_critical * std)

x1 = np.linspace(x_min, t_critical_left, 100)
y1 = norm.peda_df(x1, mean, std)
plt.fill_between(x1, y1, color='red')

x2 = np.linspace(t_critical_right, x_max, 100)
y2 = norm.peda_df(x2, mean, std)
plt.fill_between(x2, y2, color='red')

plt.scatter(sample_mean, 0)
plt.annotate("x_bar", (sample_mean, 0.7))


# In[78]:


if(t_value < t_critical):
    print("Reject Null Hypothesis")
else:
    print("Fail to reject Null Hypothesis")


# In[ ]:





# # Step - 6 - Conclusion

# ## Data understanding:
# ####  The dataset contains the employment outcomes of engineering graduates as dependent variables (Salary, Job Titles, and Job Locations) along with the standardized scores from three different areas – cognitive skills, technical skills and personality skills. 
# ## Data manipulation:
# #### Here i am obeserving the Data set contains the 4000 rows and 40 columns and the this data set having so many duplicate  values and frist we to manipulate the data set and remove unwanted rows and columns after  that check the any nan values are there or not and  after to take the cleaned data set  do visualization
# ## Data Visualization:
# ### Univariate Analysis -> Peda_df, Histograms, Boxplots, Countplots
# #### univariate analysis having many plots  and it shows the  probability and frequency distribution
# ### Bivariate Analysis -> Scatterplot,hexbinplot,pairplot,swarmplot,barplot,boxplot
# #### Here the conclusion is the total project oberverving the we are showing the employee's data set is to comparing the all percentages  and in this data set having  outliers using boxplot  find out the outliers using countplot to find out the for ex: jobcity having which city have more employee's so we can find the place ...

# # Research questions

# ###   There are two friends in a city first  friend is one of the managers of TATA company and second one is one of  managers  in AMEO company the second person started conversation by saying that ,we, the AMEO company hires or recruits the students or freshres who are having percentage of above 70 and he also mentioned that the avarege percentage who are recruited is 80.Then the frist person replied   that  am i joke to you ? and he said that is that even possible to maintain the average percentage of 80 . can you prove it
# 

# # Step - 7 - Perform feature transformation:

# ## Numerical columns -Standardization
# ### Mean-0,std-1

# In[81]:


eda_df_num = eda_df.select_dtypes(include=['int64', 'float64'])
eda_df_num


# In[82]:


from sklearn.preprocessing import StandardScaler
features = ["Salary","10percentage","12graduation","12percentage","CollegeID","CollegeTier","collegeGPA","GraduationYear","English",
      "Logical","Quant","conscientiousness","agreeableness","extraversion","nueroticism","nueroticism"]
autoscaler = StandardScaler()
eda_df[features] = autoscaler.fit_transform(eda_df[features])


# In[83]:


eda_df[features].describe()


# # Categorical columns 
# ## convert the feature to Binary.

# In[84]:


eda_df_cat = eda_df.select_dtypes(include=['object'])

eda_df_cat.head()


# In[85]:


from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Assuming eda_df_cat is your DataFrame containing categorical variables
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_data = encoder.fit_transform(eda_df_cat)

# Extracting feature names
encoded_columns = encoder.get_feature_names_out(input_features=eda_df_cat.columns)

# Creating a DataFrame with the encoded data
categorical = pd.DataFrame(encoded_data, columns=encoded_columns)

# Displaying the first few rows
categorical.head()


# #### The Standardization is done using onehot-encoding and it will be coverted into  the feature to Binary only object columns 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




