
# coding: utf-8

# # <font color='red'> *FINAL PROJECT - BLACK FRIDAY SALE*

# ### The data set consists of transactions of a super store from Tri-State Region on the Black Friday Sale.
# 
# ### We have taken this data set from www.analyticsvidhya.com

# ### <font color='grey'> *The question we ask on our data set* 
# 
# ### Can we predict the purchase trend for future Black Friday sales from the past data ?

# ### <font color='grey'> *Why it is an interesting question ?* 
# 
# ### Predicting the future purchase trend can help us to increase the inventory for the next Black Friday Sale by analysing the customer purchasing pattern of the products by considering their age, gender, product categories they buy, etc.

# ### <font color='grey'> *Assumptions* 
# 
# ### - The product categories contain many NAN values which we have filled with ‘0’  considering that the customer did not buy anything from that product category.
# ### - In the feature ‘Marital Status’ there are two values ‘0’ and ‘1’ which we consider as ‘Unmarried’ and ‘Married’ respectively.

# ### <font color='grey'> *Limitations* 
# 
# ### - There might be some features (brand, availability, etc of products) which are not present in our data and can affect the trend of future purchase.
# ### - Various Economic factors can indirectly affect the sales and lead to change in future purchase trends.

# ### <font color='grey'> *The data we are using for our analysis* 
# 
# 

# In[1]:

# We will import all packages before we import our data set

get_ipython().magic('matplotlib inline')
#'%matplotlib inline' sets the backend of matplotlib to the 'inline' backend.
#With this backend, the output of plotting commands is displayed inline within frontends like the Jupyter notebook

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import pyplot
import scipy.stats as stats
from scipy.stats import chisquare 
import statsmodels.formula.api as smf
sns.set(style='ticks', context='talk') 
#set() is a Seaborn function for proper Visualisations of graphs and we have used style ='ticks' and context='talks'
import plotly.plotly as py
import plotly 
plotly.tools.set_credentials_file(username='amod1', api_key='PXLMjlUoSkqVYd1cQbsY')


# reading data set using pandas function

d = pd.read_csv('/Users/senorpete/Desktop/DS1.csv')
d.head()


# ### We have imported our dataframe, now we clean the data before we begin with visualisation and machine learning part

# ### From the above result we see that the total number of observations are 65499 which we see missing in the columns 'Electronics' and 'Furniture'. 
# ### So we fill those NAN values.

# ### <font color='grey'> *Cleaning the Data Set* 
# 
# 

# In[41]:

#df.column.fillna function is used to fill NA values 

d.fillna(value = '0', inplace = True)
d


# In[42]:

d.info()


# ### From the above result we can confirm that our data is clean.
# ### The cleaned data set is as follows.

# In[5]:

d.head()


# ### Now we start with the visualisation part which will give us a clear picture of dependencies of features in the data set.

# ## <font color='grey'> *Visualisation* 
# 
# 

# In[6]:

sns.distplot(d['Purchase'])


# In[7]:

# using the parameter kde just in case we don't want the curve that is marked
# also using the bins so that we get a clearer image of the plot
sns.distplot(d['Purchase'], kde = False, bins = 30)


# In[8]:

#barplot for categorical vs numerical
sns.set_style('ticks')
sns.barplot(x='Gender',y='Purchase',data = d)
sns.despine(left=True,bottom=True)


# Analysis : The maximum purchase is done by males.

# In[9]:

# Countplot of Male and female(to compare the total count and the purchase done by the higher and the lower gender)
sns.countplot(x ='Gender', data = d)


# In[10]:

#Boxplot is used to check the purchase range by different age groups
sns.boxplot(x='Age',y='Purchase',data =d)


# Analysis: The maximum purchase is done by people by people falling in the age group of 51-55, followed by people of age group 0-17.

# In[11]:

#Boxplot is used to check the purchase range by different age groups (which are further segregated by Marital status)
fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(11.7, 8.27)    
sns.boxplot(x='Age',y='Purchase',data =d,hue='Marital_Status', ax=ax)


# Analysis: This boxplot along with the purchase done by different age groups shows the distinction of the marital status as well. Clearly in many cases the group of singles tend to purchase more than the married.

# In[12]:

sns.violinplot(x = 'State', y = 'Purchase', data = d, hue = 'Gender', split = True)


# Analysis : This violin plot shows the analysis of state vs purchase segregated by the gender. Clearly PA state has most purchases, and most purchases are made by men.

# Maximum Purchase Total in the Tri-state.

# In[13]:

sns.factorplot(x = 'State', y = 'Purchase', data = d, kind = 'bar')


# In[14]:

tc = d.corr()


# In[15]:

sns.heatmap(tc)


# In[16]:

g = sns.FacetGrid(data = d,col = 'State', row = 'Apparels')
g.map(plt.scatter, 'Purchase','Marital_Status')


# In[17]:

g = sns.FacetGrid(data = d,col = 'State', row = 'Electronics')
g.map(plt.scatter, 'Purchase','Marital_Status')


# In[18]:

from plotly import __version__


# In[19]:

import cufflinks as cf


# In[20]:

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot


# In[21]:

init_notebook_mode(connected = True)


# In[22]:

cf.go_offline()


# In[23]:

df = pd.DataFrame(data = d, columns = ['Apparels','Purchase'])


# In[24]:

df.iplot()


# Analysis: Interactive and informative plot of Apparels vs Purchase 

# In[25]:

d.iplot(kind = 'surface', colorscale='rdylbu')


# Analysis: This is the plot of the entire data set in 3D.

# In[26]:

df1 = pd.DataFrame(data = d, columns = ['State','Gender','Purchase'])


# In[27]:

df1.iplot()


# Analysis: This is the informative plot of State, Gender and the Purchases made by the following categories.

# In[28]:

df2 = pd.DataFrame(data = d, columns = ['State','Gender','Apparels','Purchase'])


# In[29]:

df2.iplot()


# Analysis : Plot of State, Gender, Apparels and the purchase done by them

# ##  We did some geographical ploting using plotly package 

# In[34]:

#We use pd.to_numeric function to change the data type to integer

d.Electronics = pd.to_numeric(d.Electronics, errors='coerce')
d = d.dropna(subset=['Electronics'])
d.Electronics = d.Electronics.astype(int)

d.Furniture = pd.to_numeric(d.Furniture, errors='coerce')
d = d.dropna(subset=['Furniture'])
d.Furniture = d.Furniture.astype(int)


# In[35]:

final= d[['State', 'Apparels', 'Electronics','Furniture','Purchase']].copy()
final[:5]


# In[36]:

NYC= final[(final['State'] == 'NYC')]
print('Apparels:',sum(NYC['Apparels']))
print('Electronics:',sum(NYC['Electronics']))
print('Furniture:',sum(NYC['Furniture']))
NYC_apparels=sum(NYC['Apparels'])
NYC_electronics=sum(NYC['Electronics'])
NYC_furniture=sum(NYC['Furniture'])
NYC_Purchase=NYC_apparels+NYC_electronics+NYC_furniture


# In[37]:

NJ= final[(final['State'] == 'NJ')] 
print('Apparels:',sum(NJ['Apparels']))
print('Electronics:',sum(NJ['Electronics']))
print('Furniture:',sum(NJ['Furniture']))
NJ_apparels=sum(NJ['Apparels'])
NJ_electronics=sum(NJ['Electronics'])
NJ_furniture=sum(NJ['Furniture'])
NJ_Purchase=NJ_apparels+NJ_electronics+NJ_furniture


# In[38]:

PA= final[(final['State'] == 'PA')] 
print('Apparels:',sum(PA['Apparels']))
print('Electronics:',sum(PA['Electronics']))
print('Furniture:',sum(PA['Furniture']))
PA_apparels=sum(PA['Apparels'])
PA_electronics=sum(PA['Electronics'])
PA_furniture=sum(PA['Furniture'])
PA_Purchase=PA_apparels+PA_electronics+PA_furniture


# In[39]:

f = {'State':['NY','NJ','PA'],
     'Apparels': [NYC_apparels,NJ_apparels,PA_apparels], 
     'Electronics': [NYC_electronics, NJ_electronics,PA_electronics],  
     'Furniture': [NYC_furniture,NJ_furniture,PA_furniture],
     'Purchase':[NYC_Purchase,NJ_Purchase,PA_Purchase]
    }
f1 = pd.DataFrame(data=f)
print(f1)


# In[40]:


for col in f1.columns:
    f1[col] = f1[col].astype(str)

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

f1['text'] = f1['State'] + '<br>' +    'Apparels '+f1['Apparels']+' Electronics '+f1['Electronics']+'<br>'+    'Furniture '+f1['Furniture']
    

    
data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = f['State'],
        z = f1['Purchase'].astype(float),
        locationmode = 'USA-states',
        text = f1['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "USD")
        ) ]

layout = dict(
        title = '2017 Black Friday Sale in Tri State Region<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map') 


# ### Now we proceed with the final part i.e. Machine Learning. We will use different statistical techniques to predict the future Black Friday Sale Purchase.

# ###  <font color='GREY'> *Multi-linear Regression*
# 

# In[43]:

d.describe()


# In[44]:

from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Age','State','Marital_Status','Apparels', 'Furniture', 'Electronics']
le = LabelEncoder()
for i in var_mod:
    d[i] = le.fit_transform(d[i])
d.dtypes 


# In[45]:

from sklearn.model_selection import train_test_split


# In[46]:

X = d[['Gender','Age','Apparels', 'Furniture', 'Electronics']]
y = d['Purchase']


# In[47]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[48]:

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)


# In[49]:

print(lm.intercept_)


# In[50]:

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[51]:

predictions = lm.predict(X_test)


# In[52]:

plt.scatter(y_test,predictions)


# ###  <font color='GREY'> *Corelation*
# 
# #### Hypothesis : Apparels and Purchase columns are dependent
# #### Alternate Hypothesis : They are independent

# In[54]:

db = pd.DataFrame({'Apparels': d.Apparels,'Purchase': d.Purchase})
db[:5]


# In[55]:

# Calculating the correlation coeeficient

db.corr()


# ###  <font color='GREY'> *Chi-square Test*
# 
# #### Hypothesis : The expected value is 9288 which is the mean value.
# #### Alternative Hypothesis : The expected value is not 9288

# In[56]:

observed = 10035
expected = 9288

#Using the formula for chi-square test

chi_squared_stat = (((observed-expected)**2)/expected)
chi_squared_stat


# In[57]:

expected = 9288
chisquare(d.Purchase.mean(),f_exp=expected)


# ###  <font color='GREY'> *T-Test*
# 
# #### Hypothesis : The future Purchase value is near to the mean value
# #### Alternate Hypothesis : The future Purchase value is not near to the mean value

# In[58]:

stats.ttest_1samp(d['Purchase'], 0)   


# ### <font color='grey'> *Conclusion* 
# 
# ### - At the end of our analysis we determine that there is a striking correlation between the predictors and the response variables. 
# ### - We found that the feature ‘Apparels’  were the most sold products amongst all product categories and the age group which purchased the most belongs to 0-17 and 51-55(Females).
# ### - Most Electronics and Furniture were sold in New Jersey followed by Pennsylvania and New York.

# # <font color='red'> *TEAM MEMBERS*
# 
# ## AMOD PANCHAL
# ## HEENA MANSOORI
# ## KRISHNA DHRUV
# ## SAURABH KARAMBALKAR

# In[ ]:



