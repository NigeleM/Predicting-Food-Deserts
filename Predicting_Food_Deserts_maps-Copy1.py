#!/usr/bin/env python
# coding: utf-8

# # Predicting the Emergence of Food Deserts 
# 
# - Nigele X McCoy (nmccoy9@gatech.edu), Anthony Philip Lee (alee657@gatech.edu) 

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from matplotlib.colors import ListedColormap
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# ##  Data Preparation
# 
# - **Crime Data** 

# In[2]:


## Crime 


# In[3]:


crimeData = pd.read_csv('crime_data_w_population_and_crime_rate.csv')
crimeData['county_name'] = crimeData['county_name'].str.replace('County','')
crimeData['county_name'] = crimeData['county_name'].str.replace(' ','')
crimeData.head(5)


# In[4]:


# Crime describe


# In[5]:


crimeData.describe()


# In[6]:


# Read 2015 Food Environment Atlas Data Download


# In[7]:


FoodEnvironAtlas2015 = pd.ExcelFile('2015 Food Environment Atlas Data Download.xls')


# In[8]:


FoodAtlasDict = {tabName: pd.read_excel('2015 Food Environment Atlas Data Download.xls',sheet_name=tabName) for tabName in FoodEnvironAtlas2015.sheet_names}
print(FoodAtlasDict.keys())


# In[9]:


FoodAtlasDict['ACCESS'].insert(1,'county_name','',True)
FoodAtlasDict['ACCESS']['county_name'] = FoodAtlasDict['ACCESS']['County'] + ',' + FoodAtlasDict['ACCESS']['State']
FoodAtlasDict['ACCESS'] = FoodAtlasDict['ACCESS'].drop(columns=['State','County'])
FoodAtlasDict['ACCESS'].head(5)


# In[10]:


FoodAtlasDict['SOCIOECONOMIC'].insert(1,'county_name','',True)
FoodAtlasDict['SOCIOECONOMIC']['county_name'] = FoodAtlasDict['SOCIOECONOMIC']['County'] + ',' + FoodAtlasDict['SOCIOECONOMIC']['State']
FoodAtlasDict['SOCIOECONOMIC'] = FoodAtlasDict['SOCIOECONOMIC'].drop(columns=['State','County'])
FoodAtlasDict['SOCIOECONOMIC'].head(5)


# In[11]:


FoodAtlasDict['HEALTH'].insert(1,'county_name','',True)
FoodAtlasDict['HEALTH']['county_name'] = FoodAtlasDict['HEALTH']['County'] + ',' + FoodAtlasDict['HEALTH']['State']
FoodAtlasDict['HEALTH'] = FoodAtlasDict['HEALTH'].drop(columns=['State','County','FIPS'])
FoodAtlasDict['HEALTH'].head(5)


# In[12]:


FoodAtlasDict['INSECURITY'].insert(1,'county_name','',True)
FoodAtlasDict['INSECURITY']['county_name'] = FoodAtlasDict['INSECURITY']['County'] + ',' + FoodAtlasDict['INSECURITY']['State']
FoodAtlasDict['INSECURITY'] = FoodAtlasDict['INSECURITY'].drop(columns=['State','County','FIPS'])
FoodAtlasDict['INSECURITY'].head(5)


# In[13]:


crimeDict = {
    "AGASSLT": "AGGRAVATED ASSAULTS (04)",
    "SAGASSLT": "AGGRAVATED ASSAULTS (04)",
    "ALLOTHR": "ALL OTHER OFF EXCEPT TRAFFIC (26)",
    "SALLOTHR": "ALL OTHER OFF EXCEPT TRAFFIC (26)",
    "ARSON": "ARSONS (09)",
    "SARSON": "ARSONS (09)",
    "BOOKMKG": "BOOKMAKING, HORSE AND SPORT (19A)",
    "SBOOKMKG": "BOOKMAKING, HORSE & SPORT (19A)",
    "BURGLRY": "BURGLARIES (05)",
    "SBURGLRY": "BURGLARIES (05)",
    "CPOPCRIM": "COUNTY POPULATION-AGENCIES REPORT CRIMES",
    "COVIND": "COVERAGE INDICATOR",
    "CURFEW": "CURFEW, LOITERING VIOL: JUV (28)",
    "SCURFEW": "CURFEW, LOITERING VIOL: JUV (28)",
    "DISORDR": "DISORDERLY CONDUCT (24)",
    "SDISORDR": "DISORDERLY CONDUCT (24)",
    "DUI": "DRIVING UNDER INFLUENCE (21)",
    "SDUI": "DRIVING UNDER INFLUENCE (21)",
    "DRUGTOT": "DRUG ABUSE VIOLATIONS-TOTAL (18)",
    "SDRUGTOT": "DRUG ABUSE VIOLATIONS-TOTAL (18)",
    "DRGSALE": "DRUG ABUSE-SALE/MANUFACTURE (180)",
    "SDRGSALE": "DRUG ABUSE-SALE/MANUFACTURE (180)",
    "DRGPOSS": "DRUG POSSESSION-SUBTOTAL (185)",
    "SDRGPOSS": "DRUG POSSESSION-SUBTOTAL (185)",
    "DRUNK": "DRUNKENNESS (23)",
    "SDRUNK": "DRUNKENNESS (23)",
    "EMBEZL": "EMBEZZLEMENT (12)",
    "SEMBEZL": "EMBEZZLEMENT (12)",
    "FIPS_CTY": "FIPS COUNTY CODE",
    "FIPS_ST": "FIPS STATE CODE",
    "FRGYCNT": "FORGERY AND COUNTERFEITING (10)",
    "SFRGYCNT": "FORGERY & COUNTERFEITING (10)",
    "FRAUD": "FRAUD (11)",
    "SFRAUD": "FRAUD (11)",
    "OTGAMBL": "GAMBLING-ALL OTHER (19C)",
    "SOTGAMBL": "GAMBLING-ALL OTHER (19C)",
    "GAMBLE": "GAMBLING-TOTAL (19)",
    "SGAMBLE": "GAMBLING-TOTAL (19)",
    "GRNDTOT": "GRAND TOTAL",
    "SGRNDTOT": "GRAND TOTAL",
    "STLNPRP": "HAVE STOLEN PROPERTY (13)",
    "SSTLNPRP": "HAVE STOLEN PROPERTY (13)",
    "EDITION": "ICPSR EDITION NUMBER",
    "PART": "ICPSR PART NUMBER",
    "IDNO": "ICPSR SEQUENTIAL CASE ID NUMBER",
    "STUDYNO": "ICPSR STUDY NUMBER",
    "LARCENY": "LARCENIES (06)",
    "SLARCENY": "LARCENIES (06)",
    "LIQUOR": "LIQUOR LAW VIOLATIONS (22)",
    "SLIQUOR": "LIQUOR LAW VIOLATIONS (22)",
    "MJPOSS": "MARIJUANA-POSSESSION (18F)",
    "SMJPOSS": "MARIJUANA-POSSESSION (18F)",
    "MJSALE": "MARIJUANA-SALE/MANUFACTURE (18B)",
    "SMJSALE": "MARIJUANA-SALE/MANUFACTURE (18B)",
    "MVTHEFT": "MOTOR VEHICLE THEFTS (07)",
    "SMVTHEFT": "MOTOR VEHICLE THEFTS (07)",
    "JURFLAG": "MULTI-COUNTY JURISDICTION FLAG",
    "MURDER": "MURDERS (011)",
    "SMURDER": "MURDERS (011)",
    "AG_ARRST": "NMBR OF AGENCIES IN COUNTY REPORT ARRSTS",
    "AG_OFF": "NMBR OF AGENCIES IN COUNTY REPORT CRIMES",
    "SNUMBERS": "NUMBERS & LOTTERY (19B)",
    "NUMBERS": "NUMBERS AND LOTTERY (19B)",
    "SOFAGFAM": "OFFENSES AGAINST FAMILY & CHILD (20)",
    "OFAGFAM": "OFFENSES AGAINST FAMILY/CHILD (20)",
    "COCPOSS": "OPIUM/COCAINE-POSSESSION (18E)",
    "SCOCPOSS": "OPIUM/COCAINE-POSSESSION (18E)",
    "COCSALE": "OPIUM/COCAINE-SALE/MANUFACTURE (18A)",
    "SCOCSALE": "OPIUM/COCAINE-SALE/MANUFACTURE (18A)",
    "OTHASLT": "OTHER ASSAULTS (08)",
    "SOTHASLT": "OTHER ASSAULTS (08)",
    "OTHPOSS": "OTHER DRUG-POSSESSION (18H)",
    "SOTHPOSS": "OTHER DRUG-POSSESSION (18H)",
    "OTHSALE": "OTHER:DANGEROUS NON-NARCOTICS (18D)",
    "SOTHSALE": "OTHER:DANGEROUS NON-NARCOTICS (18D)",
    "P1PRPTY": "PART 1-PROPERTY CRIMES",
    "SP1PRPTY": "PART 1-PROPERTY CRIMES",
    "P1TOT": "PART 1-TOTAL",
    "SP1TOT": "PART 1-TOTAL",
    "P1VLNT": "PART 1-VIOLENT CRIMES",
    "SP1VLNT": "PART 1-VIOLENT CRIMES",
    "SCOMVICE": "PROSTITUTION & COMMERC VICE (16)",
    "COMVICE": "PROSTITUTION AND COMMERC VICE (16)",
    "RAPE": "RAPES (02)",
    "SRAPE": "RAPES (02)",
    "ROBBERY": "ROBBERIES (03)",
    "SROBBERY": "ROBBERIES (03)",
    "RUNAWAY": "RUNAWAYS: JUVENILES (29)",
    "SRUNAWAY": "RUNAWAYS: JUVENILES (29)",
    "SEXOFF": "SEX OFFENSES (17)",
    "SSEXOFF": "SEX OFFENSES (17)",
    "SPOPARST": "STATE POPULATION",
    "SPOPCRIM": "STATE POPULATION-AGENCIES RPRT CRIMES",
    "SUSPICN": "SUSPICION (27)",
    "SSUSPICN": "SUSPICION (27)",
    "SYNSALE": "SYNTHETIC DRUG-SALE/MANUFACTURE (18C)",
    "SSYNSALE": "SYNTHETIC DRUG-SALE/MANUFACTURE (18C)",
    "SYNPOSS": "SYNTHETIC NARCOTICS-POSSESSION (18G)",
    "SSYNPOSS": "SYNTHETIC NARCOTICS-POSSESSION (18G)",
    "CPOPARST": "TOT CNTY POPULATION-AGENCIES RPRT ARRSTS",
    "PROPERTY": "TOTAL PROPERTY CRIMES",
    "SPROPERTY": "TOTAL PROPERTY CRIMES",
    "VIOL": "TOTAL VIOLENT CRIMES",
    "SVIOL": "TOTAL VIOLENT CRIMES"
}


# In[ ]:





# In[14]:


# https://www.icpsr.umich.edu/web/ICPSR/studies/37059/variables
# https://www.kaggle.com/datasets/mikejohnsonjr/united-states-crime-rates-by-county/data


# ## Food Environment Atlas & Crime Data
# 
# - Crime Data
# - Food Environment Atlas

# In[15]:


SocEconCrime = pd.merge(crimeData,FoodAtlasDict['SOCIOECONOMIC'],on='county_name')


# In[16]:


SocEconCrime.head(5)


# In[17]:


# see crime dict for meaning or follow link # https://www.kaggle.com/datasets/mikejohnsonjr/united-states-crime-rates-by-county/data
SocEconCrime.columns


# In[18]:


SocEconCrime.shape


# In[ ]:





# In[19]:


# MODINDX = BURGLRY +	LARCENY +	MVTHEFT
# INDEX = MURDER	+ RAPE +	ROBBERY +	AGASSLT
SocEconCrime = SocEconCrime.drop(columns=['index'	,'EDITION',	'PART','IDNO'])


# In[20]:


SocEconCrime.to_excel('test.xlsx',index=False)
countyName = SocEconCrime['county_name']
SocEconCrime = SocEconCrime.drop(columns='county_name')
SocEconCrime = SocEconCrime.replace('<Null>', '0')
SocEconCrime = SocEconCrime.astype(float)
SocEconCrime = pd.concat([countyName, SocEconCrime], axis=1)
SocEconCrime['Food Desert'] = (SocEconCrime['POVRATE10'] >= 30).astype(int) 


# In[21]:


SocEconCrime.head(10)


# In[22]:


# 30 as threshold
SocEconCrime['Food Desert'].value_counts()


# # MODINDX and INDEX Analysis
# 
# - **MODINDX = BURGLRY +	LARCENY +	MVTHEFT**
# - **INDEX = MURDER	+ RAPE +	ROBBERY +	AGASSLT**
# - 

# In[23]:


modindx_indx = SocEconCrime[['county_name','INDEX',	'MODINDX','population','Food Desert']].copy()
modindx_indx['MODINDX_population'] = modindx_indx['MODINDX'] / modindx_indx['population']
modindx_indx['INDEX_population'] = modindx_indx['INDEX'] / modindx_indx['population']

modindx_indx.head(5)


# In[24]:


modindx_indx.describe().T


# In[25]:


modindx_indx.sort_values(by='MODINDX_population',ascending=False).head(15)


# In[26]:


modindx_indx.sort_values(by='INDEX_population',ascending=False).head(15)


# In[27]:


modindx_indx.groupby(['Food Desert','MODINDX_population'])['Food Desert'].count()


# In[28]:


modindx_indx[modindx_indx['Food Desert'] == 1].groupby('MODINDX_population').count()


# In[29]:


# Playing around with a potential example

# Step 1: Filter rows where 'Food Desert' == 1
filtered_data = modindx_indx[modindx_indx['Food Desert'] == 1]

# Step 2: Extract the 'MODINDX_population' column
modindx_population = filtered_data['MODINDX_population']

# Step 3: Plot the distribution
plt.figure(figsize=(10, 6))

# Histogram with KDE overlay
sns.histplot(modindx_population, kde=True, bins=30, color='green')

# Add labels and title
plt.title('Distribution of MODINDX_population (Food Desert = 1)', fontsize=16)
plt.xlabel('MODINDX_population', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# Display the plot
plt.show()


# In[30]:


target = 'Food Desert'
features = SocEconCrime.drop(columns=['county_name', target])

X_train, X_test, y_train, y_test = train_test_split(features, SocEconCrime[target], test_size=0.3, random_state=42)

# Logistic Regression Model
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)

# Report
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[31]:


# KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
y_pred_knn = knn.predict(X_test)

# Report
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))


# In[32]:


target = 'Food Desert'
features = SocEconCrime.drop(columns=['county_name', target])


X_train, X_test, y_train, y_test = train_test_split(features, SocEconCrime[target], test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM classifier
svm_model = SVC(kernel='rbf', random_state=42) 
svm_model.fit(X_train_scaled, y_train)

# 
y_pred = svm_model.predict(X_test_scaled)

# Report
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[33]:


import pandas as pd


# In[34]:


FoodResearchAtlas2015 = pd.ExcelFile('2015 Food Environment Atlas Data Download.xls')
FoodResearchAtlas2015.sheet_names


# In[35]:


# Food Access Data
food_atlas_access_df=FoodResearchAtlas2015.parse('ACCESS')

# Socio Economic Data
food_atlas_socioeconomic_df=FoodResearchAtlas2015.parse('SOCIOECONOMIC')
food_atlas_socioeconomic_df=food_atlas_socioeconomic_df.drop(['State', 'County'], axis=1)

# Population Data - Choose only 2010
food_atlas_population_df=FoodResearchAtlas2015.parse('Supplemental Data - County')
food_atlas_population_df=food_atlas_population_df[['FIPS Code','State', 'County name', '2010 Census population']]
food_atlas_population_df.rename(columns={'FIPS Code': 'FIPS'}, inplace=True)
food_atlas_population_df=food_atlas_population_df.drop(['State', 'County name'], axis=1)

# Merge the data frames
food_atlas_df = pd.merge(food_atlas_access_df, food_atlas_socioeconomic_df, on='FIPS')
food_atlas_df = pd.merge(food_atlas_df, food_atlas_population_df, on='FIPS')

food_atlas_df.head()


# In[36]:


food_atlas_access_df.columns


# In[37]:


FOOD_DESERT_ACCCESS_THRESHOLD=30
# Add a LACCESS_POP10_FLAG column and set to 1 if PCT_LACCESS_POP10 > 0.3
food_atlas_df['LACCESS_POP10_FLAG'] = food_atlas_df['PCT_LACCESS_POP10'].apply(lambda x: 1 if x > FOOD_DESERT_ACCCESS_THRESHOLD else 0)
food_atlas_df['LACCESS_LOWI10_FLAG'] = food_atlas_df['PCT_LACCESS_LOWI10'].apply(lambda x: 1 if x > FOOD_DESERT_ACCCESS_THRESHOLD else 0)
food_atlas_df['LACCESS_CHILD10_FLAG'] = food_atlas_df['PCT_LACCESS_CHILD10'].apply(lambda x: 1 if x > FOOD_DESERT_ACCCESS_THRESHOLD else 0)
food_atlas_df['LACCESS_SENIORS10_FLAG'] = food_atlas_df['PCT_LACCESS_SENIORS10'].apply(lambda x: 1 if x > FOOD_DESERT_ACCCESS_THRESHOLD else 0)
food_atlas_df['LACCESS_HHNV10_FLAG'] = food_atlas_df['PCT_LACCESS_HHNV10'].apply(lambda x: 1 if x > FOOD_DESERT_ACCCESS_THRESHOLD else 0)

# Replace all '<Null>' with 0
food_atlas_df = food_atlas_df.replace('<Null>', 0)

# Show all the column types
food_atlas_df.dtypes


# In[38]:


# Load crime data
crime_df = pd.read_csv('crime_data_w_population_and_crime_rate.csv')

if 'index' in crime_df.columns:
    crime_df = crime_df.drop(['index', 'EDITION', 'PART', 'population', 'FIPS_ST', 'FIPS_CTY', 'IDNO', 'COVIND', 'INDEX', 'MODINDX'], axis=1)

if 'county_name' in crime_df.columns:
    # Split county and state from county_name
    crime_df['County'] = crime_df['county_name'].apply(lambda x: x.split(',')[0])
    crime_df['State'] = crime_df['county_name'].apply(lambda x: x.split(',')[1])

    # Drop the word 'County' from the county name
    crime_df['County'] = crime_df['County'].apply(lambda x: x.replace(' County', ''))
    crime_df['County'] = crime_df['County'].apply(lambda x: x.replace(' city', ''))

    # Strip leading and trailing spaces
    crime_df['County'] = crime_df['County'].apply(lambda x: x.strip())
    crime_df['State'] = crime_df['State'].apply(lambda x: x.strip())

    crime_df = crime_df.drop(['county_name'], axis=1)

food_atlas_crime_df = pd.merge(food_atlas_df, crime_df, on=['State', 'County'])
food_atlas_crime_df = food_atlas_crime_df.dropna()
food_atlas_crime_df.to_excel('foodATLAS.xlsx',index=False)


# In[39]:


food_atlas_crime_df.head(10)


# In[40]:


# Split the data into training and test data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = food_atlas_crime_df
# Scale the data
X = X.drop(['LACCESS_POP10', 'PCT_LACCESS_POP10', 'LACCESS_POP10_FLAG', 
            'LACCESS_LOWI10', 'PCT_LACCESS_LOWI10', 'LACCESS_LOWI10_FLAG', 
            'LACCESS_CHILD10', 'PCT_LACCESS_CHILD10', 'LACCESS_CHILD10_FLAG', 
            'LACCESS_SENIORS10', 'PCT_LACCESS_SENIORS10', 'LACCESS_SENIORS10_FLAG', 
            'LACCESS_HHNV10', 'PCT_LACCESS_HHNV10', 'LACCESS_HHNV10_FLAG', 
            'FIPS', 'State', 'County', ], axis=1)
            #'CPOPCRIM', 'CPOPARST', 'crime_rate_per_100000', 'PCT_NHBLACK10', 'PCT_HISP10'], axis=1)
column_names = X.columns
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = food_atlas_crime_df['LACCESS_POP10_FLAG']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Convert X_rain to data frames
X_train_df = pd.DataFrame(X_train, columns=column_names)
X_train_df.head(25)


# In[41]:


# Perform logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
accuracy_score(y_test, y_pred)


# In[42]:


# Display coefficients
coefficients = pd.DataFrame(logreg.coef_[0], column_names, columns=['Coefficient'])
print(coefficients.sort_values(by='Coefficient', ascending=False))

coefficients = pd.DataFrame(logreg.coef_[0], column_names, columns=['Standardized Coefficient'])
print(coefficients.sort_values(by='Standardized Coefficient', ascending=False))


# In[43]:


import statsmodels.api as sm

# Add intercept term
X_train_sm = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit(maxiter=1000)

# Print summary with p-values
print(result.summary())


# In[44]:


print(column_names)


# In[45]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
data = X_train_df
# data = data.drop(['CPOPCRIM', 'CPOPARST', 'crime_rate_per_100000', 'PCT_NHBLACK10', 'PCT_HISP10' ], axis=1)
vif = [variance_inflation_factor(data, i) for i in range(data.shape[1])]

# Display VIF values
vif_df = pd.DataFrame({'Variable': data.columns, 'VIF': vif})
print(vif_df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[46]:


import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Train the SVM model

svm_model = SVC(kernel='linear', random_state=42)  # Linear kernel for coefficients
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Get the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Extract coefficients (feature importance)
coefficients = svm_model.coef_.flatten()

# Rank features by importance
feature_importance = pd.DataFrame({
    'Feature': column_names,
    'Coefficient': coefficients,
    'Importance': abs(coefficients)  # Absolute value for importance
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)


# ## County Breakdown

# In[47]:


# Correctly define the path to the GeoJSON file
geojson_path = 'geojson-counties-fips.json'
from folium.features import GeoJsonTooltip
import folium
from IPython.display import display, IFrame


# Create the map
m = folium.Map(location=[37.8, -96.9], zoom_start=5)

# custom_scale = food_atlas_crime_df['PCT_LACCESS_POP10'].quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
# print(custom_scale)
custom_scale = [0.0, 30.0, 50.0, 75.0, 100.01]
# Add GeoJSON to the map with LACCESS_POP10_FLAG as the color field and county names
choropleth = folium.Choropleth(
    geo_data=geojson_path,               # GeoJSON file path
    name="choropleth",
    data=food_atlas_crime_df,               # Data containing LACCESS_POP10_FLAG
    columns=["FIPS", "PCT_LACCESS_POP10"],  # Columns to match FIPS and value
    # columns=["FIPS", "LACCESS_POP10_FLAG"],  # Columns to match FIPS and value
    key_on="feature.id",                # Key in GeoJSON to match FIPS
    fill_color="YlOrRd",                # Color scale
    # fill_color='PuBu',                # Color scale
    threshold_scale=custom_scale,           # Custom color scale
    fill_opacity=0.7,
    line_opacity=0.2,
    # legend_name="Percent of Population with Low Food Access",
    legend_name="Food Desert Indicator"
).add_to(m)

# Add county names to the choropleth as tooltips
choropleth.geojson.add_child(
    folium.features.GeoJsonTooltip(
        fields=["NAME"],  # Include county name and food desert flag
        aliases=["County:"],  # Label for each field
        localize=True
    )
)

# Add layer control to toggle layers
folium.LayerControl().add_to(m)



# Save and display the map
map_file_path = "Food_Desert_Map.html"
m.save(map_file_path)
map_file_path

# Display the map in the notebook
display(IFrame(map_file_path, width=800, height=600))


# In[ ]:




