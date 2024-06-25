import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os

##################################################################################################################################
### read all files ###
##################################################################################################################################

# raw trip data read
tripdf=pd.read_csv('/home/vsri/raw_data/yellow_tripdata_2021-01_raw_updated.csv')
# ratecode_id has mixed data type - to be inspected later

# surcharge json read, and prepped for use
# read json and orient objects from single line, ensure data type is read as is
surdf = pd.read_json('/home/vsri/raw_data/surcharge_data.json', orient='index', convert_axes=False) 
# set index as the new trip_id and reset an index
surdf=surdf.rename_axis('trip_id').reset_index() 

# metadata read 
payment_lookup=pd.read_csv('/home/vsri/solution_data/payment_type.csv')
vendor_lookup=pd.read_csv('/home/vsri/solution_data/vendor_id.csv')
ratecode_lookup=pd.read_csv('/home/vsri/solution_data/ratecode_id.csv')

#####################################################################################################################################
### inspect data ###
#####################################################################################################################################

# #columns #rows
tripdf.shape  # (1369765, 16)
surdf.shape # (1369765, 3)
# #entires match

# headers of columns displayed
tripdf.columns
surdf.columns 

# check data types for all columns
tripdf.dtypes # pickup/dropoff datetime, trip distance and store/fwd flag are objects, rest are int64/float64
surdf.dtypes # trip id is an object, other two are float64

# print first 5 rows of data
tripdf.head()
surdf.head()

# returns statistical summary of numerical columns only
tripdf.describe()
# null values present passenger_count and ratecode_id in tripdf
# >98k rows - not deleting, working around these for analysis
# negative values present in min of $values
surdf.describe()
# negative values present in min of surcharges

# inspect non-numerical columns
non_num_cols=['tpep_pickup_datetime','tpep_dropoff_datetime','store_and_fwd_flag']
print(tripdf[non_num_cols].count())
# store/fwd flag has null values
non_num_cols_surcharge=['trip_id']
print(surdf[non_num_cols_surcharge].count())

# count unique values in each column
unique_counts = tripdf.nunique()
print(unique_counts)
# vendor has 5 unique values - metadata says 2 types only - leaving data for further inspection
# ratecode id has 7 unique values including null value as expected
# store/fwd flag is only showing 2 uniques values - null value still present, dtype is object
unique_counts_surcharge = surdf.nunique()
print(unique_counts_surcharge)
# improvement_surcharge - 3 unique values (Including negative values)
# congestion_surcharge - 5 unique values (Including negative values)

# check for duplicates
tripdf.duplicated().sum()
surdf.duplicated().sum()
# no duplicated rows

# Ascertain that Trip IDs are unique
tripdf.duplicated(subset=['tripId']).sum()
surdf.duplicated(subset=['trip_id']).sum()
# All Trip IDs are unique

###########################################################################################################################
### basic clean of data ###
###########################################################################################################################

# rename columns
tripdf = tripdf.rename(columns={'VendorID': 'vendor_id', 'RatecodeID': 'ratecode_id', 'tripId': 'trip_id'})

# enforce datatypes
# tripdf = tripdf.astype({'vendor_id':int,'passenger_count':int, 'ratecode_id':int, 'payment_type':intt}) - null values present
tripdf = tripdf.astype({'vendor_id':int, 'payment_type':int}) 
surdf = surdf.astype({'trip_id':int}) 
# not all were floats, so enforcing type after clean of column
tripdf['trip_distance'] = tripdf['trip_distance'].replace('km', '', regex=True).astype(float)

# Extracting day from date time objects
# Convert from object to datetime
tripdf['tpep_pickup_datetime']=pd.to_datetime(tripdf['tpep_pickup_datetime'])
tripdf['tpep_dropoff_datetime']=pd.to_datetime(tripdf['tpep_dropoff_datetime'])

# Split datetimes to days
tripdf['pickup_day']=tripdf['tpep_pickup_datetime'].dt.day_name()
tripdf['dropoff_day']=tripdf['tpep_dropoff_datetime'].dt.day_name()

# Classify by payment type, vendor and ratecode
tripdf['paymenttype'] = tripdf['payment_type'].map(payment_lookup.set_index('payment_type')['payment_type_name'])
tripdf['vendor'] = tripdf['vendor_id'].map(vendor_lookup.set_index('vendor_id')['vendor'])
tripdf['ratecode'] = tripdf['ratecode_id'].map(ratecode_lookup.set_index('ratecode_id')['ratecode'])

# Check all cloumns after renaming and additions
tripdf.columns
# #columns=21
# Index(['vendor_id', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance', 'ratecode_id', 
#       'store_and_fwd_flag', 'PULocationID', 'DOLocationID', 'payment_type', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 
#       'tolls_amount', 'trip_id', 'pickup_day', 'dropoff_day', 'paymenttype', 'vendor', 'ratecode'], dtype='object')
surdf.columns
# Index(['trip_id', 'improvement_surcharge', 'congestion_surcharge'], dtype='object')  

# Save files with minimal cleaning and modifications
tripdf.to_csv('tripdf.csv', index=False) # use later
surdf.to_csv('surdf.csv', index=False) # use later     
     
# Joining the data
tdf_taxi = pd.merge(tripdf, surdf, on='trip_id', how='outer') # retain all info, no new rows formed 
# (1369765, 23) - no new trip ids, all matched
tdf_taxi.to_csv('tdf_taxi.csv', index=False)

###########################################################################################################################
### handling null and negative values ###
###########################################################################################################################

# read merged data if necessary
tdf_taxi=pd.read_csv('/home/vsri/solution_data/tdf_taxi.csv')

# check for null values
tdf_taxi.isnull().sum() # passenger_count 98352, ratecode_id 98352, store_and_fwd_flag 98352, vendor 98352, ratecode 98388
null_rows = tdf_taxi[tdf_taxi.isnull().any(axis=1)]  

# ~7% of the data has null values, in 3 specific columns, and other two as a consequence of the vendor and ratecode lookup
# these columns unlikely to be used in analysis directly, further visual insights required before removing 7% of the data 
# rows with null values saved separately for further inspection if required
tdf_taxi_null = null_rows.copy() 
tdf_taxi_null.to_csv('tdf_taxi_null.csv', index=False)

# check for negative values in trip data
num_cols = tdf_taxi.select_dtypes(include=[np.number]) # select all columns with numerical data type
neg_cols = num_cols.columns[(num_cols < 0).any(axis=0)]
# 7 columns ['fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement surchagre', 'congestion_surcharge']
neg_rows = tdf_taxi.loc[(tdf_taxi[neg_cols] < 0).any(axis=1), neg_cols] # (6775, 7)  

# save separately and exclude rows where negative values are present 
tdf_taxi_neg = tdf_taxi.loc[(tdf_taxi[neg_cols] < 0).any(axis=1)]
tdf_taxi_cleaned = tdf_taxi.loc[~(tdf_taxi[neg_cols] < 0).any(axis=1)]

# since negative values are present in less 0.5 percent of the data - removing these rows from further analysis
# upon inspection, most of these are unknown or voided trips, so safe to remove from further analysis

# save negative values for further inspection, save cleaned data fo further analysis
tdf_taxi_neg.to_csv('tdf_taxi_neg.csv', index=False)
tdf_taxi_cleaned.to_csv('tdf_taxi_cleaned.csv', index=False)

#########################################################################################################################
### Advanced Statistical Metrics and Analysis Preparation ###
#########################################################################################################################

# Correlation Analysis
numeric_cols = tdf_taxi_cleaned.select_dtypes(include=[float, int])
correlation_matrix = numeric_cols.corr()
correlation_matrix.to_csv('correlation_matrix.csv')

# Set the matplotlib backend to 'Agg' for file saving
plt.switch_backend('agg')

# Create the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')

# Save the heatmap as a file
plt.savefig('correlation_matrix_heatmap.png')

# tdf_taxi_cleaned exported to Tableau  for advanced analysis and visualisation
# Outliers cleaned in Tableau



