
# coding: utf-8

# # Basic import

# In[2]:


import pandas as pd


# # Preprocess Data

# In[3]:


def pre_process_data(data):
    # Make the necessary imports
    import datetime
    import pandas as pd
    import numpy as np
    from statistics import mean
    
    # Only select the data column for parsing
    data = data['data']
    
    
    x_data = []
    y_data = []
    z_data = []
    time_frame = []
    device_data = []
    battery_level = []
    device_battery = {}
    
    for item in data:
        decoded_item = item[2:-5]
        item_value = decoded_item.split(',')
        #print(item_value)
        
        # Take the device ID information
        device_id = item_value[0]
        
        # The time information is in index 3 of the item_value list
        time_ar = item_value[3]
        times = datetime.datetime.strptime(time_ar, "%Y-%m-%dT%H:%M:%SZ")
        
        # Need to find out why we are adding timedelta of 6 hours. Is it because Dhaka is GMT+6?
        # --> They sent date as UTC, so we've converted it into BST
        time_val = times + datetime.timedelta(hours=6) # why the timedelta? --> using timedelta we can get the time we want and also we can convert the time into GMT+6 but using timedelta we directly converted it into our time format. 
        
        # Take the battery information from index position 2 of the item_value list
        # and convert it from 2-bit to 16-bit integer.
        # --> axis data and battery information was sent as 16-bit integer, now we converted it into decimal
        battery = np.int16(int(item_value[2], 16))
        
        # Take the x, y, z information from index position 1 of the item_value list 
        # and convert it from 2-bit to 16-bit integer
        # x values between string index 0:4 for item_value[1]
        x = (np.int16(int(item_value[1][0:4], 16)))/256 
        # y values between string index 4:8 for item_value[1]
        y = (np.int16(int(item_value[1][4:8], 16)))/256 
        # x values between string index 8:12 for item_value[1]
        z = (np.int16(int(item_value[1][8:], 16)))/256 
        
        # Append the battery, x, y, z, time and device information to the previously initialized lists
        battery_level.append(battery)
        x_data.append(x)
        y_data.append(y)
        z_data.append(z)
        time_frame.append(time_val)
        device_data.append(device_id)
        
    
    # Return all the lists together in a dataframe
    #return x_data, y_data, z_data, time_frame, device_data
    # Create new dataframe df using the new list variables
    df = pd.DataFrame(list(zip(device_data, time_frame, x_data, y_data, z_data)), 
                   columns =['device_id', 'time', 'x', 'y', 'z']) 
    
    return df


# # Remove duplicates

# In[4]:


def remove_duplicates(dataframe, duplicate_row_to_keep = "first", column_to_subset_by="time"):
   
    # Resets the dataframe index, as a precaution. I don't like sudden jumps in index values
    # converts time column to datetime object and sorts values in ascending order (chronologically)
    # Removes the duplicates.
    # 'duplicate_row_to_keep' may either be "first" or "last". The default is set to "first".
    # 'column_to_subset_by' will be the column with the timestamp values. The default is set to "time".

    
    # Make the necessary imports
    import pandas as pd
    
    #convert time column from str to datetime type
    dataframe['time'] = pd.to_datetime(dataframe['time'])
    dataframe = dataframe.sort_values(by=['time'])
    
    # remove duplicates
    dataframe = dataframe.drop_duplicates(subset=column_to_subset_by, keep=duplicate_row_to_keep)
    
    # reset index
    dataframe.reset_index(drop=True, inplace=True)
    
    # return the duplicate removed dataframe
    return dataframe


# # Smoothing noise

# In[5]:


def smooth_noise(dataframe, window_size=5):
    # smoothes noise by running a rolling mean and drops the null columns
    # window_size is set to 5 as default
    
    # make necessary imports 
    import pandas as pd
    
    dataframe['x'] = dataframe['x'].rolling(window=window_size).mean()
    dataframe['y'] = dataframe['y'].rolling(window=window_size).mean()
    dataframe['z'] = dataframe['z'].rolling(window=window_size).mean()
    
    # drop rows with null values for the smoothened data
    dataframe = dataframe.dropna(axis=0)
    
    return dataframe


# # Calculating the x_diff, y_diff, z_diff and sum_diff

# In[6]:


def calculate_differential_values(dataframe):
    # renames x, y, z columns to add _diff with their names
    # calculates their differential values
    # calculates the sum of the (absolute) differential values
    
    # make necessary imports
    import pandas as pd
    
    # rename columns    
    dataframe = dataframe.rename(columns={"x":"x_diff", "y": "y_diff", "z": "z_diff"})
    
    # turn time to index
    dataframe = dataframe.set_index('time')
    
    # now calculate the differences between consecutive rows
    dataframe = dataframe.diff(axis=0, periods=1)
    
    # now drop rows with na values
    dataframe = dataframe.dropna()
    
    # now calculate the sum_diff
    dataframe['sum_diff'] = abs(dataframe['x_diff']) + abs(dataframe['y_diff']) + abs(dataframe['z_diff'])
    
    # return the changed dataframe
    return dataframe


# # K-means

# In[7]:


def clustering_function(km_df, number_of_clusters=3, init = "random", n_init=20):
    # necessary imports
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # as a precaution, in case there is timestamp as index, we should reset it 
    # so we can delete it as a column in the next try block
    try:
        # drop the time column 
        km_df.reset_index(drop=True, inplace=True) 
    except:
        pass
    
    # as a precaution, in case there is a timestamp column, we need to drop it
    try:
        # drop the time column 
        km_df = km_df.drop(columns = ['time']) 
    except:
        pass
    
    # separate the data from the dataframe and convert to np array
    X = km_df.values
    X = np.nan_to_num(X)
    
    # scale the data
    # try two different heuristics: the minmax scaler and the standard scaler 
    # and see which works better
    # my hunch is the standard scaler should work better since the variables may have covariance
    Clus_dataSet = StandardScaler().fit_transform(X)
    Clus_dataSet
    
    # Initialize the number of clusters. My personal recommendation is 2.
    clusterNum = number_of_clusters
    
    ## These are the parameters you can choose for initializing the Kmeans class
    # init : {‘k-means++’, ‘random’ or an ndarray}
    # Method for initialization, defaults to ‘k-means++’:
    # ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. 
    # See section Notes in k_init for more details.
    # ‘random’: choose k observations (rows) at random from data for the initial centroids.
    # If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
    # n_init : int, default: 10
    # Number of time the k-means algorithm will be run with different centroid seeds. 
    # The final results will be the best output of n_init consecutive runs in terms of inertia.
    # max_iter : int, default: 300
    
    # initialize the kmeans model. 
    # for the time being, just tune the init, n_clusters and n_init parameters
    # we'll find out more about the optimal number of clusters from the elbow method later
    k_means = KMeans(init = init, n_clusters = clusterNum, n_init = n_init)
    
    # fit the model with the data
    k_means.fit(X)
    
    # separate the labels
    labels = k_means.labels_
    
    # add the labels into a new column to the original dataframe
    # we need a labelled dataset to train the classifier model
    km_df["Clus_km"] = labels
    
    # return the labeled dataframe
    return km_df


# # Train SVC

# In[8]:


def train_and_export_svc(svm_df, output_filename="/svm_for_cow.pkl"):
    # This function has no return value
    # PLEASE LET ME KNOW IF THIS FUNCTION SHOULD RETURN TRAINED MODEL
    # INSTEAD OF EXPORTING MODEL TO A FILEPATH
    
    # importing the libraries
    import pandas as pd
    import numpy as np
    from sklearn.externals import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    import os
    
    
    # To divide the data into attributes and labels, execute the following code:
    X = svm_df.drop(columns=['Clus_km'])
    y = svm_df['Clus_km']
    
    # divide data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 101)
    
    # initialize the classifier
    # use gridsearch on a test dataset to find the best parameters (see the script file)
    svclassifier = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
                      decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf',
                      max_iter=-1, probability=False, random_state=None, shrinking=True,
                      tol=0.001, verbose=False)
    
    # fit the classifier into the data
    model = svclassifier.fit(X_train, y_train)
    
    # now predict on the test set using the predict method
    y_pred = model.predict(X_test)
    
    # save the svm model
    path = os.getcwd()
    joblib.dump(model, path + output_filename)
    
    
    


# # Predict using the SVC model

# In[9]:


def predict_using_svc(cow, classifier_filepath="svm_for_cow.pkl"):
    # Necessary imports
    import datetime
    import pandas as pd
    from sklearn.externals import joblib
    
    # rename columns
    cow.rename(columns={ "x": "x_diff" }, inplace = True)
    cow.rename(columns={ "y": "y_diff" }, inplace = True)
    cow.rename(columns={ "z": "z_diff" }, inplace = True)

    # turn time to index
    cow = cow.set_index('time')

    # now calculate the differences between consecutive rows
    cow = cow.diff(axis=0, periods=1)

    # now calculate the sum_diff
    cow['sum_diff'] = abs(cow['x_diff']) + abs(cow['y_diff']) + abs(cow['z_diff'])

    # now drop rows with na values
    cow = cow.dropna()
    
    # Load model from file
    classifier = joblib.load(classifier_filepath)
    
    # now predict on the test set using the predict method
    y_pred = classifier.predict(cow)
    
    # add labels to the dataframe
    labels = y_pred
    cow['labels'] = labels
    
    # returned the labeled dataframe
    return cow


# # Tabulate weights

# In[10]:


def calculate_weights(cow):
    # imports
    import pandas as pd
    from datetime import date
    from datetime import time
    from datetime import datetime
    from datetime import timedelta
    
    # initialize the weight column
    cow['weight'] = None
    
    # tabulate the values in the weight column
    # assumption is cluster 0 is low activity, cluster 1 is medium activity, cluster 2 is high activity
    for i in range(len(cow)):
        if (cow.loc[i, 'time']==0):
            cow.loc[i, 'weight']= 0
        elif (cow.loc[i, 'time']==1):
            cow.loc[i, 'weight']= 0.1
        else:
            cow.loc[i, 'weight']= 0.9
            
    # return the weight added dataframe
    return cow


# # Calculating Activity Level in 1 hour time slices

# In[11]:


def calculate_activity_level(cow):
    # imports 
    import pandas as pd
    from datetime import date
    from datetime import time
    from datetime import datetime
    from datetime import timedelta
    
    # convert time column from str to datetime type
    # sort in ascending order (chronologically)
    cow['time'] = pd.to_datetime(cow['time'])
    cow = cow.sort_values(by=['time'])
    
    # find the initial time value
    starting_time = cow.loc[0, 'time']
    
    # initialize the lists that are to be appended in the loop
    time = []
    activity_level = []
    
    # create one hour time slices
    end_time = starting_time
    while (end_time <= cow.iloc[len(cow)-1, 0]):
        # create the 1-hour time range for the data to be filtered in
        new_time = end_time
        end_time = new_time + timedelta(hours=1) # one hour slice
        
        # create date filter mask
        # greater than the start date and smaller than the end date
        # hold the data from the 1-hour timeslice in a placeholder dataframe
        mask = (cow['time'] > new_time) & (cow['time'] <= end_time)
        placeholder = cow[mask]
        
        # summarise the value counts for the labels in the placeholder dataframe
        summary = pd.DataFrame(placeholder['labels'].value_counts())
        summary['cluster'] = summary.index
        
        # rename columns
        summary.rename(columns={ summary.columns[0]: "count" }, inplace = True)

        # add weights column
        # tabulate the values in the weight column
        # IMPORTANT: assumption is cluster 0 is low activity, cluster 1 is medium activity, cluster 2 is high activity
        summary['weight'] = None
        for i in range(len(summary)):
            if (summary.iloc[i, 1]==0):
                summary.loc[i, 'weight']=0
            elif (summary.iloc[i, 1]==1):
                summary.loc[i, 'weight']=0.1
            else:
                summary.loc[i, 'weight']=0.9

        
        # initialise the hourly activity level as an empty list
        hourly_activity_level = []

        
        # keep appending the product of weight and value count for each cluster label to the hourly_activity_level list
        for i in range(len(summary)):
            activity_level.append(summary.iloc[i, 0]*summary.iloc[i, 2])

        # calculate the sum of the elements in the hourly_activity_level list
        hourly_activity_level = sum(hourly_activity_level)
    
        # append the time and sum of hourly activity level to the time and activity_level lists
        time.append(new_time)
        activity_level.append(hourly_activity_level)

        # create a dataframe using the time and activity level lists
        activity_df = pd.DataFrame(list(zip(time, activity_level)), 
                                   columns =['time', 'activity_level'])
        
        # convert the time column to datetime object
        activity_df['time'] = pd.to_datetime(activity_df['time'])
        
        # initialise the activity level columns for the previous 1, 24, 48, and 72 hours
        activity_df['activity_level_1'] = None
        activity_df['activity_level_24'] = None
        activity_df['activity_level_48'] = None
        activity_df['activity_level_72'] = None
        
        for i in range(len(activity_df)):

            timevalue_1 = activity_df.loc[i, 'time'] - timedelta(hours=1)
            timevalue_24 = activity_df.loc[i, 'time'] - timedelta(hours=24)
            timevalue_48 = activity_df.loc[i, 'time'] - timedelta(hours=48)
            timevalue_72 = activity_df.loc[i, 'time'] - timedelta(hours=72)

            # some errors arise due to duplicate time values being present in the data
            # trying to cheat my way through with the use of try except block
            try:  
                activity_df.loc[i, 'activity_level_1'] = activity_df.loc[activity_df['time']==timevalue_1]['activity_level'].values[0]
                activity_df.loc[i, 'activity_level_24'] = activity_df.loc[activity_df['time']==timevalue_24]['activity_level'].values[0]
                activity_df.loc[i, 'activity_level_48'] = activity_df.loc[activity_df['time']==timevalue_48]['activity_level'].values[0]
                activity_df.loc[i, 'activity_level_72'] = activity_df.loc[activity_df['time']==timevalue_72]['activity_level'].values[0]
            except:
                pass
            
    
    # return the activity level dataframe
    return activity_df


# # Calculate the Activity Index

# In[12]:


def calculate_activity_index(activity_df):
    # imports
    import pandas as pd
    from datetime import date
    from datetime import time
    from datetime import datetime
    from datetime import timedelta
    
    # initialize the historical comparison value, trend and activity index columns
    activity_df['historical_comparison_value'] = None
    activity_df['trend'] = None
    activity_df['activity_index'] = None
    
    # calculate the historical comparison value, trend and activity index columns
    for i in range(len(activity_df)):

        # calculate the historical comparison value
        try:
            placeholder = 3*activity_df.loc[i, 'activity_level']
            placeholder = placeholder - (activity_df.loc[i, 'activity_level_24'] + activity_df.loc[i, 'activity_level_48'] + activity_df.loc[i, 'activity_level_72'])
            historical_comparison_value = placeholder/(activity_df.loc[i, 'activity_level_24'] + activity_df.loc[i, 'activity_level_48'] + activity_df.loc[i, 'activity_level_72'])
        except:
            pass

        # calculate the trend
        try:
            trend = (activity_df.loc[i, 'activity_level'] - activity_df.loc[i, 'activity_level_1'])/activity_df.loc[i, 'activity_level_1']
        except:
            pass

        # insert the values for historical comparison value and trend into the dataframe
        try:
            activity_df.loc[i, 'historical_comparison_value'] = historical_comparison_value
            activity_df.loc[i, 'trend'] = trend
        except:
            pass


        # calculate the activity index now
        try:
            activity_df.loc[i, 'activity_index'] = historical_comparison_value + trend
        except:
            pass
    
    # create an activity index column
    # drop unnecessary columns
    activity_index_df = activity_df.drop(columns=['activity_level', 'activity_level_1',
                                             'activity_level_24', 'activity_level_48',
                                             'activity_level_72', 'historical_comparison_value',
                                             'trend'])
    
    return activity_index_df

