def featureEngineeringDelay(df_train, df_future, df_airports):
    
    # df_train=df_hdata
    
    # Calculate arrival dalay
    fmt="%Y/%m/%d %H:%M:%S"
    fmt2="%H:%M:%S"

    # build dates on each flight stage
    # 1) add date to departure time
    df_train['DEPARTURE_DELAY'][df_train['DEPARTURE_DELAY'].isna()]=0
    df_train['SCHEDULED_DEPARTURE']=pd.to_datetime(df_train['SCHEDULED_DEPARTURE'], format=fmt)
    df_train['DEP_date_time']=df_train['SCHEDULED_DEPARTURE']+pd.to_timedelta(df_train['DEPARTURE_DELAY'],'m')
    # 2) arrival_time: DEP_date_time+ tax_out (min)+airtime (min)+tax_in (min)
    df_train['Arr_date_time']=df_train['DEP_date_time']+ pd.to_timedelta(df_train['TAXI_IN'],'m')+pd.to_timedelta(df_train['TAXI_OUT'],'m')+ pd.to_timedelta(df_train['AIR_TIME'],'m')
    # 3)compute the date of schedule arrival: SCHEDULED_DEPARTURE+SCHEDULED_TIME (min)
    df_train['SCH_ARR_date_time']=pd.to_datetime(df_train['SCHEDULED_DEPARTURE'], format=fmt)+pd.to_timedelta(df_train['SCHEDULED_TIME'],'m')
    # 4) compute arrival delay: schedule arrival-Arr_date_time
    df_train['ARRIVAL_DELAY']=(df_train['Arr_date_time']-df_train['SCH_ARR_date_time']).astype('timedelta64[m]')
    # 5) for nan in AIR TIME -build wheels_on
    # 5.1) compute wheels_on, wheels_off
    df_train['WHEELS_OFF']=df_train['DEP_date_time']+pd.to_timedelta(df_train['TAXI_OUT'],'m')
    df_train['WHEELS_ON']=df_train['Arr_date_time']-pd.to_timedelta(df_train['TAXI_IN'],'m')

    # Merge Airports dataset to df_train and df_future
    df_air_or = df_airports.rename(columns={'IATA_CODE':'ORIGIN_AIRPORT', 'CITY': 'OR_CITY', 'STATE': 'OR_STATE', 'COUNTRY': 
                                            'OR_COUNTRY', 'LATITUDE': 'OR_LATITUDE', 'LONGITUDE': 'OR_LONGITUDE'}, index={'ONE': 'Row_1'})
    df_air_des = df_airports.rename(columns={'IATA_CODE':'DESTINATION_AIRPORT', 'CITY': 'DES_CITY', 'STATE': 'DES_STATE', 'COUNTRY':
                                             'DES_COUNTRY', 'LATITUDE': 'DES_LATITUDE', 'LONGITUDE': 'DES_LONGITUDE'}, index={'ONE': 'Row_1'})    
    df_final1 = df_train.merge(df_air_or, how='left', on='ORIGIN_AIRPORT', left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
    df_final2 = df_final1.merge(df_air_des, how='left', on='DESTINATION_AIRPORT', left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)        
    df_final = df_final2.drop(['AIRPORT_y','AIRPORT_x'], axis = 1)
    
    df_predict1 = df_future.merge(df_air_or, how='left', on='ORIGIN_AIRPORT', left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
    df_predict2 = df_predict1.merge(df_air_des, how='left', on='DESTINATION_AIRPORT', left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)    
    df_predict = df_predict2.drop(['AIRPORT_y','AIRPORT_x'], axis = 1)
    
    # DATE TIME (HOUR, DAY, WEEKDAY): all observations come from the same year, then not useful in the model; Month varies from 3 to 7, avoid the summer or winter vacation already, then maybe not that useful.
    df_final['DAY_DE'] = df_final['SCHEDULED_DEPARTURE'].dt.day
    df_final['WEEKDAY_DE'] = df_final['SCHEDULED_DEPARTURE'].dt.weekday
    df_final['WEEKEND_DE'] = df_final['WEEKDAY_DE'].apply(lambda x: 1 if x >= 6 else 0)
    df_final['HOUR_DE'] = df_final['SCHEDULED_DEPARTURE'].dt.hour
    df_final['HOUR_AR'] = df_final['SCH_ARR_date_time'].dt.hour

    # Future data dates
    df_predict['SCHEDULED_ARRIVAL']=pd.to_datetime(df_predict['SCHEDULED_ARRIVAL'], format=fmt2)
    df_predict['SCHEDULED_DEPARTURE']=pd.to_datetime(df_predict['SCHEDULED_DEPARTURE'])
    df_predict['DAY_DE'] = df_predict['SCHEDULED_DEPARTURE'].dt.day
    df_predict['WEEKDAY_DE'] = df_predict['SCHEDULED_DEPARTURE'].dt.weekday
    df_predict['WEEKEND_DE'] = df_predict['WEEKDAY_DE'].apply(lambda x: 1 if x >= 6 else 0)
    df_predict['HOUR_DE'] = df_predict['SCHEDULED_DEPARTURE'].dt.hour
    df_predict['HOUR_AR'] = df_predict['SCHEDULED_ARRIVAL'].dt.hour

    # Approach as classification problem: 
    #"A flight os counted as "on time" if it operated less than 15 minutes later than the scheduled time shown 
    # in the carriers' Computerozed Researvation Systems (CRS)"
    
    df_final['ARRIVAL_DELAY_15M'] = 0
    df_final['ARRIVAL_DELAY_15M'][df_final['ARRIVAL_DELAY'] > 15]=1
    
    df_final['ARRIVAL_DELAY_1HR'] = 0
    df_final['ARRIVAL_DELAY_1HR'][df_final['ARRIVAL_DELAY'] > 60]=1
    
    df_final['DEPARTURE_DELAY_15M'] = 0
    df_final['DEPARTURE_DELAY_15M'][df_final['DEPARTURE_DELAY']> 15]=1
    
    #CANCELLATION_REASON - Reason for Cancellation of flight: A - Airline/Carrier; B - Weather; C - National Air System; D - Security
    df_final['CANCELLED_Carrier'] = 0
    df_final['CANCELLED_Carrier'][df_final['CANCELLATION_REASON'] == "A"]=1
    
    df_final['CANCELLED_Weather'] = 0
    df_final['CANCELLED_Weather'][df_final['CANCELLATION_REASON'] == "B"]=1
    
    df_final['CANCELLED_System'] = 0
    df_final['CANCELLED_System'][df_final['CANCELLATION_REASON'] == "C"]=1
    
    df_final['CANCELLED_Security'] = 0
    df_final['CANCELLED_Security'][df_final['CANCELLATION_REASON'] == "D"]=1
        
    # drop varaibles
    df_final.drop(columns=['DEPARTURE_TIME', 'SCHEDULED_ARRIVAL','ARRIVAL_TIME','SCHEDULED_DEPARTURE', 'TAXI_OUT', 'WHEELS_OFF', 
                           'WHEELS_ON', 'TAXI_IN', 'ELAPSED_TIME','AIR_TIME','OR_COUNTRY','DES_COUNTRY','DEP_date_time','Arr_date_time',
                           'SCH_ARR_date_time','CANCELLATION_REASON'], inplace=True)
    
    df_predict.drop(columns=[ 'SCHEDULED_ARRIVAL','SCHEDULED_DEPARTURE','OR_COUNTRY','DES_COUNTRY'], inplace=True)

    # Further refinement: the numbers of flights on land and air on certain moment can play a role
    # number of fligths per airplane - tail number
    
    #TEMPORAL SOLUTION FOR THESE NAN: replace missing values
    df_final['DEPARTURE_DELAY'] = df_final['DEPARTURE_DELAY'].fillna(0)
    df_final['AIR_SYSTEM_DELAY'] = df_final['AIR_SYSTEM_DELAY'].fillna(0)
    df_final['SECURITY_DELAY'] = df_final['SECURITY_DELAY'].fillna(0)
    df_final['AIRLINE_DELAY'] = df_final['AIRLINE_DELAY'].fillna(0)
    df_final['LATE_AIRCRAFT_DELAY'] = df_final['LATE_AIRCRAFT_DELAY'].fillna(0)
    df_final['WEATHER_DELAY'] = df_final['WEATHER_DELAY'].fillna(0)    

    df_final = df_final.dropna(axis=0, how='any')
    
    df_final['DAY_DE'] = df_final['DAY_DE'].astype(str)
    df_final['WEEKEND_DE'] = df_final['WEEKEND_DE'].astype(str)
    df_final['WEEKDAY_DE'] = df_final['WEEKDAY_DE'].astype(str)
    df_final['FLIGHT_NUMBER'] = df_final['FLIGHT_NUMBER'].astype(str)
    
    df_predict['DAY_DE'] = df_predict['DAY_DE'].astype(str)
    df_predict['WEEKEND_DE'] = df_predict['WEEKEND_DE'].astype(str)
    df_predict['WEEKDAY_DE'] = df_predict['WEEKDAY_DE'].astype(str)
    df_predict['FLIGHT_NUMBER'] = df_predict['FLIGHT_NUMBER'].astype(str)


    return(df_final,df_predict)


def encoders_cat(X_train, X_test, y_train,df_futurefd):
    
    # create lists of numeric and categorical features  
    numerical = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical = X_train.select_dtypes(include=['object']).columns
    print(f"Categorical columns are: {categorical}")
    print(f"Numerical columns are: {numerical}")
    
    #cat_cols_index = X_train.columns.get_indexer(categorical_features )

    # create a numeric and categorical transformer to perform preprocessing steps
    numeric_trans = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                    ('scaler', StandardScaler())])

    categorical_trans= Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                       ('woe', ce.woe.WOEEncoder())])

    # use the ColumnTransformer to apply to the correct features
    preprocessor= ColumnTransformer(transformers=[('num', numeric_trans , numerical),
                                                  ('cat', categorical_trans, categorical)])

    preprocessor.fit(X_train, y_train)

    X_train=preprocessor.transform(X_train)
    X_test=preprocessor.transform(X_test)
    X_predict=preprocessor.transform(df_futurefd)
    
    over = SMOTE(sampling_strategy=0.3, k_neighbors=k)
    under = RandomUnderSampler(sampling_strategy=0.5)
    X, y = over.fit_resample(X_train, y_train)
    X_train, y_train = under.fit_resample(X, y)
    
    return (X_train,X_test,y_train, X_predict)


# Boruta for featuring selection
def feature_selection(x_train, y_train,columns_order):
    '''
    Feature selection using Boruta
    '''
    rfc = ensemble.RandomForestClassifier(random_state=1, n_estimators=1000, max_depth=10)
    boruta_selector = BorutaPy(rfc, n_estimators=1, verbose=2, random_state=1)
    boruta_selector.fit(np.array(x_train), np.array(y_train))  

    ### print results
    df= pd.DataFrame(columns_order,columns=['Features'])
    df['Ranking']=boruta_selector.ranking_

    print('Ranking: ',df.sort_values(by=['Ranking']))          
    print('No. of significant features: ', boruta_selector.n_features_) 

    # We pass the selector such that we can deal with transformations later
    return boruta_selector, df

def encoders_final_model(X_fulldata,Y_full):
    # create lists of numeric and categorical features  
    numerical = X_fulldata.select_dtypes(include=['int64', 'float64']).columns
    categorical = X_fulldata.select_dtypes(include=['object']).columns
    print(f"Categorical columns are: {categorical}")
    print(f"Numerical columns are: {numerical}")
    
    #cat_cols_index = X_train.columns.get_indexer(categorical_features )

    # create a numeric and categorical transformer to perform preprocessing steps
    numeric_trans = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                    ('scaler', StandardScaler())])

    categorical_trans= Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                       ('woe', ce.woe.WOEEncoder())])

    # use the ColumnTransformer to apply to the correct features
    preprocessor= ColumnTransformer(transformers=[('num', numeric_trans , numerical),
                                                  ('cat', categorical_trans, categorical)])

    preprocessor.fit(X_fulldata, Y_full)

    X_fulldata=preprocessor.transform(X_fulldata)
    
    over = SMOTE(sampling_strategy=0.3, k_neighbors=k)
    under = RandomUnderSampler(sampling_strategy=0.5)
    X, y = over.fit_resample(X_fulldata, Y_full)
    X_fulldata, Y_full= under.fit_resample(X, y)
    
    return (X_fulldata, Y_full)



def featureEngineeringCancellation(df_train, df_future, df_airports):
    
    # df_train=df_hdata
    # Calculate arrival dalay
    fmt="%Y/%m/%d %H:%M:%S"
    fmt2="%H:%M:%S"

    # build dates on each flight stage
    # 1) add date to departure time
    df_train['DEPARTURE_DELAY'][df_train['DEPARTURE_DELAY'].isna()]=0
    df_train['SCHEDULED_DEPARTURE']=pd.to_datetime(df_train['SCHEDULED_DEPARTURE'], format=fmt)
    df_train['DEP_date_time']=df_train['SCHEDULED_DEPARTURE']+pd.to_timedelta(df_train['DEPARTURE_DELAY'],'m')
    # 2) arrival_time: DEP_date_time+ tax_out (min)+airtime (min)+tax_in (min)
    df_train['Arr_date_time']=df_train['DEP_date_time']+ pd.to_timedelta(df_train['TAXI_IN'],'m')+pd.to_timedelta(df_train['TAXI_OUT'],'m')+ pd.to_timedelta(df_train['AIR_TIME'],'m')
    # 3)compute the date of schedule arrival: SCHEDULED_DEPARTURE+SCHEDULED_TIME (min)
    df_train['SCH_ARR_date_time']=pd.to_datetime(df_train['SCHEDULED_DEPARTURE'], format=fmt)+pd.to_timedelta(df_train['SCHEDULED_TIME'],'m')
    # 4) compute arrival delay: schedule arrival-Arr_date_time
    df_train['ARRIVAL_DELAY']=(df_train['Arr_date_time']-df_train['SCH_ARR_date_time']).astype('timedelta64[m]')
    # 5) for nan in AIR TIME -build wheels_on
    # 5.1) compute wheels_on, wheels_off
    df_train['WHEELS_OFF']=df_train['DEP_date_time']+pd.to_timedelta(df_train['TAXI_OUT'],'m')
    df_train['WHEELS_ON']=df_train['Arr_date_time']-pd.to_timedelta(df_train['TAXI_IN'],'m')

    # Merge Airports dataset to df_train and df_future
    df_air_or = df_airports.rename(columns={'IATA_CODE':'ORIGIN_AIRPORT', 'CITY': 'OR_CITY', 'STATE': 'OR_STATE', 'COUNTRY': 
                                            'OR_COUNTRY', 'LATITUDE': 'OR_LATITUDE', 'LONGITUDE': 'OR_LONGITUDE'}, index={'ONE': 'Row_1'})
    df_air_des = df_airports.rename(columns={'IATA_CODE':'DESTINATION_AIRPORT', 'CITY': 'DES_CITY', 'STATE': 'DES_STATE', 'COUNTRY':
                                             'DES_COUNTRY', 'LATITUDE': 'DES_LATITUDE', 'LONGITUDE': 'DES_LONGITUDE'}, index={'ONE': 'Row_1'})    
    df_final1 = df_train.merge(df_air_or, how='left', on='ORIGIN_AIRPORT', left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
    df_final2 = df_final1.merge(df_air_des, how='left', on='DESTINATION_AIRPORT', left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)        
    df_final = df_final2.drop(['AIRPORT_y','AIRPORT_x'], axis = 1)
    
    df_predict1 = df_future.merge(df_air_or, how='left', on='ORIGIN_AIRPORT', left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
    df_predict2 = df_predict1.merge(df_air_des, how='left', on='DESTINATION_AIRPORT', left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)    
    df_predict = df_predict2.drop(['AIRPORT_y','AIRPORT_x'], axis = 1)
    
    # DATE TIME (HOUR, DAY, WEEKDAY): all observations come from the same year, then not useful in the model; Month varies from 3 to 7, avoid the summer or winter vacation already, then maybe not that useful.
    df_final['DAY_DE'] = df_final['SCHEDULED_DEPARTURE'].dt.day
    df_final['WEEKDAY_DE'] = df_final['SCHEDULED_DEPARTURE'].dt.weekday
    df_final['WEEKEND_DE'] = df_final['WEEKDAY_DE'].apply(lambda x: 1 if x >= 6 else 0)
    df_final['HOUR_DE'] = df_final['SCHEDULED_DEPARTURE'].dt.hour
    df_final['HOUR_AR'] = df_final['SCH_ARR_date_time'].dt.hour

    # Future data dates
    df_predict['SCHEDULED_ARRIVAL']=pd.to_datetime(df_predict['SCHEDULED_ARRIVAL'], format=fmt2)
    df_predict['SCHEDULED_DEPARTURE']=pd.to_datetime(df_predict['SCHEDULED_DEPARTURE'])
    df_predict['DAY_DE'] = df_predict['SCHEDULED_DEPARTURE'].dt.day
    df_predict['WEEKDAY_DE'] = df_predict['SCHEDULED_DEPARTURE'].dt.weekday
    df_predict['WEEKEND_DE'] = df_predict['WEEKDAY_DE'].apply(lambda x: 1 if x >= 6 else 0)
    df_predict['HOUR_DE'] = df_predict['SCHEDULED_DEPARTURE'].dt.hour
    df_predict['HOUR_AR'] = df_predict['SCHEDULED_ARRIVAL'].dt.hour

    # Approach as classification problem: 
    #"A flight os counted as "on time" if it operated less than 15 minutes later than the scheduled time shown 
    # in the carriers' Computerozed Researvation Systems (CRS)"
    
    
    #CANCELLATION_REASON - Reason for Cancellation of flight: A - Airline/Carrier; B - Weather; C - National Air System; D - Security
    df_final['CANCELLED'] = df_final['CANCELLED'].fillna(0)
    df_final['CANCELLED_Carrier'] = 0
    df_final['CANCELLED_Carrier'][df_final['CANCELLATION_REASON'] == "A"]=1
    
    df_final['CANCELLED_Weather'] = 0
    df_final['CANCELLED_Weather'][df_final['CANCELLATION_REASON'] == "B"]=1
    
    df_final['CANCELLED_System'] = 0
    df_final['CANCELLED_System'][df_final['CANCELLATION_REASON'] == "C"]=1
    
    df_final['CANCELLED_Security'] = 0
    df_final['CANCELLED_Security'][df_final['CANCELLATION_REASON'] == "D"]=1
        
    # drop varaibles
    df_final.drop(columns=['DEPARTURE_TIME', 'SCHEDULED_ARRIVAL','ARRIVAL_TIME','SCHEDULED_DEPARTURE', 'TAXI_OUT', 'WHEELS_OFF', 
                           'WHEELS_ON', 'TAXI_IN', 'ELAPSED_TIME','AIR_TIME','OR_COUNTRY','DES_COUNTRY','DEP_date_time','Arr_date_time',
                           'SCH_ARR_date_time','CANCELLATION_REASON'], inplace=True)
    
    df_predict.drop(columns=[ 'SCHEDULED_ARRIVAL','SCHEDULED_DEPARTURE','OR_COUNTRY','DES_COUNTRY'], inplace=True)

    # Further refinement: the numbers of flights on land and air on certain moment can play a role
    # number of fligths per airplane - tail number
    
    #TEMPORAL SOLUTION FOR THESE NAN: replace missing values
    df_final['DEPARTURE_DELAY'] = df_final['DEPARTURE_DELAY'].fillna(0)
    df_final['AIR_SYSTEM_DELAY'] = df_final['AIR_SYSTEM_DELAY'].fillna(0)
    df_final['SECURITY_DELAY'] = df_final['SECURITY_DELAY'].fillna(0)
    df_final['AIRLINE_DELAY'] = df_final['AIRLINE_DELAY'].fillna(0)
    df_final['LATE_AIRCRAFT_DELAY'] = df_final['LATE_AIRCRAFT_DELAY'].fillna(0)
    df_final['WEATHER_DELAY'] = df_final['WEATHER_DELAY'].fillna(0)    

    
    #df_final = df_final.dropna(axis=0, how='any')
    df_final['DAY_DE'] = df_final['DAY_DE'].astype(str)
    df_final['WEEKEND_DE'] = df_final['WEEKEND_DE'].astype(str)
    df_final['WEEKDAY_DE'] = df_final['WEEKDAY_DE'].astype(str)
    df_final['FLIGHT_NUMBER'] = df_final['FLIGHT_NUMBER'].astype(str)
    
    df_predict['DAY_DE'] = df_predict['DAY_DE'].astype(str)
    df_predict['WEEKEND_DE'] = df_predict['WEEKEND_DE'].astype(str)
    df_predict['WEEKDAY_DE'] = df_predict['WEEKDAY_DE'].astype(str)
    df_predict['FLIGHT_NUMBER'] = df_predict['FLIGHT_NUMBER'].astype(str)

    return(df_final,df_predict)


def featureEngineering(df_train, df_future, df_airports):
    
    # df_train=df_hdata
    
    # Calculate arrival dalay
    fmt="%Y/%m/%d %H:%M:%S"
    fmt2="%H:%M:%S"

    # build dates on each flight stage
    # 1) add date to departure time
    df_train['DEPARTURE_DELAY'][df_train['DEPARTURE_DELAY'].isna()]=0
    df_train['SCHEDULED_DEPARTURE']=pd.to_datetime(df_train['SCHEDULED_DEPARTURE'], format=fmt)
    df_train['DEP_date_time']=df_train['SCHEDULED_DEPARTURE']+pd.to_timedelta(df_train['DEPARTURE_DELAY'],'m')
    # 2) arrival_time: DEP_date_time+ tax_out (min)+airtime (min)+tax_in (min)
    df_train['Arr_date_time']=df_train['DEP_date_time']+ pd.to_timedelta(df_train['TAXI_IN'],'m')+pd.to_timedelta(df_train['TAXI_OUT'],'m')+ pd.to_timedelta(df_train['AIR_TIME'],'m')
    # 3)compute the date of schedule arrival: SCHEDULED_DEPARTURE+SCHEDULED_TIME (min)
    df_train['SCH_ARR_date_time']=pd.to_datetime(df_train['SCHEDULED_DEPARTURE'], format=fmt)+pd.to_timedelta(df_train['SCHEDULED_TIME'],'m')
    # 4) compute arrival delay: schedule arrival-Arr_date_time
    df_train['ARRIVAL_DELAY']=(df_train['Arr_date_time']-df_train['SCH_ARR_date_time']).astype('timedelta64[m]')
    # 5) for nan in AIR TIME -build wheels_on
    # 5.1) compute wheels_on, wheels_off
    df_train['WHEELS_OFF']=df_train['DEP_date_time']+pd.to_timedelta(df_train['TAXI_OUT'],'m')
    df_train['WHEELS_ON']=df_train['Arr_date_time']-pd.to_timedelta(df_train['TAXI_IN'],'m')

    # Merge Airports dataset to df_train and df_future
    df_air_or = df_airports.rename(columns={'IATA_CODE':'ORIGIN_AIRPORT', 'CITY': 'OR_CITY', 'STATE': 'OR_STATE', 'COUNTRY': 
                                            'OR_COUNTRY', 'LATITUDE': 'OR_LATITUDE', 'LONGITUDE': 'OR_LONGITUDE'}, index={'ONE': 'Row_1'})
    df_air_des = df_airports.rename(columns={'IATA_CODE':'DESTINATION_AIRPORT', 'CITY': 'DES_CITY', 'STATE': 'DES_STATE', 'COUNTRY':
                                             'DES_COUNTRY', 'LATITUDE': 'DES_LATITUDE', 'LONGITUDE': 'DES_LONGITUDE'}, index={'ONE': 'Row_1'})    
    df_final1 = df_train.merge(df_air_or, how='left', on='ORIGIN_AIRPORT', left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
    df_final2 = df_final1.merge(df_air_des, how='left', on='DESTINATION_AIRPORT', left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)        
    df_final = df_final2.drop(['AIRPORT_y','AIRPORT_x'], axis = 1)
    
    df_predict1 = df_future.merge(df_air_or, how='left', on='ORIGIN_AIRPORT', left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
    df_predict2 = df_predict1.merge(df_air_des, how='left', on='DESTINATION_AIRPORT', left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)    
    df_predict = df_predict2.drop(['AIRPORT_y','AIRPORT_x'], axis = 1)
    
    # DATE TIME (HOUR, DAY, WEEKDAY): all observations come from the same year, then not useful in the model; Month varies from 3 to 7, avoid the summer or winter vacation already, then maybe not that useful.
    df_final['DAY_DE'] = df_final['SCHEDULED_DEPARTURE'].dt.day
    df_final['WEEKDAY_DE'] = df_final['SCHEDULED_DEPARTURE'].dt.weekday
    df_final['WEEKEND_DE'] = df_final['WEEKDAY_DE'].apply(lambda x: 1 if x >= 6 else 0)
    df_final['HOUR_DE'] = df_final['SCHEDULED_DEPARTURE'].dt.hour
    df_final['HOUR_AR'] = df_final['SCH_ARR_date_time'].dt.hour

    # Future data dates
    df_predict['SCHEDULED_ARRIVAL']=pd.to_datetime(df_predict['SCHEDULED_ARRIVAL'], format=fmt2)
    df_predict['SCHEDULED_DEPARTURE']=pd.to_datetime(df_predict['SCHEDULED_DEPARTURE'])
    df_predict['DAY_DE'] = df_predict['SCHEDULED_DEPARTURE'].dt.day
    df_predict['WEEKDAY_DE'] = df_predict['SCHEDULED_DEPARTURE'].dt.weekday
    df_predict['WEEKEND_DE'] = df_predict['WEEKDAY_DE'].apply(lambda x: 1 if x >= 6 else 0)
    df_predict['HOUR_DE'] = df_predict['SCHEDULED_DEPARTURE'].dt.hour
    df_predict['HOUR_AR'] = df_predict['SCHEDULED_ARRIVAL'].dt.hour

    # Approach as classification problem: 
    #"A flight os counted as "on time" if it operated less than 15 minutes later than the scheduled time shown 
    # in the carriers' Computerozed Researvation Systems (CRS)"
    
    df_final['ARRIVAL_DELAY_15M'] = 0
    df_final['ARRIVAL_DELAY_15M'][df_final['ARRIVAL_DELAY'] > 15]=1
    
    df_final['ARRIVAL_DELAY_1HR'] = 0
    df_final['ARRIVAL_DELAY_1HR'][df_final['ARRIVAL_DELAY'] > 60]=1
    
    df_final['DEPARTURE_DELAY_15M'] = 0
    df_final['DEPARTURE_DELAY_15M'][df_final['DEPARTURE_DELAY']> 15]=1
    
    #CANCELLATION_REASON - Reason for Cancellation of flight: A - Airline/Carrier; B - Weather; C - National Air System; D - Security
    df_final['CANCELLED_Carrier'] = 0
    df_final['CANCELLED_Carrier'][df_final['CANCELLATION_REASON'] == "A"]=1
    
    df_final['CANCELLED_Weather'] = 0
    df_final['CANCELLED_Weather'][df_final['CANCELLATION_REASON'] == "B"]=1
    
    df_final['CANCELLED_System'] = 0
    df_final['CANCELLED_System'][df_final['CANCELLATION_REASON'] == "C"]=1
    
    df_final['CANCELLED_Security'] = 0
    df_final['CANCELLED_Security'][df_final['CANCELLATION_REASON'] == "D"]=1
        
    # drop varaibles
    df_final.drop(columns=['DEPARTURE_TIME', 'SCHEDULED_ARRIVAL','ARRIVAL_TIME','SCHEDULED_DEPARTURE', 'TAXI_OUT', 'WHEELS_OFF', 
                           'WHEELS_ON', 'TAXI_IN', 'ELAPSED_TIME','AIR_TIME','OR_COUNTRY','DES_COUNTRY','DEP_date_time','Arr_date_time',
                           'SCH_ARR_date_time','CANCELLATION_REASON'], inplace=True)
    
    df_predict.drop(columns=[ 'SCHEDULED_ARRIVAL','SCHEDULED_DEPARTURE','OR_COUNTRY','DES_COUNTRY'], inplace=True)

    # Further refinement: the numbers of flights on land and air on certain moment can play a role
    # number of fligths per airplane - tail number
    
    #TEMPORAL SOLUTION FOR THESE NAN: replace missing values
    df_final['DEPARTURE_DELAY'] = df_final['DEPARTURE_DELAY'].fillna(0)
    df_final['AIR_SYSTEM_DELAY'] = df_final['AIR_SYSTEM_DELAY'].fillna(0)
    df_final['SECURITY_DELAY'] = df_final['SECURITY_DELAY'].fillna(0)
    df_final['AIRLINE_DELAY'] = df_final['AIRLINE_DELAY'].fillna(0)
    df_final['LATE_AIRCRAFT_DELAY'] = df_final['LATE_AIRCRAFT_DELAY'].fillna(0)
    df_final['WEATHER_DELAY'] = df_final['WEATHER_DELAY'].fillna(0)    

    #df_final = df_final.dropna(axis=0, how='any')
    
    df_final['DAY_DE'] = df_final['DAY_DE'].astype(str)
    df_final['WEEKEND_DE'] = df_final['WEEKEND_DE'].astype(str)
    df_final['WEEKDAY_DE'] = df_final['WEEKDAY_DE'].astype(str)
    df_final['FLIGHT_NUMBER'] = df_final['FLIGHT_NUMBER'].astype(str)
    
    df_predict['DAY_DE'] = df_predict['DAY_DE'].astype(str)
    df_predict['WEEKEND_DE'] = df_predict['WEEKEND_DE'].astype(str)
    df_predict['WEEKDAY_DE'] = df_predict['WEEKDAY_DE'].astype(str)
    df_predict['FLIGHT_NUMBER'] = df_predict['FLIGHT_NUMBER'].astype(str)


    return(df_final,df_predict)


    



