#import modules
from sklearn import ensemble,gaussian_process,linear_model,naive_bayes,neighbors,svm,tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, precision_score, recall_score, auc, classification_report, precision_recall_curve

from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt   
import seaborn as sns

#import dadta
df_airports = pd.read_csv('airports.csv')
df_hdata = pd.read_csv('historic_data.csv', low_memory=False)
df_future = pd.read_csv('future_data.csv', low_memory=False)
# preprocesing - using functions
feature_hd, feature_fd = featureEngineeringCancellation(df_hdata, df_future, df_airports)

##due to lack of computational resources. stratified samplig considering CANCELLED percentage
df_cat =feature_hd.groupby('CANCELLED', group_keys=False).apply(lambda x: x.sample(frac=0.20))


#variables to use from historical, future dataset
df_model1 = df_cat[['AIRLINE','FLIGHT_NUMBER','TAIL_NUMBER','ORIGIN_AIRPORT','DESTINATION_AIRPORT','DISTANCE','OR_CITY','OR_STATE',
                    'OR_LATITUDE','OR_LONGITUDE','DES_CITY','DES_STATE','DES_LATITUDE','DES_LONGITUDE','DAY_DE','WEEKDAY_DE',
                    'WEEKEND_DE','HOUR_DE','HOUR_AR','CANCELLED']]

df_futurefd = feature_fd[['AIRLINE','FLIGHT_NUMBER','TAIL_NUMBER','ORIGIN_AIRPORT','DESTINATION_AIRPORT','DISTANCE','OR_CITY','OR_STATE',
                    'OR_LATITUDE','OR_LONGITUDE','DES_CITY','DES_STATE','DES_LATITUDE','DES_LONGITUDE','DAY_DE','WEEKDAY_DE',
                    'WEEKEND_DE','HOUR_DE','HOUR_AR']]


#split train/test
X_train, X_test= train_test_split(df_model1, test_size=0.3, stratify=df_model1.CANCELLED)

y_train = X_train.CANCELLED
y_test= X_test.CANCELLED

X_train.drop(columns=['CANCELLED'], inplace=True)
X_test.drop(columns=['CANCELLED'], inplace=True)

#extract type of features
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns
columns_order=np.array(numeric_features.tolist()+categorical_features.tolist())

# apply encoders
X_train,X_test,y_train, X_predict = encoders_cat(X_train,X_test,y_train,df_futurefd)

# Select features using Boruta
feature_selector, selected_features = feature_selection(X_train, y_train, columns_order)

# Apply feature selection
X_boruta= feature_selector.transform(np.array(X_train))
X_test_boruta = feature_selector.transform(np.array(X_test)) 

# to run several model Reference: https://www.kaggle.com/code/rahulstephenites2/airline-flight-delaytime-prediction/notebook
MLA = [
    linear_model.LogisticRegression(random_state=11, max_iter=1000), 
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    #SVM
    svm.LinearSVC(random_state=11),
    #Trees    
    tree.DecisionTreeClassifier(random_state=11),
    tree.ExtraTreeClassifier(random_state=11),
    #Ensemble Methods
    ensemble.AdaBoostClassifier(random_state=11),
    ensemble.BaggingClassifier(random_state=11),
    ensemble.GradientBoostingClassifier(random_state=11),
    ensemble.RandomForestClassifier(n_jobs = -1,random_state=11),
    ]



MLA_columns = []
MLA_compare = pd.DataFrame(columns = MLA_columns)
results=[]

row_index = 0
for alg in MLA:

    cv_results = cross_val_score(alg, X_boruta, y_train, cv=10, scoring='f1')
    results.append(cv_results)
    predicted = alg.fit(X_boruta, y_train).predict(X_test_boruta)
    fp, tp, th = roc_curve(y_test, predicted)
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'Name'] = MLA_name
    MLA_compare.loc[row_index, 'Train F1 Score'] = round(alg.score(X_boruta, y_train), 4)
    print("model:", alg,  " Train F1 Score: ", round(alg.score(X_boruta, y_train), 4))
    MLA_compare.loc[row_index, 'Test F1 Score'] = round(alg.score(X_test_boruta , y_test), 4)
    print("model: ",alg,  " Test F1 Score: ", round(alg.score(X_test_boruta , y_test), 4))
    MLA_compare.loc[row_index, 'AUC'] = auc(fp, tp)  
    row_index+=1
    
MLA_compare.sort_values(by = ['Test F1 Score'], ascending = False, inplace = True)    
MLA_compare


########################
# Runing final modeling in the full dataset to have predictions and variable importance
######################
X_full= df_model1
Y_full= X_full.CANCELLED
X_full.drop(columns=['CANCELLED'], inplace=True)

# apply encoders
X_full,Y_full= encoders_final_model(X_full,Y_full)

# Apply feature selection from the previous Boruta selection
X_full=feature_selector.transform(np.array(X_full)) 
X_predict=feature_selector.transform(np.array(X_predict)) 

#modeling the best alternative according to F1: RandomForestClassifier 
model= ensemble.RandomForestClassifier(n_jobs = -1,random_state=11)
model.fit(X_full,Y_full)

#variable importance
model.feature_importances_

# adding prediction to the future dataset
df_future['cancellations_predClass']=model.predict(X_predict)
probs=model.predict_proba(X_predict)
df_future['cancellations_predProbs=0']=probs[:,0]
df_future['cancellations_predProbs=1']=probs[:,1]

#export prediction resutls
df_future.to_excel('cancellations_preditions.xlsx', engine='xlsxwriter')

#export importance
selected_features['importance']= model.feature_importances_
selected_features.to_excel('imporance_cancellations.xlsx', engine='xlsxwriter')


