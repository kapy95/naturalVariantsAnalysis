#Ensemble tree model
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier

#Parameterisation
from sklearn.model_selection import GridSearchCV


# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

#Data management:
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import numpy as np
from scipy.special import expit as sigmoid, logit as inverse_sigmoid
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist
from numpy.random import default_rng


#Performance measures: 
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

#Representations:
from performanceResults import generateDataMap

import warnings
warnings.filterwarnings("ignore")

import joblib

from sklearn.preprocessing import MinMaxScaler


def findBestEnsembleTree(classifier_type,X_train, y_train, param_dist):
    """
    Input: 
        -classifier_type: string indicating the classifier to be used (random forest or XGBClassifier)
        - X_train: dataframe to train an ensemble tree
        - y_train: real labels of the training data
        - param_dist: dictionary whose keys correspond to parameters and the values correspond to arrays with the parameter options to find the best ensemble tree
            (example: { {'n_estimators': [50,100,150,200,250,300,350,400,450,500],
              'max_features': ['auto', 'sqrt', 'log2'],
              'criterion':['gini','entropy'],
              'max_depth': [None]
              }})
    Output:
        - Ensemble tree with the best parameters based on those specified in param_dist
            Example: 'n_estimators':50, 'max_features':'auto', 'criterion':'gini'
    """
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    if classifier_type=="Random Forest":
        # Create a random forest classifier, setting the random state to 0 for replicability
        clf = RandomForestClassifier(random_state=0)
    else:
        clf = XGBClassifier(seed=0)
        
    #With grid search we exhaustively look for the best hyperparameters within the options indicated by the user

    #Specifically these are some of the parameters of GridSearchCV:
    #estimator: classifier model to be used
    #param_grid:values to test
    #cv: number of cross validations to be performed. If [(slice(none),slice(None))] is passed as parameter, no CV will be used even though, this is not recommended.

    grid_search = GridSearchCV(clf, 
                           param_grid = param_dist,
                           refit=True, #when set to true, the best estimator found will be trained with the original data 
                           cv=[(slice(None), slice(None))],#cv indicates cross validation
                           return_train_score=True)#when true returns training scores for each parameter setting

    # Fit the grid search object to the data
    grid_search.fit(X_train, y_train)

    #Save in a dataframe the results of the grid search
    grid_results=pd.DataFrame(grid_search.cv_results_)

    #Return the results of the search (grid_results), and the ensemble tree with the best configuaration (best_estimator_)
    le_name_mapping=pd.Series(le_name_mapping)
    return grid_results, grid_search.best_estimator_,le_name_mapping



def findBestXGBoostModel(X_train, y_train, val_perc, X_test, y_test, limit_early, limit_estimators, limit_depth):
    possible_values_early_stop=[no_rounds_early for no_rounds_early in range(5,limit_early,5)]
    possible_values_estimators=[n_est for n_est in (i*50 for i in range(1,limit_estimators))]
    possible_values_lr=[lr_sp/100 for lr_sp in range(10,50,10)]
    possible_values_depth=[deepth for deepth in range(0,limit_depth,2)]
    boosters=['gbtree']#do not use gblinear, because it omits NaN results
    
    X_train_def, X_val, y_train_def, y_val=train_test_split(X_train, y_train, test_size=val_perc, stratify=y_train, random_state=0)
    
    vals_used_early_stop=[]
    vals_used_estimators=[]
    vals_used_lr=[]
    vals_used_depth=[]
    vals_used_boosters=[]


    results_precision=[]
    results_f1=[]
    results_recall=[]
    results_acc=[]
    i=1
    max_searchs=len(possible_values_early_stop)*len(possible_values_estimators)*len(possible_values_lr)*len(possible_values_depth)
    for val_early_stop in possible_values_early_stop:

        for val_n_estimators in possible_values_estimators:
            for lr in possible_values_lr:
                for val_depth in possible_values_depth:
                    for val_booster in boosters:
                        print("Search "+str(i)+ " of "+str(max_searchs) )
                        i=i+1
                        #save the values used to build the model, so that later we know the results of each configuration
                        vals_used_early_stop.append(val_early_stop)
                        vals_used_boosters.append(val_booster)
                        vals_used_depth.append(val_depth)
                        vals_used_estimators.append(val_n_estimators)
                        vals_used_lr.append(lr)

                        #Build a model with the combination of parameters:
                        model = xgb.XGBClassifier(n_estimators=val_n_estimators,
                                objective="multi:softprob",#with objective we indicate that the classifier is for multiple classes (multi) and softprob allows us to obtain class probabilities instead of the most probable class when doing predictions
                                num_class=len(y_train), #with this parameter we indicate the number of classes
                                eta=lr,#learning rate
                                early_stopping_rounds=val_early_stop, #number of iterations without improvement required to stop the boosting
                                max_depth=val_depth,#maximum depth of the trees, with 0 value there is not limit
                                booster=val_booster,
                                seed=0,
                                random_state=0)
                        
                        # Fit model on train and use validation for early stopping
                        model.fit(X_train_def, y_train_def, eval_set=[(X_val, y_val)], verbose=False)

                        #Predict on test set
                        y_pred_test = model.predict(X_test)
                        
                        #Measure the performance of the model:
                        f1 = f1_score(y_true=y_test, y_pred=y_pred_test, average='weighted')
                        prec=precision_score(y_true=y_test, y_pred=y_pred_test,average='weighted')
                        rec=recall_score(y_true=y_test, y_pred=y_pred_test,average='weighted')
                        acc=accuracy_score(y_true=y_test, y_pred=y_pred_test)
                        
                        #save the results
                        results_f1.append(f1)
                        results_acc.append(acc)
                        results_precision.append(prec)
                        results_recall.append(rec)


    results_search=pd.DataFrame.from_dict({"early_stop":vals_used_early_stop,
                            "n_estimators":vals_used_estimators,
                            "learning_rate":vals_used_lr,
                            "max_depth":vals_used_depth,
                            "booster":vals_used_boosters,
                            "accuracy":results_acc,
                            "f1_score":results_f1,
                            "recall":results_recall,
                            "precision":results_precision
                            })
    
    return results_search


def findBestXGBoostModel_withKfold(X, y, val_perc, limit_early, limit_estimators, limit_depth, n_splits):
    possible_values_early_stop=[no_rounds_early for no_rounds_early in range(5,limit_early,5)]
    possible_values_estimators=[n_est for n_est in (i*50 for i in range(1,limit_estimators))]
    possible_values_lr=[lr_sp/100 for lr_sp in range(10,50,10)]
    possible_values_depth=[deepth for deepth in range(0,limit_depth,2)]
    boosters=['gbtree']#do not use gblinear, because it omits NaN results
    
    vals_used_early_stop=[]
    vals_used_estimators=[]
    vals_used_lr=[]
    vals_used_depth=[]
    vals_used_boosters=[]

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    results_precision=[]
    results_f1=[]
    results_recall=[]
    results_acc=[]
    i=1
    max_searchs=len(possible_values_early_stop)*len(possible_values_estimators)*len(possible_values_lr)*len(possible_values_depth)
    for val_early_stop in possible_values_early_stop:
        for val_n_estimators in possible_values_estimators:
            for lr in possible_values_lr:
                for val_depth in possible_values_depth:
                    for val_booster in boosters:
                        print("Search "+str(i)+ " of "+str(max_searchs) )
                        i=i+1
                        #save the values used to build the model, so that later we know the results of each configuration
                        vals_used_early_stop.append(val_early_stop)
                        vals_used_boosters.append(val_booster)
                        vals_used_depth.append(val_depth)
                        vals_used_estimators.append(val_n_estimators)
                        vals_used_lr.append(lr)

                        results_precision_kfold=[]
                        results_f1_kfold=[]
                        results_recall_kfold=[]
                        results_acc_kfold=[]

                        for train_index, test_index in kf.split(X, y):
                            X_train_fold, X_test = X.iloc[train_index], X.iloc[test_index]
                            y_train_fold, y_test = y[train_index], y[test_index]

                            # Split train set into train and validation
                            X_train_fold, X_val, y_train_fold, y_val = train_test_split(X_train_fold, y_train_fold, test_size=val_perc, stratify=y_train_fold, random_state=0)

                            #Build a model with the combination of parameters:
                            model = xgb.XGBClassifier(n_estimators=val_n_estimators,
                                                      objective="multi:softprob",#with objective we indicate that the classifier is for multiple classes (multi) and softprob allows us to obtain class probabilities instead of the most probable class when doing predictions
                                                      num_class=len(y_train_fold), #with this parameter we indicate the number of classes
                                                      eta=lr,#learning rate
                                                      early_stopping_rounds=val_early_stop, #number of iterations without improvement required to stop the boosting
                                                      max_depth=val_depth,#maximum depth of the trees, with 0 value there is not limit
                                                      booster=val_booster,
                                                      seed=0,
                                                      random_state=0)
                                            
                            # Fit model on train and use validation for early stopping
                            model.fit(X_train_fold, y_train_fold, eval_set=[(X_val, y_val)], verbose=False)

                            #Predict on test set
                            y_pred_test = model.predict(X_test)
                                            
                            #Measure the performance of the model:
                            f1 = f1_score(y_true=y_test, y_pred=y_pred_test, average='weighted')
                            prec=precision_score(y_true=y_test, y_pred=y_pred_test,average='weighted')
                            rec=recall_score(y_true=y_test, y_pred=y_pred_test,average='weighted')
                            acc=accuracy_score(y_true=y_test, y_pred=y_pred_test)
                            
                            #save the results in an intermediate list to calculate the mean later
                            results_f1_kfold.append(f1)
                            results_acc_kfold.append(acc)
                            results_precision_kfold.append(prec)
                            results_recall_kfold.append(rec)
                        
                        #save the results
                        results_f1.append(np.mean(results_f1_kfold))
                        results_acc.append(np.mean(results_acc_kfold))
                        results_precision.append(np.mean(results_precision_kfold))
                        results_recall.append(np.mean(results_recall_kfold))


    results_search=pd.DataFrame.from_dict({"early_stop":vals_used_early_stop,
                            "n_estimators":vals_used_estimators,
                            "learning_rate":vals_used_lr,
                            "max_depth":vals_used_depth,
                            "booster":vals_used_boosters,
                            "accuracy":results_acc,
                            "f1_score":results_f1,
                            "recall":results_recall,
                            "precision":results_precision
                            })
    
    return results_search



def createAndTrainXGB_early_stop(X_method, y_method, params, size, stratification):
    """
    Input:
        - X_method: instances to use in training and validation
        - y_method: instance labels to be used in training and validation
        - params: parameters of xgboost in dictionary format
        - size: float to indicate the size of validation dataset
        - stratification: bolean value to indicate whether the training and validation data should be stratified or not
    
    Output:
        - clf: xgboost trained based on the data, parameters. This model uses early stop to avoid overfitting

    """
    if stratification==True:
        strats=y_method
        shuff=True
    else:
        strats=None
        shuff=False

    X_train, X_val, y_train, y_val = train_test_split(X_method,
                                                      y_method,
                                                      test_size=size,
                                                      stratify=strats,
                                                      shuffle=shuff, 
                                                      random_state=0)
    clf = xgb.XGBClassifier(params)
    # Fit the model with early stopping
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

    return clf


def measureClassifierPerformance(classifiers, X_test, y_test, le_name_mapping_dataset, dirResults):
    """ Description: Function to measure the performance of a set of classifiers in different aspects for each class
        Input: 
            classifiers: dictionary whose keys are the code names of the classifiers, and the values are the classifier models
            X_test: data to be used for test
            y_test: class labels of the the test data

        Output:
            classifiers: dataframe whose rows represent the performance of a classifier for a certain class, like this:

            | Precision | Recall | Classifier | f1-score | Class
                0.01       0.87      clf40        0.5        X
                ...        ...        ...         ...       ...
            
            accus_classifiers: dictionary whose keys represent classification models and the values are the accuracies obtained in the test set
    
    """
    accus_classifiers=[]
    precisions_classifiers=[]
    recalls_classifiers=[]
    f1_classifiers=[]

    labels=[]
    classifiers_labels=[]
    values_labels=list(le_name_mapping_dataset.keys())

    for label_clf, specific_clf in classifiers.items():
        joblib.dump(specific_clf, dirResults+"/"+label_clf+".joblib")
        y_pred_test=specific_clf.predict(X_test)
        dirClfMap=dirResults+"/heatmap_"+label_clf+".png"
        generateDataMap(y_real_test_map=y_test,
                        y_pred_map=y_pred_test,
                        le_name_mapping=le_name_mapping_dataset, 
                        labels=list(le_name_mapping_dataset.values()),
                        dirMap=dirClfMap)

        precision_specific_clf=precision_score(y_true=y_test,y_pred=y_pred_test,average=None)
        accu_specific_clf=accuracy_score(y_true=y_test,y_pred=y_pred_test)
        recall_specific_clf=recall_score(y_true=y_test, y_pred=y_pred_test, average=None)
        f1_specific_clf=f1_score(y_true=y_test, y_pred=y_pred_test, average=None)

        accus_classifiers.append(accu_specific_clf)

        precisions_classifiers.extend(precision_specific_clf)
        recalls_classifiers.extend(recall_specific_clf)
        labels.extend(values_labels)
        f1_classifiers.extend(f1_specific_clf)
        
        clf_labels=[label_clf for i in range(len(values_labels))]
        classifiers_labels.extend(clf_labels)

    df_classifiers=pd.DataFrame.from_dict({"Precision":precisions_classifiers,
                                       "Recall":recalls_classifiers,
                                       "Classifier":classifiers_labels,
                                       "f1-scores":f1_classifiers,
                                       "Class":labels})
    
    df_accus_classifiers=pd.DataFrame.from_dict({"Classifier":list(classifiers.keys()), "Accuracy": accus_classifiers})
        
    return df_classifiers, df_accus_classifiers
    



def findChampionsInSpecificClassProbs(x_train_bin_class, bin_class, classifier, case_sizes):
    #mapping: dictionary that translates bin to coded classes (i.e., new_le_mapping variable).The keys are classes and the values are the coded classes. 
    #The coded classes represent the indices that will be used to obtain the predicted probability of an instance for that class 
    index_prob_pred=bin_class

    probs_y_train=classifier.predict_proba(x_train_bin_class)#we obtain the predicted probability of belonging to each for each trace
    #Example: probs_y_train=[ Trace1:[class1: 0.2, class2: 0.5, class3:0.3] , Trace2[class1: 0.9, class2: 0.1, class3:0.0], ...]
    cases=x_train_bin_class.index.to_list()#we obtain the case ids of these cases

    #we obtain a dictionary that relates each case to the predicted probability of its class, which is in the position equal to its class
    #For instance, if in the encoding the class is called 3, the fourth position of the list of the probabilities list contains the probability of the class
    bin_class_prob_per_case={case: probs[index_prob_pred] for case,probs in zip (cases, probs_y_train)}
    series_prob_case=pd.Series(bin_class_prob_per_case)

    champion=series_prob_case[series_prob_case==max(series_prob_case)]#obtain the champion, that is the instance with the highest probability to be in its correct class

    # #if there is more than one instance in the results with the same maximum probability:
    if len(champion)>1:
        sizes_possible_champions=case_sizes.loc[list(champion.index)]#get the sizes related to the possible champions 

        #find the case between the possible champios with lowest size
        shortest_champion=sizes_possible_champions[sizes_possible_champions==min(sizes_possible_champions)]

        if len(shortest_champion)>1:#again if there is more than one... (e.g. 12)
            rng = default_rng(seed=0)
            random_number=rng.integers(0, len(shortest_champion), size=1)[0]#generate a random number between the first possible champioon and the last one (e.g. between 1 and 12, it could be 5). We set the seed for reproducibility
            champion_class_id=list(shortest_champion.index)[random_number]#select it randomly based on the number, namely its identifier
            champion_class_label=x_train_bin_class.loc[[champion_class_id]]#filter the champion based on its identifier
        else:
            champion_class_label=x_train_bin_class.loc[[list(shortest_champion.index)[0]]]
    else: #if there is one champion directly, just filter it and save it
        champion_class_label=x_train_bin_class.loc[[list(champion.index)[0]]]

    return champion_class_label



def boundaryCasesBetweenTwoClasses(df_probs_class_x, df_probs_class_y, class_labelX, class_labelY, X_test):
    """
    Description: this function looks for the two closest boundary cases between cases. With boundary cases we understand cases whose prediction in the ensemble tree is the corresponding one to its real class (e.g., if the 
    class of a trace t1 is X, its class is predicted as X in the tree), but in terms of prediction probabilities another class Y is close to class X probability (e.g., prediction=[Probs to belong to X: 0.6, Probs to belong to Y: 0.4])
    
    Input: 
        - df_probs_class_x: dataframe whose rows represent cases and the columns represent the predicted probabilities to belong to a class. The class of all cases is predicted as X
        - df_probs_class_y: dataframe whose rows represent cases and the columns represent the predicted probabilities to belong to a class. The class of all cases is predicted as Y
        - class_labelY: class Y label used by the label encoder. It will be used to know the name of the column of class Y probabilities in the dataframes
        - class labelX: class X label used by the label encoder. It will be used to know the name of the column of class X probabilities in the dataframes
        - X_test: dataframe whose rows represent cases, and the columns confidences to declare rules.
    
    Output:
        - Pair of boundary cases between classes X and Y

    """

    #Firstly we calculate for each case of both dataframes the inverse absolute similarity using the predicted probabilities for class X and class Y. Basically consists in this: 1/(1+abs(probsX-probsY)). 
    # Thus, the less difference there is between probs of X and probs of Y, the greater the similarity will be. Therefore, we reward cases where probsX and probsY are close.
    df_probs_class_x["similarity_probs_labels"]=[1/(1+abs(row[class_labelX]-row[class_labelY])) for index, row in df_probs_class_x.iterrows()]
    df_probs_class_y["similarity_probs_labels"]=[1/(1+abs(row[class_labelX]-row[class_labelY])) for index, row in df_probs_class_y.iterrows()]
    
    #We sort the dataframes based on the similarity measure:
    df_sorted_classX=df_probs_class_x.sort_values(by="similarity_probs_labels",ascending=False)
    df_sorted_classY=df_probs_class_y.sort_values(by="similarity_probs_labels",ascending=False)

    #we obtain the 10 first case ids whose probabilities are most similar
    best_boundary_cases_classX=df_sorted_classX.iloc[0:10]
    best_boundary_cases_classY=df_sorted_classY.iloc[0:10]

    #we filter these cases in X_test
    filtered_cases_classX=X_test.filter(items=list(best_boundary_cases_classX.index), axis=0).fillna(-100)#to be able to calculate the distance, we fill nan values for -1 whihc is -100 in the current scale
    filtered_cases_classY=X_test.filter(items=list(best_boundary_cases_classY.index), axis=0).fillna(-100)

    #Finally we calculate the similarity between the cases based on the euclidean distance between the confidences of declare rules
    matrixBoundaryCases=cdist(filtered_cases_classX.values, filtered_cases_classY.values)

    print("Minimal distance: "+ str(np.min(matrixBoundaryCases)))

    #               case 1 class Y | case 2 class Y | case 3 class Y| ....
    #case 1 class X      0.5             0.3               0.9
    #case 2 class X      0.4             0.25              0.1
    #....               ....             ...               ...

    indexMinDistance=np.argmin(matrixBoundaryCases)#we obtain the index of the minimum distance
                        
    #Transform the index of the minimum distance to real index:
    min_idx = np.unravel_index(indexMinDistance, matrixBoundaryCases.shape)
    #The x component represents a case of classX, and the y component represents a case of classY, so we filter them:
    min_case_class1=filtered_cases_classX.loc[[list(filtered_cases_classX.index)[min_idx[0]]]]
    min_case_class2=filtered_cases_classY.loc[[list(filtered_cases_classY.index)[min_idx[1]]]]

    return min_case_class1, min_case_class2


    


if __name__ == "__main__":
    training_data=pd.read_csv("./training_data_75tr_25tt.csv",index_col=0)
    test_data=pd.read_csv("./test_data_75tr_25tt.csv",index_col=0)

    y_train=training_data['Class']
    X_train=training_data.drop(columns=['Class'])

    y_test=test_data['Class']
    X_test=test_data.drop(columns=["Class"])
    X_test=X_test[X_train.columns.to_list()]

    le = LabelEncoder()

    y_train_transformed = le.fit_transform(y_train)
    y_test_transformed = le.transform(y_test)
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    perc=0.4

    early_rounds=20
    max_estimators=8
    max_depth=12


    results_search_model = findBestXGBoostModel(X_train,
                                                y_train_transformed,
                                                perc,
                                                X_test,
                                                y_test_transformed,
                                                early_rounds,
                                                max_estimators,
                                                max_depth)

    nameFile="search_val_perc"+str(perc)+"_earlyrounds"+str(early_rounds)+"_"+"maxEstimators"+str(max_estimators)+"_"+"maxDepth"+str(max_depth)+".csv"
    results_search_model.to_csv(nameFile)


    whole_data_traces=pd.read_csv("./test_data_rules.csv",index_col=0).reset_index().drop(columns=['index'])
    y=whole_data_traces['Class']
    X=whole_data_traces.drop(columns=['Class'])

    le_kfold = LabelEncoder()

    y_transformed = le_kfold.fit_transform(y)
    n_splits=3
    results_search_model_kfold_traces = findBestXGBoostModel_withKfold(X,
                                                                       y_transformed,
                                                                       perc,
                                                                       early_rounds,
                                                                       max_estimators,
                                                                       max_depth,
                                                                       n_splits)

    nameFile_kfold="search_val_perc"+str(perc)+"_earlyrounds"+str(early_rounds)+"_"+"maxEstimators"+str(max_estimators)+"_"+"maxDepth"+str(max_depth)+"_onlytraces_kfold.csv"
    results_search_model_kfold_traces.to_csv(nameFile_kfold)