from ensembleTree import createAndTrainXGB_early_stop, measureClassifierPerformance, boundaryCasesBetweenTwoClasses, findChampionsInSpecificClassProbs
from performanceResults import representMatrixMeasure
from sklearn.model_selection import train_test_split
import os
import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np



import pm4py


#Library to load models:
import joblib

#SHAP functions:
from SHAP_utils import calculateSHAP_values, getMostAbsoluteImportantFeatures, plotGlobalFeatureImportances, plotFeatureImportanceForSpecificInstance, getSHAP_Values_for_instance

def main(X_train, y_train, X_test, y_test, case_sizes, param_dist, sizes, stratf, le_name_mapping, dirRes, training, classifier_route):
    classifiers={}
    if training==True:
        print("Training classifiers....")
        for size in sizes:
            clf_specific_size_val=createAndTrainXGB_early_stop(X_train, 
                                                            y_train, 
                                                            param_dist, 
                                                            size, 
                                                            stratf)
            nameClf="clf_val_"+str(size)
            classifiers[nameClf]=clf_specific_size_val
        
        print("Measuring performance...")
        df_performance_classifiers, df_accus=measureClassifierPerformance(classifiers, 
                                                                                X_test, 
                                                                                y_test,
                                                                                le_name_mapping,
                                                                                dirRes)
        df_performance_classifiers.to_csv(dirRes+"/performance_classifiers.csv")
        df_accus.to_csv(dirRes+"/accus_classifiers.csv")
        
        dirMatrixes=dirRes+"/"+"matrix_classes_"
        metrics=["Precision", "Recall", "f1-scores"]

        print("Representing performance results...")
        for metric in metrics:
            specific_matrix_name=dirMatrixes+metric+".pdf"
            representMatrixMeasure(dataframe=df_performance_classifiers, 
                                class_column="Class", 
                                col_num=2, 
                                performance_metric=metric, 
                                dirMatrix=specific_matrix_name)

        #Get the best classifier based on accuracy:
        best_classifier_name=df_accus.sort_values("Accuracy",ascending=False)["Classifier"][0]    
        print("The best classifier is: "+best_classifier_name)
        best_classifier=classifiers[best_classifier_name]
        
    else:
        print("Loading trained classifier...")
        best_classifier=joblib.load(classifier_route)

    print("Calculating shapley values...")

    #Use the best classifier and the test data to obtain Shapley Values that explain how the classifier works:
    shap_values_training, explainer=calculateSHAP_values(best_classifier, X_train)

    print("Finding most important features of the classifier...")
    #Based on the shap values find the most important features following the classifier criterion in absolute terms (check function if you do not understand the "absolute terms" part):
    most_important_features, dict_absolute_importance_feature_per_class=getMostAbsoluteImportantFeatures(shap_values_training,
                                                                                                         list(X_train.columns))
    #Now we invert the mapping dictionary to generate a plot with the absolute importances  
    inv_mapping = {v: k for k, v in le_name_mapping.items()}
    dirPlotShap=dirRes+"/global_importances_shap.pdf"
    plotGlobalFeatureImportances(X_train, shap_values_training, inv_mapping, dirPlotShap)#we plot the importances at global level using SHAP 


    dict_probs_cases_per_label={}
    classes=le_name_mapping.keys()

    #Calculate shap values for test:
    shap_values_test=explainer.shap_values(X_test)

    for encoded_class_label in set(y_train):#for each class
        print("Finding champion for class "+str(encoded_class_label)+"...")
        positions_train_instances_class_label=np.where(y_train == encoded_class_label)#find the indices of the instances of that class in the train data
        X_train_class_label=X_train.iloc[positions_train_instances_class_label]#filter the test data to just contain the instances related to that class

        #Find the champion of the class (i.e., the instance with a highest predicted probability to belong to its class).
        champion_class_label=findChampionsInSpecificClassProbs(X_train_class_label,
                                                      encoded_class_label,
                                                      best_classifier,
                                                      case_sizes)

        #save the champions in csv format
        champion_class_label.to_csv(dirRes+"/best_predicted_instance_class"+str(encoded_class_label)+".csv")

        #after find the champion, we prepare the test instances related to that class so that we can look later for the boundary cases:
        positions_test_instances_class_label=np.where(y_test == encoded_class_label)#we find which instances of the test data belong to that class
        X_test_class_label=X_test.iloc[positions_test_instances_class_label]#we filter them

        predicts_cases_bin_class=best_classifier.predict(X_test_class_label)#we obtain their predicted classes
        correct_predictions=[1 if prediction==encoded_class_label else 0 for prediction in predicts_cases_bin_class]#if it matches its real class, we will filter them
        x_test_filtered = X_test_class_label[[bool(x) for x in correct_predictions]]

        #we obtain the predicted probabilty of each class for each filtered instance:
        probs_y_test=best_classifier.predict_proba(x_test_filtered)
        cases=x_test_filtered.index.to_list()#we obtain their case ids
        if x_test_filtered.empty:
            print("No correctly classified cases in test set")
        #Transform the probabilities of the filtered instances into a dataframe 
        df_probs_class_label = pd.DataFrame(probs_y_test, columns= classes, index=cases)

        #format of the dataframe:
        #                                           probs to belong to class X | probs to belong to class Y | probs to belong to class Z     |  .....                                 
        #instance 1 correctly classfied in class X              0.7                         0.20                        0.10                     ...
        #instance N correctly classfied in class X              0.6                         0.35                        0.05                     ...
        #                    ...                                ...                         ...                         ...                      ...

        #and save it in a dictionary along with the instances:
        dict_probs_cases_per_label[encoded_class_label]=(x_test_filtered, df_probs_class_label)

    #Once we have the cases correctly classified for each class, and the champions, we find the boundary cases for each possible combination:
    #For instance consider that we have four classes [A,B,C,D].We will obtain all possible combinations with the following for loops:
    dirAllBoundaryCases=dirRes+"/boundaryCases"
    os.mkdir(dirAllBoundaryCases)

    for i in range(0,len(classes)-1):#e.g. [A,B,C]
        df_probs_classI = dict_probs_cases_per_label[i][1]
        for j in range(i+1,len(classes)):#e.g., [B,C,D]
            df_probs_classJ = dict_probs_cases_per_label[j][1]
            print("Finding boundary cases for classes "+str(i)+" and "+str(j)+"....")

            #for each possible combination find the boundary cases based on probabilities:
            boundaryCaseClassX, boundaryCaseClassJ=boundaryCasesBetweenTwoClasses(df_probs_class_x=df_probs_classI,
                                                                                  df_probs_class_y=df_probs_classJ,
                                                                                  class_labelX=i,
                                                                                  class_labelY=j,
                                                                                  X_test=X_test)
            #Filter only the columns of the cases that the most important
            boundaryCaseClassX=boundaryCaseClassX[most_important_features]
            boundaryCaseClassJ=boundaryCaseClassJ[most_important_features]

            #concatenate them in one dataframe
            boundaryCases=boundaryCaseClassX.append(boundaryCaseClassJ)

            #save the dataframe in a csv
            boundaryCases.to_csv(dirRes+"/boundaryCases_Classes"+str(i)+"-"+str(j)+".csv")
            
            #we will obtain the features related to each boundary case:
            identifiers=boundaryCases.index.to_list()#firstly we obtain the identifiers
            
            #using the string index we obtain the feature values of the boundary case of class X
            feature_values_instance_x=X_test.loc[identifiers[0]]

            #we will also obtain its shapley values, for that we need the numerical index inside its class (e.g., it is instance 5 of 1000 instances in class X). This is due to, in our case
            #the structure storing the shapley values has N lists corresponding to the n classes of the classifier. Inside each class list, there are more lists, each one correspond to the 
            #contributions of the features for each instance of each class which are used in a multiclassification problem to estimate the contributions (all instances are used independently of
            #their classes)
            #For instance consider the following dataframe:
            #                   Feature AB | Feature AC | .....
            #instance 1 class X    ...          ...
            #instance 1 class Y    ...          ...
            #instance 2 class X    ...          ...
            #instance 1 class Z    ...          ...
            #instance 3 class X    ...          ...
            #       ...            ...          ...
            # Then the shapley values will be stored like this (if there are three classes)
            # shap values=[ [contributions for class X], [contributions for class Y], [contributions for class Z ]]
            # shap values=[ [ [contributions instance 1],[contributions isntance 2],[contributions isntance 3],... ],  ....]    

            #we obtain the numerical identifier related to boundary case of class X (i.e., its row number)
            numerical_index_boundary_case_x=X_test.index.get_indexer([identifiers[0]])[0]

            #We also the expected value of that class to perform a representation, these values are contained in a list where each position corresponds to the 
            expected_value_x = explainer.expected_value[i]
            
            #with all shap_values, the class in numerical format, and the numerical index inside its class, we can obtain its shapley values.
            shap_values_boundary_case_x=getSHAP_Values_for_instance(shap_values_test,
                                                                    i,
                                                                    numerical_index_boundary_case_x)
            
            #we do the same for boundary case of class Y
            numerical_index_boundary_case_y=X_test.index.get_indexer([identifiers[1]])[0]
            expected_value_y = explainer.expected_value[j]
            feature_values_instance_y=X_test.loc[identifiers[1]]
            shap_values_boundary_case_y=getSHAP_Values_for_instance(shap_values_test,
                                                                    j,
                                                                    numerical_index_boundary_case_y)

            #we create a directory to store plots related to the contributions
            specific_directory=dirAllBoundaryCases+"/boundaryCases"+str(i)+"-"+str(j)
            os.mkdir(specific_directory)

            plotFeatureImportanceForSpecificInstance(shap_values_boundary_case_x, expected_value_x, feature_values_instance_x, identifiers[0], specific_directory)
            plotFeatureImportanceForSpecificInstance(shap_values_boundary_case_y, expected_value_y, feature_values_instance_y, identifiers[1], specific_directory)



dataset=pd.read_csv("./Data/road_traffic/mined_rtfm_relabelled_confidences.csv", index_col=0)
dataset = dataset.set_index("case:concept:name")
rtfm_raw=pm4py.read_xes("./Data/road_traffic/RawData/Road_Traffic_Fine_Management_Process.xes")
case_sizes=rtfm_raw.groupby(by=["case:concept:name"]).apply(lambda x: len(x)).to_dict()

series_case_sizes=pd.Series(case_sizes)

X=dataset.drop(columns=["Class"])

y=dataset['Class']
y=y.replace(to_replace = ['credit_collection', 'paid_full'], value = ['collected', 'fully_paid'])#replace some class names to adjust them to the ones in the paper
print("No. of features:"+str(len(X.columns)))

le = LabelEncoder()
print("Is na? "+str(X.isnull().values.any()))
y_transformed = le.fit_transform(y)
le_name_mapping = dict(zip(le.classes_,le.transform(le.classes_)))
cols=X.columns.to_list()
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y_transformed,
                                                    test_size=0.2,
                                                    stratify=y_transformed,
                                                    shuffle=True,#disorder the data
                                                    random_state=0)


resultsFolder="./results/Ours/"
now=datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
dirResults=resultsFolder+now
os.mkdir(dirResults)
sizes_val=[0.2,0.3,0.4]
stratf=True


param_dist={
    "objective":"multi:softprob",
    "num_class":len(set(y_transformed)),#it is necessary to indicate the number of classes in the data: set-> only obtain an instance of each class, len-> provide the number of classes
    "eta":0.2,
    "early_stopping_rounds":5,
    "max_depth":10,
    "booster":"gbtree",
    "seed":0
    }

classifier_route=""
training=True

main(X_train, y_train, X_test, y_test, series_case_sizes, param_dist, sizes_val, stratf, le_name_mapping, dirResults, training, classifier_route)

