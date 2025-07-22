import shap
import pandas as pd
import matplotlib.pyplot as plt
import os

def calculateSHAP_values(model, X):
    """
    Description: function to calculate shap values for a certain trained and loaded model using joblib.

    input: 
        model:trained model already loaded (in our case a classifier)
        X: data to obtain the shapley values (i.e., the explanations). It can be any data (traing, test...)
    
    output:
        shap_values: importance values for each feature in each instance of X based on the model (i.e., classifier)

    """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.values.tolist())

    return shap_values, explainer


def getMostAbsoluteImportantFeatures(shap_values, feature_names):
    """
    Description: function to calculate the most important features involved in the classification in absolute terms (with absolute we refer that they have a negative and a positive influence).

    Input:
        - shap_values: shapley values indicating the importance of each feature for each instance of each class. It is structured like a list of lists, 
        where the first represents the instances of class 1 involved in the classication (labeled with 1 by the the label encoder). Then inside each class list, 
        there is a list of lists where each one has a size equal to the number of features, and contains the corresponding importances:

        [class1: [importances_instance1_for_class1, importances_instance2_for_class1...],
         class2: [importances_instance1_for_class2, importances_instance2_for_class2...],
         class3: [importances_instance1_for_class3, importances_instance2_for_class3...],
         ...
         ]

         -feature names: names of the features sorted like in the data used to train the classifier
    
    Output:
        - most_importances_features: list of most important features in absolute terms
        - dict_absolute_importance_feature_per_class: dictionary whose keys are the classes in encoded format (e.g., 0 instance of normal payments), 
          and the values are dictionaries whose keys are the names of features and the values the absolute mean importance for that class.

    """

    dict_absolute_importance_feature_per_class={}
    most_important_features=[]
    i=0
    for shap_values_class in shap_values:
        importances_class=pd.DataFrame(shap_values_class, columns=feature_names)
        absolute_importances_class=abs(importances_class)
        mean_absolute_importances=absolute_importances_class.mean()
        filtered_absolute_importances=mean_absolute_importances[mean_absolute_importances>0].to_dict()
        most_important_features.extend(list(filtered_absolute_importances.keys()))
        dict_absolute_importance_feature_per_class[i]=filtered_absolute_importances
        i=i+1
    
    most_important_features=list(set(most_important_features))

    return most_important_features, dict_absolute_importance_feature_per_class


def representViolinFeatureImportancesClassX(shap_values, class_index, data_feature_names, number_features):
    """
    Description: function to plot a violin graph of the feature importances for a certain class. A limited number of features are displayed.

    Input:
        - shap_values: shapley values containing the importance of each feature for each instance of each class
        - class_index: class in encoded format (e.g., 0 instance of normal payments)
        - data_feature_names: names of the features 
        - number_features: maximal amount of features to be displayed
    
    Output:
        - graph in violin format displaying the direction (positive or negative) for each feature
    
    """
    shap.plots.violin(shap_values[class_index], feature_names=data_feature_names, max_display=number_features)



def plotGlobalFeatureImportances(X_test, shap_values, dict_class_names, dirPlot):

    """
    Description: function to plot the 10 most absolute important features involved in the classification, highlighting in which classes they have a high absolute influence. The influence is absolute, so it could have both a positive and a negative influence in a class (e.g. suppose that a class has two instances, and feature F1 has an influence of 1.5 and -0.5, its mean absolute is 1.0)
    
    Input: 
        - X_test: data used to obtain explanations
        - shap_values: values explaining the importance of each feature for each instance based on the classifier
        - dict_class_names: dictionary whose keys are the transformed labels by a label enconder, and the values are the real names of labels. Example:

          Structure of dict_class_names: ={0:'Normal process',
                                     1:'Multiple payments',
                                     2:'More than 1 positive answer',
                                     3:'More than one Send documents->receive positive answer', 
                                     4:'Bad order+multiple payments'}
    
    Output: representation of the 10 most important features relating them to the corresponding clases where they are influencing
    """
    shap.summary_plot(shap_values,
                      X_test,
                      plot_type="bar",
                      class_names=dict_class_names,
                      class_inds='original', 
                      max_display=10,
                      show=False)
    plt.savefig(dirPlot,bbox_inches='tight')
    plt.close()
    # plt.show()


def plotFeatureImportanceForSpecificInstance(shap_values_instance, expected_value, feature_values_instance, X_test_index, dirImage=None):
    """
    Description:function to plot the feature importances in the predictions for a specific instance in X_test

    Input:
        - X_test: test data
        - shap_values: shapley values calculated for X_test
        - explainer: model to explain the predictions which is obtained by SHAP library
        - class_index: index used to represent the class in the label encoder (e.g., 0->normal payments)
        - instance index: index of the instance inside its class (e.g, first instance of normal payments would be 0)
        - X_test_index: index of the instance in X_test which you want to check its importances
    
    Output:
        - Barplot containing the 20 most important features for that specific data instance

    """

    # SHAP values for the selected instance and class 
    shap.waterfall_plot(shap.Explanation(values=shap_values_instance, 
                                         base_values=expected_value, 
                                         data=feature_values_instance),
                                         max_display=10,
                                         show=False)
    if dirImage is not None:
        plt.savefig(dirImage+"/shapley_10_most_important_features_instance"+str(X_test_index)+".pdf",bbox_inches='tight')
        plt.close()



def getSHAP_Values_for_instance(shap_values, class_index, instance_index):
    shap_values_instance = shap_values[class_index][instance_index]
    return shap_values_instance

