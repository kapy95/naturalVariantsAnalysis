import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def representMatrixMeasure(dataframe, class_column, col_num, performance_metric, dirMatrix):
    sns.set_theme(font_scale=1.35) 
    g = sns.FacetGrid(dataframe, col=class_column,height=4, aspect=1.0, palette="muted", col_wrap=col_num)
    g.map(sns.barplot, "Classifier", performance_metric)
    plt.savefig(dirMatrix)
    plt.close()


def generateDataMap(y_real_test_map, y_pred_map, le_name_mapping, labels, dirMap):
    matrix_base=pd.DataFrame.from_dict({"real_labels":y_real_test_map,"predicted_labels":y_pred_map})
    mapping_series=pd.Series(le_name_mapping)
    data_map=list()
    
    for real_label in labels:
        predicted_labels_for_specific_label=matrix_base[matrix_base['real_labels']==real_label]
        vc_predicted_rl=((predicted_labels_for_specific_label['predicted_labels'].value_counts())/len(predicted_labels_for_specific_label))
        vc_predicted_rl=vc_predicted_rl*100

        for label2 in labels:
            predicted_vals=list(vc_predicted_rl.index)
            if label2 in predicted_vals:
                continue
            else:
                vc_predicted_rl[label2]=0
        vc_predicted_rl=vc_predicted_rl.sort_index()
        data_map.append(vc_predicted_rl.values)

    labels_map=[mapping_series[mapping_series==val].index[0] for val in labels]
    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(data_map, xticklabels=labels_map, yticklabels=labels_map, annot=True, cmap="coolwarm", fmt=".2f")

    # Add labels and title
    plt.xlabel("Predicted")
    plt.ylabel("Real value")
    plt.savefig(dirMap)
    plt.close()



if __name__ == "__main__":
    dataset=pd.read_csv("./Data/road_traffic/mined_rtfm_relabelled_confidences.csv", index_col=0)
    dataset = dataset.set_index("case:concept:name")

    X=dataset.drop(columns=["Class"])

    y=dataset['Class']
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
    
    specific_clf=joblib.load("./results/06-16-2025_20-39-30/clf_val_0.2.joblib")
    y_pred_test=specific_clf.predict(X_test)
    dirClfMap="./heatmap_test"+".pdf"
    generateDataMap(y_real_test_map=y_test,
                    y_pred_map=y_pred_test,
                    le_name_mapping=le_name_mapping, 
                    labels=list(le_name_mapping.values()),
                    dirMap=dirClfMap)


