from sklearn.feature_selection import f_classif
import arff, pandas as pd, numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main1():

    with open("diabetes.arff", "r") as file:

        dataset = arff.load(file)
        attribute_names = [attr[0] for attr in dataset['attributes']]
        data_dict = {attr: [] for attr in attribute_names}

        for row in dataset['data']:
            for i, value in enumerate(row):
                data_dict[attribute_names[i]].append(value)

        df = pd.DataFrame(data_dict)

        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        f_values, p_values = f_classif(X, y)

        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'F-Value': f_values,
            'p-Value': p_values
        })

        feature_scores = feature_scores.sort_values(by='F-Value', ascending=False)

        best_feature = feature_scores.iloc[0]['Feature']
        worst_feature = feature_scores.iloc[-1]['Feature']

        print(feature_scores)
main1()

