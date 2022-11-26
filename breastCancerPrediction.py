
# Imports:
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
def main():
    data = read_file("BRCA.csv")
    data = remove_null(data)
    print_data(data)

    int_data = int_key(data)

    # Splitting data
    x = np.array(int_data[['Age', 'Gender', 'Protein1', 'Protein2', 'Protein3','Protein4', 
                    'Tumour_Stage', 'Histology', 'ER status', 'PR status', 
                    'HER2 status', 'Surgery_type']])
    y = np.array(int_data[['Patient_Status']])
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
    model = generate_model(xtrain, ytrain)
    prediction(model) # type = numpy.ndarray






def read_file(file):
    data = pd.read_csv(file)
    return data

def remove_null(data):
    print(data.isnull().sum())
    data = data.dropna()
    return data


def print_data(data):
    print(data.head())
    data.info()
    print(data.Gender.value_counts())

def tumor_stage_figure(data):

    # Tumour Stage
    stage = data["Tumour_Stage"].value_counts()
    transactions = stage.index
    quantity = stage.values

    figure = px.pie(data, 
                values=quantity, 
                names=transactions,hole = 0.5, 
                title="Tumour Stages of Patients")
    figure.show()

def histology_figure(data):

# Histology
    histology = data["Histology"].value_counts()
    transactions = histology.index
    quantity = histology.values
    figure = px.pie(data, 
                values=quantity, 
                names=transactions,hole = 0.5, 
                title="Histology of Patients")
    figure.show()

def status_values(data):
    # ER status
    print(data["ER status"].value_counts())
    # PR status
    print(data["PR status"].value_counts())
    # HER2 status
    print(data["HER2 status"].value_counts())

def surgery_figure(data):
    # Surgery_type
    surgery = data["Surgery_type"].value_counts()
    transactions = surgery.index
    quantity = surgery.values
    figure = px.pie(data, 
                values=quantity, 
                names=transactions,hole = 0.5, 
                title="Type of Surgery of Patients")
    figure.show()

def int_key(data):

    data["Tumour_Stage"] = data["Tumour_Stage"].map({"I": 1, "II": 2, "III": 3})
    data["Histology"] = data["Histology"].map({"Infiltrating Ductal Carcinoma": 1, 
                                            "Infiltrating Lobular Carcinoma": 2, "Mucinous Carcinoma": 3})
    data["ER status"] = data["ER status"].map({"Positive": 1})
    data["PR status"] = data["PR status"].map({"Positive": 1})
    data["HER2 status"] = data["HER2 status"].map({"Positive": 1, "Negative": 2})
    data["Gender"] = data["Gender"].map({"MALE": 0, "FEMALE": 1})
    data["Surgery_type"] = data["Surgery_type"].map({"Other": 1, "Modified Radical Mastectomy": 2, 
                                                    "Lumpectomy": 3, "Simple Mastectomy": 4})
    print(data.head())
    return data

def generate_model(xtrain, ytrain):
    # Create model
    model = SVC()
    model.fit(xtrain, ytrain)
    return model

def prediction(model):

    # Prediction
    features = [['Age', 'Gender', 'Protein1', 'Protein2', 'Protein3','Protein4', 'Tumour_Stage', 'Histology', 'ER status', 'PR status', 'HER2 status', 'Surgery_type']]
    features1 = np.array([[36.0, 1, 0.080353, 0.42638, 0.54715, 0.273680, 3, 1, 1, 1, 2, 2,]])
    features2 = np.array([[45.0, 0, 0.000353, 0.42638, 0.54715, 0.273680, 3, 0, 1, 1, 2, 2,]])
    features3 = np.array([[13.0, 1, 0.080353, 0.42638, 0.54715, 0.273680, 3, 1, 1, 1, 2, 2,]])
    features4 = np.array([[22.0, 1, 0.080353, 0.42638, 0.54715, 0.273680, 3, 1, 1, 1, 2, 2,]])
    features5 = np.array([[32.0, 1, 0.080353, 0.42638, 0.54715, 0.273680, 3, 1, 1, 1, 2, 2,]])
    features6 = np.array([[40.0, 1, 0.080353, 0.42638, 0.54715, 0.273680, 3, 1, 1, 1, 2, 2,]])
    features7 = np.array([[53.0, 1, 0.080353, 0.42638, 0.54715, 0.273680, 3, 1, 1, 1, 2, 2,]])
    features8 = np.array([[67.0, 1, 0.080353, 0.42638, 0.54715, 0.273680, 3, 1, 1, 1, 2, 2,]])
    features9 = np.array([[49.0, 1, 0.080353, 0.42638, 0.54715, 0.273680, 3, 1, 1, 1, 2, 2,]])
    features10 = np.array([[89.0, 1, -0.036229,0.79551,-0.013525,1.6299, 3, 3, 1, 1, 1, 2,]])

    features_list = [features1, features2, features3, features4, features5, features6, features7, features8, features9, features10]
    for features in features_list:
        print(model.predict(features))

    print("\n\n\n\n\n\n\n")
    print("Dead?")
    features = np.array([[68.0, 1, 0.64903,1.424,-0.39536, 1.1848, 3, 3, 1, 1, 1, 4,]])

    # TCGA-D8-A4Z1,68,FEMALE,0.64903,1.424,-0.39536,
    # 1.1848,I,Infiltrating Ductal Carcinoma,Positive,Positive,Negative,
    # Simple Mastectomy,07-Dec-17,18-Jul-18,Dead


    print(model.predict(features))

    

main()

