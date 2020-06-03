import pandas as pd
import sklearn
from sklearn_pandas import DataFrameMapper
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt

dict_gc = {'A11':'account_lt_0',
'A12':'account_0_200',
'A13':'account_gt_200_1y',
'A14':'no_account',
'A2':'duration',
'A30':'no_credits_all_paid',
'A31':'all_credits_paid_back_duly',
'A32':'existing_credits_paid_back_duly_now',
'A33':'delay_payingoff_past',
'A34':'critical_account_other_credits_othersB',
'A40':'purpose_car_new',
'A41':'purpose_car_used',
'A42':'purpose_furniture_equipment',
'A43':'purpose_radio_television',
'A44':'purpose_domestic_appliances',
'A45':'purpose_repairs',
'A46':'purpose_education',
'A47':'vacation_or_else',
'A48':'purpose_retraining',
'A49':'purpose_business',
'A410':'purpose_others',
'A5':'credit_amount',
'A61':'saving_lt_100',
'A62':'saving_100_500',
'A63':'saving_500_1000',
'A64':'saving_gt_1000',
'A65':'unknown_no_savings_account',
'A7:':'being_employed',
'A71':'unemployed',
'A72':'employed_lt_1',
'A73':'employed_1_4',
'A74':'employed_4_7',
'A75':'employed_gt_7',
'A8':'installment_rate_disposable_income',
'A91':'male_divorced_separated',
'A92':'female_divorced_separated_married',
'A93':'male_single',
'A94':'male_married_widowed',
'A95':'female_single',
'A101':'none_apply_case',
'A102':'co-applicant_case',
'A103':'guarantor_case',
'A11:':'present_residence_since',
'A121':'property_real',
'A122':'noprop_savings_agreement_lifeinsurance',
'A123':'nopropOrsaving_car_other',
'A124':'unknown_noproperty',
'A13_':'age',
'A141':'bank_other_ins_plan',
'A142':'stores_other_ins_plan',
'A143':'none_other_ins_plan',
'A151':'housing_rent',
'A152':'housing_own',
'A153':'housing_free',
'A16':'number_existing_credits_this_bank',
'A171':'unemployed_unskilled_nonresident',
'A172':'unskilled_resident',
'A173':'skilled_employee_official',
'A174':'management_self-employed_highly_qualified_employee',
'A18':'number_liable_people',
# 'A191':'telephone_none',
# 'A192':'telephone_yes_registered',
# 'A201':'foreign_yes',
# 'A202':'foreign_no',
'A19': 'with_telephone',
'A20': 'not_foreign'
}

"""
loading the csv file
"""
def read_datafile(filepath, nfeatures = 0):
    df = pd.read_csv(filepath, sep=" ", header=None)
    names = ["A"+str(i+1) for i in range(nfeatures)]
    names.append("Class")
    df.columns = names
    return df

def read_datafile_header(filepath):
    df = pd.read_csv(filepath)
    return df

def make_mapper(list_transformed_features):
    return DataFrameMapper([
        (feature,sklearn.preprocessing.LabelBinarizer()) for feature in list_transformed_features
    ])

def make_mapperD(list_transformed_features):
    return DataFrameMapper([
        (feature,sklearn.preprocessing.KBinsDiscretizer()) for feature in list_transformed_features
    ])

def make_mapperLabel(list_transformed_features):
    return DataFrameMapper([
        (feature,sklearn.preprocessing.LabelEncoder()) for feature in list_transformed_features
    ])


def load_data(name="AC"):
    print("Loading dataset ##### " + name + " ##### ...")
    if name == "AC":
        df = read_datafile("data/australian.dat", nfeatures=14)
        df = df.dropna()
        print("Transforming the data ...")
        categorial = ["A4", "A5", "A6", "A8", "A9", "A11", "A12"]
    elif name == "GC":
        df = read_datafile("data/german.data", nfeatures=20)
        df = df.dropna()
        print("Transforming the data ...")
        df["Class"] = df["Class"] - 1
        categorial = ["A1", "A3", "A4", "A6", "A7", "A9", "A10", "A12", "A14", "A15", "A17", "A19", "A20"]
    elif name == "CC":
        df = pd.read_csv("data/risk_factors_cervical_cancer.csv")
        print("Transforming the data ...")
        categorial = [df.columns[i] for i in range(df.shape[1]-1) if i not in [0,2,8,10,3,1,12,25,26,27,5,6]]
        cname = [c for c in df.columns]
        df = df.dropna()
        cname[-1] = "Class"
        df.columns = cname
        # df = df.replace("?", np.nan)
    elif name == "HR":
        df = pd.read_csv("data/hribm_data.csv")
        print("Transforming the data ...")
        cname = [c for c in df.columns]
        df = df.dropna()
        cname[0] = "Class"
        df.columns = cname
        categorial = ["BusinessTravel", "Department", "Education", "EducationField", "EmployeeCount", "EnvironmentSatisfaction",
                      "Gender", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", "Over18", "OverTime", "PerformanceRating",
                      "RelationshipSatisfaction", "StandardHours", "StockOptionLevel", "WorkLifeBalance"]
    elif name == "ICU":
        df = pd.read_csv("data/icu_data.csv")
        print("Transforming the data ...")
        cname = [c for c in df.columns]
        df = df.dropna()
        cname[0] = "Class"
        df.columns = cname
        categorial = ["is_male", "race_white", "race_black", "race_hispanic", "race_other", "metastatic_cancer",
                      "diabetes", "vent", "sepsis_angus", "sepsis_martin", "sepsis_explicit", "septic_shock_explicit", "severe_sepsis_explicit", "sepsis_nqf", "sepsis_cdc",
                      "sepsis_cdc_simple", "sepsis-3"]
    elif name == "HD":
        df = pd.read_csv("data/heart_data.csv")
        print("Transforming the data ...")
        cname = [c for c in df.columns]
        df = df.dropna()
        cname[0] = "Class"
        df.columns = cname
        categorial = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",]
    elif name == "CHURN":
        df = pd.read_csv("data/churn_data.csv")
        print("Transforming the data ...")
        cname = [c for c in df.columns]
        df = df.dropna()
        cname[0] = "Class"
        df.columns = cname
        categorial = cname[1:-3]
        print()
    elif name == "SB":
        df = read_datafile_header("data/spambase_data.csv")
        df = df.dropna()
        print("Transforming the data ...")
        cname = [c for c in df.columns]
        df = df.dropna()
        cname[0] = "Class"
        df.columns = cname
        categorial = []
    elif name == "BC":
        df = read_datafile_header("data/breastcancer_data.csv")
        df = df.dropna()
        print("Transforming the data ...")
        cname = [c for c in df.columns]
        df = df.dropna()
        cname[0] = "Class"
        df.columns = cname
        categorial = []
    else:
        print("The dataset " + name + " does not exist!!")
        exit(0)
    numerical = list(df.columns.values)
    [numerical.remove(x) for x in categorial]
    numerical.remove("Class")
    cname = []
    cX = []
    y = np.array(df[["Class"]])
    y = y.reshape(y.shape[0], 1)
    if len(categorial) > 0:
        for c in categorial:
            index_na = np.array([i for i in range(df[c].values.shape[0]) if df[c].values[i] == "?"], dtype=np.int)
            mapper = make_mapper([c])
            df[c] = df[c].replace("?", df[c].values[0])
            if len(np.unique(df[c].values)) == 1:
                continue
            else:
                x = mapper.fit_transform(df)
                x[index_na] = 0
                for i in range(x.shape[1]):
                    # print(mapper.transformed_names_[i])
                    if name=="GC":
                        if "_" in mapper.transformed_names_[i]:
                            cname.append(dict_gc[mapper.transformed_names_[i].split("_")[1]])
                        else:
                            cname.append(dict_gc[mapper.transformed_names_[i]])
                    else:
                        cname.append(mapper.transformed_names_[i])
                cX.append(np.array(x, dtype=np.int))

        X_cdata = np.hstack(tuple(cX))
    nname = []
    nX = []
    for n in numerical:
        x_nona = np.array([val for val in df[n].values if val != "?"], dtype=np.float)
        index_na = np.array([i for i in range(df[n].values.shape[0]) if df[n].values[i] == "?"], dtype=np.int)
        df[n] = df[n].replace("?", -99)
        x = np.array(df[n].values, dtype=np.float)
        if len(np.unique(x)) == 2:
            print(np.unique(x))
            nname.append("{}_with{}_not{}".format(dict_gc[n], np.max(x), np.min(x)))
            x = (x - np.min(x))/(np.max(x)-np.min(x))
            nX.append(np.array(x.reshape(x.shape[0], 1), dtype=np.int))
        elif len(np.unique(x)) == 1:
            continue
        else:
            print(n)
            discretizer = KBinsDiscretizer(n_bins=4, encode='onehot', strategy='kmeans')
            x_nona = x_nona.reshape(x_nona.shape[0], 1)
            x = x.reshape(x.shape[0], 1)
            discretizer.fit(x_nona)
            discretizer.bin_edges_[0] = np.sort(discretizer.bin_edges_[0])
            x = discretizer.transform(x).toarray()
            x[index_na] = 0.0
            for i in range(x.shape[1]):
                arr = np.round(discretizer.bin_edges_[0], 2)
                if name == "GC":
                    cname.append(dict_gc[n] + "_" + str(arr[i]) + "to" + str(arr[i+1]))
                else:
                    cname.append(n + "_" + str(arr[i]) + "to" + str(arr[i + 1]))
            nX.append(np.array(x, dtype=np.int))
    X_ndata = np.hstack(tuple(nX))
    if len(categorial) > 0:
        datanew = np.hstack((y, X_cdata, X_ndata))
    else:
        datanew = np.hstack((y, X_ndata))
    datanew = datanew.astype(float)
    datanew = datanew.astype(int)
    if name == "SB":
        fName = ["spam"] + cname + nname
    else:
        fName = ["Class"] + cname + nname
    dfnew = pd.DataFrame(datanew)
    dfnew.columns = fName
    if name == "AC":
        dfnew.to_csv("data/new_australian_data.csv", index=False)
    elif name == "GC":
        dfnew.to_csv("data/new_german_data.csv", index=False)
    elif name == "CC":
        dfnew.to_csv("data/new_cervical_data.csv", index=False)
    elif name == "SB":
        dfnew.to_csv("data/new_spambase_data.csv", index=False)
    elif name == "BC":
        dfnew.to_csv("data/new_breastcancer_data.csv", index=False)
    elif name == "HR":
        dfnew.to_csv("data/new_hribm_data.csv", index=False)
    elif name == "ICU":
        dfnew.to_csv("data/new_icu_data.csv", index=False)
    elif name == "HD":
        dfnew.to_csv("data/new_heart_data.csv", index=False)
    elif name == "CHURN":
        dfnew.to_csv("data/new_churn_data.csv", index=False)
    print()
    # mapper = make_mapper(categorial)
    # mapperL = make_mapperLabel(categorial)
    # X1 = mapper.fit_transform(df)
    # X2 = np.array(df[numerical])
    # X_numerical = np.hstack((X1, X2))
    # X1_l = mapperL.fit_transform(df)
    # X_mixed = np.hstack((X1_l, X2))
    # y = np.array(df[["Class"]]).ravel()
    # return X_numerical, X_mixed, y, len(numerical), len(categorial)

load_data("CHURN")

# ['Age', 'Number of sexual partners', 'First sexual intercourse',
#        'Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
#        'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD',
#        'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis',
#        'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
#        'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
#        'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
#        'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
#        'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
#        'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis',
#        'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller',
#        'Citology', 'Biopsy']