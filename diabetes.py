##############################
# Diabete Feature Engineering
##############################

# Problem : Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin edebilecek bir makine
# öğrenmesi modeli geliştirilmesi istenmektedir. Modeli geliştirmeden önce gerekli olan veri analizi ve
# özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır.
# ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan
# Pima Indian kadınları  üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal
# bağımsız değişkenden oluşmaktadır. Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun
# pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# Pregnancies: Number of pregnancies
# Glucose: Glucose
# BloodPressure: Blood pressure (Diastolic)
# SkinThickness: Skin Thickness
# Insulin: Insulin
# BMI: Body mass index
# DiabetesPedigreeFunction: A function that calculates our probability of having diabetes based on our descendants.
# Age: Age (year)
# Outcome: Information whether the person has diabetes or not. Have the disease (1) or not (0)

# TASK  1: Exploratory Data Analysis
#          Step 1: Examine the dataset overall.
#          Step 2: Capture the numerical and categorical variables.
#          Step 3: Perform the analysis of numerical and categorical variables.
#          Step 4: Do the target variable analysis. (The average of the target variables according to the
#                  categorical variables, the average of the numerical variables according to the target variable)
#          Step 5: Outlier observation analysis.
#          Step 6: Missing observation analysis.
#          Step 7: Do a correlation analysis.

# TASK 2: FEATURE ENGINEERING
#          Step 1: Take the necessary actions for missing and outliers.
#                  There are no missing observations in the data set, but Glucose, Insulin, etc. observation units
#                  containing a value of 0 in variables may express the missing value For example, a person's
#                  glucose or insulin value will not be 0 aking this into account, you can assign zero values as
#                  NaN in the corresponding values and then apply operations to the missing values.
#          Step 2: Create new variables.
#          Step 3: Perform the encoding operations.
#          Step 4: Standardize for numerical variables.
#          Step 5: Create a model.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

##################################
# TASK 1: Exploratory Data Analysis
##################################

df_ = pd.read_csv(r"C:\Users\engin\PycharmProjects\MIUUL\venv\5. Feature Engineering\Week5_Homework\diabetes.csv")
df = df_.copy()


# Step 1: Examine dataset


def check_df(data, head=5):
    print("\n******Shape******")
    print(f'Shape     : {df.shape}\n'
          f'Size      : {df.size}\n'
          f'Dimension : {df.ndim}')
    print("\n******Types******")
    print(data.dtypes)
    print("\n******Head******")
    print(data.head(head))
    print("\n******Tail******")
    print(data.tail(head))
    print("\n******Random Sampling******")
    print(data.sample(head))
    print("\n******Missing Values******")
    print(data.isnull().sum())
    print("\n******Duplicated Values******")
    print(data.duplicated().sum())
    print("\n******Unique Values******")
    print(data.nunique())
    print("\n******Describe******")
    print(data.describe().T)


check_df(df)


# Step 2: Grab Columns


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables also include numerical-looking categorical variables.

    Parameters
    ------
        dataframe: dataframe
                The desired dataframe to retrieve variable names
        cat_th: int, optional
               the class threshold value for variables that are numerical but categorical is
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                List of categorical variables
        num_cols: list
               List of numerical variables
        cat_but_car: list
                A categorical-looking list of cardinal variables

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is in cat_cols.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]  # 0,1,2
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]  # name
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if
                col not in cat_but_car]  # since cat_cols holds the entire object data type, it can contain cat_but_car.

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]  # numerik_except for categories with problems

    print(f"Observations: {dataframe.shape[0]}")  # row
    print(f"Variables: {dataframe.shape[1]}")  # columns
    print(f'cat_cols: {len(cat_cols)}')  # the number of categorical variables
    print(f'num_cols: {len(num_cols)}')  # numerical variables
    print(f'cat_but_car: {len(cat_but_car)}')  # categorical, but cardinal
    print(f'num_but_cat: {len(num_but_cat)}')  # numerical-looking categorical

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Step 3: Perform the analysis of numerical and categorical variables.


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(df)}))
    if plot:
        plt.figure(figsize=(8.4, 3.3))
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.tight_layout()
        plt.show(block=True)
    print(80 * "*")


for col in cat_cols:
    if df[col].dtypes == 'bool':
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        plt.figure(figsize=(8.4, 3.3))
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

    print(80 * "*")


for col in num_cols:
    num_summary(df, col, plot=True)


# Step 4: Do the target variable analysis.
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


# Step 5: Outlier observation analysis.

def outlier_thresholds(dataframe, variable, low_quantile=0.05, up_quantile=0.95):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    print(quantile_one, quantile_three)
    print(low_limit, up_limit)
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, replace_with_thresholds(df, col))


# Step 6: Missing observation analysis.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

# Step 7: Correlation analysis.
f, ax = plt.subplots(figsize=[9, 6])
sns.heatmap(df.corr(), cmap="Blues", annot=True);
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

##################################
# TASK 2: FEATURE ENGINEERING
##################################

# Step 1: Take the necessary actions for missing and outliers.
#                  There are no missing observations in the data set, but Glucose, Insulin, etc. observation units
#                  containing a value of 0 in variables may express the missing value For example, a person's
#                  glucose or insulin value will not be 0 aking this into account, you can assign zero values as
#                  NaN in the corresponding values and then apply operations to the missing values.

missing_value = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in missing_value:
    print(col, df.loc[df[col] == 0].shape[0])
    df[col] = np.where(df[col] == 0, np.nan, df[col])

na_cols = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Outcome", na_cols)

msno.bar(df)
plt.show(block=True)

msno.matrix(df)
plt.show(block=True)

msno.heatmap(df)
plt.show(block=True)

df = df.fillna(df.groupby('Outcome').transform('median'))

# Step 2: Create new variables.

df['Glucose_CAT'] = pd.cut(x=df['Glucose'],
                           bins=[0, 140, 199, np.inf],
                           labels=["Normal", "Impaired Glucose Tolerance", "Diabetes"])

df['BMI_CAT'] = pd.cut(x=df['BMI'],
                       bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf],
                       labels=["Under Weight", "Healthy Weight", "Overweight",
                               "Obese Class 1", "Obese Class 2", "Obese Class 3"])

# Step 3: Perform the encoding operations.


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)


def one_hot_encoder(dataframe, categorical_col, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_col, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df.columns if 12 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols, drop_first=True)

# Step 4: Standardize for numerical variables.

ss = StandardScaler()
df[num_cols] = ss.fit_transform(df[num_cols])


# Step 5: Create a model.

y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X)

