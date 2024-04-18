#---------------------------------------------------------------------------------------------------------------------------------
### Authenticator
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit as st
#---------------------------------------------------------------------------------------------------------------------------------
### Template Graphics
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit.components.v1 as components
#---------------------------------------------------------------------------------------------------------------------------------
### Import Libraries
#---------------------------------------------------------------------------------------------------------------------------------
from streamlit_extras.stoggle import stoggle
#from mitosheet.streamlit.v1 import spreadsheet
#from ydata_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report
#----------------------------------------
import os
import time
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from random import randint
#----------------------------------------
import boto3
from botocore.exceptions import NoCredentialsError
#----------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#----------------------------------------
from scipy.stats import zscore
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
#----------------------------------------
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
#import sklearn.metrics as metrics
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, chi2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
#----------------------------------------
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.inspection import permutation_importance
#----------------------------------------
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as pyoff
#----------------------------------------
import sweetviz as sv
import shap
#from streamlit_shap import st_shap
#from mitosheet.streamlit.v1 import spreadsheet
#from st_aggrid import AgGrid
#----------------------------------------
# importing pages
#from degn_ewatch_eda import degn_ewatch_eda
#from degn_ewatch_model import degn_ewatch_model
#----------------------------------------
import pygwalker as pyg
import streamlit.components.v1 as components
#----------------------------------------
from sklearn.cluster import KMeans
#----------------------------------------
#from pycaret.regression import *
#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
#import custom_style()
#image = Image.open('Image_Clariant.png')
st.set_page_config(page_title="Regression App | v1.0",
                   layout="wide",
                   #page_icon=               
                   initial_sidebar_state="collapsed")
#----------------------------------------
st.title("ML | Regression App | v1.0")
st.markdown('Created by | <a href="mailto:avijit.mba18@gmail.com">Avijit Chakraborty</a>', 
            unsafe_allow_html=True)
st.info('**Disclaimer : :blue[Thank you for visiting the app] | Unauthorized uses or copying of the app is strictly prohibited | Click the :blue[sidebar] to follow the instructions to start the applications.**', icon="ℹ️")
#----------------------------------------
# Set the background image
st.divider()
#---------------------------------------------------------------------------------------------------------------------------------
stats_expander = st.expander("**Knowledge**", expanded=False)
with stats_expander: 
      st.info('''
        **Regression**
                       
        - A supervised learning method which generates a set of statistical processes for estimating the relationships between a dependent variable and one or more independent variable.
        ''')
st.divider()
#---------------------------------------------------------------------------------------------------------------------------------
### Feature Import
#---------------------------------------------------------------------------------------------------------------------------------
st.sidebar.header("Contents", divider='blue')
st.sidebar.info('Please choose from the following options and follow the instructions to start the application.', icon="ℹ️")
data_source = st.sidebar.radio("**:blue[Select Data Source]**", ["Local Machine", "Server Path"])
#---------------------------------------------------------------------------------------------------------------------------------

if data_source == "Local Machine" :
    
    file1 = st.sidebar.file_uploader("**:blue[Choose a file]**",
                                type=["xlsx","csv"],
                                accept_multiple_files=True,
                                key=0)
    if file1 is not None:
        df = pd.DataFrame()
        for file1 in file1:
            df = pd.read_csv(file1)
            #st.subheader("Preview of the Input Dataset :")
            #st.write(df.head(3))

# Dataset preview
            if st.sidebar.checkbox("**Preview Dataset**"):
                number = st.sidebar.slider("**Select No of Rows**",0,df.shape[0],3,5)
                st.subheader("**Preview of the Input Dataset :**",divider='blue')
                st.write(df.head(number))
            st.sidebar.divider()

            if st.sidebar.checkbox("**🗑️ Feature Drop**"):
                feature_to_drop = st.sidebar.selectbox("**Select Feature to Drop**", df.columns)
                #df_dropped = df.drop(columns=[feature_to_drop])
                if feature_to_drop:
                    #col1, col2, col3 = st.columns([1, 0.5, 1])
                    if st.sidebar.button("Apply", key="delete"):
                        st.session_state.delete_features = True
                        st.session_state.df = df.drop(feature_to_drop, axis=1)
            st.sidebar.divider() 

            st.sidebar.subheader("2. Variables Selection", divider='blue')
            target_variable = st.sidebar.multiselect("**2.1 Target (Dependent) Variable**", df.columns)
            #feature_columns = st.sidebar.multiselect("**2.2 Independent Variables**", df.columns)  

#---------------------------------------------------------------------------------------------------------------------------------
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["**Information**",
                                                                            "**Visualizations**",
                                                                            "**Cleaning**",
                                                                            "**Transformation**",
                                                                            "**Selection**",
                                                                            "**Development & Tuning**",
                                                                            "**Performance**",
                                                                            "**Validation**",
                                                                            "**Cross-Check**"])

#---------------------------------------------------------------------------------------------------------------------------------
### Informations
#---------------------------------------------------------------------------------------------------------------------------------

            with tab1:
                st.subheader("**Data Analysis**",divider='blue')

                col1, col2, col3, col4, col5, col6 = st.columns(6)

                col1.metric('**Number of input values (rows)**', df.shape[0], help='number of rows in the dataframe')
                col2.metric('**Number of variables (columns)**', df.shape[1], help='number of columns in the dataframe')     
                col3.metric('**Number of numerical variables**', len(df.select_dtypes(include=['float64', 'int64']).columns), help='number of numerical variables')
                col4.metric('**Number of categorical variables**', len(df.select_dtypes(include=['object']).columns), help='number of categorical variables')
                st.divider()           

                stats_expander = st.expander("**Exploratory Data Analysis (EDA)**", expanded=False)
                with stats_expander:        
                    #pr = df.profile_report()
                    #st_profile_report(pr)
                    st.table(df.head())

#---------------------------------------------------------------------------------------------------------------------------------
### Visualizations
#---------------------------------------------------------------------------------------------------------------------------------

            with tab2: 

                #stats_expander = st.expander("**Visualization**", expanded=False)
                #with stats_expander: 
                    #dabl.plot(df, 'target_variable')
                    #st.pyplot()

                st.subheader("Visualization | Playground",divider='blue')
                    
                pyg_html = pyg.to_html(df)
                components.html(pyg_html, height=1000, scrolling=True)

#---------------------------------------------------------------------------------------------------------------------------------
### Feature Cleaning
#---------------------------------------------------------------------------------------------------------------------------------

            with tab3:
                
                st.subheader("Missing Values Check & Treatment",divider='blue')
                col1, col2 = st.columns((0.2,0.8))

                with col1:
                    def check_missing_values(data):
                            missing_values = df.isnull().sum()
                            missing_values = missing_values[missing_values > 0]
                            return missing_values 
                    missing_values = check_missing_values(df)

                    if missing_values.empty:
                            st.success("**No missing values found!**")
                    else:
                            st.warning("**Missing values found!**")
                            st.write("**Number of missing values:**")
                            st.table(missing_values)

                            with col2:        
                                #treatment_option = st.selectbox("**Select a treatment option**:", ["Impute with Mean","Drop Missing Values", ])
        
                                # Perform treatment based on user selection
                                #if treatment_option == "Drop Missing Values":
                                    #df = df.dropna()
                                    #st.success("Missing values dropped. Preview of the cleaned dataset:")
                                    #st.table(df.head())
            
                                #elif treatment_option == "Impute with Mean":
                                    #df = df.fillna(df.mean())
                                    #st.success("Missing values imputed with mean. Preview of the imputed dataset:")
                                    #st.table(df.head())
                                 
                                # Function to handle missing values for numerical variables
                                def handle_numerical_missing_values(data, numerical_strategy):
                                    imputer = SimpleImputer(strategy=numerical_strategy)
                                    numerical_features = data.select_dtypes(include=['number']).columns
                                    data[numerical_features] = imputer.fit_transform(data[numerical_features])
                                    return data

                                # Function to handle missing values for categorical variables
                                def handle_categorical_missing_values(data, categorical_strategy):
                                    imputer = SimpleImputer(strategy=categorical_strategy, fill_value='no_info')
                                    categorical_features = data.select_dtypes(exclude=['number']).columns
                                    data[categorical_features] = imputer.fit_transform(data[categorical_features])
                                    return data            

                                numerical_strategies = ['mean', 'median', 'most_frequent']
                                categorical_strategies = ['constant','most_frequent']
                                st.write("**Missing Values Treatment:**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    selected_numerical_strategy = st.selectbox("**Select a strategy for treatment : Numerical variables**", numerical_strategies)
                                with col2:
                                    selected_categorical_strategy = st.selectbox("**Select a strategy for treatment : Categorical variables**", categorical_strategies)  
                                
                                #if st.button("**Apply Missing Values Treatment**"):
                                cleaned_df = handle_numerical_missing_values(df, selected_numerical_strategy)
                                cleaned_df = handle_categorical_missing_values(cleaned_df, selected_categorical_strategy)   
                                st.table(cleaned_df.head(2))

                                # Download link for treated data
                                st.download_button("**Download Treated Data**", cleaned_df.to_csv(index=False), file_name="treated_data.csv")

                #with col2:

                st.subheader("Duplicate Values Check",divider='blue') 
                if st.checkbox("Show Duplicate Values"):
                    if missing_values.empty:
                        st.table(df[df.duplicated()].head(2))
                    else:
                        st.table(cleaned_df[cleaned_df.duplicated()].head(2))

                #with col4:

                    #x_column = st.selectbox("Select x-axis column:", options = df.columns.tolist()[0:], index = 0)
                    #y_column = st.selectbox("Select y-axis column:", options = df.columns.tolist()[0:], index = 1)
                    #chart = alt.Chart(df).mark_boxplot(extent='min-max').encode(x=x_column,y=y_column)
                    #st.altair_chart(chart, theme=None, use_container_width=True)  


                st.subheader("Outliers Check & Treatment",divider='blue')
                def check_outliers(data):
                                # Assuming we're checking for outliers in numerical columns
                                numerical_columns = data.select_dtypes(include=[np.number]).columns
                                outliers = pd.DataFrame(columns=['Column', 'Number of Outliers'])

                                for column in numerical_columns:
                                    Q1 = data[column].quantile(0.25)
                                    Q3 = data[column].quantile(0.75)
                                    IQR = Q3 - Q1

                                    # Define a threshold for outliers
                                    threshold = 1.5

                                    # Find indices of outliers
                                    outliers_indices = ((data[column] < Q1 - threshold * IQR) | (data[column] > Q3 + threshold * IQR))

                                    # Count the number of outliers
                                    num_outliers = outliers_indices.sum()
                                    outliers = outliers._append({'Column': column, 'Number of Outliers': num_outliers}, ignore_index=True)

                                return outliers

                if missing_values.empty:
                    df = df.copy()
                else:
                    df = cleaned_df.copy()

                col1, col2 = st.columns((0.2,0.8))

                with col1:
                        # Check for outliers
                        outliers = check_outliers(df)

                        # Display results
                        if outliers.empty:
                            st.success("**No outliers found!**")
                        else:
                            st.warning("**Outliers found!**")
                            st.write("**Number of outliers:")
                            st.table(outliers)
                    
                with col2:
                        # Treatment options
                        treatment_option = st.selectbox("**Select a treatment option:**", ["Cap Outliers","Drop Outliers", ])

                            # Perform treatment based on user selection
                        if treatment_option == "Drop Outliers":
                                df = df[~outliers['Column'].isin(outliers[outliers['Number of Outliers'] > 0]['Column'])]
                                st.success("Outliers dropped. Preview of the cleaned dataset:")
                                st.write(df.head())

                        elif treatment_option == "Cap Outliers":
                                df = df.copy()
                                for column in outliers['Column'].unique():
                                    Q1 = df[column].quantile(0.25)
                                    Q3 = df[column].quantile(0.75)
                                    IQR = Q3 - Q1
                                    threshold = 1.5

                                    # Cap outliers
                                    df[column] = np.where(df[column] < Q1 - threshold * IQR, Q1 - threshold * IQR, df[column])
                                    df[column] = np.where(df[column] > Q3 + threshold * IQR, Q3 + threshold * IQR, df[column])

                                    st.success("Outliers capped. Preview of the capped dataset:")
                                    st.write(df.head())

#---------------------------------------------------------------------------------------------------------------------------------
### Feature Encoding
#---------------------------------------------------------------------------------------------------------------------------------

                #for feature in df.columns: 
                    #if df[feature].dtype == 'object': 
                        #print('\n')
                        #print('feature:',feature)
                        #print(pd.Categorical(df[feature].unique()))
                        #print(pd.Categorical(df[feature].unique()).codes)
                        #df[feature] = pd.Categorical(df[feature]).codes

#---------------------------------------------------------------------------------------------------------------------------------
### Feature Transformation
#---------------------------------------------------------------------------------------------------------------------------------

            with tab4:
                
                    st.subheader("Feature Encoding",divider='blue')

                    # Function to perform feature encoding
                    def encode_features(data, encoder):
                        if encoder == 'Label Encoder':
                            encoder = LabelEncoder()
                            encoded_data = data.apply(encoder.fit_transform)
                        elif encoder == 'One-Hot Encoder':
                            encoder = OneHotEncoder(drop='first', sparse=False)
                            encoded_data = pd.DataFrame(encoder.fit_transform(data), columns=encoder.get_feature_names(data.columns))
                        return encoded_data
                    
                    encoding_methods = ['Label Encoder', 'One-Hot Encoder']
                    selected_encoder = st.selectbox("**Select a feature encoding method**", encoding_methods)
                    
                    encoded_df = encode_features(df, selected_encoder)
                    st.table(encoded_df.head(2))                   
                

                    st.subheader("Feature Scalling",divider='blue') 

                    # Function to perform feature scaling
                    def scale_features(data, scaler):
                        if scaler == 'Standard Scaler':
                            scaler = StandardScaler()
                        elif scaler == 'Min-Max Scaler':
                            scaler = MinMaxScaler()
                        elif scaler == 'Robust Scaler':
                            scaler = RobustScaler()

                        scaled_data = scaler.fit_transform(data)
                        scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
                        return scaled_df
                    
                    scaling_methods = ['Standard Scaler', 'Min-Max Scaler', 'Robust Scaler']
                    selected_scaler = st.selectbox("**Select a feature scaling method**", scaling_methods)

                    if st.button("**Apply Feature Scalling**", key='f_scl'):
                        scaled_df = scale_features(encoded_df, selected_scaler)
                        st.table(scaled_df.head(2))
                    else:
                         df = encoded_df.copy()
#---------------------------------------------------------------------------------------------------------------------------------
### Feature Selection
#---------------------------------------------------------------------------------------------------------------------------------

            with tab5:

                st.subheader("Feature Selection:",divider='blue')    
                #target_variable = st.multiselect("**Target (Dependent) Variable**", df.columns)

                col1, col2 = st.columns(2) 

                with col1:
                    #st.subheader("Feature Selection (Method 1):",divider='blue')
                    st.markdown("**Method 1 : Checking VIF Values**")
                    vif_threshold = st.number_input("**VIF Threshold**", 1.5, 10.0, 5.0)

                    def calculate_vif(data):
                        X = data.values
                        vif_data = pd.DataFrame()
                        vif_data["Variable"] = data.columns
                        vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
                        vif_data = vif_data.sort_values(by="VIF", ascending=False)
                        return vif_data

                    # Function to drop variables with VIF exceeding the threshold
                    def drop_high_vif_variables(data, threshold):
                        vif_data = calculate_vif(data)
                        high_vif_variables = vif_data[vif_data["VIF"] > threshold]["Variable"].tolist()
                        data = data.drop(columns=high_vif_variables)
                        return data
                                       
                    st.markdown(f"Iterative VIF Thresholding (Threshold: {vif_threshold})")
                    #X = df.drop(columns = target_variable)
                    vif_data = drop_high_vif_variables(df, vif_threshold)
                    vif_data = vif_data.drop(columns = target_variable)
                    selected_features = vif_data.columns
                    st.write("#### Selected Features (considering VIF values in ascending orders)")
                    st.table(selected_features)
                    #st.table(vif_data)

                with col2:

                    #st.subheader("Feature Selection (Method 2):",divider='blue')                        
                    st.markdown("**Method 2 : Checking Selectkbest Method**")          
                    method = st.selectbox("**Select Feature Selection Method**", ["SelectKBest (f_classif)", "SelectKBest (f_regression)", "SelectKBest (chi2)"])
                    num_features_to_select = st.slider("**Select Number of Independent Features**", min_value=1, max_value=len(df.columns), value=5)

                    # Perform feature selection
                    if "f_classif" in method:
                            feature_selector = SelectKBest(score_func=f_classif, k=num_features_to_select)

                    elif "f_regression" in method:
                            feature_selector = SelectKBest(score_func=f_regression, k=num_features_to_select)

                    elif "chi2" in method:
                            # Make sure the data is non-negative for chi2
                            df[df < 0] = 0
                            feature_selector = SelectKBest(score_func=chi2, k=num_features_to_select)
                    
                    X = df.drop(columns = target_variable)  # Adjust 'Target' to your dependent variable
                    y = df[target_variable]  # Adjust 'Target' to your dependent variable
                    X_selected = feature_selector.fit_transform(X, y)

                    # Display selected features
                    selected_feature_indices = feature_selector.get_support(indices=True)
                    selected_features_kbest = X.columns[selected_feature_indices]
                    st.write("#### Selected Features (considering values in 'recursive feature elimination' method)")
                    st.table(selected_features_kbest)
    
#---------------------------------------------------------------------------------------------------------------------------------
### Model Development & Tuning
#---------------------------------------------------------------------------------------------------------------------------------

            with tab6:

                col1, col2, col3 = st.columns(3)   

                with col1:
                     
                    st.subheader("Dataset Splitting Criteria",divider='blue')
                    test_size = st.slider("Test Size (as %)", 10, 50, 30, 5)    
                    random_state = st.number_input("Random State", 0, 100, 42)
                    n_jobs = st.number_input("Parallel Processing (n_jobs)", -10, 10, 1)     

                with col2: 
                    st.subheader("Choose an Algorithm",divider='blue')          
                    regressors = ['linear_regression', 
                                  'decision_tree_regression', 
                                  'random_forest_regression', 
                                  'gradient_boosting',
                                  'xtreme_gradient_boosting']
                    algorithms = st.selectbox("**Choose an algorithm for predictions**", options=regressors)
  
                    progress_text = "Prediction in progress. please wait."
                    my_bar = st.progress(0, text=progress_text)
                    #st.button("Predict", key='Classify')
                    #with st.spinner():
                    for percent_complete in range(100):
                        time.sleep(0.1)
                        my_bar.progress(percent_complete + 1, text=progress_text)

    #----------------------------------------
                # Split the data into train and test sets
                X = df[selected_features]
                y = df[target_variable]
                X_train, X_test, train_labels, test_labels = train_test_split(X, y, test_size=test_size/100, random_state=random_state)
    #----------------------------------------
                from sklearn.preprocessing import MinMaxScaler
                mmscaler = MinMaxScaler()
                mmscaler_df_train = mmscaler.fit_transform(X_train)
                mmscaler_df_test = mmscaler.fit_transform(X_test)
                X_train_scaled = pd.DataFrame(mmscaler_df_train, columns=X_train.columns)
                X_test_scaled = pd.DataFrame(mmscaler_df_test, columns=X_test.columns)
    #----------------------------------------
                with col3:

                    if algorithms == 'linear_regression':
                        st.subheader("Tune the Hyperparameters",divider='blue')
                        lin_reg = LinearRegression()
                        lin_reg.fit(X_train_scaled, train_labels)
                        lin_pred = lin_reg.predict(X_test_scaled)
                        lin_pred_test = lin_reg.predict(X_test_scaled)
                        models = pd.DataFrame({'Model' : ['Linear Regression'],
                               'Test_Score (mse)' : [mean_squared_error(test_labels, lin_pred_test)],
                               'Test_Score (rmse)' : [np.sqrt(r2_score(test_labels, lin_pred_test))],
                               'Test_Score (r2)' : [r2_score(test_labels, lin_pred_test)]})
                        #actual_vs_predict = pd.DataFrame({'Actual': test_labels, 'Predicted': lin_pred})


                    if algorithms == 'decision_tree_regression':
                        st.subheader("Tune the Hyperparameters",divider='blue')
                        min_samples_leaf = st.number_input("**Minimum number of samples required to be at a leaf node**", 1, 10, step=1,key='min_samples_leaf_dt')
                        min_samples_split = st.number_input("**Minimum number of samples required to split an internal node**", 2, 10, step=1,key='min_samples_split_dt')
                        max_depth = st.number_input("**The maximum depth of the tree**", 1, 20, step=1, key='max_depth_dt')
                        max_features = st.selectbox("**The maximum features used in the model**", ["auto","sqrt","log2"], key='max_features_dt')
                        splitter = st.radio("**Choose the split at each node**", ('best', 'random'), key='splitter_dt')
                        tree_reg = DecisionTreeRegressor(min_samples_leaf = min_samples_leaf,
                                                         min_samples_split = min_samples_split,
                                                         max_depth = max_depth, 
                                                         splitter = splitter,
                                                         max_features = max_features)
                        tree_reg.fit(X_train_scaled, train_labels)
                        dt_pred = tree_reg.predict(X_test_scaled)
                        models = pd.DataFrame({'Model' : ['Decision Tree'],
                               'Test_Score (mse)' : [mean_squared_error(test_labels, dt_pred)],
                               'Test_Score (rmse)' : [np.sqrt(r2_score(test_labels, dt_pred))],
                               'Test_Score (r2)' : [r2_score(test_labels, dt_pred)]})
                        #st.subheader("Model Score")
                        #st.bar_chart(models)        
        
                    if algorithms == 'random_forest_regression':
                        st.subheader("Tune the Hyperparameters",divider='blue')
                        min_samples_leaf = st.number_input("**Minimum number of samples required to be at a leaf node**", 1, 10, step=1,key='min_samples_leaf_rf')
                        min_samples_split = st.number_input("**Minimum number of samples required to split an internal node**", 2, 10, step=1,key='min_samples_split_rf')
                        n_estimators = st.number_input("**The number of trees in the forest**", 100, 5000, step=10,key='n_estimators_rf')
                        max_depth = st.number_input("**The maximum depth of the tree**", 1, 20, step=1, key='max_depth_rf')
                        bootstrap = st.radio("**Bootstrap samples when building trees**", ('True', 'False'), key='bootstrap_rf')
                        rf_reg = RandomForestRegressor(min_samples_leaf = min_samples_leaf,
                                                       min_samples_split = min_samples_split,
                                                       n_estimators = n_estimators, 
                                                       max_depth = max_depth, 
                                                       bootstrap = bootstrap,
                                                       n_jobs = n_jobs)
                        rf_reg.fit(X_train_scaled, train_labels)
                        rf_pred = rf_reg.predict(X_test_scaled)
                        models = pd.DataFrame({'Model' : ['Random Forest'],
                               'Test_Score (mse)' : [mean_squared_error(test_labels, rf_pred)],
                               'Test_Score (rmse)' : [np.sqrt(r2_score(test_labels, rf_pred))],
                               'Test_Score (r2)' : [r2_score(test_labels, rf_pred)]})
        
                    if algorithms == 'gradient_boosting':
                        st.subheader("Tune the Hyperparameters",divider='blue')
                        n_estimators = st.number_input("The number of trees in the forest", 100, 5000, step = 10,key='n_estimators')
                        max_depth = st.number_input("The maximum depth of the tree", 1, 20, step = 1, key='max_depth')
                        learning_rate = st.number_input("Learning rate", .01, .1, step =.01, key ='learning_rate')
                        gbr = GradientBoostingRegressor(n_estimators=n_estimators, 
                                                        max_depth=max_depth, 
                                                        learning_rate=learning_rate)
                        gbr.fit(X_train_scaled, train_labels)
                        gbr_pred = gbr.predict(X_test_scaled)
                        models = pd.DataFrame({'Model' : ['Gradient Boosting'],
                               'Test_Score (mse)' : [mean_squared_error(test_labels, gbr_pred)],
                               'Test_Score (rmse)' : [np.sqrt(r2_score(test_labels, gbr_pred))],
                               'Test_Score (r2)' : [r2_score(test_labels, gbr_pred)]})
        
        
                    if algorithms == 'xtreme_gradient_boosting':
                        st.subheader("Tune the Hyperparameters",divider='blue')
                        n_estimators = st.number_input("The number of trees in the forest", 100, 5000, step = 10,key='n_estimators')
                        max_depth = st.number_input("The maximum depth of the tree", 1, 20, step = 1, key='max_depth')
                        learning_rate = st.number_input("Learning rate", .01, .1, step =.01, key ='learning_rate')
                        booster= st.radio("Boosting options", ('gbtree', 'gblinear'), key='booster')
                        xgb = XGBRegressor(n_estimators=n_estimators, 
                                           max_depth=max_depth, 
                                           learning_rate=learning_rate, 
                                           booster = booster)
                        xgb.fit(X_train_scaled, train_labels)
                        xgb_pred = xgb.predict(X_test_scaled)        
                        models = pd.DataFrame({'Model' : ['Xtreme Gradient Boosting'],
                               'Test_Score (mse)' : [mean_squared_error(test_labels, xgb_pred)],
                               'Test_Score (rmse)' : [np.sqrt(r2_score(test_labels, xgb_pred))],
                               'Test_Score (r2)' : [r2_score(test_labels, xgb_pred)]})

#---------------------------------------------------------------------------------------------------------------------------------
### Model Performance
#---------------------------------------------------------------------------------------------------------------------------------

            with tab7:

                col1, col2 = st.columns((0.3,0.7)) 
                with col1:
                    with st.container():

                        st.subheader("Model Overall Score",divider='blue')
                        st.write(models.head())

#----------------------------------------
                        
                with col2:
                    with st.container():

                        st.subheader("Score Visualization",divider='blue')
                        score = ['Model Score (MSE)', 
                                    'Model Score (RMSE)', 
                                    'Model Score (R2)', 
                                  ]
                        score = st.selectbox("**Choose an algorithm for predictions**", options=score, key = 'score')
                        
                        if score == 'Model Score (MSE)':

                            plot_data_11 = [go.Bar(x=models['Model'],
                                            y= models['Test_Score (mse)'],
                                            width = [0.5, 0.5],
                                            marker=dict(color=['green', 'blue']))]
                            plot_layout_11 = go.Layout(xaxis={"type": "category"},
                                                yaxis={"title": "Test_Score (mse)"},
                                                title='Model Score (MSE)',)
                            fig = go.Figure(data=plot_data_11, layout=plot_layout_11)
                            st.plotly_chart(fig,use_container_width = True)

                        if score == 'Model Score (RMSE)':
                              
                            plot_data_12 = [go.Bar(x=models['Model'],
                                            y= models['Test_Score (rmse)'],
                                            width = [0.5, 0.5],
                                            marker=dict(color=['red']))]
                            plot_layout_12 = go.Layout(xaxis={"type": "category"},
                                                yaxis={"title": "Test_Score (rmse)"},
                                                title='Model Score (RMSE)',)
                            fig = go.Figure(data=plot_data_12, layout=plot_layout_12)
                            st.plotly_chart(fig,use_container_width = True)

                        if score == 'Model Score (R2)':
                              
                            plot_data_13 = [go.Bar(x=models['Model'],
                                            y= models['Test_Score (r2)'],
                                            width = [0.5, 0.5],
                                            marker=dict(color=['blue']))]
                            plot_layout_13 = go.Layout(xaxis={"type": "category"},
                                                yaxis={"title": "Test_Score (r2)"},
                                                title='Model Score (R2)',)
                            fig = go.Figure(data=plot_data_13, layout=plot_layout_13)
                            st.plotly_chart(fig,use_container_width = True)

#----------------------------------------
                                                    
                st.subheader("Feature Importance",divider='blue') 

                col1, col2 = st.columns((0.3,0.7)) 
                with col1:
                    with st.container():
                          
                        if algorithms == 'linear_regression':
                            #st.write('NA')
                            #st.write("### Feature Importance:")
                            feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': lin_reg.coef_[0]})
                            feature_importance.sort_values(by='Coefficient', key=abs, ascending=False, inplace=True)
                            st.write(feature_importance)

                        if algorithms == 'decision_tree_regression':
                            feature_importance = tree_reg.feature_importances_
                            importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importance})
                            st.write(importance_df)

                        if algorithms == 'random_forest_regression':
                            feature_importance = rf_reg.feature_importances_
                            importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importance})
                            st.write(importance_df)

                        if algorithms == 'gradient_boosting':
                            feature_importance = gbr.feature_importances_
                            importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importance})
                            st.write(importance_df)

                        if algorithms == 'xtreme_gradient_boosting':
                            feature_importance = xgb.feature_importances_
                            importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importance})
                            st.write(importance_df)

#----------------------------------------                
                
                with col2:
                    with st.container():
                        
                        if algorithms == 'linear_regression':
                        # Plot feature importance
                            #st.write('NA')
                            plot_data_21 = [go.Bar(x = feature_importance['Feature'],
                                            y = feature_importance['Coefficient'])]
                            plot_layout_21 = go.Layout(xaxis = {"title": "Feature"},
                                                yaxis = {"title": "Coefficient"},
                                                title = 'Feature Importance',)
                            fig = go.Figure(data = plot_data_21, layout = plot_layout_21)
                            st.plotly_chart(fig,use_container_width = True)

                        if algorithms == 'decision_tree_regression':
                        # Plot feature importance
                            plot_data_22 = [go.Bar(x=importance_df['Feature'],
                                            y= importance_df['Importance'])]
                            plot_layout_22 = go.Layout(xaxis={"title": "Feature"},
                                                yaxis={"title": "Importance"},
                                                title='Feature Importance',)
                            fig = go.Figure(data=plot_data_22, layout=plot_layout_22)
                            st.plotly_chart(fig,use_container_width = True)
                              
                        if algorithms == 'random_forest_regression':
                        # Plot feature importance

                            plot_data_23 = [go.Bar(x=importance_df['Feature'],
                                            y= importance_df['Importance'])]
                            plot_layout_23 = go.Layout(xaxis={"title": "Feature"},
                                                yaxis={"title": "Importance"},
                                                title='Feature Importance',)
                            fig = go.Figure(data=plot_data_23, layout=plot_layout_23)
                            st.plotly_chart(fig,use_container_width = True)

                        if algorithms == 'gradient_boosting':
                        # Plot feature importance

                            plot_data_24 = [go.Bar(x=importance_df['Feature'],
                                            y= importance_df['Importance'])]
                            plot_layout_24 = go.Layout(xaxis={"title": "Feature"},
                                                yaxis={"title": "Importance"},
                                                title='Feature Importance',)
                            fig = go.Figure(data=plot_data_24, layout=plot_layout_24)
                            st.plotly_chart(fig,use_container_width = True)

                        if algorithms == 'xtreme_gradient_boosting':
                        # Plot feature importance                                                            

                            plot_data_25 = [go.Bar(x=importance_df['Feature'],
                                            y= importance_df['Importance'])]
                            plot_layout_25 = go.Layout(xaxis={"title": "Feature"},
                                                yaxis={"title": "Importance"},
                                                title='Feature Importance',)
                            fig = go.Figure(data=plot_data_25, layout=plot_layout_25)
                            st.plotly_chart(fig,use_container_width = True)

#---------------------------------------------------------------------------------------------------------------------------------
### Model Validation
#---------------------------------------------------------------------------------------------------------------------------------

            with tab8:

                    #st.sidebar.header("3. Cross Validation", divider='blue')

                    st.subheader("**Cross Validation**",divider='blue') 
                    col1, col2 = st.columns((0.3,0.7)) 
                    with col1:

                        cv = st.slider("**CV Value**", 0, 10, 5, 1)  
                        scoring = st.selectbox("**Select type of scoring**",["neg_root_mean_squared_error",
                                                                             "mean_squared_error",
                                                                             "r2"])
  
                        st.divider()
                        
                        if algorithms == 'linear_regression':
                            cv_score = cross_val_score(lin_reg, 
                                       X_train, train_labels, 
                                       cv=cv, 
                                       scoring=scoring)
                            st.write(f"**Cross-Validation R2**: {cv_score.mean():.4f} (Â±{cv_score.std():.4f})")

                        if algorithms == 'decision_tree_regression':
                            cv_score = cross_val_score(tree_reg, 
                                       X_train, train_labels, 
                                       cv=cv, 
                                       scoring=scoring)
                            st.write(f"**Cross-Validation R2**: {cv_score.mean():.4f} (Â±{cv_score.std():.4f})")

                        if algorithms == 'random_forest_regression':
                            cv_score = cross_val_score(rf_reg, 
                                       X_train, train_labels, 
                                       cv=cv, 
                                       scoring=scoring)
                            st.write(f"**Cross-Validation R2**: {cv_score.mean():.4f} (Â±{cv_score.std():.4f})")

                        if algorithms == 'gradient_boosting':
                            cv_score = cross_val_score(gbr, 
                                       X_train, train_labels, 
                                       cv=cv, 
                                       scoring=scoring)
                            st.write(f"**Cross-Validation R2**: {cv_score.mean():.4f} (Â±{cv_score.std():.4f})")

                        if algorithms == 'xtreme_gradient_boosting':
                            cv_score = cross_val_score(xgb, 
                                       X_train, train_labels, 
                                       cv=cv, 
                                       scoring=scoring)
                            st.write(f"**Cross-Validation R2**: {cv_score.mean():.4f} (Â±{cv_score.std():.4f})")

                    with col2:  
                         
                            st.write("### Cross-Validation Plot:")
                            fig, ax = plt.subplots(figsize=(12, 6))
                            ax = plt.plot(range(1, cv + 1), cv_score, marker='o')
                            #plt.title('Cross-Validation Mean Squared Error Scores')
                            plt.xlabel('Fold')
                            plt.ylabel('Negative Mean Squared Error')
                            st.pyplot(fig)

                    
                    st.subheader("Bias-Variance Tradeoff", divider='blue')
                    col1, col2 = st.columns(2) 
                    with col1:
                        degree = st.slider("**Polynomial Degree**", min_value=1, max_value=10, value=3)

                    with col2:
                        noise_level = st.slider("**Noise Level**", min_value=0.1, max_value=5.0, value=1.0)

#---------------------------------------------------------------------------------------------------------------------------------
### Model Cross Check
#---------------------------------------------------------------------------------------------------------------------------------

            #with tab9:                   
                
                #col1, col2, col3 = st.columns(3) 
                #with col1:

                    #st.subheader("Display model information", divider='blue')
                    #setup(df, target=df[target_variable])
                    #setup_df = pull()
                    #st.table(setup_df)

                #with col2:

                    #st.subheader("Analyze & Compare model output", divider='blue')
                    #best_model = compare_models()
                    #compare_df = pull()
                    #st.table(compare_df)

                #with col3:

                    #st.subheader("Visualize model output", divider='blue')

                #st.subheader("Model Summary", divider='blue')
                #----------------------------------------

                #sc = dabl.SimpleClassifier()
                #target_col = df[target_variable]
                #sc.fit(df, target_col=target_col)  # Assuming 'target' is the target column in your dataset
                #st.write(sc.model_summary)

                #----------------------------------------
                
                #from lazypredict.Supervised import LazyClassifier
                #clf = LazyClassifier(ignore_warnings=True, custom_metric=None)
                #models, predictions = clf.fit(X_train, X_test, train_labels, test_labels)
                #st.subheader("Model Performance", divider='blue')
                #st.write(models)
                
                #----------------------------------------                        
                
                #from tpot import TPOTClassifier
                #tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)
                #tpot.fit(X_train, train_labels)
                #st.subheader("Model Performance", divider='blue')
                #st.write("Best pipeline steps:", tpot.fitted_pipeline_)
                #st.write("Best pipeline score:", tpot.score(X_test, test_labels))

                #----------------------------------------
