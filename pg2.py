# Import Libraries and modules
import dash
from dash import dcc, html, callback, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

# libraries that are used for analysis and visualization
import pandas as pd
import numpy as np

# libraries used to pre-process 
from sklearn import preprocessing, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# libraries used to implement models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# libraries to evaluate performance
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error

# Library of warnings would assist in ignoring warnings issued
import warnings
warnings.filterwarnings('ignore')

# to set max column display
pd.pandas.set_option('display.max_columns',None)

# ... (le reste de tes importations)

# Initialisez votre application Dash
dash.register_page(__name__)

# Chargez l'ensemble de données
bike_df = pd.read_csv('bike.csv', encoding='latin')

X = bike_df.drop('rented_bike_count', axis=1)
y = np.sqrt(bike_df['rented_bike_count'])  # applying sqrt transformation in the target variable.

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

# Scaling Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a base decision tree regression model
dt = DecisionTreeRegressor(max_depth=12)

# Initialize the AdaBoost regression model
ada = AdaBoostRegressor(base_estimator=dt, n_estimators=60, learning_rate=1, random_state =33)

param_grid = {'n_estimators': [50,80],       # number of trees in the ensemble
             'max_depth': [15,20],           # maximum number of levels allowed in each tree.
             'min_samples_split': [5,15],    # minimum number of samples necessary in a node to cause node splitting.
             'min_samples_leaf': [3,5]}      # minimum number of samples which can be stored in a tree leaf.


# Initialize the RandomForestRegressor model
rf = RandomForestRegressor()

# Use GridSearchCV to perform a grid search over the parameter grid
grid_search = GridSearchCV(rf, param_grid=param_grid, cv=5, scoring='r2')

# Fit the model to the training data
grid_search.fit(X, y)

# Get the best parameters from the grid search
rf_optimal_model = grid_search.best_estimator_
rf_optimal_model

param_grid = {'n_estimators': [600,800],     # number of trees in the ensemble
             'max_depth': [8,10],            # maximum number of levels allowed in each tree.
             'min_samples_split': [3,5],     # minimum number of samples necessary in a node to cause node splitting.
             'min_samples_leaf': [2,3]}      # minimum number of samples which can be stored in a tree leaf.


# Initialize the RandomForestRegressor model
lgb = LGBMRegressor()

# Use GridSearchCV to perform a grid search over the parameter grid
grid_search = GridSearchCV(lgb, param_grid=param_grid, cv=5, scoring='r2')

# Fit the model to the training data
grid_search.fit(X, y)

# Get the best parameters from the grid search
lgb_optimal_model = grid_search.best_estimator_
lgb_optimal_model

options = [
    {'label': 'Regression Linéaire', 'value': 0},
    {'label': 'Lasso', 'value': 1},
    {'label': 'Ridge', 'value': 2},
    {'label': 'Elastic Net', 'value': 3},   
    {'label': 'Support Vector Machine(SVM)', 'value': 4},
    {'label': 'Decision Tree', 'value': 5},
    {'label': 'AdaBoost', 'value': 6},
    {'label': 'Random Forest', 'value': 7},
    {'label': 'Light Gradient Boosting', 'value': 8},
]
ref_models = {
    0: LinearRegression(),
    1: Lasso(alpha=0.1, max_iter=1000),
    2: Ridge(alpha=0.1, max_iter=1000),
    3: ElasticNet(alpha=0.1, max_iter=1000),
    4: SVR(kernel='rbf',C=100),
    5: DecisionTreeRegressor(min_samples_leaf=20, min_samples_split=3,max_depth=20, random_state=33),
    6: ada,
    7: rf_optimal_model,
    8: lgb_optimal_model, 
}

selected_value = 0

# empty list for appending performance metric score
model_result = []
model = ref_models[selected_value].fit(X_train, y_train)

# Définir la fonction predict
def predict(ml_model, model_name):
    try:
        # model fitting
        model = ml_model.fit(X_train, y_train)

        # predicting values
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Reverse the transformation on the predictions
        y_train_pred_original = np.power(y_train_pred, 2)
        y_test_pred_original = np.power(y_test_pred, 2)

        # Create a scatter plot
        scatter_temp = pd.DataFrame({'y_pred': y_test_pred, 'y_test': y_test})
        scatter_fig = px.scatter(scatter_temp, x='y_pred', y='y_test', trendline="ols")

        # Create a line plot
        line_fig = go.Figure()
        line_fig.add_trace(go.Scatter(x=np.arange(20), y=y_test_pred[:20], mode='lines', name='Predicted'))
        line_fig.add_trace(go.Scatter(x=np.arange(20), y=np.array(y_test)[:20], mode='lines', name='Actual'))
        line_fig.update_layout(xaxis_title='Test Data on last 20 points', legend=dict(x=0, y=1, traceorder='normal'))

        return scatter_fig, line_fig

    except Exception as e:
        # En cas d'erreur, renvoyer un message d'erreur explicite
        error_message = f"Erreur lors de la prédiction avec le modèle {model_name}: {str(e)}"
        scatter_fig = go.Figure()
        line_fig = go.Figure()

        # Vous pouvez personnaliser le message d'erreur en fonction de votre préférence
        scatter_fig.add_trace(go.Scatter(x=[0], y=[0], text=[error_message], mode='text'))
        line_fig.add_trace(go.Scatter(x=[0], y=[0], text=[error_message], mode='text'))

        return scatter_fig, line_fig

# Ajouter un callback pour mettre à jour les graphiques après la sélection du modèle
@callback(
    [Output('scatter-plot', 'figure'),
     Output('line-plot', 'figure')],
    [Input('submit-button-state', 'n_clicks')],
    [State('model-dropdown', 'value')]
)
def update_graphs(n_clicks, selected_model):
    # Utiliser la fonction predict pour obtenir les figures mises à jour
    scatter_fig, line_fig = predict(ref_models[selected_model], selected_model)
    return scatter_fig, line_fig

# Ajoutez ces lignes pour créer un tableau dans votre layout
layout = html.Div(children=[
    html.H1(children='FINAL PROJECT'),
    html.Div(
        className='row',
        style={'padding': '5%'},
        children=[
            html.Div(
                className='four columns',
                style={'margin-top': '5%'},
                children=[
                    html.H3('Visualisation of our dataset :'),
                    # Ajoutez le DataTable ici
                    dash_table.DataTable(
                        id='bike-data-table',
                        columns=[
                            {'name': col, 'id': col} for col in bike_df.columns
                        ],
                        data=bike_df.head(10).to_dict('records')  # Afficher les 10 premières lignes
                    ),
                ]
            ),
        ]
    ),

    html.Div(
    className='row',
    style={'padding': '5%', 'margin-bottom': '0%'},
    children=[
        html.Div(
            className='four columns',
            style={'margin-top': '1%'},
            children=[
                html.H3('Multi-selection model drop-down list'),  # Déplacez cette ligne ici
                # Ajoutez le bouton de sélection du modèle
                dcc.Dropdown(
                    id='model-dropdown',
                    options=options,
                    value=selected_value
                ),
                html.Button(
                    id='submit-button-state',
                    n_clicks=0,
                    children='Submit'
                )
            ]
        ),
    ]
),
    html.Div(
        className='row',
        style={'padding': '5%'},
        children=[
            html.Div(
                className='four columns',
                style={'margin-top': '0%'},
                children=[
                    # Ajoutez les graphiques Plotly ici
                    dcc.Graph(
                        id='scatter-plot',
                    ),
                    dcc.Graph(
                        id='line-plot',
                    ),
                ]
            ),
        ]
    )
])