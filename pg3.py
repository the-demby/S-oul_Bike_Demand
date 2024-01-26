import dash
from dash import dcc, html, callback, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

dash.register_page(__name__)

bike_df = pd.read_csv('bike.csv', encoding='latin')

# Créer un graphique initial avec des données factices (à remplacer avec vos propres données)
initial_line_fig = go.Figure()
initial_line_fig.add_trace(go.Scatter(x=np.arange(20), y=np.random.rand(20), mode='lines', name='Initial'))
initial_line_fig.update_layout(xaxis_title='X Axis', yaxis_title='Y Axis', title='Initial Line Plot')

layout = html.Div(
    [
        html.H1('Page 3 - Data Analysis'),
        html.P("This is a sample page for data analysis."),

        # Ajouter le bouton de sélection d'attribut
        dcc.Dropdown(
            id='attribute-dropdown',
            options=[
                {'label': 'Holiday', 'value': 0},
                {'label': 'No Holiday', 'value': 1},
                {'label': 'Both Attributes', 'value': 2}
            ],
            value=0,  # Sélectionnez 'Holiday' par défaut
            style={'width': '50%'}
        ),

        # Ajoutez le bouton submit
        html.Button(
            id='submit-button',
            n_clicks=0,
            children='Submit'
        ),

        # Ajouter le graphique Plotly initial
        dcc.Graph(
            id='bike-graph',
            figure=initial_line_fig
        ),
    ]
)


# Ajouter un callback pour mettre à jour le graphique en fonction du bouton submit
@callback(
    Output('bike-graph', 'figure'),
    [Input('submit-button', 'n_clicks')],
    [State('attribute-dropdown', 'value')]
)
def update_graph(n_clicks, selected_attribute):
    # Votre logique de gestion de clic ici
    if n_clicks > 0:
        # Filtrer les données en fonction de la sélection d'attribut
        if selected_attribute == 0:
            filtered_data = bike_df[bike_df['holiday'] == 0]
        elif selected_attribute == 1:
            filtered_data = bike_df[bike_df['holiday'] == 1]
        else:
            # Si la valeur est 2, afficher les deux catégories
            filtered_data = bike_df

        # Calculer la moyenne pour chaque heure
        avg_data = filtered_data.groupby(['hour', 'holiday'])['rented_bike_count'].mean().reset_index()

        # Mettez à jour le graphique avec les données de moyenne actualisées
        updated_line_fig = px.line(avg_data, x='hour', y='rented_bike_count', color='holiday',
                                   labels={'rented_bike_count': 'Average Rented Bike Count'},
                                   title='Average Line Plot')
        return updated_line_fig
    else:
        # Retournez le graphique initial lorsqu'aucun clic n'a été effectué
        return initial_line_fig