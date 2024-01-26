# Import Libraries and modules
import dash
from dash import dcc, html, dash_table

# Utilisez le style externe (fichier CSS) que vous avez créé
external_stylesheets = ['assets/style.css']  # Remplacez 'style.css' par le nom de votre fichier CSS

app = dash.Dash(__name__, use_pages=True, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        # main app framework
        html.Div("Seoul-Bike-Sharing-Demand-Prediction", style={'fontSize': 50, 'textAlign': 'center'}),
        html.Div([
            dcc.Link(page['name'] + "  |  ", href=page['path'])
            for page in dash.page_registry.values()
        ]),
        html.Hr(),

        # content of each page
        dash.page_container
    ]
)

if __name__ == "__main__":
    app.run(debug=True)