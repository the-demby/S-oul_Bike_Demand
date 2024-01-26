import dash
from dash import dcc, html

dash.register_page(__name__, path='/')

layout = html.Div(
    [
        html.H1('Project Description'),
        html.H2('Superivised Machine Learning- Regression'),

        html.Img(src='/assets/image1.png', style={'width': '100%'}),  # Ajoutez ici la première image

        html.H2('Description'),
        html.P("Currently Rental bikes are introduced in many urban cities for the enhancement of mobility comfort. It is important to make the rental bike available and accessible to the public at the right time as it lessens the waiting time. Eventually, providing the city with a stable supply of rental bikes becomes a major concern. The crucial part is the prediction of bike count required at each hour for the stable supply of rental bikes."),

        html.H2('Data Description'),

        html.P("The dataset contains weather information (Temperature, Humidity, Windspeed, Visibility, Dewpoint, Solar radiation, Snowfall, Rainfall), the number of bikes rented per hour and date information."),
        html.P("Attribute Information"),
        html.Ul([
            html.Li('Date : year-month-day'),
            html.Li('Rented Bike count - Count of bikes rented at each hour'),
            html.Li('Hour - Hour of he day'),
            html.Li('Temperature-Temperature in Celsius'),
            html.Li('Humidity - %'),
            html.Li('Windspeed - m/s'),
            html.Li('Visibility - 10m'),
            html.Li('Dew point temperature - Celsius'),
            html.Li('Solar radiation - MJ/m2'),
            html.Li('Rainfall - mm'),
            html.Li('Snowfall - cm'),
            html.Li('Seasons - Winter, Spring, Summer, Autumn'),
            html.Li('Holiday - Holiday/No holiday'),
            html.Li('Functional Day - NoFunc(Non Functional Hours), Fun(Functional hours)'),
        ]),
        html.Img(src='/assets/image2.png', style={'width': '100%'}),  # Ajoutez ici la deuxième image,
        html.P("Based on the adjusted R2 metric, the best-performing model is LIGHT Gradient Boosting. It will therefore be used to carry out the rest of the analysis and propose solutions."),


        html.H2('Conclusions:'),
        html.P("Here are some solutions to manage Bike Sharing Demand:"),
        html.Ul([
            html.Li("The majority of rentals are for daily commutes to workplaces and colleges. Therefore open additional stations near these landmarks to reach their primary customers."),
            html.Li("While planning for extra bikes to stations, the peak rental hours must be considered, i.e. 7–9 am and 5–6 pm."),
            html.Li("Maintenance activities for bikes should be done at night due to the low usage of bikes during the night time. Removing some bikes from the streets at night time will not cause trouble for the customers."),
        ]),
        html.P("We see 2 rental patterns across the day in bike rental count - first for a Working Day where the rental count is high at peak office hours (8 am and 5 pm) and the second for a Non-working day where the rental count is more or less uniform across the day with a peak at around noon."),
        html.P("Hour of the day: Bike rental count is mostly correlated with the time of the day. As indicated above, the count reaches a high point during peak hours on a working day and is mostly uniform during the day on a non-working day."),
        html.P("Temperature: People generally prefer to bike at moderate to high temperatures. We see the highest rental counts between 32 to 36 degrees Celsius."),
        html.P("Season: We see the highest number of bike rentals in the Spring (July to September) and Summer (April to June) Seasons and the lowest in the Winter (January to March) season."),
        html.P("Weather: As one would expect, we see the highest number of bike rentals on a clear day and the lowest on a snowy or rainy day."),
        html.P("Humidity: With increasing humidity, we see a decrease in the bike rental count."),
        html.P("I have chosen the Light GBM model which is above all, I want better predictions for the rented_bike_count, and time isn't compelling here. As a result, various linear models, decision trees, Random Forests, and Gradient Boost techniques were used to improve accuracy. I compared R2 metrics to choose a model."),
        html.P("Due to less no. of data in the dataset, the training R2 score is around 99% and the test R2 score is 92.5%. Once we get more data we can retrain our algorithm for better performance."),
    ]
)