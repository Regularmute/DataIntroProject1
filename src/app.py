# Import packages
from dash import Dash, html, dash_table, dcc, Output, Input
import plotly.express as px
from datetime import datetime, timedelta
from model_predict import get_forecasts_and_predict

# Get tomorrow's date
tomorrow = datetime.now().replace(hour=0, minute=0, second=0,
                                  microsecond=0) + timedelta(days=1)
formatted_tomorrow = tomorrow.strftime('%A, %B %d of %Y')

# Initialize the app
app = Dash()

# App layout
app.layout = html.Div([
    html.Div(
        children=f'Predicted Electricity CO2 Emissions for {formatted_tomorrow} by Hour',
        style={'textAlign': 'center', 'fontSize': 24, 'marginBottom': '30px'}
    ),
    dcc.Loading(
        id="loading-1",
        type="default",
        children=[
            html.Div(id="content"),
        ]
    )
])


@app.callback(
    Output("content", "children"),
    Input("loading-1", "children")
)
def update_content(_):
    # Incorporate data
    df = get_forecasts_and_predict(tomorrow)

    if df.empty:
        return html.P('Tomorrow\'s data is missing, apologies for the inconvenience.')
    else:
        fig = px.bar(df, x='Hour', y='Predicted CO2',
                     barmode='group', orientation='v')

        return [
            html.Div([
                dcc.Graph(id='co2-histogram', figure=fig)
            ]),
            dash_table.DataTable(data=df.to_dict('records'), page_size=24),
        ]


# Run the app
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
    # app.run(host='0.0.0.0', port=8080, debug=False)
