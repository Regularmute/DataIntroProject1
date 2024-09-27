# Import packages
from dash import Dash, html, dash_table, dcc
import plotly.express as px
from datetime import datetime, timedelta
from dummy_data import get_dummy_df

# Get tomorrow's date
tomorrow = datetime.today().date() + timedelta(days=1)
formatted_tomorrow = tomorrow.strftime('%A, %B %d of %Y')

# Incorporate data
df = get_dummy_df()

# Initialize histogram
fig = px.histogram(df, x='hour', y='CO2')

# Initialize the app
app = Dash()

# App layout
app.layout = [
    html.Div(children=f'Electricity Co2 emissions of {formatted_tomorrow} by hour'),
    dash_table.DataTable(data=df.to_dict('records'), page_size=10),
    html.Div([
        dcc.Graph(id='gdp-histogram', figure=fig)
    ]),
]

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)