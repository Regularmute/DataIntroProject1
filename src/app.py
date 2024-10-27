# Import packages
from dash import Dash, html, dash_table, dcc, Output, Input
import plotly.express as px
from datetime import datetime, timedelta
from model_predict import get_forecasts_and_predict
import pandas as pd

# Get tomorrow's date
tomorrow = datetime.now().replace(hour=0, minute=0, second=0,
                                  microsecond=0) + timedelta(days=1)
formatted_tomorrow = tomorrow.strftime('%A, %B %d of %Y')

# Initialize the app
app = Dash()


def create_bar_chart(df):
    df['color'] = df['Rank'].apply(
        lambda rank: 'green' if rank <= 4 else 'red' if rank >= 21 else 'blue')

    df['Hour'] = pd.to_numeric(df['Hour'])

    fig = px.bar(
        df, x='Hour', y='Predicted CO2',
        color='color',
        color_discrete_map={'green': 'green', 'red': 'red',
                            'blue': 'blue'},
        orientation='v'
    )

    fig.update_layout(
        showlegend=False,
        width=800,
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
        bargap=0
    )

    return fig


def create_data_table(df):
    df['Predicted CO2'] = df['Predicted CO2'].apply(lambda x: f"{x:.2f}")
    if 'color' in df.columns:
        df = df.drop(columns=['color'])
    df_transposed = df.set_index('Hour').transpose()
    df_transposed.columns = df_transposed.columns.astype(str)
    df_transposed = df_transposed.rename_axis('Hour').reset_index()
    data = df_transposed.to_dict('records')
    columns = [{'name': i, 'id': i} for i in df_transposed.columns]

    return dash_table.DataTable(
        data=data,
        columns=columns,
        style_cell={'textAlign': 'center'},
        style_cell_conditional=[
            {'if': {'column_id': col['id']}, 'width': '80px'} for col in columns
        ],
        page_size=3,
        style_table={'width': '100%', 'margin': '0 auto'},
    )


# App layout
app.layout = html.Div([
    html.H1(
        "ECO - Electricity Carbon Dioxide Emissions Optimizer",
        style={'textAlign': 'center', 'fontSize': 36, 'marginBottom': '15px'}
    ),
    html.Div(
        children=f'Predicted Electricity CO2 Emissions for {formatted_tomorrow} by Hour',
        style={'textAlign': 'center', 'fontSize': 24, 'marginBottom': '10px'}
    ),
    dcc.Loading(
        id="loading-1",
        type="default",
        children=[
            html.Div(id="content"),
        ]
    )
])

# Callback to update content


@ app.callback(
    Output("content", "children"),
    Input("loading-1", "children")
)
def update_content(_):
    # Incorporate data
    df = get_forecasts_and_predict(tomorrow)

    if df.empty:
        return html.P('Tomorrow\'s data is missing, apologies for the inconvenience. Electricity prices available typically at 14:00 EET.')
    else:
        bar_chart = create_bar_chart(df)
        table = create_data_table(df)

        return html.Div([
            html.Div([
                dcc.Graph(id='co2-histogram', figure=bar_chart)
            ], style={'marginBottom': '10px'}),
            html.Div([
                table
            ], style={'width': '400px', 'margin-left': '20', 'verticalAlign': 'top'}),
        ])


# Run the app
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
    # app.run(host='0.0.0.0', port=8080, debug=False)
