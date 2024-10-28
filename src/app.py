import os
from dash import Dash, html, dash_table, dcc, Output, Input
import plotly.express as px
from datetime import datetime, timedelta
from model_predict import get_forecasts_and_predict
import pandas as pd

# persisten storage path
STORAGE_PATH = "./storage/"  # Use a local folder for testing
# STORAGE_PATH = "/data/"  # Uncomment this line for Fly.io production version

TODAY_DATA_FILE = os.path.join(STORAGE_PATH, "today_data.csv")
TOMORROW_DATA_FILE = os.path.join(STORAGE_PATH, "tomorrow_data.csv")


today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
tomorrow = today + timedelta(days=1)
formatted_tomorrow = tomorrow.strftime('%A, %B %d of %Y')
formatted_today = today.strftime('%A, %B %d of %Y')


app = Dash()


def save_data(df, file_path, date):
    """Save the DataFrame to a CSV file with a date column."""
    df['Date'] = date.strftime('%Y-%m-%d')
    if not os.path.exists(STORAGE_PATH):
        os.makedirs(STORAGE_PATH)
    df.to_csv(file_path, index=False)


def load_data(file_path):
    """Load the DataFrame from the CSV file, if available."""
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    return None


def create_bar_chart(df):
    df['color'] = df['Rank'].apply(
        lambda rank: 'green' if rank <= 4 else 'red' if rank >= 21 else 'blue')

    df['Hour'] = pd.to_numeric(df['Hour'])

    fig = px.bar(
        df, x='Hour', y='Predicted CO2',
        color='color',
        color_discrete_map={'green': 'green', 'red': 'red', 'blue': 'blue'},
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
    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])
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


app.layout = html.Div([
    html.H1(
        "ECO - Electricity Carbon Dioxide Emissions Optimizer",
        style={'textAlign': 'center', 'fontSize': 36, 'marginBottom': '15px'}
    ),
    html.Div(
        children=f'Predicted Electricity CO2 Emissions',
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


@app.callback(
    Output("content", "children"),
    Input("loading-1", "children")
)
def update_content(_):
    today_data = load_data(TODAY_DATA_FILE)
    tomorrow_data = load_data(TOMORROW_DATA_FILE)

    content = []

    if today_data is None or today_data['Date'].iloc[0] != today.strftime('%Y-%m-%d'):
        if tomorrow_data is not None and tomorrow_data['Date'].iloc[0] == today.strftime('%Y-%m-%d'):
            save_data(tomorrow_data, TODAY_DATA_FILE, today)
            today_data = tomorrow_data
        else:
            content.append(html.P("Today's data is not available."))

    if today_data is not None:
        today_chart = create_bar_chart(today_data)
        today_table = create_data_table(today_data)
        content.append(html.Div([
            html.P(f"Today's Data: {formatted_today}",
                   style={'textAlign': 'left', 'fontSize': 16, 'marginBottom': '0px'}),
            html.Div([
                dcc.Graph(id='co2-histogram-today', figure=today_chart)
            ], style={'marginBottom': '5px'}),
            html.Div([
                today_table
            ], style={'width': '400px', 'margin-left': '20', 'verticalAlign': 'top'}),
        ], style={'marginBottom': '50px'}))

        content.append(
            html.Hr(style={'marginTop': '20px', 'marginBottom': '20px'})
        )

    if tomorrow_data is None or tomorrow_data['Date'].iloc[0] != tomorrow.strftime('%Y-%m-%d'):
        tomorrow_data = get_forecasts_and_predict(tomorrow)
        if not tomorrow_data.empty:
            save_data(tomorrow_data, TOMORROW_DATA_FILE, tomorrow)
        else:
            tomorrow_data = None

    if tomorrow_data is not None:
        tomorrow_chart = create_bar_chart(tomorrow_data)
        tomorrow_table = create_data_table(tomorrow_data)
        content.append(html.Div([
            html.P(f"Tomorrow's Data: {formatted_tomorrow}",
                   style={'textAlign': 'left', 'fontSize': 16, 'marginBottom': '0px'}),
            html.Div([
                dcc.Graph(id='co2-histogram-tomorrow', figure=tomorrow_chart)
            ], style={'marginBottom': '10px'}),
            html.Div([
                tomorrow_table
            ], style={'width': '400px', 'margin-left': '20', 'verticalAlign': 'top'})
        ], style={'marginBottom': '30px'}))
    else:
        content.append(html.P("Tomorrow's data is missing, apologies for the inconvenience. "
                              "Electricity prices are typically available at 14:00 EET."))

    return content


# Run the app
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
    # app.run(host='0.0.0.0', port=8080, debug=False)
