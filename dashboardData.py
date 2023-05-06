import pandas as pd
import requests
commodity = 'WTI'
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol=WTI&apikey=DCKF14084FJK2MEI'

response = requests.get(url)

wti_df = pd.DataFrame.from_dict(response.json()['Monthly Adjusted Time Series'], orient='index')

wti_df = wti_df[['4. close']]
wti_df.columns = ['Price']

wti_df.index = pd.to_datetime(wti_df.index)
wti_df.index = wti_df.index.strftime('%Y-%m') 
wti_df.index.names = ['Month']

wti_df['Commodity'] = commodity

wti_df = wti_df.loc['2023-03':]

print(wti_df)




url = 'https://www.alphavantage.co/query?function=COTTON&interval=monthly&apikey=YOUR_API_KEY'

response = requests.get(url)

data = response.json()['data']
cotton_df = pd.DataFrame.from_records(data)

cotton_df.replace('.', 0, inplace=True)

cotton_df.index = pd.to_datetime(cotton_df['date']).dt.to_period('M')
cotton_df.drop('date', axis=1, inplace=True)

cotton_df.rename(columns={'value': 'Price'}, inplace=True)

cotton_df['Commodity'] = 'Cotton'

cotton_df = cotton_df[['Price', 'Commodity']]

print(cotton_df)


wheatURL = 'https://www.alphavantage.co/query?function=WHEAT&interval=monthly&apikey=DCKF14084FJK2MEI'

response = requests.get(wheatURL)

data = response.json()['data']
wheat_df = pd.DataFrame.from_records(data)
wheat_df.replace('.', 0, inplace=True)

wheat_df.index = pd.to_datetime(wheat_df['date']).dt.to_period('M')
wheat_df.drop('date', axis=1, inplace=True)

wheat_df.rename(columns={'value': 'Price'}, inplace=True)

wheat_df['Commodity'] = 'WHEAT'

wheat_df = wheat_df[['Price', 'Commodity']]

print(wheat_df)

cornURL = 'https://www.alphavantage.co/query?function=CORN&interval=monthly&apikey=DCKF14084FJK2MEI'

response = requests.get(cornURL)

data = response.json()['data']
corn_df = pd.DataFrame.from_records(data)

corn_df.index = pd.to_datetime(corn_df['date']).dt.to_period('M')
corn_df.drop('date', axis=1, inplace=True)

corn_df.rename(columns={'value': 'Price'}, inplace=True)

corn_df['Commodity'] = 'CORN'

corn_df = corn_df[['Price', 'Commodity']]

corn_df = corn_df.replace('.', 0)

print(corn_df)

brentURL = 'https://www.alphavantage.co/query?function=BRENT&interval=monthly&apikey=DCKF14084FJK2MEI'

response = requests.get(brentURL)

data = response.json()['data']
brent_df = pd.DataFrame.from_records(data)

brent_df.index = pd.to_datetime(brent_df['date']).dt.to_period('M')
brent_df.drop('date', axis=1, inplace=True)

brent_df.rename(columns={'value': 'Price'}, inplace=True)

brent_df['Commodity'] = 'BRENT'

brent_df = brent_df[['Price', 'Commodity']]

print(brent_df)


brent_df.index = pd.to_datetime(brent_df.index.astype('datetime64[ns]'))
corn_df.index = pd.to_datetime(corn_df.index.astype('datetime64[ns]'))
cotton_df.index = pd.to_datetime(cotton_df.index.astype('datetime64[ns]'))
wti_df.index = pd.to_datetime(wti_df.index)
wheat_df.index = pd.to_datetime(wheat_df.index.astype('datetime64[ns]'))
combined_df = pd.concat([brent_df, corn_df, cotton_df, wti_df, wheat_df])

combined_df = combined_df.pivot_table(values='Price', index=combined_df.index, columns='Commodity')
combined_df.sort_index(inplace=True, ascending=False)
combined_df = combined_df.drop(combined_df.index[0])
print(combined_df)

correlations_df = combined_df.corr()
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

unique_month_years = combined_df.index.to_period('M').unique().to_timestamp()

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1('Commodity Prices', className='text-center mt-3 mb-3')
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.Label('Select a date range', className='mb-1'),
            dcc.Dropdown(
                id='date-range',
                options=[{'label': date.strftime('%B %Y'), 'value': date} for date in unique_month_years],
                multi=True,
                placeholder='Select a date range...'
            ),
        ], width=6),
    ], justify='center'),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='graph')
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.H2('Correlation Heatmap', className='text-center mt-5 mb-3'),
            dcc.Graph(id='correlation-heatmap')
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.H2('Scatter Plot Matrix', className='text-center mt-5 mb-3'),
            dcc.Graph(id='scatter-matrix')
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.Label('Select a commodity', className='mb-1'),
            dcc.Dropdown(
                id='commodity-selector',
                options=[{'label': commodity, 'value': commodity} for commodity in combined_df.columns],
                placeholder='Select a commodity...'
            ),
        ], width=6),
    ], justify='center'),
    dbc.Row([
        dbc.Col([
            html.H2('Top Price Changes', className='text-center mt-5 mb-3'),
            dcc.Graph(id='top-price-changes')
        ])
    ]),
], fluid=True)

@app.callback(
    Output('graph', 'figure'),
    [Input('date-range', 'value')]
)
def update_graph(date_range):
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = combined_df.loc[start_date:end_date]
    else:
        filtered_df = combined_df

    data = []
    for commodity in combined_df.columns:
        data.append({'x': filtered_df.index, 'y': filtered_df[commodity], 'name': commodity})
    layout = {'title': 'Wheat, WTI, Corn, Cotton, and Brent Prices'}
    return {'data': data, 'layout': layout}

@app.callback(
    Output('correlation-heatmap', 'figure'),
    [Input('date-range', 'value')]
)
def update_heatmap(date_range):
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = combined_df.loc[start_date:end_date]
    else:
        filtered_df = combined_df

    correlations_df = filtered_df.corr()

    heatmap = go.Figure(
        data=go.Heatmap(
            z=correlations_df,
            x=correlations_df.columns,
            y=correlations_df.index,
            colorscale='Viridis',
            zmin=-1,
            zmax=1
        ),
        layout=go.Layout(
            xaxis_title='Commodity',
            yaxis_title='Commodity',
            yaxis=dict(autorange='reversed'),
            margin=dict(t=50, l=50, b=50, r=50),
        )
    )

    for i, row in enumerate(correlations_df.index):
        for j, col in enumerate(correlations_df.columns):
            value = correlations_df.iloc[i, j]
            heatmap.add_trace(
                go.Scatter(
                    x=[col],
                    y=[row],
                    text=["{:.2f}".format(value)],
                    mode='text',
                    textfont=dict(size=12, color='white'),
                    showlegend=False,
                    hoverinfo='none'
                )
            )

    return heatmap
@app.callback(
    Output('scatter-matrix', 'figure'),
    [Input('date-range', 'value')]
)
def update_scatter_matrix(date_range):
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = combined_df.loc[start_date:end_date]
    else:
        filtered_df = combined_df

    fig = px.scatter_matrix(filtered_df, dimensions=filtered_df.columns, title='Scatter Plot Matrix')
    return fig


@app.callback(
    Output('top-price-changes', 'figure'),
    [Input('commodity-selector', 'value'),
     Input('date-range', 'value')]
)
def update_top_price_changes(commodity, date_range):
    if commodity:
        if date_range and len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = combined_df.loc[start_date:end_date]
        else:
            filtered_df = combined_df

        price_changes = filtered_df[commodity].diff().dropna().sort_values(ascending=False)
        top_n = 10
        top_gains = price_changes.head(top_n)
        top_losses = price_changes.tail(top_n)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=top_gains.index, y=top_gains, name='Top Gains', marker_color='green'))
        fig.add_trace(go.Bar(x=top_losses.index, y=top_losses, name='Top Losses', marker_color='red'))
        fig.update_layout(title=f'Top {top_n} Price Changes for {commodity}',
                          xaxis_title='Date', yaxis_title='Price Change',                          barmode='group',
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                          margin=dict(t=50, l=50, b=50, r=50)
                         )
        return fig
    else:
        return go.Figure()

if __name__ == '__main__':
    app.run_server(debug=False)
