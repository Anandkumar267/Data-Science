import pandas as pd
import plotly.express as px
import numpy as np
from dash import Dash, dcc, html, Input, Output
URL1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
data = pd.read_csv(URL1)
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

# Slider bounds and marks for PayloadMass
payload_min = int(data['PayloadMass'].min())
payload_max = int(data['PayloadMass'].max())
slider_marks = {int(v): str(int(v)) for v in np.linspace(payload_min, payload_max, 5)}

app = Dash()
app.layout = html.Div(children=[
    html.H1("SpaceX Launch Records Dashboard", style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),
    dcc.Dropdown(
        id='launch-site-success-rate',
        options=[{'label': 'ALL SITES', 'value': 'ALL SITES'}] + [{'label': str(i), 'value': str(i)} for i in np.unique(data['LaunchSite'])],
        value='ALL SITES',
        placeholder='Select a launch site here',
        searchable=True
    ),
    html.Br(),
    dcc.Loading(children=[dcc.Graph(id='success-pie-chart')], type='circle'),
    html.Br(),
    dcc.Slider(
        id='payload-mass',
        min=payload_min,
        max=payload_max,
        step=100,
        value=int(payload_max / 2),
        marks=slider_marks
    ),
    dcc.Graph(id='payload-mass-success')
])
@app.callback(
    Output(component_id='success-pie-chart', component_property='figure'),
    Input(component_id='launch-site-success-rate', component_property='value')
)
def get_pie_chart(entered_site):
    if entered_site == 'ALL SITES':
        # Sum successes per launch site (Class is 0/1)
        grouped = data.groupby('LaunchSite', as_index=False)['Class'].sum()
        fig = px.pie(grouped, values='Class', names='LaunchSite', title='Total Successful Launches by Site')
    else:
        # For a specific site, show success vs failure counts
        filtered = data[data['LaunchSite'] == entered_site]
        counts = filtered['Class'].value_counts().rename_axis('outcome').reset_index(name='count')
        counts['outcome'] = counts['outcome'].map({1: 'Success', 0: 'Failure'})
        fig = px.pie(counts, values='count', names='outcome', title=f'Success vs Failure for site {entered_site}')

    fig.update_layout(plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], font_color=colors['text'])
    return fig
@app.callback(
    Output(component_id='payload-mass-success', component_property='figure'),
    Input(component_id='payload-mass', component_property='value')
)
def scatter_plot(selected_payload):
    df = data[data['PayloadMass'] <= selected_payload]

    fig = px.scatter(df, x='PayloadMass', y='Class', title='Correlation between Payload and Success')

    fig.update_layout(plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], font_color=colors['text'])
    return fig
        
if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=False)