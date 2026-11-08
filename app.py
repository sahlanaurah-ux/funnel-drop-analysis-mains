import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# 1. app setup
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# --- login layout
# --- Glassmorphism login layout
login_layout = html.Div(style={
    'backgroundImage': 'url("https://images.unsplash.com/photo-1506318137071-a8e063b4bec0?q=80&w=1920")', # മനോഹരമായ പശ്ചാത്തല ചിത്രം
    'backgroundSize': 'cover',
    'backgroundPosition': 'center',
    'height': '100vh',
    'display': 'flex',
    'justifyContent': 'center',
    'alignItems': 'center',
    'margin': '0',
    'fontFamily': '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif'
}, children=[
    html.Div(style={
        'background': 'rgba(255, 255, 255, 0.1)', 
        'backdropFilter': 'blur(15px)', 
        'WebkitBackdropFilter': 'blur(15px)',
        'padding': '40px 50px',
        'borderRadius': '20px',
        'border': '1px solid rgba(255, 255, 255, 0.2)', 
        'boxShadow': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
        'width': '380px',
        'textAlign': 'center'
    }, children=[
        html.H2("Data Analytics Login", style={
            'color': '#ffffff', 
            'marginBottom': '30px', 
            'fontWeight': '600',
            'letterSpacing': '1px'
        }),
        
        dcc.Input(id='username', type='text', placeholder='Username', style={
            'margin': '12px 0', 
            'padding': '12px 15px', 
            'width': '100%', 
            'background': 'rgba(255, 255, 255, 0.2)', 
            'border': 'none', 
            'borderRadius': '8px', 
            'color': 'white', 
            'outline': 'none',
            'boxSizing': 'content-box'
        }),
        
        dcc.Input(id='password', type='password', placeholder='Password', style={
            'margin': '12px 0', 
            'padding': '12px 15px', 
            'width': '100%', 
            'background': 'rgba(255, 255, 255, 0.2)', 
            'border': 'none', 
            'borderRadius': '8px', 
            'color': 'white', 
            'outline': 'none',
            'boxSizing': 'content-box'
        }),
        
        html.P("Enter CSV File Path:", style={
            'textAlign': 'left', 
            'color': 'rgba(255, 255, 255, 0.8)', 
            'fontSize': '14px', 
            'marginTop': '15px', 
            'marginBottom': '5px'
        }),
        
        dcc.Input(id='file-path', type='text', placeholder=r'C:\path\to\your\data.csv', style={
            'margin': '5px 0 20px 0', 
            'padding': '12px 15px', 
            'width': '100%', 
            'background': 'rgba(255, 255, 255, 0.2)', 
            'border': 'none', 
            'borderRadius': '8px', 
            'color': 'white', 
            'outline': 'none',
            'boxSizing': 'content-box'
        }),
        
        html.Button('Login & Load Data', id='login-button', n_clicks=0, style={
            'width': '100%', 
            'padding': '12px', 
            'backgroundColor': '#ffffff', 
            'color': '#333333', 
            'border': 'none', 
            'borderRadius': '8px', 
            'fontWeight': 'bold', 
            'cursor': 'pointer',
            'transition': '0.3s',
            'boxShadow': '0 4px 15px rgba(0,0,0,0.2)'
        })
    ])
])
app.layout=login_layout
# first login page

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', children=login_layout)
])
# --create dashbord
def create_dashboard(df):

    # --- 7 & 8: Date Processing ---
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['HourOfDay'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.day_name()
    df['Date'] = df['Timestamp'].dt.date
    df['Month_Name'] = df['Timestamp'].dt.month_name()

# --- 13 & 21: Time & Session Analysis ---
    df['SessionDuration_seconds'] = df.groupby('SessionID')['Timestamp'].transform(lambda x: (x.max() - x.min()).total_seconds())
    session_count = df.groupby('UserID')['SessionID'].nunique().reset_index()
    session_count.columns = ['UserID', 'SessionCount']

    df = pd.merge(df, session_count, on='UserID')

# Select Features
    features = [
        'TimeOnPage_seconds',
        'HourOfDay',
        'SessionDuration_seconds',
        'SessionCount'
    ]

    X = df[features]

    y = df['Purchased']
    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
 )
    rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
 )

    rf_model.fit(X_train, y_train)

    rf_prediction = rf_model.predict(X_test)

    rf_accuracy = accuracy_score(
     y_test,
     rf_prediction
 )
    gb_model = GradientBoostingClassifier(
    random_state=42
 )

    gb_model.fit(X_train, y_train)

    gb_prediction = gb_model.predict(X_test)

    gb_accuracy = accuracy_score(
        y_test,
        gb_prediction
    )
    report = classification_report(
        y_test,
        rf_prediction,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    report_df = report_df.reset_index()
    report_df.columns = report_df.columns.astype(str)
    print(report_df.columns)
    print(report_df.head())

    report_df = report_df.reset_index()

    report_df.columns = [str(col) for col in report_df.columns]

    report_table = dash_table.DataTable(
        data=report_df.to_dict("records"),
        columns=[
            {"name": str(col), "id": str(col)}
            for col in report_df.columns
        ]
    )

    funnel_steps = ['home', 'product_page', 'cart', 'checkout', 'confirmation']
    # --- making graphs
    page_counts = df.groupby('PageType')['SessionID'].nunique().reindex(funnel_steps).reset_index()
    page_counts.columns = ['PageType', 'UserCount']
    page_counts['Next_Step_Conv_%'] = (100 * (page_counts['UserCount'].shift(-1) / page_counts['UserCount'])).fillna(0).round(2)
    page_counts['Overall_Conv_%'] = (100 * (page_counts['UserCount'] / page_counts['UserCount'].iloc[0])).round(2)
    
    t1 = dash_table.DataTable(data=page_counts.to_dict('records'),  columns=[
        {"name": str(i), "id": str(i)}
        for i in page_counts.columns
    ]
 )

    # 4. Average Session Duration
    df['SessionDuration_seconds'] = df.groupby('SessionID')['Timestamp'].transform(lambda x: (x.max() - x.min()).total_seconds())
    avg_duration = df.groupby('Purchased')['SessionDuration_seconds'].mean().reset_index().round(2)
    t4 = dash_table.DataTable(data=avg_duration.to_dict('records'), columns=[
    {"name": str(i), "id": str(i)}
    for i in avg_duration.columns
])

    # 5. Conversion Rate by Country
    country_conv = df.groupby('Country')['Purchased'].mean().reset_index().sort_values('Purchased', ascending=False).head(10)
    country_conv['Purchased'] = (country_conv['Purchased'] * 100).round(2).astype(str) + '%'
    t5 = dash_table.DataTable(data=country_conv.to_dict('records'), columns=[
    {"name": str(i), "id": str(i)}
    for i in country_conv.columns
])

    # 6. Referral Source Conversion
    ref_conv = df.groupby('ReferralSource')['Purchased'].mean().reset_index()
    ref_conv['Purchased'] = (ref_conv['Purchased'] * 100).round(2).astype(str) + '%'
    t6 = dash_table.DataTable(data=ref_conv.to_dict('records'), columns=[
    {"name": str(i), "id": str(i)}
    for i in ref_conv.columns
])

    # 7. Step-to-Step Conversion by Referral
    ref_segmented = df.groupby(['ReferralSource', 'PageType'])['SessionID'].nunique().unstack(level=0).reindex(funnel_steps)
    ref_segmented_conv = (100 * (ref_segmented.shift(-1) / ref_segmented)).fillna(0).round(2).reset_index()
    t7 = dash_table.DataTable(data=ref_segmented_conv.to_dict('records'), columns=[
    {"name": str(i), "id": str(i)}
    for i in ref_segmented_conv.columns
])

    # 8. Avg Time on Page
    time_stage = df.groupby(['PageType', 'Purchased'])['TimeOnPage_seconds'].mean().unstack().reindex(funnel_steps).reset_index()
    time_stage.columns = ['PageType', 'Non-Buyer(s)', 'Buyer(s)']
    t8 = dash_table.DataTable(data=time_stage.round(2).to_dict('records'), columns=[
    {"name": str(i), "id": str(i)}
    for i in time_stage.columns
])

    # 9 & 10. Device and Referral Interaction
    interaction = df.groupby(['DeviceType', 'ReferralSource'])['Purchased'].mean().unstack().reset_index()
    t9 = dash_table.DataTable(data=interaction.round(4).to_dict('records'),columns=[
    {"name": str(i), "id": str(i)}
    for i in interaction.columns
])

    # 11. User Path (Next Page)
    df_sorted = df.sort_values(['SessionID', 'Timestamp'])
    df_sorted['NextPageType'] = df_sorted.groupby('SessionID')['PageType'].shift(-1)
    path_data = df_sorted[df_sorted['PageType'] == 'home'].groupby(['NextPageType', 'Purchased']).size().unstack(fill_value=0).reset_index()
    t11 = dash_table.DataTable(data=path_data.to_dict('records'), columns=[
    {"name": str(i), "id": str(i)}
    for i in path_data.columns
])

    # Step 10 & 11: Seasonality & Day of Week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig_day = px.histogram(df, x='DayOfWeek', color='Purchased', barmode='group', category_orders={'DayOfWeek': day_order}, title='Step 10: Weekly Activity')
    fig_month = px.line(df.groupby('Month_Name')['Purchased'].mean().reset_index(), x='Month_Name', y='Purchased', title=' Monthly Seasonality')
    # Funnel Analysis (Step 12)
    page_counts = df.groupby('PageType')['SessionID'].nunique().reindex(funnel_steps).reset_index()
    page_counts.columns = ['PageType', 'UserCount']
    fig_funnel = px.funnel(page_counts, x='UserCount', y='PageType', title='User Journey Funnel')

    # Conversion by Hour (Step 9)
    hourly_conv = df.groupby('HourOfDay')['Purchased'].mean().reset_index()
    fig_hourly = px.line(hourly_conv, x='HourOfDay', y='Purchased', title='Conversion Rate by Hour of Day', markers=True)

    # Conversion by Device (Step 18)
    device_conv = df.groupby('DeviceType')['Purchased'].mean().reset_index()
    fig_device = px.bar(device_conv, x='DeviceType', y='Purchased', title='Conversion Rate by Device Type', color='DeviceType')

    # Conversion by Country (Step 14)
    country_conv = df.groupby('Country')['Purchased'].mean().reset_index().sort_values('Purchased', ascending=False).head(10)
    fig_country = px.bar(country_conv, x='Purchased', y='Country', orientation='h', title='Top 10 Countries by Conversion', color='Purchased')
    # Buyers vs Non-Buyers Time (Step 21)
    time_stage_purchase = df.groupby(['PageType', 'Purchased'])['TimeOnPage_seconds'].mean().reset_index()
    fig_time = px.bar(time_stage_purchase, x='PageType', y='TimeOnPage_seconds', color='Purchased', barmode='group', title='Avg Time on Page: Buyers vs Non-Buyers')

    # Bounce Rate (Step 25)
    pages_per_session = df.groupby('SessionID')['PageType'].count()
    bounced_sessions = pages_per_session[pages_per_session == 1].index
    first_page_df = df.loc[df.groupby('SessionID')['Timestamp'].idxmin()]
    bounces_by_first_page = first_page_df[first_page_df['SessionID'].isin(bounced_sessions)]['PageType'].value_counts()
    total_by_first_page = first_page_df['PageType'].value_counts()
    bounce_rate_by_page = (bounces_by_first_page / total_by_first_page).fillna(0).reset_index()
    bounce_rate_by_page.columns = ['LandingPage', 'BounceRate']
    fig_bounce = px.bar(bounce_rate_by_page, x='LandingPage', y='BounceRate', title='Bounce Rate by Landing Page', color='BounceRate')

    # Step 21: Avg Time on Page (Buyers vs Non-Buyers)
    avg_time_page = df.groupby(['PageType', 'Purchased'])['TimeOnPage_seconds'].mean().reset_index()
    fig_step21 = px.bar(avg_time_page, x='PageType', y='TimeOnPage_seconds', color='Purchased', barmode='group', title='Step 21: Time on Page (Buyers vs Non-Buyers)')

    # Step 22: Session Count vs Purchase
    session_counts = df.groupby('UserID')['SessionID'].nunique().reset_index()
    user_purchased = df.groupby('UserID')['Purchased'].max().reset_index()
    merged_22 = pd.merge(session_counts, user_purchased, on='UserID')
    fig_step22 = px.box(merged_22, x='Purchased', y='SessionID', title=' Number of Sessions vs Purchase')

    # Step 24: Correlation Matrix (Heatmap)
    # Numeric columns മാത്രം എടുക്കുന്നു
    corr_data = df.select_dtypes(include=[np.number]).corr()
    fig_step24 = px.imshow(corr_data, text_auto=True, title=' Feature Correlation Heatmap')

    

# -------------------------------

    importance_data = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf_model.feature_importances_
    })

    importance_data = importance_data.sort_values(
        by="Importance",
        ascending=False
    )

    fig_step28 = px.bar(
        importance_data,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance",
        color="Importance"
    )
    comparison = pd.DataFrame({
    "Model": ["Random Forest", "Gradient Boosting"],
    "Accuracy": [rf_accuracy, gb_accuracy]
})

    fig_compare = px.bar(
        comparison,
        x="Model",
        y="Accuracy",
        color="Model",
        text="Accuracy",
        title="Model Comparison"
    )

    fig_compare.update_traces(texttemplate="%{text:.2%}", textposition="outside")
# --- Dash App Layout ---
    


    # funtion last layout
    
    return html.Div(style={
        'backgroundImage': 'url("https://discovery.sndimg.com/content/dam/images/discovery/fullset/2022/4/20/GettyImages-678915345.jpg.rend.hgtvcom.406.406.suffix/1650480060448.jpeg")',
        'backgroundSize': 'cover',
        'backgroundPosition': 'center',
        'backgroundAttachment': 'fixed',
        'padding': '30px',
        'minHeight': '100vh'
    }, children=[
        html.H1("Funnel Drop Analysis", style={'textAlign': 'center', 'color': '#ffffff', 'marginBottom': '30px', 'textShadow': '2px 2px 4px rgba(0,0,0,0.6)'}),

        # --- 1. Top Metrics Section (KPIs) ---
        html.Div([
            html.Div([
                html.H3(f"{df['Purchased'].mean():.2%}"),
                html.P("Overall Conversion Rate")
            ], style={'textAlign': 'center', 'color': '#ffffff','padding': '20px', 'backgroundColor':'rgba(255,255,255,0.1)' ,'backdropFilter':'blur(10px)', 'borderRadius': '10px', 'flex': '1', 'margin': '10px', 'textAlign': 'center', 'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)'}),
            
            html.Div([
                html.H3(f"{(len(bounced_sessions)/df['SessionID'].nunique()):.2%}"),
                html.P("Overall Bounce Rate")
            ], style={'textAlign': 'center', 'color': '#ffffff','padding': '20px', 'backgroundColor':'rgba(255,255,255,0.1)' ,'backdropFilter':'blur(10px)', 'borderRadius': '10px', 'flex': '1', 'margin': '10px', 'textAlign': 'center', 'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)'}),
            
            html.Div([
                html.H3(f"{df['Purchased'].sum()}"),
                html.P("Total Conversions")
            ], style={'textAlign': 'center', 'color': '#ffffff','padding': '20px', 'backgroundColor':'rgba(255,255,255,0.1)' ,'backdropFilter':'blur(10px)', 'borderRadius': '10px', 'flex': '1', 'margin': '10px', 'textAlign': 'center', 'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)'})
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '40px'}),

        # --- 2. Graphs Section ---
        html.Div([
            html.H2("Visual Analytics", style={'textAlign': 'center', 'color': '#ffffff', 'textShadow': '1px 1px 3px rgba(0,0,0,0.5)','backgroundColor':'rgba(255,255,255,0.1)' ,'backdropFilter':'blur(10px)'}),
            dcc.Graph(figure=fig_funnel, style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(figure=fig_hourly, style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(figure=fig_day, style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(figure=fig_month, style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(figure=fig_device, style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(figure=fig_country, style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(figure=fig_time, style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(figure=fig_bounce, style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(figure=fig_step22, style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(figure=fig_step24, style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(figure=fig_step28, style={'width': '98%', 'display': 'inline-block'}),
            dcc.Graph(figure=fig_compare),
        ], style={'marginBottom': '50px'}),

        # --- 3. Data Tables Section (Report) ---
        
        html.Div([
            html.H2("Detailed Analysis Report", style={'textAlign': 'center', 'color': "#e2e8ee", 'marginBottom': '30px'}),
            
            # Card 1: Funnel Table
            html.Div([
                html.H4("📊 1. Funnel Steps & Conversion Rate"),
                t1
            ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 'marginBottom': '25px'}),

            # Card 2: Country Table
            html.Div([
                html.H4("🌍 2. Top 10 Countries by Conversion"),
                t5
            ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 'marginBottom': '25px'}),

            # Card 3: Time on Page Table
            html.Div([
                html.H4("🕒 3. Avg Time on Page (Buyers vs Non-Buyers)"),
                t8
            ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 'marginBottom': '25px'}),
             
              html.Div([
                html.H4(" 4. SessionDuration_seconds"),
                t4
            ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 'marginBottom': '25px'}),
            
            html.Div([
                html.H4(" 5.Referral Source Conversion"),
                t6
            ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 'marginBottom': '25px'}),
            
            html.Div([
                html.H4(" 6.Step-to-Step Conversion by Referral"),
                t7
            ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 'marginBottom': '25px'}),
            
            html.Div([
                html.H4("7. Device and Referral Interaction"),
                t9
            ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 'marginBottom': '25px'}),
            
            html.Div([
                html.H4("8. User Path (Next Page)"),
                t11
            ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 'marginBottom': '25px'}),

        ], style={'maxWidth': '1100px', 'margin': 'auto'}),
        html.Div([
    html.H4("Classification Report"),
    report_table
    ], style={
    'backgroundColor': 'white',
    'padding': '20px',
    'borderRadius': '10px',
    'marginBottom': '25px'
   }),
        # --- 4. Insights Section ---
        html.Div([
            html.H3("💡 Final Business Insights", style={'color': '#2c3e50'}),
            html.Ul([
                html.Li("Conversion rate varies significantly by day and hour."),
                html.Li("Buyers spend more time on key pages compared to non-buyers."),
                html.Li("The correlation heatmap shows which user behaviors lead to purchases."),
                html.Li("Machine Learning model (Random Forest) highlights TimeOnPage as a key predictor.")
            ])
        ], style={'padding': '20px', 'backgroundColor': 'rgba(232, 244, 248, 0.9)', 'borderRadius': '10px', 'marginTop': '40px'})
    ])

@app.callback(
    Output('page-content', 'children'),
    [Input('login-button', 'n_clicks')],
    [State('file-path', 'value')]
)
def update_output(n_clicks, file_path):
    if n_clicks > 0:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # login succsee
                return create_dashboard(df)
            except Exception as e:
                return html.Div(f"Error loading file: {e}", style={'color': 'red'})
        else:
            return html.Div("Invalid File Path! Please check the path.", style={'color': 'red'})
    return login_layout

if __name__ == '__main__':
    app.run(debug=True)
    