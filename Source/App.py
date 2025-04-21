from dash import Dash, dcc, html, Input, Output
import pandas as pd
import joblib
import os
import numpy as np

# Load the trained Random Forest model and the scaler used during training
model_path = os.path.join(os.path.dirname(__file__), 'randomforest.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

app = Dash(__name__)
server = app.server

# Define the layout with inputs for all 13 features
app.layout = html.Div([
    html.H1(
        "BrightPath Academy: Student Performance Predictor",
        style={
            'textAlign': 'center',
            'color': '#2c3e50',
            'fontFamily': 'Arial',
            'padding': '20px',
            'backgroundColor': 'Green'
        }
    ),
    
    html.Div(
        [
            # Age
            html.Div(
                [
                    html.Label("Age (15-18)", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Input(id='age', type='number', value=16, min=15, max=18, style={'width': '100%', 'padding': '8px'})
                ],
                style={'marginBottom': '15px'}
            ),
            # Gender
            html.Div(
                [
                    html.Label("Gender", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.RadioItems(
                        id='gender',
                        options=[
                            {'label': 'Male', 'value': 0},
                            {'label': 'Female', 'value': 1}
                        ],
                        value=0,
                        inline=True,
                        style={'marginTop': '5px'}
                    )
                ],
                style={'marginBottom': '15px'}
            ),
            # Ethnicity
            html.Div(
                [
                    html.Label("Ethnicity", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='ethnicity',
                        options=[
                            {'label': 'Caucasian', 'value': 0},
                            {'label': 'African American', 'value': 1},
                            {'label': 'Asian', 'value': 2},
                            {'label': 'Other', 'value': 3}
                        ],
                        value=0,
                        style={'width': '100%'}
                    )
                ],
                style={'marginBottom': '15px'}
            ),
            # Parental Education
            html.Div(
                [
                    html.Label("Parental Education", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='parental_education',
                        options=[
                            {'label': 'None', 'value': 0},
                            {'label': 'High School', 'value': 1},
                            {'label': 'Some College', 'value': 2},
                            {'label': "Bachelor's", 'value': 3},
                            {'label': 'Higher Study', 'value': 4}
                        ],
                        value=2,
                        style={'width': '100%'}
                    )
                ],
                style={'marginBottom': '15px'}
            ),
            # Study Time
            html.Div(
                [
                    html.Label("Weekly Study Time (hours, 0-20)", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Input(id='study_time', type='number', value=5, min=0, max=20, style={'width': '100%', 'padding': '8px'})
                ],
                style={'marginBottom': '15px'}
            ),
            # Absences
            html.Div(
                [
                    html.Label("Number of Absences (0-29)", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Input(id='absences', type='number', value=2, min=0, max=29, style={'width': '100%', 'padding': '8px'})
                ],
                style={'marginBottom': '15px'}
            ),
            # Tutoring
            html.Div(
                [
                    html.Label("Receiving Tutoring", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.RadioItems(
                        id='tutoring',
                        options=[
                            {'label': 'Yes', 'value': 1},
                            {'label': 'No', 'value': 0}
                        ],
                        value=1,
                        inline=True,
                        style={'marginTop': '5px'}
                    )
                ],
                style={'marginBottom': '15px'}
            ),
            # Parental Support
            html.Div(
                [
                    html.Label("Parental Support", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='parental_support',
                        options=[
                            {'label': 'None', 'value': 0},
                            {'label': 'Low', 'value': 1},
                            {'label': 'Moderate', 'value': 2},
                            {'label': 'High', 'value': 3},
                            {'label': 'Very High', 'value': 4}
                        ],
                        value=3,
                        style={'width': '100%'}
                    )
                ],
                style={'marginBottom': '15px'}
            ),
            # Extracurricular
            html.Div(
                [
                    html.Label("Participates in Extracurricular", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.RadioItems(
                        id='extracurricular',
                        options=[
                            {'label': 'Yes', 'value': 1},
                            {'label': 'No', 'value': 0}
                        ],
                        value=0,
                        inline=True,
                        style={'marginTop': '5px'}
                    )
                ],
                style={'marginBottom': '15px'}
            ),
            # Sports
            html.Div(
                [
                    html.Label("Participates in Sports", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.RadioItems(
                        id='sports',
                        options=[
                            {'label': 'Yes', 'value': 1},
                            {'label': 'No', 'value': 0}
                        ],
                        value=0,
                        inline=True,
                        style={'marginTop': '5px'}
                    )
                ],
                style={'marginBottom': '15px'}
            ),
            # Music
            html.Div(
                [
                    html.Label("Participates in Music", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.RadioItems(
                        id='music',
                        options=[
                            {'label': 'Yes', 'value': 1},
                            {'label': 'No', 'value': 0}
                        ],
                        value=0,
                        inline=True,
                        style={'marginTop': '5px'}
                    )
                ],
                style={'marginBottom': '15px'}
            ),
            # Volunteering
            html.Div(
                [
                    html.Label("Participates in Volunteering", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.RadioItems(
                        id='volunteering',
                        options=[
                            {'label': 'Yes', 'value': 1},
                            {'label': 'No', 'value': 0}
                        ],
                        value=0,
                        inline=True,
                        style={'marginTop': '5px'}
                    )
                ],
                style={'marginBottom': '15px'}
            ),
            # GPA
            html.Div(
                [
                    html.Label("GPA (0-4)", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Input(id='gpa', type='number', value=2, min=0, max=4, step=0.1, style={'width': '100%', 'padding': '8px'})
                ],
                style={'marginBottom': '15px'}
            ),
            # Predict Button
            html.Div(
                html.Button('Predict', id='predict-button', style={
                    'width': '100%', 'padding': '10px', 'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'cursor': 'pointer'
                }),
                style={'marginTop': '20px'}
            )
        ],
        style={
            'width': '50%',
            'margin': 'auto',
            'padding': '20px',
            'border': '1px solid #ddd',
            'borderRadius': '10px',
            'backgroundColor': 'white'
        }
    ),
    
    # Prediction Output
    html.Div(
        id='prediction-output',
        style={'textAlign': 'center', 'marginTop': '20px', 'fontSize': '20px'}
    )
], style={'backgroundColor': 'Green', 'minHeight': '100vh'})

# Callback to handle prediction
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [Input('age', 'value'),
     Input('gender', 'value'),
     Input('ethnicity', 'value'),
     Input('parental_education', 'value'),
     Input('study_time', 'value'),
     Input('absences', 'value'),
     Input('tutoring', 'value'),
     Input('parental_support', 'value'),
     Input('extracurricular', 'value'),
     Input('sports', 'value'),
     Input('music', 'value'),
     Input('volunteering', 'value'),
     Input('gpa', 'value')]
)
def predict(n_clicks, age, gender, ethnicity, parental_education, study_time, absences, tutoring, parental_support, extracurricular, sports, music, volunteering, gpa):
    if n_clicks is None:
        return ""
    
    # Validate inputs
    inputs = [age, gender, ethnicity, parental_education, study_time, absences, tutoring, parental_support, extracurricular, sports, music, volunteering, gpa]
    if any(x is None for x in inputs):
        return html.H3("Please fill in all fields.", style={'color': 'red'})
    
    # Prepare input data with the 13 original features
    input_data = pd.DataFrame(
        [[age, gender, ethnicity, parental_education, study_time, absences, tutoring, parental_support, extracurricular, sports, music, volunteering, gpa]],
        columns=['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering', 'GPA']
    )
    
    # Scale the 13 features using the scaler fitted on 13 features
    input_scaled = scaler.transform(input_data)
    input_scaled_df = pd.DataFrame(input_scaled, columns=input_data.columns)
    
    # Compute derived features from scaled data
    student_descriptors = float(input_scaled_df['Age'][0]) + float(input_scaled_df['Gender'][0]) + float(input_scaled_df['Ethnicity'][0]) + float(input_scaled_df['ParentalEducation'][0])
    activity = float(input_scaled_df['Extracurricular'][0]) + float(input_scaled_df['Sports'][0]) + float(input_scaled_df['Music'][0]) + float(input_scaled_df['Volunteering'][0])
    
    # Prepare final input with the 7 features
    completed_data = pd.DataFrame(
        [[input_scaled_df['StudyTimeWeekly'][0], input_scaled_df['Absences'][0], input_scaled_df['Tutoring'][0], input_scaled_df['ParentalSupport'][0], input_scaled_df['GPA'][0], activity, student_descriptors]],
        columns=['StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'GPA', 'Activity', 'StudentDiscriptors']
    )
    
    # Make prediction
    prediction = model.predict(completed_data)[0]
    grades = ['A', 'B', 'C', 'D', 'F']
    predicted_grade = grades[int(prediction)]
    
    # Provide additional context based on prediction
    if predicted_grade in ['D', 'F']:
        message = f"Predicted Grade: {predicted_grade} - This student may be at risk. Consider interventions like tutoring or attendance monitoring."
    else:
        message = f"Predicted Grade: {predicted_grade}"
    
    return html.H3(message, style={'color': '#2c3e50'})

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)