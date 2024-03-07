from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import requests

app = FastAPI()

class MatchPredictionRequest(BaseModel):
    home_team: str
    away_team: str

def fetch_data_from_api(api_token, competition_id, season):
    api_url = f"https://api.football-data.org/v2/competitions/{competition_id}/matches?season={season}"
    headers = {"X-Auth-Token": api_token}
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data['matches']
    else:
        raise HTTPException(status_code=500, detail=f"Error: {response.status_code} while retrieving data for season {season}")

def preprocess_api_data(api_data):
    api_df = pd.DataFrame(api_data)
    api_df = api_df[['homeTeam', 'awayTeam', 'score']]

    api_df['HomeTeam'] = api_df['homeTeam'].apply(lambda x: x['name'])
    api_df['AwayTeam'] = api_df['awayTeam'].apply(lambda x: x['name'])
    
    api_df['FTHG'] = api_df['score'].apply(lambda x: x['fullTime']['homeTeam'])
    api_df['FTAG'] = api_df['score'].apply(lambda x: x['fullTime']['awayTeam'])
    
    api_df['FTR'] = api_df.apply(lambda row: f"{row['FTHG']}-{row['FTAG']}", axis=1)
    
    return api_df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]

competition_id = "PL"

all_data = []
for season in range(2020, 2024):
    api_token = "8c27542ed83f4b0492db46a921ba8dd1"
    season_data = fetch_data_from_api(api_token, competition_id, season)
    all_data.extend(season_data)

api_df = preprocess_api_data(all_data)

df = api_df.dropna(subset=['FTHG', 'FTAG'])

X = df[['HomeTeam', 'AwayTeam']]
y_classification = df['FTR']
y_regression = df[['FTHG', 'FTAG']]

X_train, X_test, y_train_regression, y_test_regression = train_test_split(X, y_regression, test_size=0.2, random_state=42)

label_encoder_X = LabelEncoder()
X.loc[:, 'HomeTeam'] = label_encoder_X.fit_transform(X['HomeTeam'])
X.loc[:, 'AwayTeam'] = label_encoder_X.transform(X['AwayTeam'])

label_encoder_y_classification = LabelEncoder()
y_classification_encoded = label_encoder_y_classification.fit_transform(y_classification)

X_train, X_test, y_train_classification, y_test_classification = train_test_split(X, y_classification_encoded, test_size=0.2, random_state=42)

classifier_model = RandomForestClassifier(n_estimators=100, random_state=42)
classifier_model.fit(X_train, y_train_classification)

regressor_models = {}
for column in y_regression.columns:
    regressor_model = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor_model.fit(X_train, y_train_regression[column])
    regressor_models[column] = regressor_model



def predict_goals(home_team, away_team):
    try:
        home_team_encoded = label_encoder_X.transform([home_team])[0]
        away_team_encoded = label_encoder_X.transform([away_team])[0]
        
        match_data = [[home_team_encoded, away_team_encoded]]
        
        predictions = {}
        for column, model in regressor_models.items():
            prediction = model.predict(match_data)
            rounded_prediction = round(prediction[0])
            predictions[column] = rounded_prediction
        
        return predictions
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

def determine_overall_winner(home_team, away_team):
    home_team_goals = 0
    away_team_goals = 0
    
    num_simulations = 100
    for _ in range(num_simulations):
        goals = predict_goals(home_team, away_team)
        if goals is not None:
            home_team_goals += goals['FTHG']
            away_team_goals += goals['FTAG']
    
    if home_team_goals > away_team_goals:
        return home_team
    elif home_team_goals < away_team_goals:
        return away_team
    else:
        return "Draw"

@app.get("/", response_class=HTMLResponse)
def prediction_form():
    return '''
    <html>
    <body>
    <form method="post">
    <label for="home_team">Home Team:</label><br>
    <input type="text" id="home_team" name="home_team"><br>
    <label for="away_team">Away Team:</label><br>
    <input type="text" id="away_team" name="away_team"><br><br>
    <input type="submit" value="Predict">
    </form>
    </body>
    </html>
    '''

@app.post("/", response_class=HTMLResponse)
async def predict_match(request: Request, home_team: str = Form(...), away_team: str = Form(...)):
    #predicted_result = predict_match_result(home_team, away_team)
    predicted_goals = predict_goals(home_team, away_team)
    overall_winner = determine_overall_winner(home_team, away_team)
    
    return f'''
    <html>
    <body>
    <h2>Prediction Results</h2>
    <p>Home Team: {home_team}</p>
    <p>Away Team: {away_team}</p>
    
    <p>Predicted Goals: {predicted_goals}</p>
    <p>Overall Winner: {overall_winner}</p>
    </body>
    </html>
    '''
