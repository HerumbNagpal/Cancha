import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load engineered dataset
df = pd.read_csv("engineered_matches.csv")

# Features
X = df[
    [
        "home_avg_goals",
        "away_avg_goals",
        "home_form",
        "away_form"
    ]
]

# Target
y = df["result"]

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# -----------------------------
# USER INPUT
# -----------------------------
home_team = input("Enter Home Team: ")
away_team = input("Enter Away Team: ")

# Load original matches dataset
matches_df = pd.read_csv("./data/processed/matches.csv")

# -----------------------------
# HOME TEAM HISTORY
# -----------------------------
home_history = matches_df[
    (matches_df["HomeTeam"] == home_team) |
    (matches_df["AwayTeam"] == home_team)
]

# -----------------------------
# AWAY TEAM HISTORY
# -----------------------------
away_history = matches_df[
    (matches_df["HomeTeam"] == away_team) |
    (matches_df["AwayTeam"] == away_team)
]

# -----------------------------
# AVG GOALS FUNCTION
# -----------------------------
def calculate_avg_goals(team_history, team_name):

    goals = []

    for _, match in team_history.iterrows():

        if match["HomeTeam"] == team_name:
            goals.append(match["FTHG"])

        else:
            goals.append(match["FTAG"])

    return sum(goals) / len(goals)


# -----------------------------
# FORM FUNCTION
# -----------------------------
def calculate_form(team_history, team_name):

    recent_matches = team_history.tail(5)

    points = 0

    for _, match in recent_matches.iterrows():

        if match["HomeTeam"] == team_name:

            if match["FTR"] == "H":
                points += 3

            elif match["FTR"] == "D":
                points += 1

        elif match["AwayTeam"] == team_name:

            if match["FTR"] == "A":
                points += 3

            elif match["FTR"] == "D":
                points += 1

    return points


# Calculate features
home_avg_goals = calculate_avg_goals(home_history, home_team)

away_avg_goals = calculate_avg_goals(away_history, away_team)

home_form = calculate_form(home_history, home_team)

away_form = calculate_form(away_history, away_team)

# Create prediction input
prediction_data = pd.DataFrame([
    {
        "home_avg_goals": home_avg_goals,
        "away_avg_goals": away_avg_goals,
        "home_form": home_form,
        "away_form": away_form
    }
])

# Predict
prediction = model.predict(prediction_data)[0]

# Output
if prediction == "H":
    print(f"\nPredicted Winner: {home_team}")

elif prediction == "A":
    print(f"\nPredicted Winner: {away_team}")

else:
    print("\nPredicted Result: Draw")