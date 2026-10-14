import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv("engineered_matches.csv")
matches_df = pd.read_csv("./data/processed/matches.csv")

# -----------------------------
# TRAIN MODEL
# -----------------------------

X = df[
    [
        "home_avg_goals",
        "away_avg_goals",
        "home_form",
        "away_form"
    ]
]

y = df["result"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# -----------------------------
# UPCOMING FIXTURES
# -----------------------------

upcoming_matches = matches_df[matches_df["FTR"].isna()]

fixtures = []

for _, row in upcoming_matches.iterrows():
    fixtures.append((row["HomeTeam"], row["AwayTeam"]))

print("\n==============================")
print("UPCOMING FIXTURES")
print("==============================\n")

if len(fixtures) == 0:
    print("No upcoming fixtures found.")
    exit()

for i, (home, away) in enumerate(fixtures):
    print(f"{i+1}. {home} vs {away}")

# -----------------------------
# USER INPUT
# -----------------------------

print("\n==============================")
print("ENTER MATCH FOR PREDICTION")
print("==============================\n")

home_team = input("Enter Home Team: ").strip()
away_team = input("Enter Away Team: ").strip()

# -----------------------------
# VALIDATION
# -----------------------------

if (home_team, away_team) not in fixtures:
    print("\n⚠️ WARNING: This match is NOT in upcoming fixtures.")

    confirm = input("Do you still want to proceed? (y/n): ").strip().lower()

    if confirm != "y":
        print("Exiting...")
        exit()

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
# FEATURE FUNCTIONS
# -----------------------------

def calculate_avg_goals(team_history, team_name):

    goals = []

    for _, match in team_history.iterrows():

        if match["HomeTeam"] == team_name:
            goals.append(match["FTHG"])
        else:
            goals.append(match["FTAG"])

    return sum(goals) / len(goals)


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

# -----------------------------
# FEATURE CALCULATION
# -----------------------------

home_avg_goals = calculate_avg_goals(home_history, home_team)
away_avg_goals = calculate_avg_goals(away_history, away_team)

home_form = calculate_form(home_history, home_team)
away_form = calculate_form(away_history, away_team)

# -----------------------------
# PREDICTION INPUT
# -----------------------------

prediction_data = pd.DataFrame([
    {
        "home_avg_goals": home_avg_goals,
        "away_avg_goals": away_avg_goals,
        "home_form": home_form,
        "away_form": away_form
    }
])

# -----------------------------
# PREDICT
# -----------------------------

prediction = model.predict(prediction_data)[0]

# -----------------------------
# OUTPUT
# -----------------------------

print("\n==============================")
print("MATCH PREDICTION RESULT")
print("==============================\n")

print(f"{home_team} vs {away_team}\n")

if prediction == "H":
    print(f"🏆 Predicted Winner: {home_team}")

elif prediction == "A":
    print(f"🏆 Predicted Winner: {away_team}")

else:
    print("🤝 Predicted Result: Draw")

print("\n==============================\n")