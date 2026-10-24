"""Step 1: load NFL schedule data and compute the home-team baseline.

Run:  python step1_load_data.py
"""
import nflreadpy as nfl

# nflreadpy returns a polars DataFrame. We convert to pandas because
# that's what you'll use with scikit-learn.
games = nfl.load_schedules([2021, 2022, 2023, 2024]).to_pandas()

# Regular season only, and only games that have been played.
games = games[(games["game_type"] == "REG") & games["home_score"].notna()].copy()

# Ties (result == 0) are rare. We drop them so every game is a clear
# home win or home loss. Mention this in your README.
n_ties = int((games["result"] == 0).sum())
games = games[games["result"] != 0].copy()
print("Ties dropped:", n_ties)

# Target: 1 if the home team won, 0 if the away team won.
# "result" is home_score minus away_score.
games["home_win"] = (games["result"] > 0).astype(int)

print("Rows:", len(games))
print(games[["season", "week", "home_team", "away_team", "home_score", "away_score", "home_win"]].head())

# Split by time:
#   train 2021-22, validation 2023: use these to try features and models.
#   test 2024: the final exam. Don't look at it while building.
train = games[games["season"] <= 2022]
valid = games[games["season"] == 2023]
test = games[games["season"] == 2024]
print("Train (2021-22):", len(train), "Validation (2023):", len(valid), "Test (2024):", len(test))

# Baseline: always predict "home team wins".
# Any model has to beat this number to be worth anything.
# It is fine to print the baseline for 2023. Leave 2024 alone until the end.
print("Baseline accuracy on 2023 validation:", round(valid["home_win"].mean(), 3))
