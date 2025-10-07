import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = {
    "date": ["2024-08-01", "2024-08-02", "2024-08-03", "2024-08-04", "2024-08-05"],
    "home_team": ["Manchester City", "Real Madrid", "Bayern Munich", "Liverpool", "Barcelona"],
    "away_team": ["Arsenal", "Chelsea", "PSG", "Juventus", "Inter Milan"],
    "home_goals": [3, 2, 1, 0, 4],
    "away_goals": [1, 2, 2, 3, 2],
    "possession_home": [65, 58, 61, 49, 70],
    "shots_on_target_home": [7, 5, 4, 2, 8],
    "shots_on_target_away": [3, 4, 5, 6, 4],
}
df = pd.DataFrame(data)
df.to_csv("football_matches.csv", index=False)
print("football_matches.csv created successfully!")
print(df)

print(df.head())
print("\nData Info:")
print(df.info())

df.dropna(inplace=True)  
df['date'] = pd.to_datetime(df['date'])

# Add new column: match_result (Win/Draw/Loss for home team)
def result(row):
    if row['home_goals'] > row['away_goals']:
        return 'Win'
    elif row['home_goals'] == row['away_goals']:
        return 'Draw'
    else:
        return 'Loss'

df['match_result'] = df.apply(result, axis=1)
print("\n Added match_result column:")
print(df[['home_team', 'away_team', 'home_goals', 'away_goals', 'match_result']].head())

#  Win/Draw/Loss Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='match_result', data=df, palette='coolwarm')
plt.title("Distribution of Match Results")
plt.show()

# Possession vs Goals
plt.figure(figsize=(6,4))
sns.scatterplot(x='possession_home', y='home_goals', hue='match_result', data=df)
plt.title("Possession vs Goals (Home Team)")
plt.show()

# Average Goals per Team
avg_goals = df.groupby('home_team')['home_goals'].mean().sort_values(ascending=False)
plt.figure(figsize=(8,5))
avg_goals.head(10).plot(kind='bar', color='orange')
plt.title("Top 10 Teams by Average Home Goals")
plt.ylabel("Average Goals")
plt.show()

# Prepare Features and Target
X = df[['possession_home', 'shots_on_target_home', 'shots_on_target_away']]
y = df['match_result']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)

print(" Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Win','Draw','Loss'], yticklabels=['Win','Draw','Loss'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Predict New Match Outcome
new_match = pd.DataFrame({
    'possession_home': [62],
    'shots_on_target_home': [8],
    'shots_on_target_away': [3]
})

pred = model.predict(new_match)
print(f"Predicted Result for the New Match:  {pred[0]}")