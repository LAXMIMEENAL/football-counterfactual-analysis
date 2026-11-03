
import pickle
import requests
import time
from collections import defaultdict

RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# Load the complete original dataset
with open("records.pkl", "rb") as f:
    records = pickle.load(f)

# Group records by match
match_ids = sorted(set(r["match"] for r in records))

# Download goal events for each match
session = requests.Session()
goals_by_match = {}

for i, match_id in enumerate(match_ids, start=1):

    url = f"{RAW}/events/{match_id}.json"
    events = None

    for attempt in range(5):
        try:
            response = session.get(url, timeout=60)

            if response.status_code == 200:
                events = response.json()
                break

            if response.status_code == 404:
                break

            print(
                f"HTTP {response.status_code} for match "
                f"{match_id}, attempt {attempt + 1}"
            )

        except requests.RequestException as error:
            print(f"Request error for {match_id}: {error}")

        time.sleep(2 * (attempt + 1))

    if events is None:
        raise RuntimeError(
            f"Could not download match {match_id}. "
            "Stopping without saving an incomplete dataset."
        )

    goals = defaultdict(list)

    for event in events:
        if event["type"]["name"] != "Shot":
            continue

        if event["shot"]["outcome"]["name"] != "Goal":
            continue

        key = (
            event["possession"],
            event["team"]["id"],
            event["period"]
        )

        goal_time = event["minute"] * 60 + event["second"]
        goals[key].append(goal_time)

    goals_by_match[match_id] = goals

    if i % 20 == 0 or i == len(match_ids):
        print(f"Processed {i}/{len(match_ids)} matches", flush=True)

# Add goal-within-10-seconds label
for record in records:

    key = (
        record["poss"],
        record["team"],
        record["period"]
    )

    goal_times = goals_by_match[record["match"]].get(key, [])

    record["y_goal"] = int(
        any(
            record["time"] <= goal_time <= record["time"] + 10
            for goal_time in goal_times
        )
    )

# Save only after all matches have been processed
with open("records2.pkl", "wb") as f:
    pickle.dump(records, f)

print("\nDataset generated successfully.")
print("Total records:", len(records))
print("Total matches:", len(match_ids))
print("Shots:", sum(r["type"] == "Shot" for r in records))
print("Passes:", sum(r["type"] == "Pass" for r in records))
print("Records with y_goal:", sum("y_goal" in r for r in records))
