import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# Generate synthetic website event data
users = [f"user_{i}" for i in range(1, 51)]
pages = ["Home", "About", "Products", "Product_Details", "Cart", "Checkout"]
events = ["page_view", "click", "add_to_cart", "purchase"]

data = []
start_time = datetime(2025, 10, 1, 8, 0, 0)

for _ in range(1000):
    user = np.random.choice(users)
    page = np.random.choice(pages, p=[0.3, 0.1, 0.25, 0.15, 0.1, 0.1])
    event = np.random.choice(events, p=[0.6, 0.25, 0.1, 0.05])
    timestamp = start_time + timedelta(minutes=np.random.randint(0, 5000))
    data.append([user, page, event, timestamp])

df = pd.DataFrame(data, columns=["user_id", "page", "event_type", "timestamp"])
df.to_csv("click_data.csv", index=False)

print(df.head())

#Load and Explore Data
df = pd.read_csv("click_data.csv", parse_dates=["timestamp"])
print(df.info())
print(df['event_type'].value_counts())

# Analyze website actiivity
page_views = df['page'].value_counts()
print(page_views)           #most visited page

event_counts = df['event_type'].value_counts()
print(event_counts)         #Most common event types

#peak user activity by hour
df['hour'] = df['timestamp'].dt.hour
activity_by_hour = df.groupby('hour')['event_type'].count()
activity_by_hour.plot(kind='bar', title="Website Activity by Hour")

#Visualization
plt.figure(figsize=(10,5))
sns.countplot(data=df, x='page', order=df['page'].value_counts().index, palette="viridis")
plt.title("Most Visited Pages")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(data=df, x='event_type', palette="mako")
plt.title("Event Type Distribution")
plt.show()

# Conversion Funnel Analysis
conversion = df.groupby('user_id')['event_type'].apply(list)

purchase_users = sum(['purchase' in events for events in conversion])
add_to_cart_users = sum(['add_to_cart' in events for events in conversion])
page_view_users = sum(['page_view' in events for events in conversion])

funnel = pd.DataFrame({
    'Stage': ['Page View', 'Add to Cart', 'Purchase'],
    'Users': [page_view_users, add_to_cart_users, purchase_users]
})

sns.barplot(data=funnel, x='Stage', y='Users', palette='cool')
plt.title("User Conversion Funnel")
plt.show()
