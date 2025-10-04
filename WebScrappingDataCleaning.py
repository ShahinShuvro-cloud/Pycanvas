import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
url = "http://quotes.toscrape.com/page/1/"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Extract data
quotes = []
authors = []
for q in soup.find_all("div", class_="quote"):
    quote_text = q.find("span", class_="text").get_text(strip=True)
    author = q.find("small", class_="author").get_text(strip=True)
    
    quotes.append(quote_text)
    authors.append(author)

# Store in DataFrame
df = pd.DataFrame({"Quote": quotes, "Author": authors})
print(df.head())

print( 'Remove duplicates')
df = df.drop_duplicates()
print(df)

print('Handle missing values')
df = df.dropna()
print(df)

print('String cleaning')
df["Quote"] = df["Quote"].str.replace("“", "").str.replace("”", "")
df["Author"] = df["Author"].str.strip()

print('Example of standardizing text')
df["Author"] = df["Author"].str.title()

print(df.to_csv("cleaned_quotes.csv", index=False))