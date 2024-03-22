import pandas as pd
import re

# Load the dataset
file_name = 'True.csv'  
data = pd.read_csv(file_name)

# Function to clean text
def clean_text(text):
    # Text lowercased
    text = text.lower()
    # Remove URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'\W+|\d+', ' ', text)
    return text

# Clean the text
data['clean_text'] = data['text'].apply(clean_text)

# Save the cleaned text to a new CSV file
output_file_name = 'Cleaned_Text.csv'
data.to_csv(output_file_name, index=False)

