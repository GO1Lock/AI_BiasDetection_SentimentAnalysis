import torch
from afinn import Afinn
from transformers import BertTokenizer, BertForSequenceClassification

# Load the pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('./fine_tuned_bert_model')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define label mapping based on the model configuration
label_mapping = {0: 'negative', 1: 'positive', 2: 'neutral'}

def clean_text(text):
    
    return text

def get_sentiment(text):
    # Clean the input text
    text = clean_text(text)

    # Prepare the text for the model
    encodings = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors="pt").to(device)

    # Evaluate the text
    model.eval()
    with torch.no_grad():
        output = model(**encodings)

    # Convert output logits to sentiment prediction
    prediction = torch.argmax(output.logits, dim=-1).item()

    # Map the numeric prediction back to a string label
    sentiment_label = label_mapping[prediction]

    return sentiment_label

afinn = Afinn()

def get_word_level_sentiment(text):
    words = text.split()
    word_sentiments = []
    positive_count, negative_count, neutral_count = 0, 0, 0

    for word in words:
        score = afinn.score(word)
        if score > 0:
            sentiment = 'positive'
            positive_count += 1
        elif score < 0:
            sentiment = 'negative'
            negative_count += 1
        else:
            sentiment = 'neutral'
            neutral_count += 1
        word_sentiments.append({'word': word, 'sentiment': sentiment})

    return word_sentiments, positive_count, negative_count, neutral_count