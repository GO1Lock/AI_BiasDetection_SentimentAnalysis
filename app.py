from flask import Flask, request, jsonify, render_template
import os
import sys
import openai
from inference import get_sentiment, get_word_level_sentiment


current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

openai.api_key = '______API_______OPENAI_________KEY______ARGON______ALIU________'

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 



@app.route('/analyze_with_bert', methods=['POST'])
def analyze_with_bert():
    data = request.json
    text = data['text']
    
    # Split the text into sentences
    sentences = text.split('.')  
    
    # Analyze each sentence
    sentiments = {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}
    for sentence in sentences:
        if sentence.strip():  # skip empty sentences
            sentiment = get_sentiment(sentence)
            sentiments[sentiment] += 1
            sentiments['total'] += 1
    
    return jsonify(sentiments)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data['text']
    word_sentiments, positive_count, negative_count, neutral_count = get_word_level_sentiment(text)
    return jsonify(word_sentiments=word_sentiments, positive=positive_count, negative=negative_count, neutral=neutral_count)

@app.route('/generate_summaries', methods=['POST'])
def generate_summaries():
    print("generate_summaries endpoint was called")
    data = request.json
    text = data['text']
    try:
        # Constructing the prompts
        conservative_prompt = f"Summarize this article as a conservative: {text}"
        liberal_prompt = f"Summarize this article as a liberal: {text}"

        # Making OpenAI API calls using the new interface
        conservative_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": conservative_prompt}
            ]
        )
        liberal_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": liberal_prompt}
            ]
        )

        # Debugging print statements
        print("Conservative response:", conservative_response)
        print("Liberal response:", liberal_response)

        # Extracting summaries
        conservative_summary = conservative_response['choices'][0]['message']['content']
        liberal_summary = liberal_response['choices'][0]['message']['content']
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)})

    # Formatting and returning the response
    response = {
        'conservative_summary': conservative_summary,
        'liberal_summary': liberal_summary
    }
    return jsonify(response)





def calculate_bias_percentages(data):
    left_positive = data['left_positive']
    left_negative = data['left_negative']
    left_total = data['left_total']
    right_positive = data['right_positive']
    right_negative = data['right_negative']
    right_total = data['right_total']

    try:
        # Calculate proportions and bias for word-level
        left_bias = (left_positive - left_negative) / left_total
        right_bias = (right_positive - right_negative) / right_total

        # Normalize and convert to percentages for word-level bias
        total_bias = abs(left_bias) + abs(right_bias)

        if total_bias == 0:
            left_percentage = right_percentage = 0
        else:
            left_percentage = (abs(left_bias) / total_bias) * 100
            right_percentage = (abs(right_bias) / total_bias) * 100

        # Calculate sentence-level bias percentages
        left_positive_sentences = data['left_positive_sentences']
        left_negative_sentences = data['left_negative_sentences']
        right_positive_sentences = data['right_positive_sentences']
        right_negative_sentences = data['right_negative_sentences']

        total_sentences = left_positive_sentences + left_negative_sentences + right_positive_sentences + right_negative_sentences
        left_sentence_bias = (left_positive_sentences - left_negative_sentences) / total_sentences
        right_sentence_bias = (right_positive_sentences - right_negative_sentences) / total_sentences

        total_sentence_bias = abs(left_sentence_bias) + abs(right_sentence_bias)

        if total_sentence_bias == 0:
            left_sentence_percentage = right_sentence_percentage = 0
        else:
            left_sentence_percentage = (abs(left_sentence_bias) / total_sentence_bias) * 100
            right_sentence_percentage = (abs(right_sentence_bias) / total_sentence_bias) * 100

        return {
            'left_bias': left_percentage,
            'right_bias': right_percentage,
            'left_sentence_bias': left_sentence_percentage,
            'right_sentence_bias': right_sentence_percentage
        }

    except Exception as error:
        print('Error calculating bias:', error)
        # Return default values in case of an error
        return {
            'left_bias': 0,
            'right_bias': 0,
            'left_sentence_bias': 0,
            'right_sentence_bias': 0
        }
    
@app.route('/calculate_bias', methods=['POST'])
def calculate_bias():
    data = request.json
    return jsonify(calculate_bias_percentages(data))

if __name__ == '__main__':
    app.run(debug=True)