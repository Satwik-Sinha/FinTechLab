from flask import Flask, jsonify
from sec_edgar_downloader import Downloader
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import openai
import os

# Set the API key for OpenAI
api_key = "your_openai_api_key"
client = openai.OpenAI(api_key=api_key)

# Function to analyze text using OpenAI's davinci-002 model
def analyze_text_with_openai(text):
    # Create a completion request with the given text and model
    response = client.completions.create(
        model="davinci-002",
        prompt=text,
        max_tokens=1024
    )
    # Return the analyzed text
    return response.choices[0].text

# Company information for SEC EDGAR downloader
company_name = "FinTechSubmission"
email_address = "ssinha20@hawk.iit.edu"
download_directory = "/Users/satwiksinha/Downloads/FinTechSummerLab"

# Initialize the SEC EDGAR downloader
downloader = Downloader(company_name, email_address, download_directory)

# Define the companies and the period for analysis
companies = ['TSLA', 'MSFT', 'V']
start_year = 1995
end_year = 2023

# List to store insights for each company
insights = []
for company in companies:
    # Construct the company directory path
    company_dir = os.path.join(download_directory, "sec-edgar-filings", company, "10-K")
    if os.path.exists(company_dir):
        # Get all subdirectories in the company directory
        all_subdirs = [d for d in os.listdir(company_dir) if os.path.isdir(os.path.join(company_dir, d))]
        for subdir in all_subdirs:
            # Extract the accession year from the subdirectory name
            acc_year = subdir.split('-')[1]
            # Convert the accession year to a filing year
            filing_year = "20" + acc_year if int(acc_year) < 50 else "19" + acc_year

            # Construct the filing path
            filing_path = os.path.join(company_dir, subdir, "full-submission.txt")
            if os.path.exists(filing_path):
                # Read the filing text
                with open(filing_path, 'r') as file:
                    filing_text = file.read()

                # Limit the filing text to a chunk size
                chunk_size = 890
                if len(filing_text) > chunk_size:
                    filing_text = filing_text[:chunk_size]

                # Analyze the filing text using OpenAI
                insight = analyze_text_with_openai(filing_text)
                # Store the insight
                insights.append({'company': company, 'year': filing_year, 'insight': insight})

# Create a DataFrame from the insights
df_insights = pd.DataFrame(insights, columns=['company', 'year', 'insight'])

# Visualization using Plotly
if not df_insights.empty:
    # Create a bar chart of insights per company
    fig = px.bar(df_insights, x='company', y='year', color='company', title='Number of Insights per Company')
    fig.update_layout(xaxis_title='Company', yaxis_title='Number of Insights')
    fig.show()

    # Additional visualization: Sentiment analysis over time
    # Calculate sentiment scores for each insight
    sentiment_scores = df_insights['insight'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df_insights['sentiment'] = sentiment_scores
    # Create a line chart of sentiment scores over time
    fig2 = px.line(df_insights, x='year', y='sentiment', color='company', title='Sentiment Analysis Over Time')
    fig2.update_layout(xaxis_title='Year', yaxis_title='Sentiment Score')
    fig2.show()
else:
    print("No insights to visualize.")

@app.route('/get_insights', methods=['GET'])
def get_insights():
    # Convert DataFrame to JSON
    insights_json = df_insights.to_json(orient='records')
    return jsonify(insights_json)

if __name__ == '__main__':
    app.run(debug=True)