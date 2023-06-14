import requests
import os
import csv
from dotenv import load_dotenv

load_dotenv()  # Load the environment variables from the .env file

api_key = os.getenv("SERP_API_KEY")  # Set the API key as an environment variable
query = 'neurotechnology'
num_results = 20  # Number of results per page
total_searches = 1500  # Total number of searches

params = {
    'q': query,
    'api_key': api_key,
    'engine': 'google_scholar',
    'scholar_year_start': 2013,  # Set the start year (e.g., 2013 for the past ten years)
    'num': num_results,
}

search_count = 0  # Counter for the number of searches made
page = 0  # Page number

# Open the CSV file in append mode or create a new file if it doesn't exist
with open('neurotechnology_papers.csv', 'a', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)

    while search_count < total_searches:
        params['start'] = page * num_results

        response = requests.get('https://serpapi.com/search', params=params)

        if response.status_code == 200:
            data = response.json()

            # Process the response data
            results = data.get('organic_results', [])

            sorted_results = sorted(
                results, key=lambda x: x.get('inline_links', {}).get('cited_by', {}).get('total', 0) or 0, reverse=True
            )

            for result in sorted_results:
                title = result.get('title', '')
                abstract = result.get('snippet', '')
                citations = result.get('inline_links', {}).get('cited_by', {}).get('total', 0) or 0
                # Retrieve other desired data fields

                # Write the data row to the CSV file
                writer.writerow([title, abstract, citations])

                search_count += 1
                if search_count >= total_searches:
                    break

        else:
            # Handle error cases
            print(f"Error occurred during API request: {response.status_code} - {response.text}")
            break

        page += 1

    print(f"Data collection completed. Total search count: {search_count}")
