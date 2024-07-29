import gradio as gr
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load titles from CSV
def load_titles_from_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    return df[0].tolist()

# Analyze keywords in the titles
def analyze_keywords(titles):
    words = []
    stop_words = set(stopwords.words('english'))
    for title in titles:
        tokens = word_tokenize(title.lower())
        words.extend([word for word in tokens if word.isalpha() and word not in stop_words])
    return Counter(words)

# Plot keywords function
def plot_keywords(keywords_data):
    colors = {
        'ICRA 2023': 'red',
        'ICRA 2024': 'orange',
        'IROS 2022': 'blue',
        'IROS 2023': 'lightblue',
        'IROS 2024': 'purple',
        'ICRA 2022': 'yellow'
    }

    data_for_plotting = defaultdict(lambda: defaultdict(int))
    for label, keywords in keywords_data.items():
        for word, count in keywords.items():
            data_for_plotting[word][label] += count

    sorted_words = sorted(data_for_plotting, key=lambda x: sum(data_for_plotting[x].values()), reverse=True)[:50]
    fig, ax = plt.subplots(figsize=(12, 10))
    y_positions = range(len(sorted_words))

    legend_handles = {}
    for idx, word in enumerate(sorted_words):
        left = 0
        for conference in sorted(data_for_plotting[word], key=lambda x: data_for_plotting[word][x], reverse=True):
            count = data_for_plotting[word][conference]
            handle = ax.barh(y_positions[idx], count, left=left, color=colors[conference], label=conference if conference not in legend_handles else "")
            left += count
            if conference not in legend_handles:
                legend_handles[conference] = handle

    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_words)
    ax.set_xlabel('Frequency')
    ax.set_title('Segmented Top 50 Common Keywords from Multiple Conferences')
    ax.legend(handles=[legend_handles[conf] for conf in sorted(legend_handles)])
    plt.tight_layout()
    plt.gca().invert_yaxis()

    return plt

# Set up Gradio interface
def process_selection(selected_combinations):
    # Define paths to your CSV files
    file_paths = {
        "ICRA 2022": "conferences_lists/ICRA-2022.csv",
        "ICRA 2023": "conferences_lists/ICRA-2023.csv",
        "ICRA 2024": "conferences_lists/ICRA-2024.csv",
        "IROS 2022": "conferences_lists/IROS-2022.csv",
        "IROS 2023": "conferences_lists/IROS-2023.csv",
        "IROS 2024": "conferences_lists/IROS-2024.csv"
    }

    keywords_data = {}
    for combo in selected_combinations:
        titles = load_titles_from_csv(file_paths[combo])
        keywords = analyze_keywords(titles)
        keywords_data[combo] = keywords

    fig = plot_keywords(keywords_data)
    return fig

with gr.Blocks() as app:
    with gr.Row():
        selection = gr.CheckboxGroup(
            label="Select Conference-Year",
            choices=["ICRA 2022", "ICRA 2023", "ICRA 2024", "IROS 2022", "IROS 2023", "IROS 2024"]
        )
        submit_button = gr.Button("Generate Plot")

    plot_output = gr.Plot()

    submit_button.click(
        fn=process_selection,
        inputs=selection,
        outputs=plot_output
    )

if __name__ == "__main__":
    app.launch()
