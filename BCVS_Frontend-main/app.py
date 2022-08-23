from flask import Flask, request, render_template
# from scrapping_keywords import extract
from news_generation import news_generation
from recommendations import recommendations
from BCVS import bcvs_generation
import torch
from Competitors import competitors
from Femalenews import female_generation
from MaleNews import male_generation
from summarization import Tweets_Summarizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

import jinja2
app = Flask(__name__)

env = jinja2.Environment()
env.globals.update(zip=zip)
app.jinja_env.filters['zip'] = zip


@app.route('/', methods=["POST", "GET"])
def index():
    return render_template('home.html')


@app.route('/search', methods=["POST", "GET"])
def search():
    if request.method == 'POST':
        # print(request.form)
        output = news_generation(request.form.get('query'))
        trends = output['trends'][:13]
        url = output['URL']
        snippet = output['Snippet']
        keywords = output['Keywords']
        title = output['title']
        zipped_output = zip(url, title, snippet, keywords)
        for u, t, sni, k in zipped_output:
            print('------------------- ', u, t, sni, k)
            break
        # print('--------------------- ', zipped_output)
        return render_template('home.html', trends=trends, zipped_output=zipped_output)


@app.route('/recommendation', methods=["POST", "GET"])
def recommendation():
    if request.method == 'POST':
        print(request.form)
        output=recommendations()
        url = output['URL']
        snippet = output['Snippet']
        keywords = output['Keywords']
        title = output['title']
        portfolio_options = output['portfolio_optoins']
        portfolio_values = output['portfolio_values']
        portfolio_values = portfolio_values*10
        zipped_portfolio = zip(portfolio_values, portfolio_options)
        zipped_output = zip(url, title, snippet, keywords, portfolio_options)
        # print(len(zipped_output))
        # for u, t, sni, k in zipped_output:
        #     print('------------------- ', u,t,sni,k)
        return render_template('recommendation.html', trends=[], zipped_output=zipped_output, portfolio=zipped_portfolio)

    return render_template('recommendation.html')

@app.route('/BCVS', methods=["POST", "GET"])
def BCVS():
    output=bcvs_generation()
    url = output['URL']
    snippet = output['Snippet']
    keywords = output['Keywords']
    title = output['title']
    zipped_output = zip(url, title, snippet, keywords)
    return render_template('BCVS.html', zipped_output=zipped_output)


@app.route('/competitors', methods=["POST", "GET"])
def competitor():
    output=competitors()
    url = output['URL']
    snippet = output['Snippet']
    keywords = output['Keywords']
    title = output['title']
    zipped_output = zip(url, title, snippet, keywords)
    return render_template('competitors.html', zipped_output=zipped_output)


@app.route('/male', methods=["POST", "GET"])
def male():
    output=male_generation()
    url = output['URL']
    snippet = output['Snippet']
    keywords = output['Keywords']
    title = output['title']
    zipped_output = zip(url, title, snippet, keywords)
    return render_template('male.html', zipped_output=zipped_output)

@app.route('/female', methods=["POST", "GET"])
def female():
    output=female_generation()
    url = output['URL']
    snippet = output['Snippet']
    keywords = output['Keywords']
    title = output['title']
    zipped_output = zip(url, title, snippet, keywords)
    return render_template('female.html', zipped_output=zipped_output)


# --------------------------------------------------------------------------------------------------- #
# define model checkpoints
model_checkpoints_english = "facebook/bart-large-cnn"
model_checkpoints_french = "csebuetnlp/mT5_multilingual_XLSum"
# check for cuda availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize model and tokenizer
french_tokenizer = AutoTokenizer.from_pretrained(model_checkpoints_french)
french_model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoints_french).to(device)

# # english_tokenizer = BartTokenizer.from_pretrained(model_checkpoints_english)
# # english_model = BartForConditionalGeneration.from_pretrained(model_checkpoints_english).to(device)

# # define Tweets_Summarizer object
tweets_summarizer = Tweets_Summarizer(
    gsheets_url = "https://docs.google.com/spreadsheets/d/1Q8Cj8zx5-iOqrnaRB2H1dZCHYcXjkjdGXwGjBhpBPMU/edit",
    french_tokenizer = french_tokenizer,
    french_model = french_model
)

# generate batch summaries
social_media_analysis_output = tweets_summarizer.generate_batch_summaries()
# --------------------------------------------------------------------------------------------------- #

@app.route('/social_media_analysis', methods=["POST", "GET"])
def social_media_analysis():
    keywords = social_media_analysis_output["keywords"]
    summaries = social_media_analysis_output["french_summary"]
    zipped_output = zip(keywords, summaries)
    return render_template('social_media_analysis.html', zipped_output=zipped_output)

@app.route('/graphs', methods=["POST", "GET"])
def graphs():
    path_to_graphs_folder = os.path.join(os.getcwd(), "Graphs")
    graphs_file_paths = [os.path.join(path_to_graphs_folder, file) for file in os.listdir('Graphs')]
    graphs_file_keywords = [file.replace('.png', '') for file in os.listdir(path_to_graphs_folder)]
    return render_template('graphs.html', zipped_output=zip(graphs_file_keywords, graphs_file_paths))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True)
