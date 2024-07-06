import nltk
from collections import Counter
from nltk.tokenize import word_tokenize

#pip install nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('stopwords')

def download_nltk_resources():
    # Download necessary NLTK resources if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def most_frequent_noun(strings):

    download_nltk_resources()
    # Tokenize the text into words
    words = word_tokenize(strings)
    
    # Part-of-speech tagging
    pos_tags = nltk.pos_tag(words)
    
    # Filter out the nouns
    nouns = [word for word, pos in pos_tags if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
    
    # Count the frequency of each noun
    noun_counts = Counter(nouns)
    
    # Get the most common noun
    if noun_counts:
        most_common_noun, _ = noun_counts.most_common(1)[0]
        return most_common_noun
    else:
        return None

