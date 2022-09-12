from keras.preprocessing.text import text_to_word_sequence
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import nltk
import uvicorn
from fastapi import FastAPI
import pickle

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('wordnet\r')

new_sort_model = pickle.load(open("news_sorting.pkl", "rb"))

app = FastAPI(
    title="News Classification",
    description="A simple API that use NLP model to categorize news",
    version="0.1")


def preprocessing(train_text):

    train_text = str(train_text)
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    tokenized_train_set = text_to_word_sequence(train_text,
                                                filters=filters,
                                                lower=True,
                                                split=" ")

    stop_words = set(stopwords.words('english'))
    stopwordremove = [i for i in tokenized_train_set if not i in stop_words]

    stopwordremove_text = ' '.join(stopwordremove)

    # remove numbers
    numberremove_text = ''.join(c for c in stopwordremove_text if not c.isdigit())

    # --Stemming--
    stemmer = PorterStemmer()
    stem_input = nltk.word_tokenize(numberremove_text)
    stem_text = ' '.join([stemmer.stem(word) for word in stem_input])

    def get_wordnet_pos(word):

        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    lem_input = nltk.word_tokenize(stem_text)
    lemmatizer = WordNetLemmatizer()
    lem_text = ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in lem_input])

    return lem_text


@app.get("/predict-news")
def predict_news(news: str):

    preprocessed_text = preprocessing(news)

    prediction = (new_sort_model.predict([preprocessed_text]))
    prediction = "".join(prediction)

    types = {'business': 'business',
             'tech': 'tech',
             'politics': 'politics',
             'sport': 'sport',
             'entertainment': 'entertainment'}

    return {"prediction": types[prediction]}


#if __name__=="__main__":
    #uvicorn.run(app, port=8000, host='127.0.0.1')
