import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pickle
import joblib
import re
import requests
import time
from transformers import BertTokenizer, BertModel

# My Google Books API key
API_KEY = "AIzaSyBPqNfkZpqu6kdLMsoyLAEbOXmzGW3Lqg4"



class BookVAE(nn.Module):
    def __init__(self, metadata_dim, latent_dim=32):
        super(BookVAE, self).__init__()
        self.latent_dim = latent_dim
        self.bert_dim = 768
        self.metadata_dim = metadata_dim
        self.input_dim = self.bert_dim + metadata_dim

        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert_model.parameters():
            param.requires_grad = False

        # Encoder part
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Latent space
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder part (need this even though we don't use it)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.input_dim),
        )

    def get_bert_embeddings(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, 0, :]

    def encode(self, bert_embeddings, metadata):
        x = torch.cat([bert_embeddings, metadata], dim=1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# Load my trained models (only load once)
@st.cache_resource
def load_my_models():
    try:
        # Load the files I created during training
        with open('vae_preprocessing.pkl', 'rb') as f:
            vae_stuff = pickle.load(f)

        baseline_stuff = joblib.load('baseline_models.pkl')

        # Load the neural network
        model_data = torch.load('book_vae_model.pth', map_location='cpu')
        my_model = BookVAE(
            metadata_dim=model_data['metadata_dim'],
            latent_dim=model_data['latent_dim']
        )
        my_model.load_state_dict(model_data['model_state_dict'])
        my_model.eval()

        return vae_stuff, baseline_stuff, my_model

    except Exception as e:
        st.error(f"Can't load models: {e}")
        return None, None, None


# Clean up text (same as training)
def clean_text(text):
    if not text:
        return ""
    # Remove punctuation and make lowercase
    text = re.sub(r'[^a-zA-Z\s]', ' ', str(text).lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Figure out what genres the user wants
def guess_genres(description, baseline_stuff):
    try:
        # Use the trained model to guess genres
        vectorizer = baseline_stuff['tfidf_vectorizer']
        model = baseline_stuff['nb_model']
        encoder = baseline_stuff['genre_encoder']

        # Process the text
        clean_desc = clean_text(description)
        text_vector = vectorizer.transform([clean_desc])
        probabilities = model.predict_proba(text_vector)[0]

        # Get the genres with high probability
        good_genres = []
        for i, prob in enumerate(probabilities):
            if prob > 0.1:  # Only keep if >10% chance
                genre_name = encoder.inverse_transform([i])[0]
                good_genres.append((genre_name, prob))

        # Sort by probability
        good_genres.sort(key=lambda x: x[1], reverse=True)
        return good_genres[:5]  # Return top 5

    except Exception as e:
        st.error(f"Error guessing genres: {e}")
        return [('fiction', 0.5)]


# Make search terms for Google Books
def make_search_terms(genres):
    searches = []

    # Simple mapping of genres to search words
    genre_words = {
        'mystery': ['mystery', 'detective'],
        'thriller': ['thriller', 'suspense'],
        'romance': ['romance', 'love'],
        'fantasy': ['fantasy', 'magic'],
        'science fiction': ['science fiction', 'sci-fi'],
        'horror': ['horror'],
        'historical': ['historical fiction'],
        'literary': ['literary fiction'],
        'young adult': ['young adult'],
        'biography': ['biography'],
        'self help': ['self help'],
        'business': ['business'],
        'science': ['science'],
        'history': ['history']
    }

    # Build search terms
    for genre, confidence in genres:
        if confidence > 0.15:  # Only use if confident enough
            words = genre_words.get(genre, [genre])
            for word in words:
                searches.append(f"{word} books")
                searches.append(f"best {word}")

    # Remove duplicates and limit
    unique_searches = list(set(searches))
    return unique_searches[:5]


# Get books from Google
def get_books_from_google(search_terms):
    all_books = []

    for search in search_terms:
        try:
            # Call Google Books API
            url = "https://www.googleapis.com/books/v1/volumes"
            params = {
                'q': search,
                'key': API_KEY,
                'maxResults': 6,
                'orderBy': 'relevance',
                'printType': 'books',
                'langRestrict': 'en'
            }

            response = requests.get(url, params=params)
            data = response.json()

            # Process the books we got back
            if 'items' in data:
                for item in data['items']:
                    book = process_book(item)
                    if book:
                        all_books.append(book)

            time.sleep(0.5)  # Be nice to Google

        except:
            continue  # Skip if error

    return all_books


# Process one book from Google API
def process_book(item):
    try:
        info = item.get('volumeInfo', {})

        # Get description
        description = info.get('description', '')
        if not description or len(description) < 50:
            return None  # Skip if no good description

        # Skip very old books
        pub_date = info.get('publishedDate', '')
        if pub_date and len(pub_date) >= 4:
            try:
                year = int(pub_date[:4])
                if year < 1980:
                    return None  # Skip old books
            except:
                pass

        # Create book info
        book = {
            'title': info.get('title', 'Unknown'),
            'author': ', '.join(info.get('authors', ['Unknown'])),
            'summary': description,
            'genre': ', '.join(info.get('categories', ['Fiction'])),
            'published': pub_date,
            'rating': info.get('averageRating', 0),
            'num_ratings': info.get('ratingsCount', 0),
            'link': info.get('previewLink', '')
        }

        return book

    except:
        return None


# Remove duplicates and rank books
def rank_books(books, genres):
    # Remove duplicate books
    seen = set()
    unique_books = []

    for book in books:
        book_id = f"{book['title']}_{book['author']}"
        if book_id not in seen:
            seen.add(book_id)
            unique_books.append(book)

    # Give each book a score
    for book in unique_books:
        score = 0

        # Points for matching genre
        book_genre = book['genre'].lower()
        for genre, confidence in genres:
            if genre.lower() in book_genre:
                score += confidence * 2

        # Points for good rating
        if book['rating'] > 0:
            score += book['rating'] / 5.0

        # Points for popularity
        if book['num_ratings'] > 0:
            score += min(np.log(book['num_ratings']) / 10, 0.5)

        book['score'] = score

    # Sort by score (best first)
    unique_books.sort(key=lambda x: x.get('score', 0), reverse=True)
    return unique_books


# Main app
def main():
    st.title("Book Recommender")
    st.write("Tell me what you want to read and I'll find some good books!")

    # Load my trained models
    vae_stuff, baseline_stuff, my_model = load_my_models()
    if baseline_stuff is None:
        st.error("Can't load the AI models. Make sure all files are here.")
        return

    st.success("Models loaded! Ready to find books.")

    # Get user input
    st.subheader("What kind of book do you want?")
    user_input = st.text_area(
        "Describe it:",
        placeholder="Example: I want a mystery book with a detective",
        height=80
    )

    # How many books to show
    num_books = st.slider("How many books?", 1, 8, 5)

    # Find books button
    if st.button("Find Books!"):
        if not user_input:
            st.warning("Please tell me what you want!")
        else:
            with st.spinner("Looking for books..."):
                # Step 1: Guess what genres they want
                genres = guess_genres(user_input, baseline_stuff)

                # Step 2: Make search terms
                searches = make_search_terms(genres)

                # Step 3: Get books from Google
                books = get_books_from_google(searches)

                # Step 4: Rank the books
                ranked_books = rank_books(books, genres)

                # Show results
                if ranked_books:
                    # Show what genres we think they want
                    st.subheader("I think you want:")
                    cols = st.columns(3)
                    for i, (genre, conf) in enumerate(genres[:3]):
                        with cols[i]:
                            st.metric(genre.title(), f"{conf * 100:.0f}%")

                    # Show the books
                    st.subheader(f"Here are {min(len(ranked_books), num_books)} books:")

                    for i, book in enumerate(ranked_books[:num_books], 1):
                        st.write(f"**{i}. {book['title']}**")
                        st.write(f"By: {book['author']}")

                        if book['rating'] > 0:
                            stars = "⭐" * int(book['rating'])
                            st.write(f"{stars} {book['rating']:.1f}/5")

                        st.write(f"Genre: {book['genre']}")

                        if book['published']:
                            st.write(f"Published: {book['published'][:4]}")

                        # Show summary in expandable box
                        with st.expander("Read Summary"):
                            st.write(book['summary'][:300] + "...")
                            if book['link']:
                                st.link_button("View on Google Books", book['link'])

                        st.write("---")

                else:
                    st.error("Sorry, couldn't find any books. Try describing differently.")


if __name__ == "__main__":
    main()