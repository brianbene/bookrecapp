import requests
import pandas as pd
import time
import json
import os

GOOGLE_BOOKS_API_KEY = "AIzaSyBPqNfkZpqu6kdLMsoyLAEbOXmzGW3Lqg4"
def get_books_from_google(search_term, max_books=50):
    print(f"Searching for books about: {search_term}")

    books = []
    start_index = 0

    while len(books) < max_books:
        # Build the API request URL
        url = "https://www.googleapis.com/books/v1/volumes"
        params = {
            'q': search_term,
            'key': GOOGLE_BOOKS_API_KEY,
            'maxResults': 40,
            'startIndex': start_index,
            'orderBy': 'relevance',
            'printType': 'books',
            'langRestrict': 'en'
        }

        try:
            # Make the API request
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Check if we got any books
            if 'items' not in data:
                print(f"No more books found for '{search_term}'")
                break

            # Process each book
            for item in data['items']:
                book_info = extract_book_info(item)
                if book_info:
                    books.append(book_info)

                # Stop if we have enough books
                if len(books) >= max_books:
                    break

            start_index += 40
            time.sleep(0.5)  # Faster API calls

        except Exception as e:
            print(f"Error getting books: {e}")
            break

    print(f"Found {len(books)} books for '{search_term}'")
    return books


def extract_book_info(api_item):
    try:
        volume_info = api_item.get('volumeInfo', {})

        # Get the book's description/summary
        description = volume_info.get('description', '')

        # Skip books without a good description (reduced minimum length)
        if not description or len(description) < 50:
            return None

        # Skip books that are too old (focus on more recent books)
        pub_date = volume_info.get('publishedDate', '')
        if pub_date and len(pub_date) >= 4:
            try:
                year = int(pub_date[:4])
                if year < 1950:  # Skip very old books
                    return None
            except:
                pass

        # Extract all the book information
        book_data = {
            'title': volume_info.get('title', 'Unknown'),
            'authors': ', '.join(volume_info.get('authors', ['Unknown'])),
            'summary': description,
            'genres': ', '.join(volume_info.get('categories', ['Unknown'])),
            'published_date': pub_date,
            'page_count': volume_info.get('pageCount', 0),
            'average_rating': volume_info.get('averageRating', 0),
            'ratings_count': volume_info.get('ratingsCount', 0),
            'language': volume_info.get('language', 'en'),
            'publisher': volume_info.get('publisher', ''),
            'isbn': extract_isbn(volume_info.get('industryIdentifiers', [])),
            'preview_link': volume_info.get('previewLink', ''),
            'info_link': volume_info.get('infoLink', '')
        }

        return book_data

    except Exception as e:
        print(f"Error processing book: {e}")
        return None


def extract_isbn(identifiers):
    for identifier in identifiers:
        if identifier.get('type') in ['ISBN_13', 'ISBN_10']:
            return identifier.get('identifier', '')
    return ''


def main():
    """
    Main function that collects books and saves them to a CSV file.
    """
    print("Starting book data collection...")

    # Check if API key is set
    if GOOGLE_BOOKS_API_KEY == "YOUR_API_KEY_HERE":
        print("❌ Error: Please set your Google Books API key!")
        print("1. Go to: https://console.cloud.google.com/")
        print("2. Create a project and enable the Books API")
        print("3. Create an API key")
        print("4. Replace 'YOUR_API_KEY_HERE' in this script with your key")
        return

    print("✅ Using Google Books API key")
    print("Testing API connection...")

    # List of book genres/topics to search for - EXPANDED LIST
    search_terms = [
        # Fiction genres
        "mystery fiction",
        "science fiction",
        "fantasy adventure",
        "romance novel",
        "thriller suspense",
        "historical fiction",
        "literary fiction",
        "horror stories",
        "dystopian fiction",
        "contemporary fiction",
        "classic literature",
        "young adult fiction",
        "crime fiction",
        "detective stories",
        "psychological thriller",
        "urban fantasy",
        "epic fantasy",
        "space opera",
        "time travel fiction",
        "vampire fiction",
        "zombie fiction",
        "western fiction",
        "war fiction",
        "family saga",
        "coming of age",
        "love story",
        "adventure fiction",
        "action thriller",
        "espionage fiction",
        "legal thriller",
        "medical thriller",

        # Non-fiction genres
        "biography memoir",
        "self help books",
        "business books",
        "psychology books",
        "history books",
        "science books",
        "philosophy books",
        "health fitness",
        "cooking books",
        "travel books",
        "true crime",
        "politics books",
        "religion spirituality",
        "personal development",
        "economics books",
        "technology books",
        "nature books",
        "art books",
        "music books",
        "sports books",

        # Specific popular topics
        "artificial intelligence",
        "climate change",
        "entrepreneurship",
        "meditation mindfulness",
        "productivity",
        "leadership",
        "parenting books",
        "relationships",
        "financial planning",
        "world war",
        "ancient history",
        "space exploration",
        "neuroscience",
        "evolution biology"
    ]

    all_books = []

    # Collect books for each genre - MORE books per genre
    for term in search_terms:
        books = get_books_from_google(term, max_books=50)  # Increased from 30
        all_books.extend(books)
        time.sleep(1)  # Reduced wait time to speed up collection

    print(f"\nTotal books collected: {len(all_books)}")

    # Remove duplicate books (same title and author)
    df = pd.DataFrame(all_books)

    if len(df) == 0:
        print("❌ No books collected!")
        return

    print(f"Before removing duplicates: {len(df)} books")

    # Remove duplicates more carefully
    df = df.drop_duplicates(subset=['title', 'authors'], keep='first')

    # Remove books with very short summaries
    df = df[df['summary'].str.len() >= 50]  # Reduced from 100

    # Remove books without proper titles
    df = df[df['title'] != 'Unknown']
    df = df[df['authors'] != 'Unknown']

    # Keep books published after 1950
    def extract_year(date_str):
        if pd.isna(date_str) or not date_str:
            return 2000  # Default year
        try:
            return int(str(date_str)[:4])
        except:
            return 2000

    df['pub_year'] = df['published_date'].apply(extract_year)
    df = df[df['pub_year'] >= 1950]

    print(f"After cleaning: {len(df)} books")

    # Save to CSV file
    filename = os.path.join(BASE_PATH, "book_dataset.csv")
    df.to_csv(filename, index=False)

    print(f"✅ Saved {len(df)} books to {filename}")
    print("Data collection complete!")


if __name__ == "__main__":
    main()