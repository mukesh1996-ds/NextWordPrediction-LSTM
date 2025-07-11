import requests

# List of Gutenberg book IDs and names
books = {
    "1342": "pride_and_prejudice.txt",         
    "1661": "sherlock_holmes.txt",             
    "11":   "alice_in_wonderland.txt",         
    "84":   "frankenstein.txt",                
    "2701": "moby_dick.txt"                    
}

def download_books():
    for book_id, filename in books.items():
        url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
        print(f"Downloading: {filename}")
        r = requests.get(url)
        if r.status_code == 200:
            with open(f"data/{filename}", 'w', encoding='utf-8') as f:
                f.write(r.text)
        else:
            print(f"Failed to download {filename} (status code: {r.status_code})")

def merge_books(output_path="data/gutenberg_corpus.txt"):
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for filename in books.values():
            with open(f"data/{filename}", 'r', encoding='utf-8') as infile:
                outfile.write(infile.read() + "\n\n")

if __name__ == "__main__":
    download_books()
    merge_books()
    print("All books downloaded and merged into: data/gutenberg_corpus.txt")
