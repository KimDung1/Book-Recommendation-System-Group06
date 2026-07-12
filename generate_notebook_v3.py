import nbformat as nbf

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("""# Book Recommendation System - Data Preparation & EDA\n**Group 06**\n- **Member:** Lê Kim Dũng\n- **ID:** 28211452455\n- **Role:** Data & Analysis Lead\n\nThis notebook covers:\n1. Data collection and loading (Goodbooks-10k)\n2. Data cleaning and preprocessing\n3. Exploratory Data Analysis (EDA)\n4. Feature engineering / data preparation\n5. References"""),
    
    nbf.v4.new_markdown_cell("""## 0. Install and Import Libraries"""),
    nbf.v4.new_code_cell("""!pip install pandas numpy matplotlib seaborn scikit-learn kaggle"""),
    nbf.v4.new_code_cell("""import os\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.feature_extraction.text import TfidfVectorizer\n\n# Setup visualization styles\nsns.set_theme(style="whitegrid")\nplt.rcParams['figure.figsize'] = (10, 6)"""),

    nbf.v4.new_markdown_cell("""## 1. Data Collection & Loading\nDownload the Goodbooks-10k dataset from Kaggle.\nMake sure you have `kaggle.json` uploaded to your current directory or configured at `~/.kaggle/kaggle.json`."""),
    nbf.v4.new_code_cell("""# Setup kaggle.json for API access\nimport shutil\nfrom pathlib import Path\nimport os\n\nkaggle_dir = Path.home() / '.kaggle'\nkaggle_dir.mkdir(exist_ok=True)\n\nif os.path.exists('kaggle.json'):\n    shutil.copy('kaggle.json', kaggle_dir / 'kaggle.json')\n    try:\n        os.chmod(kaggle_dir / 'kaggle.json', 0o600)\n    except:\n        pass\n    print("kaggle.json configured successfully.")\nelse:\n    print("Make sure kaggle.json is configured at ~/.kaggle/kaggle.json")"""),
    
    nbf.v4.new_code_cell("""# Download dataset using Kaggle CLI\n# Note: Sometimes !kaggle fails on Windows. If it does, you can run:\n# import kaggle; kaggle.api.authenticate(); kaggle.api.dataset_download_files('zygmunt/goodbooks-10k', path='data', unzip=True)\n!kaggle datasets download -d zygmunt/goodbooks-10k\n\n# Unzip the downloaded file\nimport zipfile\nif os.path.exists('goodbooks-10k.zip'):\n    with zipfile.ZipFile('goodbooks-10k.zip', 'r') as zip_ref:\n        zip_ref.extractall('data')\n    print("Dataset extracted to 'data' folder.")"""),
    
    nbf.v4.new_code_cell("""# Load datasets into Pandas DataFrames\ndata_dir = 'data'\nbooks = pd.read_csv(f'{data_dir}/books.csv')\nratings = pd.read_csv(f'{data_dir}/ratings.csv')\ntags = pd.read_csv(f'{data_dir}/tags.csv')\nbook_tags = pd.read_csv(f'{data_dir}/book_tags.csv')\n\nprint(f"Books: {books.shape}")\nprint(f"Ratings: {ratings.shape}")\nprint(f"Tags: {tags.shape}")\nprint(f"Book Tags: {book_tags.shape}")"""),

    nbf.v4.new_markdown_cell("""## 2. Data Cleaning & Preprocessing"""),
    nbf.v4.new_code_cell("""# Display missing values in books dataframe\nprint("Missing values in books:")\nprint(books.isnull().sum()[books.isnull().sum() > 0])"""),
    
    nbf.v4.new_code_cell("""# Fill missing values\n# For original_publication_year, we can fill with the median\nbooks['original_publication_year'] = books['original_publication_year'].fillna(books['original_publication_year'].median())\n\n# For isbn and isbn13, missing values can be filled with 'Unknown'\nbooks['isbn'] = books['isbn'].fillna('Unknown')\nbooks['isbn13'] = books['isbn13'].fillna('Unknown')\n\n# For original_title, if missing, we use title\nbooks['original_title'] = books['original_title'].fillna(books['title'])\n\n# Drop rows with language_code missing as it's very few, or just fill with 'eng'\nbooks['language_code'] = books['language_code'].fillna('eng')\n\nprint("After cleaning, missing values:")\nprint(books.isnull().sum().sum())"""),

    nbf.v4.new_code_cell("""# Check for duplicates in ratings\nprint("Duplicate ratings:", ratings.duplicated().sum())\n# Drop duplicates if any\nratings = ratings.drop_duplicates()"""),

    nbf.v4.new_markdown_cell("""## 3. Exploratory Data Analysis (EDA)"""),
    nbf.v4.new_code_cell("""# 3.1 Distribution of ratings\nplt.figure(figsize=(8, 5))\nsns.countplot(data=ratings, x='rating', palette='viridis')\nplt.title('Distribution of User Ratings')\nplt.xlabel('Rating')\nplt.ylabel('Count')\nplt.show()"""),
    
    nbf.v4.new_code_cell("""# 3.2 Top 10 most rated books\ntop_books = books.sort_values('ratings_count', ascending=False).head(10)\nplt.figure(figsize=(10, 6))\nsns.barplot(data=top_books, x='ratings_count', y='title', palette='magma')\nplt.title('Top 10 Books by Number of Ratings')\nplt.xlabel('Number of Ratings')\nplt.ylabel('Book Title')\nplt.show()"""),
    
    nbf.v4.new_code_cell("""# 3.3 Top 10 highly rated books (with at least 10,000 ratings)\nhighly_rated = books[books['ratings_count'] > 10000].sort_values('average_rating', ascending=False).head(10)\nplt.figure(figsize=(10, 6))\nsns.barplot(data=highly_rated, x='average_rating', y='title', palette='crest')\nplt.title('Top 10 Highest Rated Books (Min 10,000 ratings)')\nplt.xlabel('Average Rating')\nplt.ylabel('Book Title')\nplt.xlim(4.0, 5.0)\nplt.show()"""),
    
    nbf.v4.new_code_cell("""# 3.4 Distribution of Publication Years\nplt.figure(figsize=(10, 6))\nsns.histplot(books[books['original_publication_year'] > 1900]['original_publication_year'], bins=50, kde=True, color='teal')\nplt.title('Distribution of Books by Publication Year (Post 1900)')\nplt.xlabel('Publication Year')\nplt.ylabel('Count')\nplt.show()"""),

    nbf.v4.new_markdown_cell("""## 4. Feature Engineering / Data Preparation\nWe will prepare data for two types of recommendation systems:\n1. **Collaborative Filtering:** Creating a user-item interaction matrix.\n2. **Content-Based Filtering:** Combining text features like authors and tags into a single text representation for each book."""),
    
    nbf.v4.new_code_cell("""# 4.1 Collaborative Filtering Preparation\nfrom scipy.sparse import csr_matrix\n\nn_users = ratings['user_id'].nunique()\nn_books = ratings['book_id'].nunique()\n\nprint(f"Number of unique users: {n_users}")\nprint(f"Number of unique books: {n_books}")\n\n# Create sparse user-item interaction matrix\nuser_item_matrix = csr_matrix((ratings['rating'], (ratings['user_id'], ratings['book_id'])))\nprint("User-Item Matrix shape:", user_item_matrix.shape)"""),
    
    nbf.v4.new_code_cell("""# 4.2 Content-Based Filtering Preparation\n# Merge tags with book_tags\nmerged_tags = pd.merge(book_tags, tags, on='tag_id')\n\n# Group tags by book\nbook_tags_grouped = merged_tags.groupby('goodreads_book_id')['tag_name'].apply(lambda x: ' '.join(x)).reset_index()\n\n# Merge with books dataset\n# Note: In books.csv, the goodreads ID is stored in 'book_id', while 'id' is the 1-10000 index\nbooks_with_tags = pd.merge(books, book_tags_grouped, left_on='book_id', right_on='goodreads_book_id', how='left')\nbooks_with_tags['tag_name'] = books_with_tags['tag_name'].fillna('')\nbooks_with_tags['authors'] = books_with_tags['authors'].fillna('')\n\n# Combine authors and tags for text processing\nbooks_with_tags['content_features'] = books_with_tags['authors'] + " " + books_with_tags['tag_name']\n\nprint("Sample content features:")\nprint(books_with_tags[['title', 'content_features']].head(3))"""),
    
    nbf.v4.new_code_cell("""# Save prepared data for the Modeling step (Thành viên 2)\nimport os\nos.makedirs('data', exist_ok=True)\nbooks_with_tags.to_csv('data/books_cleaned_with_features.csv', index=False)\nprint("Data preparation complete and saved to data/books_cleaned_with_features.csv")"""),

    nbf.v4.new_markdown_cell("""## 5. References\n*(APA Format)*\n\n- McKinney, W. (2010). Data structures for statistical computing in python. In *Proceedings of the 9th Python in Science Conference* (Vol. 445, pp. 51-56).\n- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of machine learning research*, 12(Oct), 2825-2830.\n- Waskom, M. L. (2021). Seaborn: statistical data visualization. *Journal of Open Source Software*, 6(60), 3021. https://doi.org/10.21105/joss.03021\n- Zygmunt, Z. (2017). Goodbooks-10k [Data set]. Kaggle. https://www.kaggle.com/datasets/zygmunt/goodbooks-10k""")
]

with open(r'e:\BigData\Group 06\Group06_BookRecommendation_LeKimDung.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generated cleanly using nbformat v4.")
