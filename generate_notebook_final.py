import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

# ─────────────────────────────────────────────
# CELL 0 – Title
# ─────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
# 📚 Book Recommendation System — Data Preparation & EDA
**Course:** DS 423 — Machine Learning with Large Datasets  
**Group:** 06 | **University:** Duy Tan University, Da Nang

| Member | ID | Role |
|---|---|---|
| Lê Kim Dũng | 28211452455 | Data & Analysis Lead |
| Võ Minh Chính | 28212306297 | Modeling & Evaluation Lead |

---

### Notebook Contents (Lê Kim Dũng's tasks)
1. Environment Setup
2. Data Collection & Loading  
3. Data Cleaning & Preprocessing  
4. Exploratory Data Analysis (EDA)  
5. Feature Engineering / Data Preparation  
6. APA References
"""))

# ─────────────────────────────────────────────
# SECTION 0 – Setup
# ─────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 0. Environment Setup"))

cells.append(nbf.v4.new_code_cell("""\
# Install required libraries (run once)
!pip install pandas numpy matplotlib seaborn scikit-learn kaggle scipy wordcloud --quiet
"""))

cells.append(nbf.v4.new_code_cell("""\
import os, shutil, zipfile
from pathlib import Path

import numpy  as np
import pandas as pd
import matplotlib.pyplot   as plt
import matplotlib.ticker   as mticker
import seaborn             as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

# ── Visualization defaults ──────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

print("All libraries imported successfully.")
"""))

# ─────────────────────────────────────────────
# SECTION 1 – Data Collection
# ─────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 1. Data Collection & Loading
We use the **Goodbooks-10k** dataset from Kaggle  
([zygmunt/goodbooks-10k](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k)).

> **Pre-requisite:** place `kaggle.json` in the same directory as this notebook  
> (or at `~/.kaggle/kaggle.json`).
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── 1.1  Configure Kaggle credentials ───────
kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(exist_ok=True)

local_creds = Path("kaggle.json")
target_creds = kaggle_dir / "kaggle.json"

if local_creds.exists():
    shutil.copy(local_creds, target_creds)
    try:
        os.chmod(target_creds, 0o600)
    except Exception:
        pass          # Windows doesn't enforce chmod — that's fine
    print("✅  kaggle.json configured.")
elif target_creds.exists():
    print("✅  kaggle.json already present at ~/.kaggle/.")
else:
    print("⚠️  kaggle.json not found. Please place it in the current directory.")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── 1.2  Download & extract dataset ─────────
DATA_DIR = Path("data")

if DATA_DIR.exists() and any(DATA_DIR.glob("*.csv")):
    print("✅  Dataset already extracted in 'data/'.")
else:
    print("Downloading dataset from Kaggle …")
    os.system("kaggle datasets download -d zygmunt/goodbooks-10k")

    zip_path = Path("goodbooks-10k.zip")
    if zip_path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        print("✅  Dataset extracted to 'data/'.")
    else:
        print("❌  Download failed. Check your kaggle.json credentials.")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── 1.3  Load CSV files ──────────────────────
books     = pd.read_csv(DATA_DIR / "books.csv")
ratings   = pd.read_csv(DATA_DIR / "ratings.csv")
tags      = pd.read_csv(DATA_DIR / "tags.csv")
book_tags = pd.read_csv(DATA_DIR / "book_tags.csv")

print(f"books     : {books.shape[0]:>7,} rows × {books.shape[1]} cols")
print(f"ratings   : {ratings.shape[0]:>7,} rows × {ratings.shape[1]} cols")
print(f"tags      : {tags.shape[0]:>7,} rows × {tags.shape[1]} cols")
print(f"book_tags : {book_tags.shape[0]:>7,} rows × {book_tags.shape[1]} cols")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── 1.4  Quick peek at each table ───────────
print("=== books (first 3 rows) ===")
display(books.head(3))

print("\\n=== ratings (first 3 rows) ===")
display(ratings.head(3))

print("\\n=== tags (first 3 rows) ===")
display(tags.head(3))

print("\\n=== book_tags (first 3 rows) ===")
display(book_tags.head(3))
"""))

# ─────────────────────────────────────────────
# SECTION 2 – Cleaning
# ─────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 2. Data Cleaning & Preprocessing"))

cells.append(nbf.v4.new_code_cell("""\
# ── 2.1  Missing-value overview ─────────────
print("Missing values in books.csv:")
mv = books.isnull().sum()
print(mv[mv > 0].to_string())
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── 2.2  Impute / fill missing values ───────

# Numeric: publication year → median
med_year = books["original_publication_year"].median()
books["original_publication_year"] = (
    books["original_publication_year"].fillna(med_year).astype(int)
)

# Identifiers: unknown placeholder
books["isbn"]   = books["isbn"].fillna("Unknown")
books["isbn13"] = books["isbn13"].fillna("Unknown")

# Title: fall back to translated title
books["original_title"] = books["original_title"].fillna(books["title"])

# Language: default to English
books["language_code"] = books["language_code"].fillna("eng")

remaining = books.isnull().sum().sum()
print(f"✅  Missing values remaining: {remaining}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── 2.3  Duplicate ratings ───────────────────
n_dup = ratings.duplicated().sum()
print(f"Duplicate rows in ratings: {n_dup:,}")
ratings = ratings.drop_duplicates()
print(f"✅  Ratings after dedup: {len(ratings):,}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── 2.4  Data-type summary ───────────────────
print("books dtypes:")
print(books.dtypes)
print()
print("ratings dtypes:")
print(ratings.dtypes)
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── 2.5  Basic statistics ────────────────────
print("=== books.describe() ===")
display(books[["average_rating", "ratings_count",
               "original_publication_year"]].describe().round(2))

print("\\n=== ratings.describe() ===")
display(ratings["rating"].describe().round(2))
"""))

# ─────────────────────────────────────────────
# SECTION 3 – EDA
# ─────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 3. Exploratory Data Analysis (EDA)"))

# 3.1 Rating distribution
cells.append(nbf.v4.new_markdown_cell("### 3.1 Distribution of User Ratings"))
cells.append(nbf.v4.new_code_cell("""\
fig, ax = plt.subplots(figsize=(8, 5))
counts = ratings["rating"].value_counts().sort_index()
bars = ax.bar(counts.index, counts.values,
              color=sns.color_palette("viridis", 5))
ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=10)
ax.set_xlabel("Rating (1–5)")
ax.set_ylabel("Number of Ratings")
ax.set_title("Distribution of User Ratings")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.tight_layout()
plt.show()
"""))

# 3.2 Top 10 most-rated
cells.append(nbf.v4.new_markdown_cell("### 3.2 Top 10 Most-Rated Books"))
cells.append(nbf.v4.new_code_cell("""\
top10_count = books.nlargest(10, "ratings_count")[["title", "ratings_count"]].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 6))
palette = sns.color_palette("magma", 10)[::-1]
ax.barh(top10_count["title"], top10_count["ratings_count"], color=palette)
ax.set_xlabel("Number of Ratings")
ax.set_title("Top 10 Books by Number of Ratings")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.invert_yaxis()
plt.tight_layout()
plt.show()
"""))

# 3.3 Top 10 highest-rated
cells.append(nbf.v4.new_markdown_cell("### 3.3 Top 10 Highest-Rated Books (≥ 10 000 ratings)"))
cells.append(nbf.v4.new_code_cell("""\
top10_avg = (
    books[books["ratings_count"] >= 10_000]
    .nlargest(10, "average_rating")[["title", "average_rating", "authors"]]
    .reset_index(drop=True)
)

fig, ax = plt.subplots(figsize=(10, 6))
palette = sns.color_palette("crest", 10)[::-1]
bars = ax.barh(top10_avg["title"], top10_avg["average_rating"], color=palette)
ax.bar_label(bars, fmt="{:.2f}", padding=4, fontsize=10)
ax.set_xlabel("Average Rating")
ax.set_title("Top 10 Highest-Rated Books (min 10 000 ratings)")
ax.set_xlim(3.9, 4.7)
ax.invert_yaxis()
plt.tight_layout()
plt.show()
"""))

# 3.4 Publication year
cells.append(nbf.v4.new_markdown_cell("### 3.4 Books Published Per Year (post-1900)"))
cells.append(nbf.v4.new_code_cell("""\
modern = books[books["original_publication_year"] > 1900]["original_publication_year"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(modern, bins=60, color="#2196F3", edgecolor="white", alpha=0.85)
ax.set_xlabel("Publication Year")
ax.set_ylabel("Number of Books")
ax.set_title("Distribution of Books by Publication Year (post-1900)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.show()
"""))

# 3.5 Language
cells.append(nbf.v4.new_markdown_cell("### 3.5 Language Distribution"))
cells.append(nbf.v4.new_code_cell("""\
lang_counts = books["language_code"].value_counts().head(10)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=lang_counts.values, y=lang_counts.index,
            hue=lang_counts.index, palette="tab10", legend=False, ax=ax)
ax.set_xlabel("Number of Books")
ax.set_title("Top 10 Language Codes")
plt.tight_layout()
plt.show()
"""))

# 3.6 Average rating distribution
cells.append(nbf.v4.new_markdown_cell("### 3.6 Average Rating Distribution per Book"))
cells.append(nbf.v4.new_code_cell("""\
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(books["average_rating"], bins=40, color="#9C27B0", edgecolor="white", alpha=0.85)
ax.axvline(books["average_rating"].mean(), color="red", linestyle="--",
           linewidth=1.5, label=f'Mean = {books["average_rating"].mean():.2f}')
ax.set_xlabel("Average Rating")
ax.set_ylabel("Number of Books")
ax.set_title("Distribution of Average Ratings per Book")
ax.legend()
plt.tight_layout()
plt.show()
"""))

# 3.7 Top tags
cells.append(nbf.v4.new_markdown_cell("### 3.7 Top 20 Most Popular Tags"))
cells.append(nbf.v4.new_code_cell("""\
# Merge tag counts
merged_tags_eda = pd.merge(book_tags, tags, on="tag_id")
top_tags = (merged_tags_eda.groupby("tag_name")["count"]
            .sum().nlargest(20).reset_index())

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=top_tags, x="count", y="tag_name",
            hue="tag_name", palette="rocket", legend=False, ax=ax)
ax.set_xlabel("Total Tag Count")
ax.set_title("Top 20 Most Popular Tags")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.tight_layout()
plt.show()
"""))

# 3.8 Ratings per user
cells.append(nbf.v4.new_markdown_cell("### 3.8 Ratings per User (Activity Distribution)"))
cells.append(nbf.v4.new_code_cell("""\
user_activity = ratings.groupby("user_id")["rating"].count()

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(user_activity, bins=50, color="#00BCD4", edgecolor="white", alpha=0.85)
ax.set_xlabel("Number of Ratings per User")
ax.set_ylabel("Number of Users")
ax.set_title("User Activity Distribution")
ax.set_xlim(0, user_activity.quantile(0.99))  # trim extreme tail
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.show()

print(f"Median ratings per user : {user_activity.median():.0f}")
print(f"Max ratings per user    : {user_activity.max():,}")
"""))

# ─────────────────────────────────────────────
# SECTION 4 – Feature Engineering
# ─────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 4. Feature Engineering / Data Preparation

Two output artefacts are produced here for the modelling stage (Võ Minh Chính):

| Artefact | Used for |
|---|---|
| **Sparse User-Item Matrix** | Collaborative Filtering (SVD, KNN, …) |
| `books_cleaned_with_features.csv` | Content-Based Filtering (TF-IDF cosine similarity) |
"""))

# 4.1 User-item matrix
cells.append(nbf.v4.new_markdown_cell("### 4.1 User-Item Interaction Matrix (Collaborative Filtering)"))
cells.append(nbf.v4.new_code_cell("""\
n_users = ratings["user_id"].max() + 1   # keep original IDs as indices
n_books = ratings["book_id"].max() + 1

user_item_matrix = csr_matrix(
    (ratings["rating"].values,
     (ratings["user_id"].values, ratings["book_id"].values)),
    shape=(n_users, n_books)
)

sparsity = 1 - user_item_matrix.nnz / (n_users * n_books)
print(f"Matrix shape : {user_item_matrix.shape}")
print(f"Non-zero entries: {user_item_matrix.nnz:,}")
print(f"Sparsity     : {sparsity:.4%}")
"""))

# 4.2 Content features
cells.append(nbf.v4.new_markdown_cell("### 4.2 Content Features (Content-Based Filtering)"))
cells.append(nbf.v4.new_code_cell("""\
# Merge tags → one string per book
merged_tags_feat = pd.merge(book_tags, tags, on="tag_id")
book_tags_grouped = (
    merged_tags_feat.groupby("goodreads_book_id")["tag_name"]
    .apply(lambda x: " ".join(x))
    .reset_index()
)

# Join with books (books.book_id == goodreads_book_id)
books_feat = pd.merge(
    books,
    book_tags_grouped,
    left_on="book_id",
    right_on="goodreads_book_id",
    how="left"
)
books_feat["tag_name"] = books_feat["tag_name"].fillna("")
books_feat["authors"]  = books_feat["authors"].fillna("")

# Combined text feature
books_feat["content_features"] = books_feat["authors"] + " " + books_feat["tag_name"]

print("Sample content_features:")
display(books_feat[["title", "authors", "content_features"]].head(4))
"""))

cells.append(nbf.v4.new_code_cell("""\
# Build TF-IDF matrix (ready for cosine-similarity in modelling notebook)
tfidf = TfidfVectorizer(stop_words="english", max_features=8_000)
tfidf_matrix = tfidf.fit_transform(books_feat["content_features"])

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"Vocabulary size    : {len(tfidf.vocabulary_):,}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Save cleaned dataset for the modelling notebook
DATA_DIR.mkdir(exist_ok=True)
out_path = DATA_DIR / "books_cleaned_with_features.csv"
books_feat.to_csv(out_path, index=False)
print(f"✅  Saved → {out_path}")
print(f"    Shape : {books_feat.shape}")
"""))

# ─────────────────────────────────────────────
# SECTION 5 – References
# ─────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 5. References *(APA 7th Edition)*

- Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, *9*(3), 90–95. https://doi.org/10.1109/MCSE.2007.55  
- McKinney, W. (2010). Data structures for statistical computing in Python. In S. van der Walt & J. Millman (Eds.), *Proceedings of the 9th Python in Science Conference* (pp. 51–56). https://doi.org/10.25080/Majora-92bf1922-00a  
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830.  
- Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender systems handbook* (2nd ed.). Springer. https://doi.org/10.1007/978-1-4899-7637-6  
- Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., … SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, *17*, 261–272. https://doi.org/10.1038/s41592-019-0686-2  
- Waskom, M. L. (2021). Seaborn: Statistical data visualization. *Journal of Open Source Software*, *6*(60), 3021. https://doi.org/10.21105/joss.03021  
- Zygmunt, Z. (2017). *Goodbooks-10k* [Data set]. Kaggle. https://www.kaggle.com/datasets/zygmunt/goodbooks-10k  
"""))

nb["cells"] = cells

out = Path(r"e:\BigData\Group 06\Group06_BookRecommendation_LeKimDung.ipynb")
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"[OK] Notebook written to {out}")
print(f"   Total cells: {len(nb['cells'])}")
