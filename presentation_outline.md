# Presentation Slide Outline: Book Recommendation System
---

## Slide 1: Title Slide
**[Main Title]** Book Recommendation System
**[Subtitle]** DS 423 – Machine Learning with Large Datasets
**[Group Information]**
- Group 6 (Interest Area: Recommendation System)
- Lê Kim Dũng (ID: 28211452455)
- Võ Minh Chính (ID: 28212306297)
**[Institution]** Duy Tan University, Da Nang

---

## Slide 2: Problem Statement
**Real-world Problem:** 
- The massive volume of published books causes "information overload", making it difficult for readers to discover books that align with their personal tastes.
- Online bookstores rely on automated systems to analyze user behavior and recommend relevant titles, which enhances user experience and drives sales.

**Our Goal:** 
- To build a personalized recommendation engine that outputs a Top-N book list tailored to individual users based on their preferences.

---

## Slide 3: Dataset
**Dataset Information:**
- **Name:** Goodbooks-10k
- **Source:** Kaggle (zygmunt/goodbooks-10k)
- **Size:** 10,000 unique books and approximately 1,000,000 user ratings.

**Key Features:**
- **Ratings:** Explicit user feedback on a 1-5 scale.
- **Metadata:** Book details including authors, publication year, and average rating.
- **Tags:** User-generated genres and topics assigned to various books.

---

## Slide 4: Solution and Approach
**Hybrid Approach:** 
We developed a recommendation system combining two core methods:
1. **Collaborative Filtering (CF):** Utilizes Singular Value Decomposition (SVD) to analyze the sparse User-Item interaction matrix. It discovers latent factors to recommend books based on similar user tastes.
2. **Content-Based Filtering (CBF):** Utilizes TF-IDF and Cosine Similarity on book tags and authors to recommend books that share similar content, themes, and genres.

---

## Slide 5: Key Parts of Code
**Implementation Highlights:**
- **Data Preparation:** Handling missing values, removing duplicates, and aggregating Tags and Authors into a unified `content_features` column.
- **Modeling:** 
  - Using `scikit-surprise` for the SVD algorithm (with an 80/20 train-test split).
  - Using `scikit-learn` for TF-IDF vectorization and cosine similarity.
- **Top-N Recommendation:** The `get_top_n_recommendations_for_user` function filters out books the user has already read, predicts scores for the rest, and returns the top 10 highest-scoring recommendations.
*(Tip: Insert a screenshot of the Jupyter Notebook output showing the Top 10 recommended books here).*

---

## Slide 6: Results and Evaluation
**SVD Model Evaluation (on 20% Test Set):**
- **Root Mean Square Error (RMSE):** 0.8402
- **Mean Absolute Error (MAE):** 0.6566
- **Precision@10:** 0.6639 (66% of recommended books are highly relevant to the user).
- **Recall@10:** 0.7101

**Conclusion:** 
An RMSE of ~0.84 indicates highly accurate predictions (less than 1-star error on average). The matrix factorization approach proves highly effective for sparse rating datasets.
*(Tip: Insert a chart from the EDA phase here, such as the "Distribution of User Ratings" chart, to make the slide more visually appealing).*

---

## Slide 7: Individual Contribution
**Total Work Split: 50% - 50%**
- **Lê Kim Dũng (50%):** Data Collection, Data Cleaning, Exploratory Data Analysis (EDA), and Feature Engineering.
- **Võ Minh Chính (50%):** Model Implementation (Collaborative & Content-Based), System Evaluation (RMSE, MAE, Precision/Recall), and the Top-N Recommendation function.
- **Shared Tasks:** Code integration, testing, LaTeX report writing, and Presentation design.

---

## Slide 8: References
*(APA 7th Edition Format)*
- Hug, N. (2020). Surprise: A Python library for recommender systems. *Journal of Open Source Software*, 5(52), 2174. https://doi.org/10.21105/joss.02174
- McKinney, W. (2010). Data structures for statistical computing in Python. In S. van der Walt & J. Millman (Eds.), *Proceedings of the 9th Python in Science Conference* (pp. 51–56).
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Zygmunt, Z. (2017). *Goodbooks-10k* [Data set]. Kaggle. https://www.kaggle.com/datasets/zygmunt/goodbooks-10k
