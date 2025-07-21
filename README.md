# README-Natural Language Processing- "BookRecApp"

## Brian Benedicks

<!-- Add your content below -->

### Ethics Statement

This application prioritizes educational development. The purpose of this application is to aid in knowledge exploration for avid readers. No private information is collected for the purposes of this application. 

### Objective- To aid avid readers in finding recommended novels based on user input data.


### 1. Data Sources

- A script was used to collect books falling under various genres along with metadata. A google API was used due to being free of charge.


### 2. Data/Feature Engineering 

1. The BERT tokenizer was used feed into the Bert model. 
2. preprocessed text to normalize book summaries.
3. TF-IDF was used to convert text to vectors for use in the multinomial NB modeling.
4. Label encoding and one-hot encoding for categorical genre information

### 3. Modeling Approach
1. Matrix Factorization: Non negative matrix factorization for collaborative filtering recommendations
2. The TF-IDF was paired with mulinomial NB to determine if word frequency improved accuracy in predicting categories.
3. Variational Autoencoder- Bert Embeddings with metadata



### 4. Model Evaluation- 
1. NB- F1 Score and accuracy
2. NCF-used RMSE for penalizing larger errors more severely. Precision@5 for top five relevant recommendations.
3. VAE-Semantic quality score to measure how well the model separates books based on genre in the semantic space.

### Model Comparison:
1. Naive Bayes Accuracy: 0.543
2. Matrix Factorization RMSE: 2.686
3. VAE + BERT Semantic Quality: 0.500
   


### Dependencies- The project employed the following dependencies:
streamlit>=1.25.0
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.3.0
pandas>=1.5.0
numpy>=1.21.0
requests>=2.28.0
joblib>=1.2.0

### Application:
![BookRecApp](bookrecapp.png)

### To run:
1. Place all files in the same directory and run the script


