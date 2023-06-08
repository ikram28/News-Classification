

# # Importing the necessary libraries:



import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import NuSVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# # Defining the list of class labels




class_labels = ['business', 'entertainment', 'politics', 'sport', 'tech']


# # Initializing empty lists to store the preprocessed texts and corresponding labels



preprocessed_texts = []
labels = []


# # Iterating through each class folder and prepocessing the texts :
# * Transorm to LowerCase
# * Remove Punctuation
# * Remove dates
# * Tokenize
# * Remove StopWords


for label in class_labels:
    folder_path = f'C:/Users/ASUS ROG STRIX/Desktop/NewsClassification/bbc/{label}'  
    
    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Read the contents of the file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Preprocess the text
        # Lowercase the text
        text = text.lower()
        # Remove punctuations, quotations, and dates
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Tokenize the text
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [word for word in tokens if word not in stopwords.words('english')]
                
        # Append the preprocessed text and label to the lists
        preprocessed_texts.append(tokens)
        labels.append(label)


# ## Converting the preprocessed texts back to string format:



preprocessed_texts = [' '.join(tokens) for tokens in preprocessed_texts]


# ## Creating a DataFrame with 'labels' and 'preprocessed_texts' as columns



data = pd.DataFrame({'labels': labels, 'preprocessed_texts': preprocessed_texts})
data


# # Feature Extraction Based On TF-IDF :

# ## Initialize the TF-IDF vectorizer



vectorizer = TfidfVectorizer()


# ## Fit the vectorizer on the preprocessed texts to learn the vocabulary and IDF values




vectorizer.fit(preprocessed_texts)


# ## Transform the preprocessed texts into TF-IDF feature vectors




features = vectorizer.transform(preprocessed_texts)


# # Classification using SVM :

# ## Splitting into Train-Test 



X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True, random_state=42)


# ## Initialize the Nu-SVC classifier with the specified maximum value of nu (0.5 in this case)



svm = NuSVC(nu=0.5)


# ## Train the SVM classifier 



svm.fit(X_train, y_train)


# ## Make predictions on the testing data


y_pred = svm.predict(X_test)


# ## Calculate precision



precision = precision_score(y_test, y_pred, average='weighted')
print(f"Precision: {precision:.4f}")


# ## Calculate recall



recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall: {recall:.4f}")


# ## Calculate F1-score



f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1-score: {f1:.4f}")


# ## Calculate the classification report



report = classification_report(y_test, y_pred)

print(report)

