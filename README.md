# Project Report on Reproduction of the Work Presented in the Article "A Novel Text Mining Approach Based on TF-IDF and Support Vector Machine for News Classification"

**<h2>Table of contents</h2>**

   [1.Introduction](#Introduction)
   
   [Problem Statement and Objectives of the Paper](#Problem-Statement-and-Objectives-of-the-Paper)
  
   [3. The state of the art on which the work is based](#The-state-of-the-art-on-which-the-work-is-based)

   [4. The methodology followed in the article](#The-methodology-followed-in-the-article)

   [5. Techniques](#Techniques)

   [6.	Results obtained with our implementation](#Results-obtained-with-our-implementation)

   [7. Comparison of Results](#Comparison-of-Results)
   
   [8. Critiques of the work](#Critiques-of-the-work)
   
   [9. Conclusion](#Conclusion)
   

<h2>1. Introduction:</h2>
In this report, we present our project aiming to perform work similar to that presented in the article titled "A Novel Text Mining Approach Based on TF-IDF and Support Vector Machine for News Classification". We begin by defining the problem statement and objectives addressed in the original article. Next, we discuss the related state-of-the-art upon which the work is based, followed by a description of the research methodology we followed. We also explain the techniques used to reproduce the approach described in the article. Subsequently, we present the results obtained through our implementation and compare them with the results presented in the original article. Finally, we provide constructive critiques of the work and conclude with the main findings of our project.

<h2>2. Problem Statement and Objectives of the Paper</h2>
The original article focuses on addressing the challenge of automatically classifying news headlines by their content into distinct groups. The main goal of the study is to introduce a unique text mining approach for news classification utilizing TF-IDF and Support Vector Machine. This approach aims to effectively categorize news headlines and enable users to identify the most prominent news group within a specific country at any given moment. To evaluate the proposed approach, the study utilizes two datasets from BBC and five groups of 20Newsgroup datasets.

<h2>3. The state of the art on which the work is based:</h2>
Before proposing their approach, the authors of the article studied the state of the art in the field of text classification. In fact, the article refers to several previous studies in the field of text classification. One study used the TF-IDF algorithm to classify news articles in Bahasa Indonesia, which calculates the weight of each word based on its repetition in the text and the number of files in which it appears. Another study made modifications to the Bayesian algorithm to improve its efficiency in classifying spam messages. A supervised queue selection method was proposed to enhance the efficiency of text classification by assigning a score to each word based on its similarity to each class. SVMs were used in another study to classify human emotions, demonstrating their effectiveness in emotion recognition in sentences. Additionally, a news customization system based on SVM was proposed to recommend favorite articles to users based on their predefined interests. Lastly, a hybrid SVM-NN method combining the nearest neighbor algorithm and SVM classifier was implemented to minimize the impact of parameters on classification accuracy.

These previous studies have highlighted the advantages and limitations of different approaches used in text classification. They have also demonstrated the potential benefits of combining techniques such as TF-IDF and SVM to improve the precision and efficiency of news classification. It is on this basis that the authors developed their innovative approach, aiming to leverage the semantic information of words and phrases in the text for more accurate classification of news headlines.

<h2>4. The methodology followed in the article:</h2>
The authors followed a well-defined research methodology to develop their approach.

 ### a. Data collection: 
 
 The dataset used in this study consisted of two main sources: the 20Newsgroup dataset and the BBC dataset:
 
- The BBC dataset comprised 2225 news texts collected between 2004 and 2005, covering five topics: business, entertainment, politics, sports, and technology. This dataset provided a diverse range of news articles for evaluation. 

- The 20Newsgroup dataset contained 19997 news articles gathered from the internet, classified into 20 distinct classes. Only the subject and body of each text were utilized in the study. The authors focused on five specific classes only: Graphics, Forsale, Sport.baseball, Religion.christian, and Politics.guns. This subset of the consists of 5070 news articles.

### b. Text preprocessing: 

The authors preprocess the raw text data by converting all words to lowercase, tokenizing and removing stop words.

### c. Feature selection: 

The authors use TF-IDF to select the most important features (words) in the text data for classification. 

### d. Classification: 

The authors use Support Vector Machine (SVM) to classify news headlines into different groups based on their content. 

### e. Evaluation: 

The authors evaluate the performance of the proposed approach using various metrics such as precision, recall, and F1-score. 
  
  ![Diagramme vierge (12)](https://github.com/ikram28/News-Classification/assets/86806466/0a7b6443-9bfe-48c2-b131-08c40817c0c6)
  
  <h2>5. Techniques :</h2>
  
In the context of our project, we employed the same techniques described in the original article. We also utilized a similar dataset and followed the data preprocessing methodology outlined in the article. We implemented the TF-IDF algorithm to compute the feature vectors and utilized the SVM classifier to train our classification model.
Here we explain how each step was implemented:

### a. Data Preprocessing:

   - Lowercasing: The text was transformed to lowercase using the `lower()` method.
   - Punctuation Removal: Punctuation marks were removed from the text using regular expressions (`re.sub()`) with the pattern `[^\w\s]`. This pattern matches any character that is not a word character or whitespace and replaces it with an empty string.
   - Date Removal: Dates were removed from the text using the same `re.sub()` method, but this time with the pattern `\d+`, which matches one or more digits.
   - Tokenization: The `word_tokenize()` method from the `nltk.tokenize` module was used to tokenize the text into individual words.
   - Stopword Removal: Stopwords, which are commonly occurring words with little semantic value, were removed from the tokenized words. The `stopwords.words('english')` method from the `nltk.corpus` module provided a list of English stopwords. A list comprehension was used to filter out stopwords from the tokenized words.

### b. Feature Extraction with TF-IDF:
   - The `TfidfVectorizer` from scikit-learn was initialized without any specific parameters.
   - The vectorizer was then fitted on the preprocessed texts using the `fit()` method. This step learned the vocabulary and IDF values from the texts.
   - The preprocessed texts were transformed into TF-IDF feature vectors using the `transform()` method. This step utilized the learned vocabulary and IDF values to convert the texts into numerical representations.

### c. Classification using Nu-SVC:
   - The dataset was split into training and testing sets using the `train_test_split()` function from scikit-learn.
   - The Nu-SVC classifier was initialized with a specified maximum value of nu (0.5) using the `NuSVC()` class from scikit-learn.
   - The SVM classifier was trained on the training data using the `fit()` method, with the feature vectors (`X_train`) and corresponding labels (`y_train`) as inputs.
   - Predictions were made on the testing data using the `predict()` method of the trained SVM classifier.

### d. Performance Evaluation:
   - Various performance metrics were calculated to evaluate the classification model's performance.
   - Precision, recall, and F1-score were computed using the `precision_score()`, `recall_score()`, and `f1_score()` functions, respectively, from scikit-learn.
   - The classification report, which includes precision, recall, F1-score, and support for each class, was generated using the `classification_report()` function from scikit-learn.

<h2>6.	Results obtained with our implementation:</h2>
In our implementation, we evaluated the performance of the classification models on two datasets: BBC and 20 Newsgroups. The results obtained are as follows:



**BBC Dataset:**

- Precision: The precision values for each class in the BBC dataset were high, ranging from 0.93 to 0.99. This indicates that the model had a high level of accuracy in predicting the correct class for each news article.
- Recall: The recall values were also high, indicating that the model effectively captured a high proportion of the relevant instances for each class.
- F1-Score: The F1-scores, which consider both precision and recall, were also excellent, ranging from 0.95 to 0.99. This demonstrates a balanced performance of the model in terms of both precision and recall.
- Accuracy: The overall accuracy of the model on the BBC dataset was 97%, indicating that the model correctly classified 97% of the news articles.
  
  ![image](https://github.com/ikram28/News-Classification/assets/86806466/6bd84043-32c1-41fd-bfc0-b5e416b59fd0)

  
**20 Newsgroups Dataset:**

- Precision: The precision values for each class in the 20 Newsgroups dataset were also quite good, ranging from 0.86 to 0.97. This indicates that the model achieved a high level of precision in classifying the news articles into their respective categories.
- Recall: The recall values were also high, indicating that the model effectively captured a high proportion of the relevant instances for each class in the dataset.
- F1-Score: The F1-scores, which provide a balanced measure of precision and recall, were consistently high, ranging from 0.89 to 0.92. This suggests that the model achieved a good balance between precision and recall in classifying the news articles.
- Accuracy: The overall accuracy of the model on the 20 Newsgroups dataset was 91%, indicating that the model correctly classified 91% of the news articles.
  
  ![image](https://github.com/ikram28/News-Classification/assets/86806466/6fc9541e-7ead-4b9d-8bec-8b0d176e0bd9)



<h2>7. Comparison of Results: </h2>

Based on the obtained results, a comparison can be made between the precision values achieved in this implementation and those reported in the referenced article. For the BBC dataset, the precision achieved in this implementation was 0.9668, which closely aligns with the precision of 97.84% (0.9784) reported in the article. This indicates that the implemented approach performs at a similar level of precision in classifying news articles into categories such as business, entertainment, politics, sport, and tech.

Moving on to the 20 Newsgroups dataset, the implemented model achieved a precision of 0.9124, while the article reported precision values ranging from 0.9165 to 0.9704 for specific classes such as graphics, forsale, sport.baseball, religion.christian, and politics.guns. Although the overall precision obtained in this implementation is slightly lower, the values remain within a reasonable range. Therefore, it can be concluded that the implemented approach successfully classifies news articles into their respective categories with good accuracy.

It's important to note that precision focuses on the exactness of the classification results, particularly the proportion of correctly predicted positive instances. While precision provides valuable insights into the model's performance, it's advisable to consider other metrics such as recall, F1-score, and accuracy for a comprehensive evaluation.

In summary, the results achieved in this implementation demonstrate precision values that are comparable to those reported in the article. This suggests that the implemented approach effectively categorizes news articles for both the BBC and 20 Newsgroups datasets.
Overall, our implementation demonstrates performance that is consistent with the results reported in the article, indicating the effectiveness and reliability of the techniques and methodology utilized. The similarities in accuracy and F1-scores between our implementation and the article's results validate the robustness and generalizability of the approach, highlighting its potential for accurate classification of news articles in real-world scenarios.

<h2>8. Critiques of the work:</h2>
Although the original article provided a promising approach for news classification, there are some criticisms that can be raised. Firstly, the method used relies solely on the analysis of textual content, which may limit its ability to capture contextual information and more complex semantic relationships. Secondly, the article did not extensively address the selection of model hyperparameters, which could have had an influence on the final performance. Lastly, the authors did not explore the limitations or potential failure cases of their approach, which could have provided additional insights for a more thorough evaluation.
  
  
 <h2> 9. Conclusion:</h2>
In conclusion, our project aimed to reproduce the results of the original article on news classification. We followed the same techniques and preprocessing methodology described in the article, using a similar dataset. Our implementation utilized the TF-IDF algorithm to calculate feature vectors and employed the SVM classifier for training our classification model. 

Overall, our results demonstrated high precision and recall values for both the BBC and 20 Newsgroup datasets. We achieved comparable precision scores to those reported in the original article for most classes. However, there were some variations in accuracy, with our implementation achieving slightly lower accuracy values compared to the reported values in the article. 

It is worth noting that while our implementation achieved promising results, there were certain limitations and areas for improvement. The reliance on textual content analysis alone may have limited our ability to capture complex semantic relationships and contextual information. Additionally, the article's lack of detailed discussion on hyperparameter selection leaves room for potential performance enhancements through fine-tuning. Moreover, exploring the limitations and potential failure cases of our approach would have provided valuable insights for a more comprehensive evaluation.

Despite these limitations, our project successfully reproduced the classification methodology described in the article and obtained satisfactory results. The knowledge gained from this reproduction study contributes to a deeper understanding of news classification techniques and serves as a foundation for future research in the field.



 
