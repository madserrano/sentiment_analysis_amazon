## Sentiment Analysis for Amazon Product Reviews

In this project, a machine learning model was built and trained to generate analysis on streaming twitter data relating to amazon product reviews. A publicly available dataset, holding more than one million amazon reviews, was used to develop and train a reliable sentiment classifier. The training dataset has live amazon reviews that spans more than a decade. Vader Sentiment Analysis was used additionally to perform a more intuitive classification using lexical features of samples. These results were appended to a separate column for comparison.


### **Train and Test data**
A reliable sentiment classifier requires an equally reliable training dataset. To train our model, a scraped amazon review dataset from http://jmcauley.ucsd.edu/ was utilized. It originally contains 142.8 million reviews spanning May 1996 to July 2014 but in this project, data was reduced to 1.05 million- this is to minimize the time and resources needed for pre-preprocessing.

**Amazon review dataset info:** This dataset includes reviews for various amazon products including books, electronics, movies & tv, CDs & vinyl, clothing, shoes, jewelry and more.
<br>**Twitter data:** The main goal of this project is to provide real-time analysis and sentiment classification of amazon product reviews from Twitter. Using twitter API, tweets relating to amazon reviews or amazon products were extracted and exported to a csv file for analysis.

### **Data Pre-processing**
Data pre-processing involved various tasks to ease the development of our classifier. Train data is already labeled, hence pre-processing only involved feature extraction. Sklearn’s train_test_split was used to split the dataset randomly with test size of 30%. To fit these data into our model, they were transformed into sparse matrix of n-gram counts by initializing NLTK Count Vectorizer. Max features were gradually increased from 2000 to 1 million for better accuracy. After vectorization, these raw frequency counts have been converted into TF-IDF values. TFIDF or “term frequency-inverse document frequency” was used to calculate the relevance of a word from the collection or dataset. Since our train dataset is already labeled, stemming and other NLTK techniques were no longer required and thus, have proceeded on to building the model.
<br><br>**Twitter data pre-processing:** Twitter data is unstructured hence, there are more tasks done to this dataset compared to the train data. Data cleaning task, which includes the following steps, is important because generating predictions and analysis on a noisy dataset may adversely affect the result of our predictions.
1. removal of special characters such as hashtag (#) and at sign(#) using regex.
2. removal of stopwords, (at, the, with, on)
3. removal of whitespaces
4. conversion to lower case
5. lemmatization

### **Experiments and Result**
Our model was built using a training data with enough samples to classify sentiments accurately. The process involved two key phases – Training and Prediction. In this phase, experiments were conducted on the following machine leaning models to determine the best classifier to use in our twitter analysis.
1. Multinomial Naïve Bayes
2. Logistic Regression
3. Random Forest
4. Support Vector Matrix

These models require numerical data for classification; hence, it is imperative to vectorize and convert the dataset to TF-IDF respectively. Next step was to scale, fit the model and predict the classes/labels of each product review using the listed classifiers. The output revealed that Logistic Regression and SVM have higher accuracy compared to other classifiers.
Refer to the code to see results graph.

### **Code Contributors**
1. https://github.com/madserrano/
2. https://github.com/jonatasaguiar
