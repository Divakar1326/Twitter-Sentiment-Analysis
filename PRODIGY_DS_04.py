import warnings
warnings.filterwarnings('ignore')
# Twitter Sentiment Analysis
#### Import Necessary Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
#### Load The Data
training_data = pd.read_csv('C:/Users/diva1/OneDrive/Documents/twitter_training.csv')
validation_data = pd.read_csv('C:/Users/diva1/OneDrive/Documents/twitter_validation.csv')
#### Data Preprocessing
training_data.columns = ['ID', 'Entity', 'Sentiment', 'Message']
validation_data.columns = ['ID', 'Entity', 'Sentiment', 'Message']
training_data['Message'] = training_data['Message'].astype(str).fillna('')
validation_data['Message'] = validation_data['Message'].astype(str).fillna('')
#### Explanatory Data Analysis
print(training_data)
print(training_data.info())
print(training_data.head())
print(training_data.shape)
print(training_data.dtypes)
print(training_data.duplicated().sum())
training_data.drop_duplicates(inplace=True)
print(training_data.isna().sum())
print((training_data.isna().sum()/len(training_data))*100)
print(training_data['Entity'].unique())
print(training_data['Sentiment'].unique())
print(validation_data)
print(validation_data.info())
print(validation_data.head())
print(validation_data.shape)
print(validation_data.dtypes)
print(validation_data.duplicated().sum())
validation_data.drop_duplicates(inplace=True)
print(validation_data.isna().sum())
print((validation_data.isna().sum()/len(validation_data))*100)
print(validation_data['Entity'].unique())
print(validation_data['Sentiment'].unique())
# EDA 1: Sentiment Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment', data=training_data)
plt.title('Sentiment Distribution in Training Data')
plt.show()

trend_counts = training_data['Entity'].value_counts()
plt.figure(figsize=(15,10))
sns.barplot(x=trend_counts.values, y=trend_counts.index)
plt.xlabel('Count')
plt.ylabel('Entity')
plt.title('Bar Plot ofPplatform Column')
plt.show()

sentiment_counts = training_data['Sentiment'].value_counts()
plt.figure(figsize=(8,8))
plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99','#ff9999','#ffcc99'])
plt.title('Pie Chart of Sentiment')
plt.axis('equal')
plt.show()

trend_sentiment = pd.crosstab(training_data['Entity'], training_data['Sentiment'])
trend_sentiment.plot(kind='bar', stacked=True, figsize=(10,7), color=['#66b3ff','#99ff99','#ff9999'])
plt.xlabel('Entity')
plt.ylabel('Count')
plt.title('Stacked Bar Chart ofPplatform and Sentiment')
plt.show()
top_10_trends = training_data['Entity'].value_counts().nlargest(10).index

# Filter the dataset for only the top 10 trends
top_10_data = training_data[training_data['Entity'].isin(top_10_trends)]

# Create subplots
fig, axes = plt.subplots(4, 1, figsize=(14, 24))
fig.suptitle('Sentiment Distribution Across Top 10 Trends', fontsize=18)

# Sentiment categories to plot
sentiments = ['Positive', 'Negative', 'Neutral', 'Irrelevant']

# Iterate over each sentiment and create count plots
for sentiment, ax in zip(sentiments, axes.flatten()):
    # Filter the data by sentiment
    filtered_data = top_10_data[top_10_data['Sentiment'] == sentiment]
    
    # Plot the data with seaborn
    sns.countplot(data=filtered_data, x='Entity', ax=ax, palette='viridis', order=top_10_trends)
    
    # Set plot titles and labels
    ax.set_title(f'{sentiment} Sentiment')
    ax.set_xlabel('Entity')
    ax.set_ylabel('Count')

# Adjust layout for the plot
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

sns.histplot(training_data['Sentiment'])
plt.show()

plt.pie(training_data['Sentiment'].value_counts(), labels=training_data['Sentiment'].unique(), autopct='%1.1f%%',
            startangle=90, wedgeprops={'linewidth': 0.5}, textprops={'fontsize': 12},
            explode=[0.1, 0.1, 0.1, 0.1], shadow=True)
plt.show()
plt.figure(figsize=(10,3))
sns.histplot(training_data['Entity'])
plt.xticks(rotation = 90)
plt.show()

plt.figure(figsize=(20,7))
sns.countplot(x="Entity", hue="Sentiment", data=training_data)
plt.title("jop by deposit")
plt.xticks(rotation = 90)
plt.show()

negative_df = training_data[training_data['Sentiment'] == 'Negative']

# Create the plot
plt.figure(figsize=(20, 7))
sns.countplot(x="Entity", hue="Sentiment", data=negative_df,palette='flare')
plt.title("'Entity by Negative Sentiment")
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(10, 9))
count_table = pd.crosstab(index=training_data['Entity'], columns=training_data['Sentiment'])
sns.heatmap(count_table, cmap='YlOrRd', annot=True, fmt='d',linewidths=0.5, linecolor='black')
plt.title('Sentiment Distribution by Entity')
plt.xlabel('Sentiment')
plt.ylabel('Entity')
plt.show()

sentiment_counts = training_data['Sentiment'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
plt.title('Sentiment Distribution in Training Data')
plt.show()
plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment', data=validation_data)
plt.title('Sentiment Distribution in Validation Data')
plt.show()
sentiment_counts = validation_data['Sentiment'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
plt.title('Sentiment Distribution in Validation Data')
plt.show()
# EDA 2: Frequency of All Entities in Training Data
entity_counts = training_data['Entity'].value_counts()

plt.figure(figsize=(10, len(entity_counts) / 2))  
sns.barplot(y=entity_counts.index, x=entity_counts.values, orient='h')
plt.title('Frequency of All Entities in Training Data')
plt.xlabel('Count')
plt.ylabel('Entity')
plt.show()
# EDA 3: Stacked Bar Chart of Sentiment Distribution by Entity
entity_sentiment_counts = pd.crosstab(training_data['Entity'], training_data['Sentiment'])

plt.figure(figsize=(12, 8))
entity_sentiment_counts.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('Stacked Bar Chart of Entity and Sentiments')
plt.xlabel('Entity')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
# EDA 4: Message Length Distribution
training_data['Message_Length'] = training_data['Message'].apply(len)
validation_data['Message_Length'] = validation_data['Message'].apply(len)
plt.figure(figsize=(8, 6))
sns.histplot(training_data['Message_Length'], kde=True, bins=30)
plt.title('Message Length Distribution in Training Data')
plt.show()
plt.figure(figsize=(8, 6))
sns.histplot(validation_data['Message_Length'], kde=True, bins=30)
plt.title('Message Length Distribution in Validation Data')
plt.show()
# EDA 5: Wordcloud
# Filter messages based on sentiment categories
positive_messages = ' '.join(training_data[training_data['Sentiment'] == 'Positive']['Message'])
negative_messages = ' '.join(training_data[training_data['Sentiment'] == 'Negative']['Message'])
neutral_messages = ' '.join(training_data[training_data['Sentiment'] == 'Neutral']['Message'])
irrelevant_messages = ' '.join(training_data[training_data['Sentiment'] == 'Irrelevant']['Message'])
# Create word clouds for each sentiment category
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_messages)
wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_messages)
wordcloud_neutral = WordCloud(width=800, height=400, background_color='white').generate(neutral_messages)
wordcloud_irrelevant = WordCloud(width=800, height=400, background_color='white').generate(irrelevant_messages)
# Set up subplots to show all word clouds
plt.figure(figsize=(16, 12))

# Plot Positive WordCloud
plt.subplot(2, 2, 1)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.title('Word Cloud for Positive Sentiment')
plt.axis('off')

# Plot Negative WordCloud
plt.subplot(2, 2, 2)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.title('Word Cloud for Negative Sentiment')
plt.axis('off')

# Plot Neutral WordCloud
plt.subplot(2, 2, 3)
plt.imshow(wordcloud_neutral, interpolation='bilinear')
plt.title('Word Cloud for Neutral Sentiment')
plt.axis('off')

# Plot Irrelevant WordCloud
plt.subplot(2, 2, 4)
plt.imshow(wordcloud_irrelevant, interpolation='bilinear')
plt.title('Word Cloud for Irrelevant Sentiment')
plt.axis('off')

# Display the plots
plt.tight_layout()
plt.show()
#### TF-IDF Vectorization
# Preprocess the text using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
# Transform the training and validation messages
X_train_tfidf = tfidf_vectorizer.fit_transform(training_data['Message'])
X_validation_tfidf = tfidf_vectorizer.transform(validation_data['Message'])
# Target labels
y_train = training_data['Sentiment']
y_validation = validation_data['Sentiment']
#### K-Means++ Clustering
wcss = []
for i in range(1, 11):  
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_train_tfidf)
    wcss.append(kmeans.inertia_) 
# Plot the WCSS 
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Optimal Number of Clusters by Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
# Initialize the KMeans model
kmeans_model = KMeans(n_clusters=6, init='k-means++', random_state=42)
# Train the model
kmeans_model.fit(X_train_tfidf)
# 3. Predict cluster labels for the validation set
cluster_labels = kmeans_model.predict(X_train_tfidf)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_tfidf.toarray())
# Plot the PCA result with cluster labels
plt.figure(figsize=(10, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=cluster_labels, cmap='rainbow', s=50, alpha=0.7)
plt.title('KMeans++ Clustering of Validation Set (PCA Visualization)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()
import numpy as np
np.unique(cluster_labels, return_counts=True)
#### Naive Bayes Classifier
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_validation_tfidf)
accuracy_nb = accuracy_score(y_validation, y_pred_nb)
report_nb = classification_report(y_validation, y_pred_nb)
print("Naive Bayes Accuracy:", accuracy_nb)
print("Naive Bayes Classification Report:\n", report_nb)

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)

#### Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_validation_tfidf)
accuracy_lr = accuracy_score(y_validation, y_pred_lr)
report_lr = classification_report(y_validation, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr)
print("Logistic Regression Classification Report:\n", report_lr)
#### Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth= 100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
y_pred_rf = rf_model.predict(X_validation_tfidf)
accuracy_rf = accuracy_score(y_validation, y_pred_rf)
report_rf = classification_report(y_validation, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)
print("Random Forest Classification Report:\n", report_rf)
#### Decision Tree
# Initialize the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
# Train the model
dt_model.fit(X_train_tfidf, y_train)
# Make predictions on the validation set
y_pred_dt = dt_model.predict(X_validation_tfidf)
# Evaluate the model
accuracy_dt = accuracy_score(y_validation, y_pred_dt)
report_dt = classification_report(y_validation, y_pred_dt)
# Classification Report 
print("Decision Tree Model Accuracy:", accuracy_dt)
print("Decision Tree Model Classification Report:\n", report_dt)
#### ANN
# Encode Sentiments into numerical values
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_validation_encoded = encoder.transform(y_validation)
# One-hot encoding
y_train_categorical = to_categorical(y_train_encoded)
y_validation_categorical = to_categorical(y_validation_encoded)
# Build the neural network model
model = Sequential()

model.add(Dense(512, input_dim=X_train_tfidf.shape[1], activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model
history = model.fit(X_train_tfidf.toarray(), y_train_categorical, epochs=10, batch_size=64, validation_data=(X_validation_tfidf.toarray(), y_validation_categorical))
# Make predictions on the validation set
y_pred_ann = model.predict(X_validation_tfidf.toarray())
y_pred_ann_labels = np.argmax(y_pred_ann, axis=1)
# Decode the predicted labels
y_pred_labels = encoder.inverse_transform(y_pred_ann_labels)
# Evaluate the model
accuracy_ann = accuracy_score(y_validation, y_pred_labels)
report_ann = classification_report(y_validation, y_pred_labels)
# Classification Report 
print("ANN Model Accuracy:", accuracy_ann)
print("ANN Model Classification Report:\n", report_ann)
#Plot the loss over epochs
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Epoch vs Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()