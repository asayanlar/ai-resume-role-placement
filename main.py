# from IPython.display import display
import string
from pathlib import Path
# import re
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import joblib

nltk.download('stopwords')
nltk.download('punkt')

mpl.use('tkagg')

cmap = plt.get_cmap('coolwarm')

resume_data_set = pd.read_csv('data_set/UpdatedResumeDataSet.csv', encoding='utf-8')

print(resume_data_set.head())
print(resume_data_set.info())

# prepare x and y axis to be plotted
categories_list = resume_data_set['Category'].value_counts().index.tolist()
num_same_category = resume_data_set['Category'].value_counts().tolist()

#set up a figure with multiple subplots to be places within it
fig, axs = plt.subplots(1, 2, figsize=(17, 9))

# configure a bar graph and plot it
plot_bar = resume_data_set['Category'].value_counts().plot(ax=axs[0], kind='bar')
axs[0].set_xlabel('Categories')
axs[0].set_ylabel("Count")

# configure a pie graph and plot it 
axs[1].plot(title='CATEGORY DISTRIBUTION')
axs[1].pie(num_same_category, labels=categories_list, autopct='%1.1f%%', shadow=True)

plt.tight_layout()
plt.show()

# create a extra column for the cleaned up resume data so we can use it for our data set
resume_data_set['Cleaned Resumed'] = ''
print(resume_data_set.head())
print(resume_data_set.info())

#retrieve all resumes from resume column
resume = resume_data_set['Resume']

# remove any non-alphanumerical characters and double spaces due to the position of the removed
# characters from the resume row and copy it over to a new column called Cleaned Resume
resume_data_set['Cleaned Resumed'] = resume.str.replace(r'[^a-zA-Z0-9\' ]+', ' ', regex=True).str.replace(r'[\s\s]+', ' ', regex=True)

# define the stop words not wanted in the resume column. This is since stop words are
# insignificant because of how frequent they appear
stop_words_set = set(stopwords.words('english')+['``',"''"])

total_words =[]
str_all_resumes = []

#put all strings from each row of the Cleaned Resume column into a single string
for resume in resume_data_set['Cleaned Resumed']:
    str_all_resumes += [resume]
str_all_resumes = ''.join(str_all_resumes)

# tokenize all words in the cleaned resume
tokenize_resume_words = nltk.word_tokenize(str_all_resumes)

# within those tokens remove all stop words
for word in tokenize_resume_words:
    if word not in stop_words_set:
        total_words.append(word)

# Frequency distribution for all tokens found in all cleaned resumes
word_freq_dist = nltk.FreqDist(total_words)

most_common_words = word_freq_dist.most_common(50)

wc = WordCloud().generate(str_all_resumes)
plt.figure(figsize=(10,10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

# Label the resumes based on their category using a numerical label
le = LabelEncoder()
resume_data_set['Category'] = le.fit_transform(resume_data_set['Category'])