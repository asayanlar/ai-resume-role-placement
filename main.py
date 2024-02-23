import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt 
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
import string
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

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
# plt.show()

# create a extra column for the cleaned up resume data so we can use it for our data set
resume_data_set['cleaned_resume'] = ''
print(resume_data_set.head())
print(resume_data_set.info())

# print(''.join(word.strip(string.punctuation) for word in resume_data_set.at[0, 'Resume']))

# for word in resume_data_set.at[0, 'Resume']: 
#     char_index = resume_data_set.at[0, 'Resume'].index(word)
#     if resume_data_set.at[0, 'Resume'][char_index + 1] != ' ':
#         print(' '.join(word.strip(string.punctuation)))
#     else:
#         print(''.join(word.strip(string.punctuation)))
resume = resume_data_set['Resume']
resume_data_set['cleaned_resume'] = resume.str.replace(r'[^a-zA-Z0-9\' ]+', ' ', regex=True).str.replace(r'[\s\s]+', ' ', regex=True)

# print(resume_data_set.at[0, 'cleaned_resume'])
# print(resume_data_set.head())
# for x in range (0, 30):
#     print(resume_data_set['cleaned_resume'][x])
# print(resume_data_set.info())

resume_data_set_copy=resume_data_set.copy()

stop_words_set = set(stopwords.words('english')+['``',"''"])

total_words =[]

str_all_resumes = []
for resume in resume_data_set['cleaned_resume']:
    str_all_resumes += [resume]

str_all_resumes = ''.join(str_all_resumes)

token_resume_words = nltk.word_tokenize(''.join(str_all_resumes))

for word in token_resume_words:
    if word not in stop_words_set and word not in string.punctuation:
        total_words.append(word)

word_freq_dist = nltk.FreqDist(total_words)

most_common_words = word_freq_dist.most_common(50)
print(most_common_words)

wc = WordCloud().generate(str_all_resumes)
plt.figure(figsize=(10,10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()