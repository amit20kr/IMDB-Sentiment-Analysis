temp_df = pd.read_csv('Datasets/IMDB_Dataset.csv')
df = temp_df.iloc[:10000]
df.duplicated().sum()

df.drop_duplicates(inplace=True)

# remove stopwords, html tags and lowercase
from nltk.corpus import stopwords

exclude= string.punctuation

def text_format(text):
    text=text.lower()
    soup= BeautifulSoup(text,'html.parser')
    text=soup.get_text().replace("\\","")

    text=text.translate(str.maketrans('','',exclude))
    pattern = re.compile(r'https?://\S+|www\.\S+')

    mid_str = pattern.sub(r'',text)

    new_text = []

    for word in mid_str.split():
        if word not in stopwords.words('english'):
            new_text.append(word)
    x = new_text[:]
    new_text.clear()
    return " ".join(x)


df['review']=df['review'].apply(text_format)

X= df.iloc[:,0]
y = df.iloc[:,1]

y

X_train,X_test,Y_train,Y_test= train_test_split(X,y,test_size=0.2,random_state=1)

X_train.shape

# Using bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X_train_bow= cv.fit_transform(X_train).toarray()
X_test_bow= cv.transform(X_test).toarray()

# using gaussian naive bayes as a model 
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb.fit(X_train_bow,Y_train)

y_pred = gnb.predict(X_test_bow)

accuracy_score(Y_test,y_pred)

confusion_matrix(Y_test,y_pred)

# using random forest model 
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

rf.fit(X_train_bow,Y_train)
y_pred = rf.predict(X_test_bow)
accuracy_score(Y_test,y_pred)

cv = CountVectorizer(max_features= 3000)

X_train_bow= cv.fit_transform(X_train).toarray()
X_test_bow= cv.transform(X_test).toarray()

rf = RandomForestClassifier()

rf.fit(X_train_bow,Y_train)
y_pred = rf.predict(X_test_bow)
accuracy_score(Y_test,y_pred)

cv = CountVectorizer(max_features= 10000,ngram_range=(2,3))

X_train_bow= cv.fit_transform(X_train).toarray()
X_test_bow= cv.transform(X_test).toarray()

rf = RandomForestClassifier()

rf.fit(X_train_bow,Y_train)
y_pred = rf.predict(X_test_bow)
accuracy_score(Y_test,y_pred)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf= tfidf.transform(X_test).toarray()


rf = RandomForestClassifier()

rf.fit(X_train_tfidf,Y_train)
y_pred = rf.predict(X_test_tfidf)
accuracy_score(Y_test,y_pred)

# using word2vec

story = []
for doc in df['review']:
    raw_sent = sent_tokenize(doc)
    for sent in raw_sent:
        story.append(simple_preprocess(sent))





model = gensim.models.Word2Vec(
    window = 10,
    min_count = 2
)

model.build_vocab(story)

model.train(story,total_examples=model.corpus_count,epochs = model.epochs)

len(model.wv.index_to_key)

# we have now created a vector of words now we will create a vector for the reviews

def document_vector(doc):
    # remove out of vocab words 
    doc = [word for word in doc.split() if word in model.wv.index_to_key]
    return np.mean(model.wv[doc],axis= 0)

document_vector(df['review'].values[0])
# this is the vector of first sentence 

X= []
from tqdm import tqdm 
for doc in tqdm(df['review'].values):
    X.append(document_vector(doc))

X=np.array(X)

X.shape
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

y = encoder.fit_transform(df['sentiment'])

y
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=1)

mnb = GaussianNB()
mnb.fit(X_train,Y_train)
y_pred = mnb.predict(X_test)
accuracy_score(Y_test,y_pred)



