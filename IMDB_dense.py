df = pd.read_csv("Datasets/IMDB_Dataset.csv")
df = df.iloc[:10000,:]
df.shape

df.duplicated().sum()

df.drop_duplicates(inplace=True)

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

X_train,X_test,Y_train,Y_test= train_test_split(X,y,test_size=0.2,random_state=1)

df.index

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=50000,ngram_range=(2,3))

X_train_bow= cv.fit_transform(X_train).toarray()
X_test_bow= cv.transform(X_test).toarray()


y_train=pd.DataFrame(Y_train.values,columns=['review'])
y_test=pd.DataFrame(Y_test.values,columns=['review'])


y_train['review']=pd.get_dummies(y_train['review'],drop_first=True)
y_test['review']=pd.get_dummies(y_test['review'],drop_first=True)


y_train['review'] = y_train['review'].astype(float)
y_test['review'] = y_test['review'].astype(float)


model = Sequential()
model.add(Dense(512,input_dim=50000,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])


model.fit(X_train_bow,y_train,epochs=10,validation_data=(X_test_bow,y_test))


def predict_comment(text):
    text_formatted = [text_format(text)]

    text_transformed = (cv.transform(text_formatted).toarray())
    output = model.predict(text_transformed)
    if (output>0.5):
        print("Postive review")
    else:
        print("Negative review")



text = "dummy text"
predict_comment(text)
