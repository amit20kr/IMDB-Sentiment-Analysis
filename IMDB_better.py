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

X_train_bow

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

rf.fit(X_train_bow,Y_train)
y_pred = rf.predict(X_test_bow)
accuracy_score(Y_test,y_pred)


text = "It was a complete waste of time. As a massive fan of Joker, I expected a strong comeback five years later. The movie was a complete drag. Whenever I thought the movie was taking a turn for the better, it got worse. Save yourself the time and money. Joaquin Phoenix's performance was excellent, but the script was terrible. Through no fault of his own, Joker Folie A Deux is in line as one of the worst sequels ever. The movie seemed more like a display of Lady Gaga's singing and acting ability, which isn't great. I believe this may be the biggest box office flop of 2024. This movie should have been released on Tubi for free."
text_formatted = [text_format(text)]

text_transformed = (cv.transform(text_formatted).toarray())
rf.predict(text_transformed)

