
#Create Data
# - Covert DataBase
# - predict if someone will go to school tomorow
# if no send a message to the teacher saying no!


import matplotlib.pyplot as plt

# trained on synthetic very small data
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,consensus_score,classification_report
from sklearn.naive_bayes import MultinomialNB

text = [
    "I like PE",
    "I dont feel like coming to school tomorrow",
    "I like school sports",
    "School isn't fun",
    "Lunch at school is the best",
    "I find class so boring",
    "Recess is always fun",
    "Tests stress me out",
    "I like doing group projects",
    "I hate doing homework",
    "I enjoy learning new things",
    "School is stressful",
    "I had a great time in science today",
    "I want to skip school",
    "I love my classes",
    "I don’t like writing essays",
    "School makes me happy",
    "I don’t understand the lessons",
    "I like computer class",
    "Studying is so hard",
    "My school has fun activities",
    "School days are too long",
    "Today at school was awesome",
    "I wish I could stay home",
    "I like helping my classmates",
    "My classes are really hard",
    "I did well on my test",
    "I don’t want to do any schoolwork",
    "School assemblies are fun",
    "I don’t like sitting in class all day",
    "school is so fun",
    "School makes me tired",
    "I made new friends today",
    "I don’t like my teacher",
    "I like reading books in class",
    "I can’t wait for school to be over",
    "My teacher is really nice",
    "I like to play games in school",
    "tomorrow is a test I'm going to skip",
    "Art class is my favorite",
    "I like coming to school every day",
    "when is school going to end",
    "I look forward to field trips",
    "I feel lazy about school",
    "I like math class",
    "should I come to school tomorrow or no?",
    "I like to talk to my friends",
    "I’m always bored in class",
    "school is so fun",
    "I don’t like sitting in class all day"
]


y_data = [
    0,1,0,1,0,1,0,1,0,1,
    0,1,0,1,0,1,0,1,0,1,
    0,1,0,1,0,1,0,1,0,1,
    0,1,0,1,0,1,0,1,0,1,
    0,1,0,1,0,1,0,1,0,1
]





vector = CountVectorizer()


data_x = vector.fit_transform(text)


#Logistic Regression or use multinomial nb
mb = MultinomialNB()

x_train,x_test,y_train,y_test = train_test_split( data_x , y_data, random_state=42)



mb.fit(x_train,y_train)

# ww= vector.transform(["school is fun"])
# y_pred = mb.predict(x_test)
# print(accuracy_score(y_pred,y_test))
# print(mb.predict(ww))



data = {
    "days_absent": np.random.randint(0,50,100),
    "likes_school": np.random.randint(1,10,100),
    "friends": np.random.randint(0,20,100),
    "average_mood": np.random.randint(0,10,100),
    "text_" : np.random.randint(0,len(text),100),



}
df = pd.DataFrame(data)


df["text_vectorized"] = df["text_"].apply(lambda x: vector.transform([text[x]]))

df["no_school"] = df["text_vectorized"].apply(lambda i: mb.predict(i)[0])
model = LogisticRegression()




# for i in range(len(df["text_"])):
#   df.loc[i,"op"] = model.predict(vector.transform(df["text_"][i]))




df["will_skip"] = ((  df["days_absent"]>10).astype(int) + (df["likes_school"]<8).astype(int) + (df["friends"]<4).astype(int) + (df["average_mood"] < 6).astype(int)  + (df["no_school"] > 0) .astype(int) )
df["will_skip"] = (df["will_skip"]>2).astype(int)


x  = df.drop(["will_skip", "text_vectorized"] ,axis=1)
y = df["will_skip"]


std = StandardScaler()

x_scaled = std.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,random_state=42)


gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,   max_depth=4, random_state=42 )

gbc . fit(x_train,y_train)
y_pred= gbc.predict(x_test)
gbc.score(x_test,y_test)

y_test_prob = gbc.predict_proba(x_test)[:,1]
fpr,tpr,value = roc_curve(y_test,y_test_prob)

plt.plot([0,1],[0,1],'--',color="gray" , label="line")

plt.plot(fpr,tpr,label ="matrix", color="orange")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.legend()
plt.title("ROC- CURVE")

# boxplot
import seaborn as sns

plt.show()

import pandas as pd


sns.boxplot(data=data)