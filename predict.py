#Create Data
# - Covert DataBase
# - predict if someone will go to school tomorow
# if no send a message to the teacher saying no!


import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI
from pydantic import BaseModel
import lightgbm as lgb
import tkinter as tk
from tkinter import messagebox

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

data = {
    "days_absent": np.random.randint(0,50,100),
    "likes_school": np.random.randint(1,10,100),
    "friends": np.random.randint(0,20,100),
    "average_mood": np.random.randint(0,10,100),
    "text" : text * 2
}
df = pd.DataFrame(data)

y = np.random.randint(0,2,100)

numeric_features = ["days_absent", "likes_school", "friends", "average_mood"]
numeric_transformer = StandardScaler()

text_features = "text"
text_transformer = TfidfVectorizer()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("text", text_transformer, text_features),
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", lgb.LGBMClassifier())
])

pipeline.fit(df, y)

y_pred = pipeline.predict(df)
y_test_prob = pipeline.predict_proba(df)[:,1]
fpr,tpr,value = roc_curve(y, y_test_prob)

plt.plot([0,1],[0,1],'--',color="gray" , label="line")
plt.plot(fpr,tpr,label ="matrix", color="orange")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.legend()
plt.title("ROC- CURVE")
plt.show()

sns.boxplot(data=df[numeric_features])

app = FastAPI()

class person(BaseModel):
  days_absent:int
  likes_school:int
  friends:int
  average_mood:int
  text:str

@app.put("/")
def func(iter: person):
    my_data = {
        "days_absent": [iter.days_absent],
        "likes_school": [iter.likes_school],
        "friends": [iter.friends],
        "average_mood": [iter.average_mood],
        "text": [iter.text]
    }
    my_df = pd.DataFrame(my_data)
    store = pipeline.predict(my_df)
    return {"will_skip": int(store[0])}

def predict_skip(days_absent, likes_school, friends, average_mood, text):
    my_data = {
        "days_absent": [days_absent],
        "likes_school": [likes_school],
        "friends": [friends],
        "average_mood": [average_mood],
        "text": [text]
    }
    my_df = pd.DataFrame(my_data)
    store = pipeline.predict(my_df)
    return store[0]

def show_result():
    try:
        days_absent = int(entry_days_absent.get())
        likes_school = int(entry_likes_school.get())
        friends = int(entry_friends.get())
        average_mood = int(entry_average_mood.get())
        text = entry_text.get("1.0", tk.END)
        result = predict_skip(days_absent, likes_school, friends, average_mood, text)
        if result == 1:
            messagebox.showinfo("Prediction", "The student is likely to skip school.")
        else:
            messagebox.showinfo("Prediction", "The student is not likely to skip school.")
    except ValueError:
        messagebox.showerror("Error", "Please fill in all fields correctly.")

root = tk.Tk()
root.title("Student Skip School Prediction")

tk.Label(root, text="Days Absent").grid(row=0, column=0)
entry_days_absent = tk.Entry(root)
entry_days_absent.grid(row=0, column=1)

tk.Label(root, text="Likes School").grid(row=1, column=0)
entry_likes_school = tk.Entry(root)
entry_likes_school.grid(row=1, column=1)

tk.Label(root, text="Friends").grid(row=2, column=0)
entry_friends = tk.Entry(root)
entry_friends.grid(row=2, column=1)

tk.Label(root, text="Average Mood").grid(row=3, column=0)
entry_average_mood = tk.Entry(root)
entry_average_mood.grid(row=3, column=1)

tk.Label(root, text="Text").grid(row=4, column=0)
entry_text = tk.Text(root, height=5, width=20)
entry_text.grid(row=4, column=1)

predict_button = tk.Button(root, text="Predict", command=show_result)
predict_button.grid(row=5, column=0, columnspan=2)

root.mainloop()