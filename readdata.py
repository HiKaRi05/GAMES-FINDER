import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Trandata.csv')

cols = ['รับสถาณการณ์กดดันได้ดี',
        'ช่างสังเกต', 'รักอิสระ',
        'ละเอียดรอบคอบ', 'มีความสนใจในความรู้รอบตัว',
        'มีระเบียบวินัย', 'ชอบลองผิดลองถูก',
        'ชอบความท้าทาย', 'มีความเป็นตัวของตัวเองสูง',
        'ขี้ระแวง', 'เข้าสังคมเก่ง',
        'มีความหนักแน่นมั่นคง', 'เป็นคนสองบุคลิก',
        'ไม่ชอบเที่ยวโลดโผน', 'ใจกว้าง',
        'ขี้เกรงใจคนอื่น']

from sklearn.model_selection import train_test_split

df['รับสถาณการณ์กดดันได้ดี'] = df.รับสถาณการณ์กดดันได้ดี.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['ช่างสังเกต'] = df.ช่างสังเกต.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['ชอบทดลอง'] = df.ชอบทดลอง.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['รักอิสระ'] = df.รักอิสระ.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['นิยมธรรมชาติอนุรักษ์'] = df.นิยมธรรมชาติอนุรักษ์.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['ละเอียดรอบคอบ'] = df.ละเอียดรอบคอบ.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['มีความสนใจในความรู้รอบตัว'] = df.มีความสนใจในความรู้รอบตัว.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['เจ้าคนนายคน'] = df.เจ้าคนนายคน.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['มีระเบียบวินัย'] = df.มีระเบียบวินัย.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['นิสัยไม่ยอมคน'] = df.นิสัยไม่ยอมคน.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['คนที่ทันคน'] = df.คนที่ทันคน.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['ชอบลองผิดลองถูก'] = df.ชอบลองผิดลองถูก.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['ชอบความท้าทาย'] = df.ชอบความท้าทาย.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['รักสนุก'] = df.รักสนุก.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['รักการอ่าน'] = df.รักการอ่าน.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['มีความเป็นตัวของตัวเองสูง'] = df.มีความเป็นตัวของตัวเองสูง.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['ขี้ระแวง'] = df.ขี้ระแวง.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['เข้าสังคมเก่ง'] = df.เข้าสังคมเก่ง.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['มีความหนักแน่นมั่นคง'] = df.มีความหนักแน่นมั่นคง.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['เป็นคนสองบุคลิก'] = df.เป็นคนสองบุคลิก.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['รักสวยรักงาม'] = df.รักสวยรักงาม.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['ไม่ชอบเที่ยวโลดโผน'] = df.ไม่ชอบเที่ยวโลดโผน.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['ไม่ค่อยกล้าแสดงออก'] = df.ไม่ค่อยกล้าแสดงออก.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['ใจกว้าง'] = df.ใจกว้าง.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['ขี้เกรงใจคนอื่น'] = df.ขี้เกรงใจคนอื่น.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['มองโลกในหลายแง่มุม'] = df.มองโลกในหลายแง่มุม.replace(['ใช่', 'ไม่ใช่'], [1, 0])
df['เป็นคนโรแมนติก'] = df.เป็นคนโรแมนติก.replace(['ใช่', 'ไม่ใช่'], [1, 0])


# X = df.drop(['ประทับเวลา','ประเภทเกมส์ที่สนใจ','นำประสบการณ์มาปรับใช้','ความจำดี'],axis=1) #features
X = df[cols]
y = df['ประเภทเกมส์ที่สนใจ']  # label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

from sklearn.preprocessing import StandardScaler

ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)

ss_test = StandardScaler()
X_test = ss_test.fit_transform(X_test)

models = {}

# Logistic Regression
from sklearn.linear_model import LogisticRegression

models['Logistic Regression'] = LogisticRegression()

# Support Vector Machines
from sklearn.svm import LinearSVC

models['Support Vector Machines'] = LinearSVC()

# Decision Trees
from sklearn.tree import DecisionTreeClassifier

models['Decision Trees'] = DecisionTreeClassifier()

# Random Forest
from sklearn.ensemble import RandomForestClassifier

models['Random Forest'] = RandomForestClassifier()

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

models['Naive Bayes'] = GaussianNB()

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

models['K-Nearest Neighbor'] = KNeighborsClassifier()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy, precision, recall, f1 = {}, {}, {}, {}

for key in models.keys():
    # Fit the classifier
    models[key].fit(X_train, y_train)

    # Make predictions
    predictions = models[key].predict(X_test)

    # Calculate metrics
    accuracy[key] = accuracy_score(predictions, y_test)
    precision[key] = precision_score(predictions, y_test, average='macro')
    recall[key] = recall_score(predictions, y_test, average='macro')
    f1[key] = f1_score(predictions, y_test, average='macro')

    df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall', 'F1'])
    df_model['Accuracy'] = accuracy.values()
    df_model['Precision'] = precision.values()
    df_model['Recall'] = recall.values()
    df_model['F1'] = f1.values()

    print(df_model)