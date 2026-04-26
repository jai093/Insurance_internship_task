import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# load data
df = pd.read_csv('insurance.csv')

# encode
df['Smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df['sex_female'] = (df['sex'] == 'female').astype(int)
df['sex_male'] = (df['sex'] == 'male').astype(int)
region_dict = {'southwest': 0, 'northwest': 1, 'northeast': 2, 'southeast': 3}
df['Region'] = df['region'].map(region_dict)

# features & target (log transform charges to match original model)
X = df[['age', 'bmi', 'children', 'Smoker', 'sex_female', 'sex_male', 'Region']]
y = np.log(df['charges'])

# scale age and bmi
scaler = MinMaxScaler()
X = X.copy()
X[['age', 'bmi']] = scaler.fit_transform(X[['age', 'bmi']])

# train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# save model and scaler
pickle.dump(model, open('model_gb.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("Model and scaler saved successfully.")
print(f"R2 score on test set: {model.score(X_test, y_test):.4f}")
