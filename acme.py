# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
import pandas as pd
import calendar


# %%
df = pd.read_csv("term-deposit-marketing-2020.csv")


# %%
temp = df["y"].map(dict(yes=1, no=0))
df.drop("y", 1, inplace=True)
df.insert(0, "y", temp)

df.head(10)

# %% [markdown]
# ## Feature Extraction
# #### At first glance, there are two features that I wish to extract:
# - Explicit feature of whether balance is negative or positive, so that it isn't lost by normalization
# - Circular values for time features, such as day of month and month of year. Using sine and cosine values representing time fetaures that loop back to their beginning, we can capture that beggining and end of sequences are in fact close. January is in fact close to December.

# %%
# Is balance negative
df["neg_balance"] = df["balance"] < 0


# %%
# Convert textual month data to integers 1-12
month_map = {m.lower(): i for i, m in enumerate(calendar.month_abbr)}
df["month"] = df["month"].apply(lambda x: month_map[x])

df.head()


# %%
day_in_month = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}

# Accumulate circular day features for each month seperately, Since each month has varying mount of days.
days_sin = pd.Series(dtype="float64")
days_cos = pd.Series(dtype="float64")

for month in day_in_month.keys():
    days = df[df["month"] == month]["day"]
    sin = np.sin((days - 1) * (2.0 * np.pi / day_in_month[month]))
    cos = np.cos((days - 1) * (2.0 * np.pi / day_in_month[month]))

    days_sin = pd.concat([days_sin, sin])
    days_cos = pd.concat([days_cos, cos])

days_sin.shape


# %%
df["days_sin"] = days_sin
df["days_cos"] = days_cos

df["month_sin"] = np.sin((df["month"] - 1) * (2.0 * np.pi / len(day_in_month)))
df["month_cos"] = np.cos((df["month"] - 1) * (2.0 * np.pi / len(day_in_month)))

df.drop(["day", "month"], 1, inplace=True)
df.head(20)


# %%
df.nunique()


# %%
df.describe()

# %% [markdown]
# ## Preprocess
# We split our train and test sets, normalize the numeric features, and one-hot encode categoricals.
# One-hot encoding is picked since the dataset is small, and categories aren't wide anyways. We can afford the performance hit of one-hot encoding. Even binary features are one-hot encoded, to better visualize which value is actually important.

# %%
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

print(train_data.shape)
print(test_data.shape)


# %%
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric = [
    "age",
    "balance",
    "duration",
    "campaign",
    "days_sin",
    "days_cos",
    "month_sin",
    "month_cos",
]
categorical = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "neg_balance",
]

one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)

one_hot.fit(df[categorical])
one_hot.categories_


# %%
categorical_features_train = one_hot.transform(train_data[categorical])
categorical_features_test = one_hot.transform(test_data[categorical])

categorical_features_train.shape, categorical_features_test.shape


# %%
normalize = StandardScaler()
normalize.fit(df[numeric])

numeric_features_train = normalize.transform(train_data[numeric])
numeric_features_test = normalize.transform(test_data[numeric])
numeric_features_train.shape, numeric_features_test.shape


# %%
train_features = np.concatenate(
    (numeric_features_train, categorical_features_train), axis=1
)
test_features = np.concatenate(
    (numeric_features_test, categorical_features_test), axis=1
)

train_features.shape, test_features.shape

# %% [markdown]
# #### Name the one-hot encoded features with format: 
# _categoricalname\_categoricalvalue_
# 
# Also maintain a list of lists for the new categorical features, for feature importance graphs

# %%
# name new features
expanded_features = numeric.copy()
grouped_features = numeric.copy()
one_hot_features = [
    [f"{feature}_{c}" for c in cats]
    for feature, cats in zip(categorical, one_hot.categories_)
]
for new_features in one_hot_features:
    expanded_features.extend(new_features)

grouped_features.extend(one_hot_features)
expanded_features


# %%
train_label, test_label = np.array(train_data["y"]), np.array(test_data["y"])
train_label.shape, test_label.shape

# %% [markdown]
# ## Model training and validation
# I have tried a couple sklearn models in a naive manner, and found that *GradientBoosting* is a top contender. Since it also allows for strong feature importance insights, I decided to use it.

# %%
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score


# %%
# Try GradientBoosting with deafult parameters
grad_clf = GradientBoostingClassifier()
rand_scores = cross_val_score(grad_clf, train_features, train_label, cv=5)
rand_mean = rand_scores.mean()

y_predict = cross_val_predict(grad_clf, train_features, train_label, cv=5)
print(f"Gradient Boost accuracy is {accuracy_score(train_label, y_predict)}")

print(confusion_matrix(train_label, y_predict))


# %%
grad_clf.get_params()


# %%
from sklearn.model_selection import GridSearchCV

params = dict(
    max_depth=[3, 5, 7, 10],
    n_estimators=[20, 50, 100, 200],
)

clf = GridSearchCV(
    estimator=GradientBoostingClassifier(),
    param_grid=params,
    cv=5,
    n_jobs=-1,
)

clf.fit(train_features, train_label)

print(clf.best_score_)
hypers = clf.best_params_
hypers


# %%
from sklearn.metrics import accuracy_score, roc_auc_score

final_clf = GradientBoostingClassifier(**hypers)

final_clf.fit(train_features, train_label)

test_predict = final_clf.predict(test_features)
print(f"The final accuracy is: {accuracy_score(test_label, test_predict)}")

# %% [markdown]
# ## Feature Importance
# A plot encompassing all features reveals that numeric features are much more significant than categorical ones.
# Last call duration is by far the most important feature. Followed by the circular time features. 
# I believe time features can be augmented further, deriving more seasonality. Without a year value, day of week cannot be calculated. And I don't have much time to deep dive into more features.
# 
# Categorical and numerical features are also plotted individually, to reveal which value is a stronger signal

# %%
# Calculate feature importance
import matplotlib.pyplot as plt

f_imp = final_clf.feature_importances_

y_pos = np.arange(len(expanded_features))

plt.figure(figsize=(15, 10)) 

plt.bar(y_pos, f_imp, align='edge', width=0.3, alpha=0.5)
plt.xticks(y_pos, expanded_features, rotation=45, ha='right')
plt.ylabel('Importance')
plt.title('Feature Importances (All)')

plt.show()


# %%
# Numerical features

y_pos = np.arange(len(numeric))

plt.bar(y_pos, f_imp[:len(numeric)], align='edge', width=0.3, alpha=0.5)
plt.xticks(y_pos, expanded_features[:len(numeric)], rotation=45, ha='right')
plt.ylabel('Importance')
plt.title('Feature Importances (Numeric)')

plt.show()


# %%
current_index = len(numeric)

for name, subcats in zip(categorical, grouped_features[len(numeric):]):
    next_index = current_index + len(subcats)

    cat_importances = f_imp[current_index:next_index]

    y_pos = np.arange(len(subcats))

    plt.bar(y_pos, cat_importances, align='edge', width=0.3, alpha=0.5)
    plt.xticks(y_pos, subcats, rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.title(f'Feature Importances ({name})')

    plt.show()
    current_index = next_index


# %%



