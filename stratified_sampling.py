import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

def get_dataset_partitions_pd(df, train_split=0.8, val_split=0.1, test_split=0.1, target_variable=None):
    assert (train_split + test_split + val_split) == 1
    
    # Only allows for equal validation and test splits
    assert val_split == test_split 

    # Shuffle
    df_sample = df.sample(frac=1, random_state=12)

    # Specify seed to always have the same split distribution between runs
    # If target variable is provided, generate stratified sets
    if target_variable is not None:
      grouped_df = df_sample.groupby(target_variable)
      arr_list = [np.split(g, [int(train_split * len(g)), int((1 - val_split) * len(g))]) for i, g in grouped_df]

      train_ds = pd.concat([t[0] for t in arr_list])
      val_ds = pd.concat([t[1] for t in arr_list])
      test_ds = pd.concat([v[2] for v in arr_list])

    else:
      indices_or_sections = [int(train_split * len(df)), int((1 - val_split) * len(df))]
      train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)
    
    return train_ds, val_ds, test_ds

dataset = datasets.load_iris()
X = pd.DataFrame(dataset.data)
y = pd.DataFrame(dataset.target)
print(f'Distribution in original set:  \n{y[0].value_counts().sort_index() / len(y)}')

train_ds, val_ds, test_ds = get_dataset_partitions_pd(y)
print(f'Distribution in training set: \n{train_ds[0].value_counts().sort_index() / len(train_ds)}\n\n'+
      f'Distribution in validation set: \n{val_ds[0].value_counts().sort_index() / len(val_ds)}\n\n'+
      f'Distribution in testing set: \n{test_ds[0].value_counts().sort_index() / len(test_ds)}')

train_ds, val_ds, test_ds = get_dataset_partitions_pd(y, target_variable=0)
print(f'Distribution in training set: \n{train_ds[0].value_counts().sort_index() / len(train_ds)}\n\n'+
      f'Distribution in validation set: \n{val_ds[0].value_counts().sort_index() / len(val_ds)}\n\n'+
      f'Distribution in testing set: \n{test_ds[0].value_counts().sort_index() / len(test_ds)}')
     