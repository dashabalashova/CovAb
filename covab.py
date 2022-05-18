import streamlit as st
import pandas as pd
import numpy as np

#!pip install sklearn
#from sklearn.model_selection import train_test_split

st.title('Predictive profiling of SARS-CoV-2 variants by Transformer Neural Network')


#df = pd.read_csv('data/LY16_train_data.csv')
#df_train, df_val = train_test_split(df, test_size=0.1, random_state=18)
#df_test = pd.read_csv('data/LY16_test_data.csv')


#print(df_train.shape[0], df_train.Label.mean().round(3))
#print(df_val.shape[0], df_val.Label.mean().round(3))
#print(df_test.shape[0], df_test.Label.mean().round(3))

#df5 = pd.DataFrame([[df_train.shape[0], df_train.Label.mean().round(3)], [df_val.shape[0], df_val.Label.mean().round(3)], \
#                   [df_test.shape[0], df_test.Label.mean().round(3)]], index = ['train', 'validation', 'test'], \
#                  columns = ['count', 'ratio'])

df5 = pd.DataFrame([[24192, 0.5], [2689, 0.503], \
                   [2987, 0.494]], index = ['train', 'validation', 'test'], \
                  columns = ['count', 'ratio'])

st.subheader('Data')
st.text('(Taft et al., 2021)')
st.write(df5)


#########################

from PIL import Image
image = Image.open('F1.large.jpg')

# https://www.biorxiv.org/content/10.1101/2021.05.24.445464v1.full
st.subheader('The ProteinBERT architecture')
st.image(image, caption='(Nadav Brandes et al., 2021) ProteinBERT’s architecture is inspired by BERT. Unlike standard Transformers, ProteinBERT supports both local (sequential) and global data. The model consists of 6 transformer blocks manipulating local (left side) and global (right side) representations. Each such block manipulates these representations by fully-connected and convolutional layers (in the case of local representations), with skip connections and normalization layers between them. The local representations affect the global representations through a global attention layer, and the global representations affect the local representations through a broadcast fully-connected layer.')


#########################

d1 = {'test_loss': 0.4495324492454529, 'test_accuracy': 0.8339120370370371, 'test_f1': 0.8371037055055541, 'test_precision': 0.8219745222929936, 'test_recall': 0.8528002643317363, 'test_runtime': 1393.3884, 'test_samples_per_second': 17.362, 'test_steps_per_second': 1.737}
d2 = {'test_loss': 0.4339987635612488, 'test_accuracy': 0.8438081071030122, 'test_f1': 0.8484848484848484, 'test_precision': 0.828752642706131, 'test_recall': 0.8691796008869179, 'test_runtime': 155.6353, 'test_samples_per_second': 17.278, 'test_steps_per_second': 1.728}
d3 = {'test_loss': 0.4493945240974426, 'test_accuracy': 0.8336123200535654, 'test_f1': 0.8350481247925656, 'test_precision': 0.8179453836150845, 'test_recall': 0.8528813559322034, 'test_runtime': 172.8313, 'test_samples_per_second': 17.283, 'test_steps_per_second': 1.73}

df1 = pd.DataFrame.from_dict(d1, orient='index', columns=['train']).T
df2 = pd.DataFrame.from_dict(d1, orient='index', columns=['validation']).T
df3 = pd.DataFrame.from_dict(d1, orient='index', columns=['test']).T

df4 = df1.append(df2).append(df3)[['test_accuracy', 'test_f1', 'test_precision', 'test_recall']]
df4.columns = ['accuracy', 'f1', 'precision', 'recall']

st.subheader('Accuracy')
st.write(df4)



