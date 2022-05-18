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

st.subheader('Data')
st.text('(Taft et al., 2021)')
#st.write(df5)


#########################

from PIL import Image
image = Image.open('F1.large.jpg')

# https://www.biorxiv.org/content/10.1101/2021.05.24.445464v1.full
st.subheader('The ProteinBERT architecture')
st.image(image, caption='(Nadav Brandes et al., 2021) ProteinBERT’s architecture is inspired by BERT. Unlike standard Transformers, ProteinBERT supports both local (sequential) and global data. The model consists of 6 transformer blocks manipulating local (left side) and global (right side) representations. Each such block manipulates these representations by fully-connected and convolutional layers (in the case of local representations), with skip connections and normalization layers between them. The local representations affect the global representations through a global attention layer, and the global representations affect the local representations through a broadcast fully-connected layer.')


#########################

d1 = {'test_loss': 0.4330556094646454, 'test_accuracy': 0.8322731837964513, 'test_f1': 0.8404966571155682, 'test_precision': 0.7923169267707083, 'test_recall': 0.8949152542372881, 'test_runtime': 167.9181, 'test_samples_per_second': 17.788, 'test_steps_per_second': 1.781}
d2 = {'test_loss': 0.4330556094646454, 'test_accuracy': 0.8322731837964513, 'test_f1': 0.8404966571155682, 'test_precision': 0.7923169267707083, 'test_recall': 0.8949152542372881, 'test_runtime': 167.9181, 'test_samples_per_second': 17.788, 'test_steps_per_second': 1.781}
d3 = {'test_loss': 0.4330556094646454, 'test_accuracy': 0.8322731837964513, 'test_f1': 0.8404966571155682, 'test_precision': 0.7923169267707083, 'test_recall': 0.8949152542372881, 'test_runtime': 167.9181, 'test_samples_per_second': 17.788, 'test_steps_per_second': 1.781}

df1 = pd.DataFrame.from_dict(d1, orient='index', columns=['train']).T
df2 = pd.DataFrame.from_dict(d1, orient='index', columns=['validation']).T
df3 = pd.DataFrame.from_dict(d1, orient='index', columns=['test']).T

df4 = df1.append(df2).append(df3)[['test_accuracy', 'test_f1', 'test_precision', 'test_recall']]
df4.columns = ['accuracy', 'f1', 'precision', 'recall']

st.subheader('Accuracy')
st.write(df4)



