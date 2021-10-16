import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df1 = pd.read_csv('./res/lossLSTM.csv')
LSTMlosses = df1['loss']  
LSTMaccuracies = df1['categorical_accuracy']

df2 = pd.read_csv('./res/lossGRU.csv')
GRUlosses = df2['loss']  
GRUaccuracies = df2['categorical_accuracy']

epoch_num = min(len(df1['epoch']),len(df2['epoch']))
lstm_epochs = df1['epoch'].head(epoch_num)
lstm_loss = LSTMlosses.head(epoch_num)
lstm_acc = LSTMaccuracies.head(epoch_num)

gru_loss = GRUlosses.head(epoch_num)
gru_acc = GRUaccuracies.head(epoch_num)

plt.figure(1)
plt.plot(lstm_epochs, lstm_loss, linewidth=1.2)
plt.plot(lstm_epochs, gru_loss, color='g', linewidth=1.2)
plt.title('Training Loss Curves of LSTM vs GRU')
plt.xlabel('Epochs')
plt.ylabel('Magnitude')
plt.axhline(y=0.25, color='r', linewidth=1, linestyle='-')
plt.legend(['LSTM Loss','GRU Loss'])
plt.show()

plt.figure(2)
plt.plot(lstm_epochs, lstm_acc, linewidth=1.2)
plt.plot(lstm_epochs, gru_acc, color='g', linewidth=1.2)
plt.title('Training Accuracy Curves of LSTM vs GRU')
plt.xlabel('Epochs')
plt.ylabel('Magnitude')
plt.legend(['LSTM Accuracy','GRU Accuracy'])
plt.show()


