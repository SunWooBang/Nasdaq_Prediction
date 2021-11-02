import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import datetime
import tensorflow as tf
from google.colab import drive
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# 랜덤시드 고정
np.random.seed(3)


drive.mount('/content/gdrive')
data = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/stock/nas16_20.csv')
data = data.dropna(axis=0)
print(data.head())
print(data.tail())
print('\n\n')


#데이터 정규화
close_value = data['Close'].values
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_value.reshape(-1, 1))


# 데이터셋 정의
seq_len = 50            # 50개 데이터를 가지고
seq_length = seq_len+1  # 다음날 예측(51일째)
ans = []

for i in range(len(scaled_data) - seq_length):
    ans.append(scaled_data[i:i+seq_length])

ans = np.array(ans)

print('\n\n')
print('<분리된 데이터 확인>')
print("전체 데이터 Length : ", len(data))
print("나눈 데이터 Length : ", len(ans))
print("나눈 데이터 0 번째 값: \n", ans[0])
print("\n\n 나눈 데이터 1번째 값: \n",ans[1])
print('\n\n')

row = int(round(ans.shape[0] * 0.7))  #7대3로
train = ans[:row, :]
np.random.shuffle(train)  #shuffle을 사용해서 데이터를 섞어주면 학습이 좋아진다. 고 한다..



# 데이터셋 분류
#train data와 test data를 1차원으로, 각 데이터 0~49번째 가격정보는 x_data, 마지막 50번째는 y_data로 설정
x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]

x_test = ans[row:, -1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = ans[row:, -1]

print('<데이터셋 분류>')
print(x_train.shape, x_test.shape)
print('\n\n')

# Early Stopping
callbacks_list = [
    EarlyStopping(
    monitor='val_loss',
    patience = 25,
    )
]


#모델
#dropout 사용해서 과적합 방지.
#사용 데이터의 기간이 길기에 신경망을 복잡하게 만듬
model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape=(50,1)))
model.add(LSTM(128, return_sequences = False)) #128
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu')) #16
model.add(Dropout(0.4)) #0.3
model.add(Dense(1, activation='relu'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary() #모델의 개략적인 스트럭쳐를 알려줌
print('\n\n')


#모델 학습.
#loss가 작을 수록 학습이 잘되는 것. 그러나 오버피팅 조심!!
hist = model.fit(x_train, y_train,
          validation_data = (x_test, y_test),
          batch_size = 40, epochs = 1000, callbacks=callbacks_list)



#학습 시각화
plt.figure(figsize=(12, 4))
fig, loss_ax = plt.subplots()
fig = plt.figure(facecolor='White')
loss_ax.plot(hist.history['loss'],'g', label='train loss')
loss_ax.plot(hist.history['val_loss'],'b', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='lower left')
plt.show()


#예측 시각화
pred = model.predict(x_test)
fig = plt.figure(facecolor='White')
plt.title('Prediction')
fig.set_size_inches(12, 5)
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()


#RMSE 및 y_predict
pred = scaler.inverse_transform(pred.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
rmse = np.sqrt(mean_squared_error(pred, y_test))
print("\nRMSE: %3f \n"%(rmse))

df = pd.DataFrame(y_test)
df.insert(1,'y_predict',pred)
df.rename(columns={0:'y_test'}, inplace=True)
print(df)
print('\n\n')


#미래 주가 예측
y_test = scaler.fit_transform(y_test.reshape(-1, 1))
step = 50 # 50개 데이터로
future = 14 #향후14일 예측하기
lastData = y_test

dx = np.copy(lastData)
estimate = [dx[-1]]
for i in range(future):
    px = dx[-step:].reshape(1, step, 1) #마지막 step만큼 입력데이터로 다음 값을 예측
    yhat = model.predict(px)[0][0]      #다음 값 예측
    estimate.append(yhat)               #예측값 저장
    dx = np.insert(dx, len(dx), yhat)   #예측값 저장 후 다음 예측



#미래예측 시각화
ax1 = np.arange(1, len(lastData) + 1)
ax2 = np.arange(len(lastData), len(lastData) + len(estimate))
plt.figure(figsize=(12, 5))
plt.title('Future Prediction')
plt.plot(ax1, lastData, color='dodgerblue', label='Time series', linewidth=1.5)
plt.plot(ax2, estimate, 'b-o', color='orange', markersize=1, label='Prediction')
plt.axvline(x=ax1[-1],  linestyle='dashed', linewidth=1)
plt.legend()
plt.show()

estimate = np.array(estimate)
predictions = scaler.inverse_transform(estimate.reshape(-1, 1))
print(predictions[:])



# 학습이 너무 안되면 Early Stopping 에서 patience를 조정할 것.
# 모델의 신경망 층이 많아서 학습이 오래걸림 -> 다만 단순화 할 경우보다 좋은 결과.
# batch_size를 크게하는 것도 하나의 방법일지도.