from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import sklearn.metrics as metrics
import datasample
import matplotlib.pyplot as plt

# 거리에 따라 weight 유무에 따른 차이 확인

# RandomOverSampling data
Y_train_rand = np.ravel(datasample.Y_train_ros)
#Y_test_rand = np.ravel(Y_test_rand)
# ADASYN data
Y_train_ada = np.ravel(datasample.Y_train_ada)
#Y_test_ada = np.ravel(Y_test_ada)
# SMOTE data
Y_train_smote = np.ravel(datasample.Y_train_smt)
#Y_test_smote = np.ravel(Y_test_smote)

print("distance weight O")
print("--------------------------k=1--------------------------")
# k=1인 거리에 따라 가중치가 주어지는 KNN, 거리 측정 방식: 유클리드
classifier = KNeighborsClassifier(n_neighbors=1, weights='distance')

# 학습 후 test 하기
model_rand = classifier.fit(datasample.X_train_ros, Y_train_rand)
# 정확도 측정
#score_rand = model_rand.score(X_test, Y_test_rand)
# 예측 값
y_pred_rand = model_rand.predict(datasample.X_test_ros)

print("RandomOverSampling")
#print(metrics.accuracy_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.precision_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.recall_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.f1_score(datasample.Y_test_ros, y_pred_rand))

# 학습 후 test 하기
model_ada = classifier.fit(datasample.X_train_ada, Y_train_ada)
# 정확도 측정
#score_ada = model_ada.score(X_test, Y_test_ada)
y_pred_ada = model_ada.predict(datasample.X_test_ada)

print("ADASYN")
#print(metrics.accuracy_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.precision_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.recall_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.f1_score(datasample.Y_test_ada, y_pred_ada))

# 학습 후 test 하기
model_smote = classifier.fit(datasample.X_train_smt, Y_train_smote)
# 정확도 측정
#score_smote = model_smote.score(X_test, Y_test_smote)
y_pred_smote = model_smote.predict(datasample.X_test_smt)

print("SMOTE")
#print(metrics.accuracy_score(datasample.Y_test_smt, y_pred_smote))
print(metrics.precision_score(datasample.Y_test_smt, y_pred_smote))
print(metrics.recall_score(datasample.Y_test_smt, y_pred_smote))
print(metrics.f1_score(datasample.Y_test_smt, y_pred_smote))
print("")


print("--------------------------k=3--------------------------")
# k=3인 거리에 따라 가중치가 주어지는 KNN, 거리 측정 방식: 유클리드
classifier = KNeighborsClassifier(n_neighbors=3, weights='distance')

# 학습 후 test 하기
model_rand = classifier.fit(datasample.X_train_ros, Y_train_rand)
# 정확도 측정
#score_rand = model_rand.score(X_test, Y_test_rand)
# 예측 값
y_pred_rand = model_rand.predict(datasample.X_test_ros)

print("RandomOverSampling")
#print(metrics.accuracy_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.precision_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.recall_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.f1_score(datasample.Y_test_ros, y_pred_rand))

# 학습 후 test 하기
model_ada = classifier.fit(datasample.X_train_ada, Y_train_ada)
# 정확도 측정
#score_ada = model_ada.score(X_test, Y_test_ada)
y_pred_ada = model_ada.predict(datasample.X_test_ada)

print("ADASYN")
#print(metrics.accuracy_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.precision_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.recall_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.f1_score(datasample.Y_test_ada, y_pred_ada))

# 학습 후 test 하기
model_smote = classifier.fit(datasample.X_train_smt, Y_train_smote)
# 정확도 측정
#score_smote = model_smote.score(X_test, Y_test_smote)
y_pred_smote = model_smote.predict(datasample.X_test_smt)

print("SMOTE")
#print(metrics.accuracy_score(datasample.Y_test_smt, y_pred_smote))
print(metrics.precision_score(datasample.Y_test_smt, y_pred_smote))
print(metrics.recall_score(datasample.Y_test_smt, y_pred_smote))
print(metrics.f1_score(datasample.Y_test_smt, y_pred_smote))
print("")

print("--------------------------k=5--------------------------")
# k=5인 거리에 따라 가중치가 주어지는 KNN, 거리 측정 방식: 유클리드
classifier = KNeighborsClassifier(n_neighbors=5, weights='distance')

# 학습 후 test 하기
model_rand = classifier.fit(datasample.X_train_ros, Y_train_rand)
# 정확도 측정
#score_rand = model_rand.score(X_test, Y_test_rand)
# 예측 값
y_pred_rand = model_rand.predict(datasample.X_test_ros)

print("RandomOverSampling")
#print(metrics.accuracy_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.precision_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.recall_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.f1_score(datasample.Y_test_ros, y_pred_rand))

# 학습 후 test 하기
model_ada = classifier.fit(datasample.X_train_ada, Y_train_ada)
# 정확도 측정
#score_ada = model_ada.score(X_test, Y_test_ada)
y_pred_ada = model_ada.predict(datasample.X_test_ada)

print("ADASYN")
#print(metrics.accuracy_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.precision_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.recall_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.f1_score(datasample.Y_test_ada, y_pred_ada))

# 학습 후 test 하기
model_smote = classifier.fit(datasample.X_train_smt, Y_train_smote)
# 정확도 측정
#score_smote = model_smote.score(X_test, Y_test_smote)
y_pred_smote = model_smote.predict(datasample.X_test_smt)

print("SMOTE")
#print(metrics.accuracy_score(datasample.Y_test_smt, y_pred_smote))
print(metrics.precision_score(datasample.Y_test_smt, y_pred_smote))
print(metrics.recall_score(datasample.Y_test_smt, y_pred_smote))
print(metrics.f1_score(datasample.Y_test_smt, y_pred_smote))
print("")

print("--------------------------k=10--------------------------")
# k=10인 거리에 따라 가중치가 주어지는 KNN, 거리 측정 방식: 유클리드
classifier = KNeighborsClassifier(n_neighbors=10, weights='distance')

# 학습 후 test 하기
model_rand = classifier.fit(datasample.X_train_ros, Y_train_rand)
# 정확도 측정
#score_rand = model_rand.score(X_test, Y_test_rand)
# 예측 값
y_pred_rand = model_rand.predict(datasample.X_test_ros)

print("RandomOverSampling")
#print(metrics.accuracy_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.precision_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.recall_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.f1_score(datasample.Y_test_ros, y_pred_rand))

# 학습 후 test 하기
model_ada = classifier.fit(datasample.X_train_ada, Y_train_ada)
# 정확도 측정
#score_ada = model_ada.score(X_test, Y_test_ada)
y_pred_ada = model_ada.predict(datasample.X_test_ada)

print("ADASYN")
#print(metrics.accuracy_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.precision_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.recall_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.f1_score(datasample.Y_test_ada, y_pred_ada))

# 학습 후 test 하기
model_smote = classifier.fit(datasample.X_train_smt, Y_train_smote)
# 정확도 측정
#score_smote = model_smote.score(X_test, Y_test_smote)
y_pred_smote = model_smote.predict(datasample.X_test_smt)

print("SMOTE")
#print(metrics.accuracy_score(datasample.Y_test_smt, y_pred_smote))
print(metrics.precision_score(datasample.Y_test_smt, y_pred_smote))
print(metrics.recall_score(datasample.Y_test_smt, y_pred_smote))
print(metrics.f1_score(datasample.Y_test_smt, y_pred_smote))
print("")


print("distance weight X")
print("--------------------------k=1--------------------------")
# k=1인 거리에 따라 가중치가 주어지는 KNN, 거리 측정 방식: 유클리드
classifier = KNeighborsClassifier(n_neighbors=1)

# 학습 후 test 하기
model_rand = classifier.fit(datasample.X_train_ros, Y_train_rand)
# 정확도 측정
#score_rand = model_rand.score(X_test, Y_test_rand)
# 예측 값
y_pred_rand = model_rand.predict(datasample.X_test_ros)

print("RandomOverSampling")
#print(metrics.accuracy_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.precision_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.recall_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.f1_score(datasample.Y_test_ros, y_pred_rand))

# 학습 후 test 하기
model_ada = classifier.fit(datasample.X_train_ada, Y_train_ada)
# 정확도 측정
#score_ada = model_ada.score(X_test, Y_test_ada)
y_pred_ada = model_ada.predict(datasample.X_test_ada)

print("ADASYN")
#print(metrics.accuracy_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.precision_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.recall_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.f1_score(datasample.Y_test_ada, y_pred_ada))

# 학습 후 test 하기
model_smote = classifier.fit(datasample.X_train_smt, Y_train_smote)
# 정확도 측정
#score_smote = model_smote.score(X_test, Y_test_smote)
y_pred_smote = model_smote.predict(datasample.X_test_smt)

print("SMOTE")
#print(metrics.accuracy_score(datasample.Y_test_smt, y_pred_smote))
print(metrics.precision_score(datasample.Y_test_smt, y_pred_smote))
print(metrics.recall_score(datasample.Y_test_smt, y_pred_smote))
print(metrics.f1_score(datasample.Y_test_smt, y_pred_smote))
print("")


print("--------------------------k=3--------------------------")
# k=3인 거리에 따라 가중치가 주어지는 KNN, 거리 측정 방식: 유클리드
classifier = KNeighborsClassifier(n_neighbors=3)

# 학습 후 test 하기
model_rand = classifier.fit(datasample.X_train_ros, Y_train_rand)
# 정확도 측정
#score_rand = model_rand.score(X_test, Y_test_rand)
# 예측 값
y_pred_rand = model_rand.predict(datasample.X_test_ros)

print("RandomOverSampling")
#print(metrics.accuracy_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.precision_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.recall_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.f1_score(datasample.Y_test_ros, y_pred_rand))

# 학습 후 test 하기
model_ada = classifier.fit(datasample.X_train_ada, Y_train_ada)
# 정확도 측정
#score_ada = model_ada.score(X_test, Y_test_ada)
y_pred_ada = model_ada.predict(datasample.X_test_ada)

print("ADASYN")
#print(metrics.accuracy_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.precision_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.recall_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.f1_score(datasample.Y_test_ada, y_pred_ada))

# 학습 후 test 하기
model_smote = classifier.fit(datasample.X_train_smt, Y_train_smote)
# 정확도 측정
#score_smote = model_smote.score(X_test, Y_test_smote)
y_pred_smote = model_smote.predict(datasample.X_test_smt)

print("SMOTE")
#print(metrics.accuracy_score(datasample.Y_test_smt, y_pred_smote))
print(metrics.precision_score(datasample.Y_test_smt, y_pred_smote))
print(metrics.recall_score(datasample.Y_test_smt, y_pred_smote))
print(metrics.f1_score(datasample.Y_test_smt, y_pred_smote))
print("")

print("--------------------------k=5--------------------------")
# k=5인 거리에 따라 가중치가 주어지는 KNN, 거리 측정 방식: 유클리드
classifier = KNeighborsClassifier(n_neighbors=5)

# 학습 후 test 하기
model_rand = classifier.fit(datasample.X_train_ros, Y_train_rand)
# 정확도 측정
#score_rand = model_rand.score(X_test, Y_test_rand)
# 예측 값
y_pred_rand = model_rand.predict(datasample.X_test_ros)

print("RandomOverSampling")
#print(metrics.accuracy_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.precision_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.recall_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.f1_score(datasample.Y_test_ros, y_pred_rand))

# 학습 후 test 하기
model_ada = classifier.fit(datasample.X_train_ada, Y_train_ada)
# 정확도 측정
#score_ada = model_ada.score(X_test, Y_test_ada)
y_pred_ada = model_ada.predict(datasample.X_test_ada)

print("ADASYN")
#print(metrics.accuracy_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.precision_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.recall_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.f1_score(datasample.Y_test_ada, y_pred_ada))

# 학습 후 test 하기
model_smote = classifier.fit(datasample.X_train_smt, Y_train_smote)
# 정확도 측정
#score_smote = model_smote.score(X_test, Y_test_smote)
y_pred_smote = model_smote.predict(datasample.X_test_smt)

print("SMOTE")
#print(metrics.accuracy_score(datasample.Y_test_smt, y_pred_smote))
#print(metrics.precision_score(datasample.Y_test_smt, y_pred_smote))
#print(metrics.recall_score(datasample.Y_test_smt, y_pred_smote))
print(metrics.f1_score(datasample.Y_test_smt, y_pred_smote))
print("")

print("--------------------------k=10--------------------------")
# k=10인 거리에 따라 가중치가 주어지는 KNN, 거리 측정 방식: 유클리드
classifier = KNeighborsClassifier(n_neighbors=10)

# 학습 후 test 하기
model_rand = classifier.fit(datasample.X_train_ros, Y_train_rand)
# 정확도 측정
#score_rand = model_rand.score(X_test, Y_test_rand)
# 예측 값
y_pred_rand = model_rand.predict(datasample.X_test_ros)

print("RandomOverSampling")
#print(metrics.accuracy_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.precision_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.recall_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.f1_score(datasample.Y_test_ros, y_pred_rand))

# 학습 후 test 하기
model_ada = classifier.fit(datasample.X_train_ada, Y_train_ada)
# 정확도 측정
#score_ada = model_ada.score(X_test, Y_test_ada)
y_pred_ada = model_ada.predict(datasample.X_test_ada)

print("ADASYN")
#print(metrics.accuracy_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.precision_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.recall_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.f1_score(datasample.Y_test_ada, y_pred_ada))

# 학습 후 test 하기
model_smote = classifier.fit(datasample.X_train_smt, Y_train_smote)
# 정확도 측정
#score_smote = model_smote.score(X_test, Y_test_smote)
y_pred_smote = model_smote.predict(datasample.X_test_smt)

print("SMOTE")
#print(metrics.accuracy_score(datasample.Y_test_smt, y_pred_smote))
#print(metrics.precision_score(datasample.Y_test_smt, y_pred_smote))
#print(metrics.recall_score(datasample.Y_test_smt, y_pred_smote))
print(metrics.f1_score(datasample.Y_test_smt, y_pred_smote))
print("")

print("--------------------------k=20--------------------------")
# k=20인 거리에 따라 가중치가 주어지는 KNN, 거리 측정 방식: 유클리드
classifier = KNeighborsClassifier(n_neighbors=20)

# 학습 후 test 하기
model_rand = classifier.fit(datasample.X_train_ros, Y_train_rand)
# 정확도 측정
#score_rand = model_rand.score(X_test, Y_test_rand)
# 예측 값
y_pred_rand = model_rand.predict(datasample.X_test_ros)

print("RandomOverSampling")
#print(metrics.accuracy_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.precision_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.recall_score(datasample.Y_test_ros, y_pred_rand))
print(metrics.f1_score(datasample.Y_test_ros, y_pred_rand))

# 학습 후 test 하기
model_ada = classifier.fit(datasample.X_train_ada, Y_train_ada)
# 정확도 측정
#score_ada = model_ada.score(X_test, Y_test_ada)
y_pred_ada = model_ada.predict(datasample.X_test_ada)

print("ADASYN")
#print(metrics.accuracy_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.precision_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.recall_score(datasample.Y_test_ada, y_pred_ada))
print(metrics.f1_score(datasample.Y_test_ada, y_pred_ada))

# 학습 후 test 하기
model_smote = classifier.fit(datasample.X_train_smt, Y_train_smote)
# 정확도 측정
#score_smote = model_smote.score(X_test, Y_test_smote)
y_pred_smote = model_smote.predict(datasample.X_test_smt)

print("SMOTE")
#print(metrics.accuracy_score(datasample.Y_test_smt, y_pred_smote))
#print(metrics.precision_score(datasample.Y_test_smt, y_pred_smote))
#print(metrics.recall_score(datasample.Y_test_smt, y_pred_smote))
print(metrics.f1_score(datasample.Y_test_smt, y_pred_smote))
print("")
