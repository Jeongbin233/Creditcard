from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import datasample as dp


# datasample.py에서 데이터 불러오기
# standard scaler 이용
X_train_ada = dp.X_train_ada
X_train_smt = dp.X_train_smt
X_train_ros = dp.X_train_ros
Y_train_ada = dp.Y_train_ada
Y_train_smt = dp.Y_train_smt
Y_train_ros = dp.Y_train_ros

# # test data 오버샘플링한 데이터
X_test_smt = dp.X_test_smt
X_test_ada = dp.X_test_ada
X_test_ros = dp.X_test_ros
Y_test_smt = dp.Y_test_smt
Y_test_ada = dp.Y_test_ada
Y_test_ros = dp.Y_test_ros

# # test data 그대로인 데이터
X_test = dp.X_test
Y_test = dp.Y_test

# 모델링
model_ada = LogisticRegression(class_weight={0:1, 1:0.01})
model_smt = LogisticRegression(class_weight={0:1, 1:0.01})
model_ros = LogisticRegression(class_weight={0:1, 1:0.01})
model_ada_t = LogisticRegression()
model_smt_t = LogisticRegression()
model_ros_t = LogisticRegression()

model_ada.fit(X_train_ada, Y_train_ada)
model_smt.fit(X_train_smt, Y_train_smt)
model_ros.fit(X_train_ros, Y_train_ros)
model_ada_t.fit(X_train_ada, Y_train_ada)
model_smt_t.fit(X_train_smt, Y_train_smt)
model_ros_t.fit(X_train_ros, Y_train_ros)


# 각 모델의 성능 확인
print("Logistic Regression, ADAYN, class_weight=0:1, 1:0.01")
print(model_ada.score(X_test, Y_test))
Y_test_ada_pre = model_ada.predict(X_test)
print(classification_report(Y_test,Y_test_ada_pre))

print("Logistic Regression, SMOTE, class_weight=0:1, 1:0.01")
print(model_smt.score(X_test, Y_test))
Y_test_smt_pre = model_smt.predict(X_test)
print(classification_report(Y_test,Y_test_smt_pre))

print("Logistic Regression, Random, class_weight=0:1, 1:0.01")
print(model_ros.score(X_test, Y_test))
Y_test_ros_pre = model_ros.predict(X_test)
print(classification_report(Y_test,Y_test_ros_pre))

print("Logistic Regression, ADAYN, test data oversampling")
print(model_ada_t.score(X_test_ada, Y_test_ada))
Y_test_ada_t_pre = model_ada_t.predict(X_test_ada)
print(classification_report(Y_test_ada,Y_test_ada_t_pre))

print("Logistic Regression, SMOTE, test data oversampling")
print(model_smt_t.score(X_test_smt, Y_test_smt))
Y_test_smt_t_pre = model_smt_t.predict(X_test_smt)
print(classification_report(Y_test_smt,Y_test_smt_t_pre))

print("Logistic Regression, Random, test data oversampling")
print(model_ros_t.score(X_test_ros, Y_test_ros))
Y_test_ros_t_pre = model_ros_t.predict(X_test_ros)
print(classification_report(Y_test_ros,Y_test_ros_t_pre))

