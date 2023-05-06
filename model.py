import sklearn
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# load data
X_train_e = np.load(r'data\Embedding_feature_train.npy')
X_test_e = np.load(r'data\Embedding_feature_test.npy')
X_train_b = np.load(r'data\Bow-feature_train.npy')
X_test_b = np.load(r'data\Bow-feature_test.npy')
Y_train = np.load(r'data\train_label.npy')
Y_test = np.load(r'data\test_label.npy')

# train random forest classifier
def train_and_test(model, train, test):
    model.fit(train[0], train[1])
    preds = model.predict(test[0])
    acc = accuracy_score(test[1], preds)


svme = SVC(C = 10, kernel='rbf')
svmb = SVC(C = 1, kernel='linear')
base_estimator1 = DecisionTreeClassifier(max_depth=60, random_state=42)
base_estimator2 = DecisionTreeClassifier(max_depth=100, random_state=42)
adab = AdaBoostClassifier(estimator=base_estimator1, n_estimators=100, learning_rate=0.1, random_state=42)
adae = AdaBoostClassifier(estimator=base_estimator2, n_estimators=100, learning_rate=0.05, random_state=42)
rfcb = RandomForestClassifier(n_estimators=100, max_features='sqrt', bootstrap=False, n_jobs=-1, random_state=42)
rfce = RandomForestClassifier(criterion='entropy', n_estimators=100, max_features='sqrt', bootstrap=False, n_jobs=-1, random_state=42)

rfcb.fit(X_train_b, Y_train)
svmb.fit(X_train_b, Y_train)
adab.fit(X_train_b, Y_train)
rfce.fit(X_train_e, Y_train)
svme.fit(X_train_e, Y_train)
adae.fit(X_train_e, Y_train)
# print([estimator.tree_.max_depth for estimator in rfcb.estimators_])
# print([estimator.tree_.max_depth for estimator in rfce.estimators_])

y_pred_b_rfc = rfcb.predict(X_test_b)
y_pred_e_rfc = rfce.predict(X_test_e)
y_pred_b_svm = svmb.predict(X_test_b)
y_pred_e_svm = svme.predict(X_test_e)
y_pred_b_ada = adab.predict(X_test_b)
y_pred_e_ada = adae.predict(X_test_e)

cm_rfc_b = confusion_matrix(Y_test, y_pred_b_rfc)
cm_rfc_e = confusion_matrix(Y_test, y_pred_e_rfc)
cm_svm_b = confusion_matrix(Y_test, y_pred_b_svm)
cm_svm_e = confusion_matrix(Y_test, y_pred_e_svm)
cm_ada_b = confusion_matrix(Y_test, y_pred_b_ada)
cm_ada_e = confusion_matrix(Y_test, y_pred_e_ada)

acc_rfc_b = accuracy_score(Y_test, y_pred_b_rfc)
acc_rfc_e = accuracy_score(Y_test, y_pred_e_rfc)
acc_svm_b = accuracy_score(Y_test, y_pred_b_svm)
acc_svm_e = accuracy_score(Y_test, y_pred_e_svm)
acc_ada_b = accuracy_score(Y_test, y_pred_b_ada)
acc_ada_e = accuracy_score(Y_test, y_pred_e_ada)
# report_b = classification_report(Y_test, Y_pred_b)
# report_e = classification_report(Y_test, Y_pred_e)

print('The acc of RFC with Bow and Embedding is: {} and {}'.format(acc_rfc_b, acc_rfc_e))
print('The acc of SVM with Bow and Embedding is: {} and {}'.format(acc_svm_b, acc_svm_e))
print('The acc of SVM with Bow and Embedding is: {} and {}'.format(acc_ada_b, acc_ada_e))

# random forest 
disp_b = ConfusionMatrixDisplay(confusion_matrix=cm_rfc_b, display_labels=rfcb.classes_)
disp_b.plot()
plt.title('random forest-20 test data-bow')
plt.savefig(r'result\cm_rfc_b.png')
plt.show()

disp_e = ConfusionMatrixDisplay(confusion_matrix=cm_rfc_e, display_labels=rfce.classes_)
disp_e.plot()
plt.title('random forest-20 test data-w2v')
plt.savefig(r'result\cm_rfc_e.png')
plt.show()


# support vector 
disp_b = ConfusionMatrixDisplay(confusion_matrix=cm_svm_b, display_labels=svmb.classes_)
disp_b.plot()
plt.title('svm-20 test data-bow')
plt.savefig(r'result\cm_svm_b.png')
plt.show()

disp_e = ConfusionMatrixDisplay(confusion_matrix=cm_svm_e, display_labels=svme.classes_)
disp_e.plot()
plt.title('svm-20 test data-w2v')
plt.savefig(r'result\cm_svm_e.png')
plt.show()


# adaboost
disp_b = ConfusionMatrixDisplay(confusion_matrix=cm_ada_b, display_labels=adab.classes_)
disp_b.plot()
plt.title('adaboost-20 test data-bow')
plt.savefig(r'result\cm_ada_b.png')
plt.show()

disp_e = ConfusionMatrixDisplay(confusion_matrix=cm_ada_e, display_labels=adae.classes_)
disp_e.plot()
plt.title('adaboost-20 test data-w2v')
plt.savefig(r'result\cm_ada_e.png')
plt.show()


class_labels = np.load(r'data\class_name.npy')
class_nums = np.load(r'data\class_nums.npy')
for i in range(len(class_labels)):
    class_labels[i] = class_labels[i]+' '+str(class_nums[i])
    if class_labels[i] == 'Emergency Response':
        class_labels[i] = class_labels[i][0:-4]+' '+str(class_nums[i])

plt.bar(class_labels, np.diag(cm_rfc_b))
plt.xticks(rotation=30, fontsize=8)
plt.yticks(range(0, 20, 2))
plt.ylim(0, 20)

plt.bar(class_labels, np.diag(cm_rfc_e))
plt.xticks(rotation=45, fontsize=8)
plt.yticks(range(0, 30, 2))
plt.ylim(0, 20)
plt.title('random forest-20 test data acc')
plt.legend(['Bag of Words','FastText Embedding'])
plt.savefig(r'result\rndf_clf_acc.png')
plt.show()


plt.bar(class_labels, np.diag(cm_svm_b))
plt.yticks(range(0, 30, 2))
plt.xticks(rotation=45, fontsize=8)
plt.ylim(0, 20)

plt.bar(class_labels, np.diag(cm_svm_e))
plt.xticks(rotation=30, fontsize=8)
plt.yticks(range(0, 20, 2))
plt.ylim(0, 20)
plt.title('svm-20 test data acc')
plt.legend(['Bag of Words','FastText Embedding'])
plt.savefig(r'result\svm_clfs_acc.png')
plt.show()


plt.bar(class_labels, np.diag(cm_ada_b))
plt.yticks(range(0, 30, 2))
plt.xticks(rotation=45, fontsize=8)
plt.ylim(0, 20)

plt.bar(class_labels, np.diag(cm_ada_e))
plt.xticks(rotation=30, fontsize=8)
plt.yticks(range(0, 20, 2))
plt.ylim(0, 20)
plt.title('adaboost-20 test data acc')
plt.legend(['Bag of Words','FastText Embedding'])
plt.savefig(r'result\adaboost_acc.png')
plt.show()
