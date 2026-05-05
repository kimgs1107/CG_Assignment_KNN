import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter

# 1. 로컬 CIFAR-10 데이터 로드 함수 정의 (Pickle 사용)
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        # 데이터를 10000 x 32 x 32 x 3 형태로 변환
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float32")
        Y = np.array(Y)
        return X, Y

def load_cifar10(root_folder):
    xs = []
    ys = []
    # 5개의 훈련 배치 파일 로드
    for b in range(1, 6):
        f = os.path.join(root_folder, f'data_batch_{b}')
        X, Y = load_cifar_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    # 테스트 배치 파일 로드
    Xte, Yte = load_cifar_batch(os.path.join(root_folder, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

# 데이터셋 폴더 경로 지정
dataset_dir = 'cifar-10-batches-py'

print(f"Loading CIFAR-10 dataset from local folder: {dataset_dir} ...")
try:
    X_train_full, y_train_full, X_test_full, y_test_full = load_cifar10(dataset_dir)
except FileNotFoundError:
    print(f"Error: '{dataset_dir}' 폴더를 찾을 수 없습니다. 파이썬 파일과 같은 위치에 데이터 폴더가 있는지 확인해주세요.")
    exit()

# 2. 부분집합(Subset) 추출
num_training = 5000
num_test = 1000

X_train = X_train_full[:num_training]
y_train = y_train_full[:num_training]
X_test = X_test_full[:num_test]
y_test = y_test_full[:num_test]

# 3. 이미지 평탄화 (Flatten: 32x32x3 -> 3072)
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# 4. KNN 클래스 수동 구현
class KNearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def compute_distances_l1(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i, :] = np.sum(np.abs(self.X_train - X[i, :]), axis=1)
        return dists

    def compute_distances_l2(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i, :] = np.sqrt(np.sum(np.square(self.X_train - X[i, :]), axis=1))
        return dists

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            y_pred[i] = Counter(closest_y).most_common(1)[0][0]
        return y_pred

# 5. 모델 훈련 및 평가 진행
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

k_choices = [1, 3, 5, 7, 9]
best_k = 0
best_y_pred = None
best_accuracy = 0

for metric in ['L1', 'L2']:
    print(f"\nComputing distances using {metric} metric...")
    if metric == 'L1':
        dists = classifier.compute_distances_l1(X_test)
    else:
        dists = classifier.compute_distances_l2(X_test)
        
    print(f"--- {metric} Results ---")
    for k in k_choices:
        y_test_pred = classifier.predict_labels(dists, k=k)
        accuracy = np.mean(y_test_pred == y_test)
        print(f"K = {k}, Accuracy = {accuracy * 100:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            best_y_pred = y_test_pred

# 6. 최적의 결과로 confusion matrix 시각화
print(f"\nDisplaying Confusion Matrix for the best model (K={best_k})...")
cm = confusion_matrix(y_test, best_y_pred)

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cifar10_classes)
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
plt.title(f"Confusion Matrix (Best K={best_k})")
plt.tight_layout()
plt.show()