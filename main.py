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

# 2. 부분집합(Subset) 추출 (Train: 5000, Test: 1000)
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


# ---------------------------------------------------------
# 5. 5-Fold Cross Validation 수행
# ---------------------------------------------------------
num_folds = 5
k_choices = [1, 3, 5, 7, 9]

# 훈련 데이터를 5개의 폴드(Fold)로 분할
X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

# 각 K값에 대한 5번의 정확도를 저장할 딕셔너리
k_to_accuracies = {k: [] for k in k_choices}

classifier = KNearestNeighbor()

print("\nStarting 5-Fold Cross Validation (This may take a few minutes)...")
# 5번의 폴드에 대해 반복
for i in range(num_folds):
    # 검증(Validation) 폴드 지정
    X_val_fold = X_train_folds[i]
    y_val_fold = y_train_folds[i]
    
    # 훈련(Training) 폴드 지정 (검증 폴드를 제외한 나머지 4개 폴드 병합)
    X_train_fold = np.concatenate(X_train_folds[:i] + X_train_folds[i+1:])
    y_train_fold = np.concatenate(y_train_folds[:i] + y_train_folds[i+1:])
    
    # 4개의 폴드로 모델 훈련
    classifier.train(X_train_fold, y_train_fold)
    
    # L2 거리를 사용하여 검증 폴드와의 거리 계산 (L1으로 변경 가능)
    dists = classifier.compute_distances_l2(X_val_fold)
    
    # 각 K 값에 대해 검증 정확도 계산 및 저장
    for k in k_choices:
        y_val_pred = classifier.predict_labels(dists, k=k)
        accuracy = np.mean(y_val_pred == y_val_fold)
        k_to_accuracies[k].append(accuracy)
    
    print(f"Finished fold {i+1}/{num_folds}")

# ---------------------------------------------------------
# 6. 평균 정확도 계산 및 그래프 시각화 (Plotting)
# ---------------------------------------------------------
print("\n--- Cross Validation Results ---")
mean_accuracies = []

for k in sorted(k_choices):
    accuracies = k_to_accuracies[k]
    mean_acc = np.mean(accuracies)
    mean_accuracies.append(mean_acc)
    print(f"K = {k}, Mean Accuracy = {mean_acc * 100:.2f}%")

# 최적의 K 찾기
best_k = k_choices[np.argmax(mean_accuracies)]
print(f"\n=> Best K selected from Cross Validation is: {best_k}")

# 그래프 그리기
plt.figure(figsize=(10, 6))

# 각 K마다 5개의 fold 정확도를 점(scatter)으로 찍어줌
for k in k_choices:
    plt.scatter([k] * len(k_to_accuracies[k]), k_to_accuracies[k], color='blue', alpha=0.5)

# 평균 정확도를 꺾은선 그래프(line)로 그림
plt.plot(k_choices, mean_accuracies, marker='o', color='red', linestyle='-', linewidth=2, label='Mean Accuracy')

plt.title('Cross-Validation on K')
plt.xlabel('K value')
plt.ylabel('Mean cross-validation accuracy')
plt.xticks(k_choices)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 7. 최적의 K로 전체 훈련 데이터(5000장) 학습 및 최종 평가
# ---------------------------------------------------------
print(f"\nRetraining the model on all 5000 training images with Best K={best_k}...")
classifier.train(X_train, y_train)

# 테스트 데이터(1000장)에 대해 평가
dists_test = classifier.compute_distances_l2(X_test)
best_y_pred = classifier.predict_labels(dists_test, k=best_k)

final_accuracy = np.mean(best_y_pred == y_test)
print(f"Final Test Accuracy with K={best_k}: {final_accuracy * 100:.2f}%")

# 최적의 결과로 혼동 행렬 시각화
print(f"\nDisplaying final Confusion Matrix...")
cm = confusion_matrix(y_test, best_y_pred)

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cifar10_classes)
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
plt.title(f"Final Confusion Matrix (K={best_k})")
plt.tight_layout()
plt.show()