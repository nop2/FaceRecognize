from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier as KNN
import joblib

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def collate_fn(x):
    return x[0]


def train(path):
    dataset = datasets.ImageFolder(path)
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=0, batch_size=2)

    print(dataset.idx_to_class)

    aligned = []
    names = []
    i = 0
    for x, y in loader:
        try:
            x_aligned, prob = mtcnn(x, return_prob=True)
            if x_aligned is not None:
                # print('Face detected with probability: {:8f}'.format(prob))
                print(f'batch {i}')
                i += 1
                aligned.append(x_aligned)
                # print(type(x_aligned))
                # print(x_aligned)
                names.append(dataset.idx_to_class[y])
        except Exception as e:
            print(e)

    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu()

    a = pd.DataFrame(np.array(embeddings), index=names)
    a.reset_index(inplace=True)
    a.columns = ['label'] + [f'v{i}' for i in range(512)]

    return a


if __name__ == '__main__':
    face_data = pd.concat([train(r'F:\Code\Python\人脸识别\计科四班人脸数据库\Part1'),
                           train(r'F:\Code\Python\人脸识别\计科四班人脸数据库\Part2')])

    face_data.to_csv('face_feature.csv', index=False, encoding='utf8')

    x = face_data.drop(columns=['label'])
    y = face_data['label']
    knn = KNN(n_neighbors=5)
    knn.fit(x, y)

    if os.path.exists('knn_model.pkl'):
        os.remove('knn_model.pkl')

    joblib.dump(knn, 'knn_model.pkl')
    print('导出模型')
