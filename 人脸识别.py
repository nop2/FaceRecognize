import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# 选择设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# 读取训练好的人脸特征数据
data = pd.read_csv('face_data.csv')
x = data.drop(columns=['label'])
y = data['label']


# 根据特征向量识别人脸，使用欧氏距离，如果距离大于1则认为识别失败
def recognize(v):
    dis = np.sqrt(sum((v[0] - x.iloc[0]) ** 2))
    name = y[0]

    for i in range(1, x.shape[0]):
        temp_dis = np.sqrt(sum((v[0] - x.iloc[i]) ** 2))
        if temp_dis < dis:
            dis = temp_dis
            name = y[i]

    return name, dis


def test():
    # mtcnn检测人脸位置
    mtcnn = MTCNN(device=device, keep_all=True)
    # 用于生成人脸512维特征向量
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # 加载训练好的KNN分类器模型
    # knn = joblib.load('knn_model.pkl')

    # 初始化视频窗口
    windows_name = 'face'
    cv2.namedWindow(windows_name)
    cap = cv2.VideoCapture(0)

    while True:
        # 从摄像头读取一帧图像
        success, image = cap.read()
        if not success:
            break

        # 检测人脸位置
        boxes, probs = mtcnn.detect(image)
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                # 设置人脸检测阈值
                if prob < 0.85:
                    continue

                x1, y1, x2, y2 = [int(p) for p in box]
                # 框出人脸位置
                cv2.rectangle(image, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), color=(0, 255, 0), thickness=2)
                # cv2.putText(image, str(round(prob, 3)), (x1, y1 - 30), cv2.FONT_ITALIC, 1, (255, 0, 255), 4)

                # 导出人脸图像
                face = mtcnn.extract(image, [box], None).to(device)
                # 生成512维特征向量
                embeddings = resnet(face).detach().cpu().numpy()
                # KNN预测
                # name = knn.predict(embeddings)
                # prop = knn.predict_proba(embeddings)

                # 获得预测姓名和距离
                name, dis = recognize(embeddings)
                # 如果距离大于0.99则识别失败
                if dis > 1.0:
                    name = 'unknown'
                # 将识别出的名字和距离显示在图像上
                cv2.putText(image, f'{name}({round(dis, 2)})', (x1 - 20, y1 - 20), cv2.FONT_ITALIC, 1, (255, 0, 255), 4)

        # 显示处理后的图片
        cv2.imshow(windows_name, image)

        # 保持窗口
        key = cv2.waitKey(1)
        if key & 0xff == ord(' '):
            break

    # 释放设备资源，销毁窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test()
