import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont


# 人脸识别器
class FaceRecognizer:
    # 初始化，加载数据
    def __init__(self):
        # 选择设备
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(self.device))

        # 读取训练好的人脸特征数据
        self.data = pd.read_csv('face_feature.csv')
        self.x = self.data.drop(columns=['label'])
        self.y = self.data['label']

        # 字体文件，用于在图片上正确显示中文
        self.font = ImageFont.truetype('simsun.ttc', size=30)

        # 加载训练好的KNN分类器模型
        self.knn_model = joblib.load('knn_model.pkl')

    # 根据特征向量识别人脸，使用欧氏距离，如果距离大于1则认为识别失败
    def _recognize(self, v):
        dis = np.sqrt(sum((v[0] - self.x.iloc[0]) ** 2))
        name = self.y[0]

        for i in range(1, self.x.shape[0]):
            temp_dis = np.sqrt(sum((v[0] - self.x.iloc[i]) ** 2))
            if temp_dis < dis:
                dis = temp_dis
                name = self.y[i]

        return name, dis

    # 人脸识别主函数
    def start_recognize(self):
        # mtcnn检测人脸位置
        mtcnn = MTCNN(device=self.device, keep_all=True)
        # 用于生成人脸512维特征向量
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # 初始化视频窗口
        windows_name = 'face'
        cv2.namedWindow(windows_name)
        cap = cv2.VideoCapture(0)

        while True:
            # 从摄像头读取一帧图像
            success, image = cap.read()
            if not success:
                break

            img = Image.fromarray(image)
            draw = ImageDraw.Draw(img)

            # 检测人脸位置
            boxes, probs = mtcnn.detect(image)
            if boxes is not None:
                for box, prob in zip(boxes, probs):
                    # 设置人脸检测阈值
                    if prob < 0.9:
                        continue

                    x1, y1, x2, y2 = [int(p) for p in box]
                    # 框出人脸位置
                    draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
                    # cv2.rectangle(image, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), color=(0, 255, 0), thickness=2)
                    # cv2.putText(image, str(round(prob, 3)), (x1, y1 - 30), cv2.FONT_ITALIC, 1, (255, 0, 255), 4)

                    # 导出人脸图像
                    face = mtcnn.extract(image, [box], None).to(self.device)
                    # 生成512维特征向量
                    embeddings = resnet(face).detach().cpu().numpy()
                    # KNN预测
                    name_knn = self.knn_model.predict(embeddings)

                    # 获得预测姓名和距离
                    _, dis = self._recognize(embeddings)
                    # 如果距离过大则认为识别失败
                    if dis > 0.9:
                        # 框出人脸位置并写上名字
                        draw.rectangle((x1, y1, x2, y2), outline=(0, 0, 255), width=2)
                        draw.text((x1, y1 - 40), f'未知', font=self.font, fill=(0, 0, 255))
                        # cv2.putText(image, 'unknown', (x1 - 20, y1 - 20), cv2.FONT_ITALIC, 1, (255, 0, 255), 4)
                    else:
                        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
                        draw.text((x1, y1 - 40), f'{name_knn[0]}({round(dis, 2)})', font=self.font, fill=(0, 255, 0))
                        # cv2.putText(image, f'{name_knn[0]}({round(dis, 2)})', (x1 - 20, y1 - 20),
                        #             cv2.FONT_ITALIC, 1, (255, 0, 255), 4)

            # 显示处理后的图片
            cv2.imshow(windows_name, np.array(img))

            # 保持窗口
            key = cv2.waitKey(1)
            if key & 0xff == 27:
                break

        # 释放设备资源，销毁窗口
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        fr = FaceRecognizer()
        fr.start_recognize()
    except Exception as e:
        print(e)
