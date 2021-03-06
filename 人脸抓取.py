import cv2
import os


def get_face(name: str, num: int):
    cv2.namedWindow('face')
    cap = cv2.VideoCapture(0)
    classfier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    if not os.path.exists(f'faces/{name}'):
        os.makedirs(f'faces/{name}')

    count = 0
    while True:
        success, image = cap.read()

        # 将当前帧转换成灰度图像
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=5, minSize=(32, 32))
        if len(faceRects) > 0:  # 大于0则检测到人脸
            for faceRect in faceRects:  # 单独框出每一张人脸
                x, y, w, h = faceRect

                # 将当前人脸保存为图片
                face_image_name = f'faces/{name}/{name}_{count}.jpg'
                print(face_image_name)
                face_image = image[y: y + h + 10, x: x + w]
                cv2.imwrite(face_image_name, face_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

                count += 1
                if count >= num:
                    break

                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, f'Count:{count}', (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)

        if count >= num:
            break

        cv2.imshow('face', image)
        key = cv2.waitKey(10)
        if key & 0xFF == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        get_face('mamingda', 200)
    except Exception as e:
        print(e)
