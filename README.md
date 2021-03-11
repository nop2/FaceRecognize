# FaceRecognize
pytorch实现实时人脸识别，使用opencv+facenet+mtcnn+knn

核心功能使用facenet_pytorch库，Github https://github.com/timesler/facenet-pytorch

字体文件：https://github.com/sonatype/maven-guide-zh/raw/master/content-zh/src/main/resources/fonts/simsun.ttc

代码仅供参考

# 使用方法
1.运行"模型训练.py"，生成人脸特征向量CSV文件和KNN模型文件
2.运行"人脸识别.py"识别人脸
