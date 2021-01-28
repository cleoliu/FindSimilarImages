from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import sys
import networkDrive
##不顯示log
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


#--自目錄將所有JEG檔案, 分析每張圖片像素點轉成矩陣--
def loadfile(filepath):                         
    picNameList=[] #存放所有圖片List
    vggInputList=[] #存放VGG矩陣List

    ##圖庫中所有圖片List
    folderAllFile = os.listdir(filepath) 
    for img_path in folderAllFile:
        if img_path.endswith(".jpg"):
            picNameList.append(img_path)
            img = image.load_img(filepath + img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            if len(vggInputList) > 0:
                vggInputList = np.concatenate((vggInputList,x))
            else:
                vggInputList = x

    return picNameList, vggInputList

#---訓練model--
def train(vggInputList):                    
    model = VGG16(weights='imagenet', include_top=False)
    # model.save('my_model.h5')
    print(model.to_json())
    return model

#--從model載入--
def loadmodel(model, picNameList, vggInputList):   
    # model = load_model('my_model.h5')
    ##使用預訓練模型進行圖像分類
    features = model.predict(vggInputList)
    ##計算相似矩陣
    features_compress = features.reshape(len(picNameList),7*7*512)
    sim = cosine_similarity(features_compress)
    return sim

#--計算餘弦相似度--
def cosine_similarity(ratings):             
    sim = ratings.dot(ratings.T)
    if not isinstance(sim, np.ndarray):
        sim = sim.toarray()
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

#--尋找相似圖排名
def choosePic(sim, picNameList):
    inputNo = np.random.randint(low=0, high=len(picNameList), size=1)[0]
    top = np.argsort(-sim[inputNo], axis=0)[1:2]
    recommend = [picNameList[i] for i in top]
    
    print(picNameList[inputNo]) #目標圖
    print(recommend[0]) #找到最像的圖


def main(count):
    #連線網路磁碟機
    localfilepath = networkDrive.ConnectNetworkDrive()
    #自目錄將所有JEG檔案, 分析每張圖片像素點轉成矩陣
    picNameList, vggInputList = loadfile(localfilepath)
    #將所有圖片訓練為model
    model = train(vggInputList)
    sim = loadmodel(model, picNameList, vggInputList)
    #從model隨機找一張圖片,輸出相似圖
    for i in range(0, int(count) , 1):
        choosePic(sim, picNameList)
    #中斷網路磁碟
    networkDrive.InterruptNetworkDrive()


if __name__ == "__main__":
    count = sys.argv[1]
    try:
        main(count)
    except:
        os.system("net use * /delete /yes") ##強置中斷網路磁碟
        print("已重新連線至網路磁碟")
        main(count)

'''
指令輸入格式:
python mapPic.py {需要組數} {圖片資料夾路徑}
範例:
python mapPic.py 2 \\192.168.10.251\share\_sevenstar\20200106
'''