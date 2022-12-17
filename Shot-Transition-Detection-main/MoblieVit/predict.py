import os,shutil
import json
import sys
sys.path.append('../')
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
from model import mobile_vit_xx_small as create_model

def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
def main(img_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    #img_path = "data/flower_photos/tulip.jpg"
    img_name=img_path.split("/")[-1]
    img_cv2 = cv2.imread(img_path, 1)
    assert  os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    # read class_indict
    json_path = 'D:\zstudy\Projec_badminton_video_segmetation\Shot-Transition-Detection-main\MoblieVit\class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=2).to(device)
    # load model weights
    model_weight_path = r"D:\zstudy\Projec_badminton_video_segmetation\Shot-Transition-Detection-main\MoblieVit\weights\best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    prob=0
    clfd=''
    for i in range(len(predict)):

        #print("class: {:10}   prob: {:.3}  for Predcit".format(class_indict[str(i)],predict[i].numpy()))
        if prob<predict[i].numpy():
            prob=predict[i].numpy()
            clfd=class_indict[str(i)]
        if i == len(predict)-1 and clfd=='1':
            #cv2.imwrite(r"data/classifd/{}".format(img_name), img_cv2)
            shutil.copy(img_path,r"D:\zstudy\Projec_badminton_video_segmetation\Shot-Transition-Detection-main\MoblieVit\data\classifd")
            #print(r"data/classifd/{}".format(img_name))
            #cv2.imwrite(os.path.join('D:\zstudy\Project_badminton\deep-learning-for-image-processing\pytorch_classification\MobileViT\data\classifd'+class_indict[str(i)],'waka.jpg'), img_cv2)
    #plt.show()

def predict_by_import(path):
    list_name=[]
    #path='data/to_classifd'   #文件夹路径
    listdir(path,list_name)
    #print(list_name)
    for i in range(len(list_name)):
        #print("#####################")
        main(list_name[i])



if __name__ == '__main__':
    list_name=[]
    path=r'D:\zstudy\Projec_badminton_video_segmetation\Shot-Transition-Detection-main\MoblieVit\data\to_classifd'   #文件夹路径
    listdir(path,list_name)
    #print(list_name)
    for i in range(len(list_name)):
        #print("#####################")
        main(list_name[i])
