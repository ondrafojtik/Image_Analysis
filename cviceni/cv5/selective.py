#http://www.huppelen.nl/publications/selectiveSearchDraft.pdf
import cv2
from torchvision import transforms
import torchvision.models as models
import torch

alex_net = models.alexnet(weights = "AlexNet_Weights.IMAGENET1K_V1").eval()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cv2.namedWindow("result", 0)

one_img = cv2.imread("2150758084.jpg")#cv2.imread("cute-shepherd-dog-posing-isolated-white-background.jpg")
one_img_res = cv2.resize(one_img, None, fx = 0.7, fy = 0.7 )
one_img_blur = cv2.medianBlur(one_img_res, 7)
one_img_res_paint = one_img_res.copy()

s_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
s_search.setBaseImage(one_img_blur)
#s_search.switchToSelectiveSearchQuality()
#s_search.switchToSelectiveSearchFast()
s_search.switchToSingleStrategy()

rects = s_search.process()
print(len(rects))

for i in range(0, len(rects)):
    x,y,w,h = rects[i]
    if w < 30 or h < 30: continue

    cropp = one_img_res[y:y+h, x:x+w]

    image_rgb = cv2.cvtColor(cropp, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(image_rgb)
    input_tensor = input_tensor.unsqueeze(0)
    output = alex_net(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 1)
    #cv2.putText()
    cat_name, cat_prob = categories[top5_catid], top5_prob.item()
    if cat_prob < 0.7: continue
    cv2.rectangle(one_img_res_paint, (x,y), (x+w, y+h), (0, 255, 0), 3)
    print(f"cat_name: {cat_name} : {cat_prob}")

    #cv2.imshow("result", one_img_res_point)
    #cv2.waitKey()


cv2.imshow("result", one_img_res_paint)
cv2.waitKey()

#mrl/data/ano2
