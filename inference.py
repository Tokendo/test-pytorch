import torch
import cv2
import torchvision.transforms as transforms
import argparse
from model import CNNModel, SmallCNN
import os
os.environ["XDG_SESSION_TYPE"] = "xcb"

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', 
    default='/home/marla/repo/test pytorch/input/4.bmp',
    help='path to the input image')
args = vars(parser.parse_args())

#the computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
# list containing all the class labels
labels = [
    'ligne','mat'
    ]

    # initialize the model and load the trained weights
model = CNNModel().to(device)
checkpoint = torch.load('/home/marla/repo/test pytorch/outputs/model.pth', map_location=device,weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# define preprocess transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])  


# read and preprocess the image
image = cv2.imread(args['input'])
# get the ground truth class
orig_image = image.copy()
# convert to RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = transform(image)
# add batch dimension
image = torch.unsqueeze(image, 0)
with torch.no_grad():
    outputs = model(image.to(device))
    sm = torch.nn.functional.softmax(outputs, dim=1)
    probability1 = sm.data.max(1, keepdim=True)[0].item()
    probability2 = sm.data.max(1, keepdim=True)[1].item()

    
print(round(probability1*100,2))
print(sm.data)
output_label = torch.topk(outputs, 1)
pred_class = labels[int(output_label.indices)]

cv2.putText(orig_image, 
    f"Pred: {pred_class}",
    (10, 55),
    0, 
    0.6, (0, 0, 255), 2, cv2.LINE_AA
)
cv2.putText(orig_image, 
    f"prob 1: {probability1}",
    (10, 80),
    0, 
    0.6, (0, 0, 255), 2, cv2.LINE_AA
)
cv2.putText(orig_image, 
    f"prob 2: {probability2}",
    (10, 100),
    0, 
    0.6, (0, 0, 255), 2, cv2.LINE_AA
)

cv2.imshow('Result', orig_image)
cv2.waitKey(0)
cv2.imwrite(f"/home/marla/repo/test pytorch/outputs/{gt_class}{args['input'].split('/')[-1].split('.')[0]}.png",
    orig_image)