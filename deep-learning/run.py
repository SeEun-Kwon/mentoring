import os
import cv2
from network import CNN
import torch
import numpy as np

# 입력 폴더에 분류할 이미지 저장

path = 'input_images'

print('=== 언어 종류 분류 프로그램 ===\n\n')
print('분류할 언어가 포함된 이미지를 input_images 폴더에 저장한 후 0을 입력하세요.\n')
inp = input('')



# model
model = CNN(4)
model.load_state_dict(torch.load('model.pt'))

img_names = os.listdir('input_images')
if inp == '0':
    for name in img_names:
        img = cv2.imread(os.path.join(path, name)).astype(np.float32)
        img = torch.tensor(img).unsqueeze(dim=0).permute(0, 3, 1, 2)
        out = model(img)[0]
        pred = np.argmax(out.detach().numpy())
        if pred == 0:
            lan = '중국어'
        elif pred == 1:
            lan = '영어'
        elif pred == 2:
            lan = '일본어'
        elif pred == 3:
            lan = '한국어'

        print(f'이미지 이름: {name}, 예측된 언어: {lan}, 예측 확률: {max(out)}')
