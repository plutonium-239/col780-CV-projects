import numpy as np
from find_intrinsic_params import find_intrinsic_matrix
from overlay_object import overlay
import cv2
import json
import glob
import imutils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--color', action='store_true', default=False, help="To give colors to faces")
parser.add_argument('--model', type=int, default=-1, help="Which model to show: -1=Simple Cube (default) 0=Prism 1=Cybertruck 2=Building 3=Katana")
parser.add_argument('--scale', type=float, default=3, help="Scale of model (default=3)")
args = parser.parse_args()

get_xy = lambda s: list(map(float, s.split(",")))
all_images = glob.glob('photos/resized/*.jpg')
points_coords = json.load(open('photos/resized/points_images.json'))

num_images = len(all_images)

P_stacked_imgs = []
for img in points_coords:
	# P_stacked = [get_xy(s)[::-1]+[1] for s in points_coords[img]]
	P_stacked = [get_xy(s)+[1] for s in points_coords[img]]
	P_stacked_imgs.append(P_stacked)
P_stacked_imgs = np.array(P_stacked_imgs)

X_stacked = []
points_real_coords = json.load(open('photos/points_real.json'))
X_stacked= [get_xy(points_real_coords[pt])+[1] for pt in points_real_coords]
X_stacked_imgs = np.array([X_stacked]*num_images)
X_stacked_imgs2= X_stacked_imgs.copy()
X_stacked_imgs2[:,:,2] = 0

np.set_printoptions(suppress=True)

K, all_Hs = find_intrinsic_matrix(P_stacked_imgs[:3,:,:], X_stacked_imgs[:3,:,:])
# K, all_Hs = find_intrinsic_matrix(P_stacked_imgs, X_stacked_imgs)
print('K', K)

# color = []
# import random
# for _ in range(P_stacked_imgs.shape[1]):
# 	color.append([random.randint(0,255), random.randint(0,255), random.randint(128,255)])

# all_Hs_new = [cv2.findHomography(X_stacked_imgs[i], P_stacked_imgs[i], cv2.RANSAC, 5.0)[0] for i in range(num_images)]
# for i in range(num_images):
# # 	print('H', all_Hs[i])
# # 	print('Hcv', all_Hs_new[i])
# 	print(all_images[i])
# 	img = cv2.imread(all_images[i])
# 	for j in range(P_stacked_imgs[i].shape[0]):
# 		x1, y1 = int(P_stacked_imgs[i,j,0]), int(P_stacked_imgs[i,j,1])
# 		print(x1,y1)
# 		img[x1-7:x1+7, y1-7:y1+7] = color[j]
# 	# img[96, 841] = color[0]
# 	cv2.imshow('img', imutils.resize(img, width=800))
# 	cv2.waitKey(0)

model_names = {
	-1: 'simple',
	0: '3dmodels/Triangular_Prism/20255_Triangular_Prism_V1.obj',
	1: '3dmodels/cybertruck/Tesla Cybertruck.obj',
	2: '3dmodels/building/Building - 6.obj',
	3: '3dmodels/katana/CYBERPUNK KATANA.obj'
} 

# if args.model > -1:
_, all_Hs = find_intrinsic_matrix(P_stacked_imgs[:num_images,:,:], X_stacked_imgs[:num_images,:,:])
overlay(model_names[args.model], all_images[:num_images], all_Hs, X_stacked_imgs2, P_stacked_imgs, args.color, scale=args.scale)
