import argparse
import glob, os
from datetime import datetime
import cv2
import numpy as np
from det import HarrisCornerDetector, standardize, normalize, g2rgb, match_corners

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default=1, help="1-6")
# parser.add_argument('--seed', type=int, default='-1', help="Random Seed for initialization")
parser.add_argument('--show', type=int, default='0', help="0/1/2, 0=Only Panorama, 1=Panorama+Matching Points, 2=Panorama+Matching Points+All Corners")
parser.add_argument('--scale', type=int, default='5', help="(H,W)/scale, as a lot of images are very large (more than 3000*2000)")
parser.add_argument('--patch_size', type=int, default='60', help="x: a patch of size (x,x) with patch_spacing is matched around corners")
parser.add_argument('--patch_spacing', type=int, default='5', help="spacing: a patch of size (x,x) with patch_spacing is matched around corners")

args = parser.parse_args()

all_images = glob.glob(f'CV_assignment_2_dataset/{args.dataset}/*')
# all_images = glob.glob(f'New Dataset/{args.dataset}/*.jpg')
optimal_thresholds = {
	1: 0.20,
	2: 0.05,
	3: 0.05,
	4: 0.05,
	5: 0.05,
	6: 0.05,
	7: 0.05,
	8: 0.05,
}

corners_frame = []
images = []
panorama = None
shape0 = None
foldername = f"results/{args.dataset}/" + datetime.now().strftime('%d-%b-%y - %H.%M.%S') 
os.makedirs(foldername)

for i, imgfile in enumerate(all_images):
	img = cv2.imread(imgfile)
	# if i==0:
	shape = [img.shape[0]//args.scale, img.shape[1]//args.scale]
	img = cv2.resize(img, shape)
	images.append(img)

	resimg, corners = HarrisCornerDetector(img, args, 0.06, optimal_thresholds[args.dataset])
	corners_frame.append(corners)
	# print(imgfile,':',len(corners[0]))


	if i==0:
		panorama = img
		continue
	pan_resimg, pan_corners = HarrisCornerDetector(panorama, args, 0.06, optimal_thresholds[args.dataset])
	panorama = match_corners(panorama, img, pan_corners, corners, args.dataset, args)

	if args.show >= 2:
		# cv2.imshow('img | resimg', np.concatenate([img, resimg], 1))
		cv2.imshow('pan_c', np.concatenate([pan_resimg], 1))
		# cv2.imshow('d', np.concatenate([out2,  cv2.convertScaleAbs(out2)], 1))
	if args.show >= 1:
		cv2.imshow('pan', np.concatenate([panorama], 1))
		cv2.waitKey(0)

	cv2.imwrite(foldername + f'/pan_{i}.png', panorama)


