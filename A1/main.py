import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from tqdm import tqdm
from gmm import OnlineGMM, NMS_vectorized, plot

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='3', help="0='Candela_m1.10', 1='CAVIAR1', 2='HallAndMonitor', 3='HighwayI', 4='IBMtest2'")
parser.add_argument('--seed', type=int, default='-1', help="Random Seed for initialization of gaussians")
parser.add_argument('--num_gaussians', type=int, default='-1', help="Number of gaussians")
parser.add_argument('--show', type=int, default='0', help="0/1/2")
parser.add_argument('--alpha', type=float, default='0.05', help="The exponentially decaying weight alpha")

args = parser.parse_args()


dataset_dict = {0:'Candela_m1.10', 1:'CAVIAR1', 2:'HallAndMonitor', 3:'HighwayI', 4:'IBMtest2', 5:'8'}
seed_dict = {0:3, 1:15, 2:7, 3:3, 4:13, 5:7}
gaussian_dict = {0:5, 1:3, 2:5, 3:3, 4:7, 5:3}
if args.seed == -1:
	args.seed = seed_dict[args.dataset]
if args.num_gaussians == -1:
	args.num_gaussians = gaussian_dict[args.dataset]
dataset = dataset_dict[args.dataset]

print(f'Dataset {dataset}\nUsing params:')
print(args)
# dataset_filename_dict = {0:'Candela_m1.10_', 1:'in', 2:'in', 3:'in', 4:'in'}
# dataset_filename = dataset_filename_dict[args.dataset]
filename = 'Candela_m1.10_' if args.dataset==0 else 'in'

video = cv2.VideoCapture(f'COL780 Dataset/{dataset}/input/{filename}%06d.png')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
h, w = int(video.get(3)), int(video.get(4))
out = cv2.VideoWriter(f'outputs/{dataset}-{args.seed}-{args.num_gaussians}.avi', fourcc, 10, (h, w))



r, X = video.read()
model = OnlineGMM(args)
# model = BackgroundSubtractorAGMM(num_gaussians=5)

count = 0
for count in tqdm(range(int(video.get(7)) - 1)):
	if not r:
		break	
	blur = cv2.GaussianBlur(X, (3,3), 2)
	prraw = model.step_frame(blur)
	predicted = (prraw*255/(args.num_gaussians-1)).astype('uint8')
	predicted = cv2.bitwise_not(predicted)
	# _,masked_img = cv2.threshold(predicted, 150, 255, cv2.THRESH_BINARY)
	# masked_img = masked_img.astype('uint8')
	# cv2.imshow('masked', masked_img)

	# blur = cv2.GaussianBlur(predicted, (3,3), 2)
	# cv2.imshow('blur', )

	# element = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
	# mask = cv2.erode(blur, element, iterations = 3)
	# element = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
	# mask = cv2.dilate(mask, element, iterations = 3)
	# mask = morphology.remove_small_objects(mask.astype(bool), min_size=32, connectivity=2).astype('uint8')
	# mask = cv2.erode(mask, element)
	# cv2.imshow('erode', mask)

	kernel = np.ones((7,7),np.uint8)
	mask = cv2.morphologyEx(predicted, cv2.MORPH_CLOSE, kernel)
	
	mask_threshd = cv2.threshold(mask, 255//((args.num_gaussians+1)/2), 255, cv2.THRESH_BINARY)[1].astype('uint8')

	# CONNECTED COMPONENTS
	nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(predicted, None, None, None, 8, cv2.CV_32S)
	areas = stats[1:,cv2.CC_STAT_AREA]
	result = np.zeros((labels.shape), np.uint8)
	for i in range(0, nlabels - 1):
		if areas[i] >= 100:   #keep
			result[labels == i + 1] = 255

	contours,_ = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
	img = cv2.cvtColor(predicted, cv2.COLOR_GRAY2RGB)
	# img = np.copy(X)
	# print(img.shape)
	# img = cv2.drawContours(img, contours, -1, (0, 255, 0))

	boxes = []
	for i, c in enumerate(contours):
		x,y,w,h = cv2.boundingRect(c)
		# if w*h > 50 and w/h < 3 and h/w < 3:
		if w*h > 50:
			boxes.append([x,y,w,h])
			# print(w*h)
	
	boxes = np.array(boxes)
	# start = time.time()
	# nms_boxes = NMS(boxes)
	# end = time.time()
	# tqdm.write(f'NMS took {end - start}')	

	# start = time.time()
	nms_boxes_2 = NMS_vectorized(boxes)
	# end = time.time()
	# tqdm.write(f'VecNMS took {end - start}')	
	# print(f'NMS: {b-a}\t NMS_vectorized: {d-c}')
	
	for x,y,w,h in nms_boxes_2:
		cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1)

	if args.show == 1:
		g2rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
		plt.imshow(np.concatenate([X, g2rgb(predicted), g2rgb(mask), img], 1), cmap='gray')
		plt.show()
	if args.show == 2:

		g2rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
		cv2.imshow('orig | pred | mask | ConnComps | box', 
			np.concatenate([X, g2rgb(predicted), g2rgb(result), img], 1))
		cv2.waitKey(0)
	
	# out.write(prraw)
	out.write(img.astype('uint8'))
	# out.write(X[:,:,0])
	# print(X[:,:,0].shape, X[:,:,0].dtype)
	# print(predicted.shape, predicted.dtype, prraw.min(), prraw.max(), predicted.min(), predicted.max())
	r, X = video.read()
	if not r:
		break

video.release()
out.release()

# from gmm0 import GMM

# r, X = video.read()
# model = GMM(2)

# count = 0
# while r:
# 	break
# 	X_f = X.reshape(-1, X.shape[-1])
# 	model.fit(X_f)
# 	plt.imshow(model.predict(X_f).reshape(X.shape[:-1]), cmap='gray')
# 	plt.show()
# 	r, X = video.read()
# 	count += 1 

# from sklearn.mixture import GaussianMixture
# model = GaussianMixture(2)
# count = 0

# while r:
# 	X_f = X.reshape(-1, X.shape[-1])
# 	model.fit(X_f)
# 	# plot(2, model.means_, model.covariances_, X_f)
# 	plt.imshow(model.predict(X_f).reshape(X.shape[:-1]), cmap='gray')
# 	plt.show()
# 	r, X = video.read()
# 	count += 1 
 


