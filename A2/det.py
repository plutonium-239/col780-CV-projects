import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

g2rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

def standardize(x):
	return (x - x.mean())/x.std()

def normalize(x):
	return (x - x.min())/(x.max() - x.min())

sigmoid = lambda x: 1/(1 + np.exp(-x))
# def standardize(x):
# 	return ()

def HarrisCornerDetector(img_orig, args, k=0.04, thresh=0.05, smoothing=False):
	def on_trackbar(val):
		corners = cv2.dilate(C, None) > (val/100)*C.max()
		i3 = img_orig.copy()
		i3[corners] = [255,0,128]
		cv2.imshow('thresh', i3)

	img = img_orig.astype('float32')
	if smoothing or args.dataset in []:
		img = cv2.GaussianBlur(img_orig, (3,3), 0).astype('float32')

	def get_corner_matrix(img):
		grimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# sobel_x = cv2.getDerivKernels(1, 0, 3, normalize=True)
		# sobel_y = cv2.getDerivKernels(0, 1, 3, normalize=True)

		sobel_x = ( np.array([[-0.5],[0],[0.5]]), np.array([[0.25],[0.5],[0.25]]) )
		sobel_y = ( np.array([[0.25],[0.5],[0.25]]), np.array([[-0.5],[0],[0.5]]) )

		I_x = cv2.sepFilter2D(grimg, cv2.CV_32F, *sobel_x, borderType=cv2.BORDER_ISOLATED)
		I_y = cv2.sepFilter2D(grimg, cv2.CV_32F, *sobel_y, borderType=cv2.BORDER_ISOLATED)

		# return

		# I_x = cv2.Sobel(grimg, -1, 1, 0)
		# I_y = cv2.Sobel(grimg, -1, 0, 1)
		

		I_x_sq = I_x**2
		I_y_sq = I_y**2
		I_xI_y = I_x*I_y

		G = cv2.getGaussianKernel(3, None)

		H_xx = cv2.filter2D(I_x_sq, -1, G)
		H_yy = cv2.filter2D(I_y_sq, -1, G)
		H_xy = cv2.filter2D(I_xI_y, -1, G)

		trace = (H_xx + H_yy)
		trace[trace == 0] = np.inf
		# C = (H_xx*H_yy - H_xy**2) / trace # det() / tr()
		C = (H_xx*H_yy - H_xy**2) - k*(H_xx + H_yy)**2 # det() / tr()
		# corner_matrix = cv2.convertScaleAbs(corner_matrix)
		corner_matrix = standardize(C)
		return C, corner_matrix
	# corner_matrix = normalize(C)
	# corner_matrix = sigmoid(corner_matrix)
	# print('before', C.min(), C.max(), C.mean())
	# print('after', corner_matrix.min(), corner_matrix.max(), corner_matrix.mean())

	# NMS
	C, corner_matrix = get_corner_matrix(img)
	C_dil, corner_matrix_dil = get_corner_matrix(cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))))
	# import ipdb; ipdb.set_trace()
	# C_dil = cv2.dilate(C, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
	# C = np.where(C==C_dil, C, 0)

	corner_matrix = np.where(corner_matrix==corner_matrix_dil, 0, corner_matrix)

	i2 = img_orig.copy()
	i2[cv2.dilate(C, None) < 0] = [0,255,0]
	# i2 = cv2.dilate(i2, None)

	i3 = img_orig.copy()
	corners = C > thresh*C.max()
	corners = np.argsort(C, axis=None)[::-1][:500] # argsort by min -> look at reverse for max -> take largest 500
	corners = corners//C.shape[1], corners%C.shape[1]
	i3[corners] = [128,0,255]
	# print(i3.min(), i3.max(), i3.mean())
	# i3 = cv2.dilate(i3, None)

	i2 = cv2.convertScaleAbs(i2)
	i3 = cv2.convertScaleAbs(i3)
	img = cv2.convertScaleAbs(img)
	# cv2.imshow('C', corner_matrix)
	if args.show == 3:
		plt.imshow(corner_matrix)
		plt.show()
		cv2.namedWindow('thresh')
		cv2.createTrackbar('threshold', 'thresh', 0, 100, on_trackbar)
		on_trackbar(5)
	if args.show >= 2:
		cv2.imshow('thresh', i3)
		# cv2.imshow('Ix, Iy', np.concatenate([I_x, I_y], 1))

		# cv2.imshow('Ix0, Iy0', np.concatenate([I_x0, I_y0], 1))
		# cv2.imshow('i3, C', np.concatenate([img, i3], 1))
		# cv2.imshow('edges, corners', np.concatenate([i2, i3], 1))
		cv2.waitKey(0)
	# return i3, np.where(C > thresh*C.max())
	return i3, corners



def match_corners(im1, im2, im1_pts, im2_pts, dataset, args, thresh=60):
	im1_features = extract_features(im1, im1_pts, args)
	im2_features = extract_features(im2, im2_pts, args)
	best_pts1, best_pts2, m = feature_matching(im1_features, im2_features, thresh=thresh)

	aff_pts1, aff_pts2 = find_best_pairs(best_pts1, best_pts2)

	# print(aff_pts1)
	# print(aff_pts2)

	w1,h1 = im1.shape[0], im1.shape[1]
	w2,h2 = im2.shape[0], im2.shape[1]
	m1 = cv2.copyMakeBorder(im1, 0, max(w1,w2) - w1, 0, max(h1,h2)-h1, borderType=cv2.BORDER_CONSTANT)
	m2 = cv2.copyMakeBorder(im2, 0, max(w1,w2) - w2, 0, max(h1,h2)-h2, borderType=cv2.BORDER_CONSTANT)
	matched = np.concatenate([m1, m2], 1)
	# print('atmched', matched.shape)
	color = []
	for pt1,pt2 in zip(aff_pts1, aff_pts2):
		x1, y1 = int(pt1[0]), int(pt1[1])
		x2, y2 = int(pt2[0]), int(pt2[1])
		color.append([random.randint(0,255), random.randint(0,255), random.randint(128,255)])
		# print(f"({x1},{y1}) -> ({x2},{y2}) [=({x2},{h+y2})]")
		matched[x1-7:x1+7, y1-7:y1+7] = color[-1]
		matched[x2-7:x2+7, max(h1,h2)+y2-7:max(h1,h2)+y2+7] = color[-1]

	M = cv2.getAffineTransform(aff_pts1, aff_pts2)
	print(M)
	H = np.eye(3)
	H[:2,:] = M
	# if old_M is not None:
	# 	H_old = np.eye(3)
	# 	H_old[:2,:] = old_M
	# 	H_mult = H@H_old
	# 	print('mult H', H_mult)
	# 	M = H_mult[:2,:]


	# print('Affine transform', M)
	# print((M[:, :2].dot(best_pts1[:3].T) + M[:, 2].reshape(2, 1)).T)

	d_p, s_w, shifted_M, pad_widths = warp_and_pad(im1, im2, M, dataset=dataset)
	new_best_pts2 = (shifted_M[:, :2].dot(aff_pts1.T) + shifted_M[:, 2].reshape(2, 1)).T
	# print(new_best_pts2)

	# print(d_p.shape, s_w.shape)
	# dest = cv2.warpAffine(im1, M, (d_p.shape[0], d_p.shape[1]))
	c = 0
	# for pt2 in new_best_pts2:
	# 	x2, y2 = int(pt2[0]), int(pt2[1])
	# 	# color = [random.randint(128,255), random.randint(128,255), random.randint(128,255)]
	# 	# print(f"({x1},{y1}) -> ({x2},{y2}) [=({x2},{h+y2})]")
	# 	# matched[x1-5:x1+5, y1-5:y1+5] = color
	# 	s_w[x2-7:x2+7, y2-7:y2+7] = color[c]
	# 	c+=1
	# print(dest.shape)
	# blended = cv2.addWeighted(d_p, 0.5, s_w, 0.75, -1.25)
	blended = cv2.addWeighted(d_p, 0.75, s_w, 0.25, 0)
	blended = cv2.addWeighted(d_p, 0.25, s_w, 0.75, 0)
	# b2 = cv2.addWeighted(dest, 0.5, s_w, 0.5, 1)

	min_y, max_y, min_x, max_x = pad_widths
	max_x = h2 + min_x
	max_y = w2 + min_y

	# print(min_x, max_x, min_y, max_y)
	# cv2.imshow('dp', d_p[min_y:max_y, min_x:max_x])
	result = s_w.copy()
	result[min_y:max_y, min_x:max_x] = d_p[min_y:max_y, min_x:max_x]

	# stitcher = cv2.Stitcher_create(mode = cv2.STITCHER_SCANS)
	# res = stitcher.stitch([im1, im2])
	# print(res[0])

	# res = warpTwoImages(im1, im2, H)
	if args.show >=1:
		# cv2.imshow('im1 im2 dest', np.concatenate([res[1]], 1))
		# cv2.imshow('s_w d_p', np.concatenate([s_w, d_p], 1))
		# plt.imshow(np.concatenate([s_w], 1))
		cv2.imshow('matched', np.concatenate([matched], 1))
		# cv2.imshow('s_w', np.concatenate([result], 1))
		# cv2.imshow('blended', np.concatenate([blended], 1))
		# plt.show()
		# cv2.waitKey(0)


	return result


def warp_and_pad(img1, img2, M, dataset=1):
	img1_h, img1_w = img1.shape[:2]
	img1_corners = np.array([
		[0, img1_w, img1_w, 0],
		[0, 0, img1_h, img1_h]])

	shifted_img1_corners = M[:, :2].dot(img1_corners) + M[:, 2].reshape(2, 1)
	# print(shifted_img1_corners)

	min_x = np.floor(np.min(shifted_img1_corners[0])).astype(int)
	min_y = np.floor(np.min(shifted_img1_corners[1])).astype(int)
	max_x = np.ceil(np.max(shifted_img1_corners[0])).astype(int)
	max_y = np.ceil(np.max(shifted_img1_corners[1])).astype(int)

	anchor_x = - min_x if (min_x < 0) else 0
	anchor_y = - min_y if (min_y < 0) else 0
	
	shifted_transf = M + [[0, 0, anchor_x], [0, 0, anchor_y]]
	# H = np.eye(3)
	# H[:2,:] = M
	# H_t = np.eye(3)
	# H_t[0,2], H_t[1,2] = anchor_x, anchor_y
	# shifted_transf = (H@H_t)[:2,:]
	# print('SHIFTED Affine transform', shifted_transf)
	# print(min_x, max_x, min_y, max_y)

	img2_h, img2_w = img2.shape[:2]

	left = anchor_y
	right = (max_y - img2_h) if max_y > img2_h else 0
	top = anchor_x
	bottom = (max_x - img2_w) if max_x > img2_w else 0

	pad_widths = [top, bottom, left, right]

	# print('pw', pad_widths)

	img2_padded = cv2.copyMakeBorder(img2, *pad_widths, borderType=cv2.BORDER_CONSTANT, value=0)

	img2_pad_h, img2_pad_w = img2_padded.shape[:2]

	img1_warped = cv2.warpAffine(img1, shifted_transf, (img2_pad_w, img2_pad_h),\
	 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

	return img2_padded, img1_warped, shifted_transf, pad_widths

def extract_features(im, pts, args):
	x_pts,y_pts = pts

	patch_size = (args.patch_size, args.patch_size)
	spacing = args.patch_spacing
	
	offset_x, offset_y = int(patch_size[0]/2), int(patch_size[1]/2)
	# x_pts = x_pts[x_pts > offset_x]
	# x_pts = x_pts[x_pts < im.shape[0] - 1 - offset_x]
	# y_pts = y_pts[y_pts > offset_y]
	# y_pts = y_pts[y_pts < im.shape[1] - 1 - offset_y]
	# patches = np.copy(im[x-offset_x:x+offset_x, y-offset_y:y+offset_y])

	features_vec = np.zeros((len(x_pts), im.shape[2]*(patch_size[0]//spacing)**2))
	mask_features = np.ones(len(x_pts), dtype=bool)

	for i,(x,y) in enumerate(zip(x_pts, y_pts)):
		# x, y = int(pts[0][i]), int(pts[1][i])
		if x < offset_x or y < offset_y or x > im.shape[0] - 1 - offset_x or y > im.shape[1] - 1 - offset_y:
			mask_features[i] = False
			continue
		# patch = im[x-offset_x:x+offset_x, y-offset_y:y+offset_y]
		x_low, x_high = max(0, x - offset_x), min(im.shape[0] - 1, x + offset_x)
		y_low, y_high = max(0, y - offset_y), min(im.shape[1] - 1, y + offset_y)
		patch = im[x_low:x_high, y_low:y_high]
		
		patch_compressed = cv2.resize(patch, [patch_size[0]//spacing, patch_size[1]//spacing])
		patch_compressed = standardize(patch_compressed)
		
		features_vec[i,:] = patch_compressed.flatten()

	return (pts[0][mask_features], pts[1][mask_features]), features_vec[mask_features]

def feature_matching(features_1, features_2, thresh=60):
	# import ipdb; ipdb.set_trace()

	distances = dist2(features_1[1], features_2[1])
	min_dists = distances.min(axis=1)
	m = distances.argmin(axis=1)
	matches = np.argsort(min_dists)

	x1, y1 = features_1[0][0][matches], features_1[0][1][matches]
	x2, y2 = features_2[0][0][m[matches]], features_2[0][1][m[matches]]

	best_pts1 = np.stack([x1, y1], axis=1)
	best_pts2 = np.stack([x2, y2], axis=1)

	return best_pts1, best_pts2, m

def find_best_pairs(best_pts1, best_pts2):
	def diff(arr1, arr2, val=0):
		return (np.abs(arr1 - arr2) <= val).all()
	
	# def angle(pt1, pt2):

	# def okay(pt1, pt2):
	# 	angle1 = np.arctan2(*pt1[::-1])
	# 	angle2 = np.arctan2(*pt2[::-1])
	# 	# return (angle1 - angle2)%(2*np.pi)
	# 	angle_diff = np.rad2deg((angle1 - angle2) % (2 * np.pi))
	# 	print(angle_diff)
	# 	return (30 <= angle_diff <= 150) or (210 <= angle_diff <= 330)

	# def okay(pt1, pt2):
	# 	return np.dot(pt1, pt2) == 0

	# CHECK IF ANY POINTS ARE THE SAME -> WONT BE ABLE TO FIND AFFINE TRANSFORM
	idx1, idx2, idx3 = 0,1,2
	for p in [best_pts1, best_pts2]:
		# if p[idx1] == p[idx2] == p[idx3]:
		# 	idx2 += 1
		# 	idx3 += 2
		# if dataset ==1:
		# 	break
		# print(idx1)
		# while not okay(p[idx1], p[idx2]):
		# 	idx2 += 1
		# print(idx2)
		# while not (okay(p[idx2], p[idx3]) and okay(p[idx3], p[idx1])):
		# 	idx3 += 1
		# print(idx3)
		min_angle = 20
		min_dist = 10 # only find points more than 10 pixels away from each other in both dirs
		def calc(a,b,c):
			if (np.abs(a - b) < min_dist).all() or (np.abs(b - c) < min_dist).all() or (np.abs(a - c) < min_dist).all():
				# print('equal')
				return 0
			cos1 = np.abs(np.dot(b - a, c - a))/(np.linalg.norm(b - a)*np.linalg.norm(c - a))
			cos2 = np.abs(np.dot(b - c, a - c))/(np.linalg.norm(b - c)*np.linalg.norm(a - c))
			# print(cos1, cos2, end=' -> ')
			cos1, cos2 = np.rad2deg(np.arccos(cos1)%(2*np.pi)), np.rad2deg(np.arccos(cos2)%(2*np.pi)) 
			cos3 = 180 - cos1 - cos2
			# print(cos1, cos2, cos3)
			return min(cos1, cos2, cos3)

		flag = False
		def min_gr_thresh(i1, i2, i3):
			global flag
			if i2 >= len(p) or i3 >= len(p):
				return False
			m = calc(p[i1], p[i2], p[i3])
			m_angs[i2, i3] = m
			return m > min_angle

		m_angs = np.zeros((len(p), len(p)))
		# try:
		# USE FOR INSTEAD
		# while calcw(idx1, idx2, idx3):
		for idx1 in range(0, len(p)):
			for idx2 in range(idx1+1, len(p)):
				for idx3 in range(idx2+1, len(p)):
					if min_gr_thresh(idx1, idx2, idx3):
						flag = True
						break
				if flag:
					break
			if flag:
				break
		if not flag:
			raise RuntimeError("No matching points found, try by reducing the min_angle or min_dist")
			# while calcw(idx1, idx2, idx3):
			# 	idx3 += 1
			# if calcw(idx1, idx2, idx3):
			# 	idx2 += 1
			# 	idx3 = idx2 + 1
		# print('FOUND :', flag)
		# except:
		# 	print('FAILED')
		# 	max_index = m_angs.argmax(axis=None)
		# 	idx2, idx3 = max_index//len(p), max_index%len(p)
		# 	break


		# while ( diff(p[idx1], p[idx3]) or diff(p[idx1], p[idx2]) or diff(p[idx2], p[idx3]) ):
			# print(diff(p[idx1], p[idx3]))
			# if diff(p[idx1], p[idx3]):
			# 	idx3 += 1
			# # print(diff(p[idx1], p[idx2]))
			# if diff(p[idx1], p[idx2]):
			# 	idx2 += 1
			# # print(diff(p[idx2], p[idx3]))
			# if diff(p[idx2], p[idx3]):
			# 	idx3 += 1
		# print(idx1, idx2, idx3)
	aff_pts1 = best_pts1[[idx1, idx2, idx3]].astype(np.float32)
	aff_pts2 = best_pts2[[idx1, idx2, idx3]].astype(np.float32)
	# print(aff_pts1)
	# print(aff_pts2)
	return aff_pts1, aff_pts2

def dist2(x, c):
	"""
	Adapted from code by Christopher M Bishop and Ian T Nabney.
	"""	
	ndata, dimx = x.shape
	ncenters, dimc = c.shape
	assert dimx == dimc, 'Data dimension does not match dimension of centers'

	return (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T + \
			np.ones((   ndata, 1)) * np.sum((c**2).T, axis=0)    - \
			2 * np.inner(x, c)
