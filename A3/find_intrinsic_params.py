import numpy as np
# import cv2

def find_intrinsic_matrix(P_stacked_imgs, X_stacked_imgs):
	'''
	P_stacked_imgs: a vector of size (num_images, n, 3) representing n points on the 2d image plane, each [u v 1]^T for num_images images.
	X_stacked_imgs: a vector of size (num_images, n, 3) representing n in the 3d world on the same plane, each [x y 1]^T for num_images images.
	'''
	num_images = P_stacked_imgs.shape[0]
	assert num_images >= 3
	# print('n',num_images)
	all_Hs = []
	for img in range(num_images):
		H = find_homography(P_stacked_imgs[img], X_stacked_imgs[img])
		# H2 = cv2.findHomography(X_stacked_imgs[img], P_stacked_imgs[img], cv2.RANSAC, 5.0)[0]
		# print('H_manual', H2)
		# print('H_opencv', H)
		all_Hs.append(H)
	K = find_K_from_Hs(all_Hs)

	return K, all_Hs

def find_homography(P_stacked, X_stacked):
	'''
	P_stacked: a vector of size (n, 3) representing n points on the 2d image plane, each [u v 1]^T.
	X_stacked: a vector of size (n, 3) representing n points in the 3d world on the same plane, each [x y 1]^T.
	'''
	n = P_stacked.shape[0]
	assert n >= 4
	# Part 1: find H (homography mapping)
	M = np.zeros((2*n, 9))
	assert (P_stacked[:,2] == 1).all() and (X_stacked[:,2] == 1).all()
	u,v = P_stacked[:,0], P_stacked[:,1]
	x,y = X_stacked[:,0], X_stacked[:,1]
	M[::2, 0] = -x
	M[::2, 1] = -y
	M[::2, 2] = -1
	M[::2, 6] = u*x
	M[::2, 7] = u*y
	M[::2, 8] = u
	M[1::2, 3] = -x
	M[1::2, 4] = -y
	M[1::2, 5] = -1
	M[1::2, 6] = v*x
	M[1::2, 7] = v*y
	M[1::2, 8] = v
	U,S,V_T = np.linalg.svd(M)
	# print('S',S)
	# print('res', M@V_T[-1])
	H = V_T[-1].reshape(3,3) # last singular value is closest to 0 (real measurements have noise)

	return H/H[-1,-1]

def find_K_from_Hs(all_Hs):
	# Part 2: find K (intrinsic parameters) from Hs of multiple images
	V = np.zeros((2*len(all_Hs), 6))
	for img in range(len(all_Hs)):
		H = all_Hs[img]
		v_11, v_12, v_22 = make_v(H, 1, 1), make_v(H, 1, 2), make_v(H, 2, 2)
		V[2*img] = v_12
		V[2*img + 1] = v_11 - v_22
	M,S,N_T = np.linalg.svd(V)
	b = N_T[-1]
	# print(N_T)
	# print(V@b)
	B = np.zeros((3,3))
	# B[[0,0], [0,1], [0,2], [1,1], [1,2], [2,2]] = b
	B.ravel()[[0,1,2,4,5,8]] = b # upper triangle
	B[1,0] = B[0,1]
	B[2,0] = B[0,2]
	B[2,1] = B[1,2]

	# print(B)
	# import ipdb; ipdb.set_trace()
	# ensure B is positive definite
	lambdas, Q = np.linalg.eig(B)
	eps = 1e-12
	lambdas[lambdas < eps] = eps
	B_pd = Q@np.diag(lambdas)@Q.T
	K_inv_T = np.linalg.cholesky(B_pd)
	K = np.linalg.inv(K_inv_T)
	if np.allclose(K[0,2], 0):
		K = K.T
	return K/K[-1,-1]

def find_K_from_Hs_2(all_Hs):
	# Part 2: find K (intrinsic parameters) from Hs of multiple images
	V = np.zeros((2*len(all_Hs), 4))
	for img in range(len(all_Hs)):
		H = all_Hs[img]
		v_11, v_12, v_22 = make_v_2(H, 1, 1), make_v_2(H, 1, 2), make_v_2(H, 2, 2)
		V[2*img] = v_12
		V[2*img + 1] = v_11 - v_22
	M,S,N_T = np.linalg.svd(V)
	b = N_T[-1]
	print(N_T)
	# print(V@b)

	K = np.zeros((3,3))
	K[0,0] = 1/np.sqrt(b[0])
	K[1,1] = 1/np.sqrt(b[2])
	K[0,2] = -b[1]/b[0]
	K[1,2] = -b[3]/b[2]
	K[-1,-1] = 1
	print(-b[1] + -b[3] + 1, 'should be 1')

	# print(B)
	# import ipdb; ipdb.set_trace()
	# ensure B is positive definite
	# lambdas, Q = np.linalg.eig(B)
	# eps = 1e-12
	# lambdas[lambdas < eps] = eps
	# B_pd = Q@np.diag(lambdas)@Q.T
	# K_inv_T = np.linalg.cholesky(B_pd)
	# K = np.linalg.inv(K_inv_T.T)
	return K

def make_v(H, i, j):
	v = np.zeros(6)
	v[0] = H[0,i]*H[0,j]
	v[1] = H[0,i]*H[1,j] + H[1,i]*H[0,j] 
	v[2] = H[0,i]*H[2,j] + H[2,i]*H[0,j] 
	v[3] = H[1,i]*H[1,j]
	v[4] = H[1,i]*H[2,j] + H[2,i]*H[1,j] 
	v[5] = H[2,i]*H[2,j]
	return v

def make_v_2(H, i, j):
	v = np.zeros(4)
	v[0] = H[0,i]*H[0,j]
	v[1] = H[0,i]*H[2,j] + H[2,i]*H[0,j] 
	v[2] = H[1,i]*H[1,j]
	v[3] = H[1,i]*H[2,j] + H[2,i]*H[1,j] 
	return v