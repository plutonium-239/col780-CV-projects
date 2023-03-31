import numpy as np
import cv2
import imutils

def find_projection(P_stacked, X_stacked):
	'''
	P_stacked: a vector of size (n, 3) representing n points on the 2d image plane, each [u v 1]^T.
	X_stacked: a vector of size (n, 4) representing n points in the 3d world, each [x y z 1]^T.
	'''
	n = P_stacked.shape[0]
	assert n >= 6

	M = np.zeros((2*n, 12))
	assert (P_stacked[:,2] == 1).all() and (X_stacked[:,3] == 1).all()
	u,v = P_stacked[:,0], P_stacked[:,1]
	x,y,z = X_stacked[:,0], X_stacked[:,1], X_stacked[:,2]
	M[::2, 0] = -x
	M[::2, 1] = -y
	M[::2, 2] = -z
	M[::2, 3] = -1
	M[::2, 8] = u*x
	M[::2, 9] = u*y
	M[::2, 10] = u*z
	M[::2, 11] = u
	M[1::2, 4] = -x
	M[1::2, 5] = -y
	M[1::2, 6] = -z
	M[1::2, 6] = -1
	M[1::2, 8] = v*x
	M[1::2, 9] = v*y
	M[1::2, 10] = v*z
	M[1::2, 11] = v
	U,S,V_T = np.linalg.svd(M)
	P = V_T[-1].reshape(3,4) # last singular value is closest to 0 (real measurements have noise)

	return P

def overlay(obj_fname, img_fnames, all_Hs, X_stacked_imgs2, P_stacked_imgs, use_color=False):
	# obj = pywavefront.Wavefront(obj_fname)
	obj = OBJ(obj_fname, swapyz=True)
	ret, K, _, _, _ = cv2.calibrateCamera(X_stacked_imgs2.astype(np.float32), P_stacked_imgs[:,:,:2].astype(np.float32), (2016, 930), None, None)


	for i in range(len(img_fnames)):
		img = cv2.imread(img_fnames[i])
		# K = np.array([[800, 0, 1000], [0, 800, 350], [0, 0, 1]])
		P = projection_matrix(K, all_Hs[i])
		# print(P)
		out = render(img, obj, P, color=use_color)
		cv2.imshow('out', imutils.resize(out, width=800))
		cv2.waitKey(0)


def projection_matrix(K, H):
	# homography = -homography
	Rt = np.linalg.inv(K)@H # KH = [R|t] = [R1 R2 R3 t]
	l = np.sqrt(np.linalg.norm(Rt[:, 0], 2)*np.linalg.norm(Rt[:, 1], 2))
	R1 = Rt[:, 0]/l
	R2 = Rt[:, 1]/l
	t = Rt[:, 2]/l
	c = R1 + R2
	p = np.cross(R1, R2)
	d = np.cross(c, p)
	c_norm = np.linalg.norm(c, 2)
	d_norm = np.linalg.norm(d, 2)
	R1 = (c/c_norm + d/d_norm)/np.sqrt(2)
	R2 = (c/c_norm - d/d_norm)/np.sqrt(2)
	R3 = np.cross(R1, R2)
	P = np.stack((R1, R2, R3, t)).T
	return K @ P

def render(img, obj, projection, color=False):
	'''
	Adapted from Bites of Code blog, this simply renders a 3d obj on a 2d image given the projection matrix. 
	'''
	vertices = np.array(obj.vertices)
	H,W = img.shape[:2]
	vertices = 3*(vertices - vertices.min())/(vertices.max() - vertices.min())
	h, w = 0,0
	i = 0
	for face in obj.faces:
		face_vertices = face[0]
		points = np.array([vertices[vertex - 1] for vertex in face_vertices])
		# render model in the middle of the reference surface. To do so,
		# model points must be displaced
		points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
		dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
		# if i < 5:
		# 	print(dst[0])
		imgpts = np.int32(dst)
		i += 1
		if color is False:
			cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
		else:
			color = face[-1]
			if isinstance(color, str) and color.startswith('#'):
				color = hex_to_rgb(face[-1])
			color = color[::-1] # reverse
			try:
				cv2.fillConvexPoly(img, imgpts, color[:3])
			except:
				cv2.fillConvexPoly(img, imgpts, (137, 27, 211))

	return img

def hex_to_rgb(hex_color):
	hex_color = hex_color.lstrip('#')
	h_len = len(hex_color)
	return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

class OBJ:
	''''
	Taken from PyGame's OBJLoader, removed references to OpenGL etc.
	'''
	def __init__(self, filename, swapyz=False):
		"""Loads a Wavefront OBJ file. """
		self.vertices = []
		self.normals = []
		self.texcoords = []
		self.faces = []
		material = None
		for line in open(filename, "r"):
			if line.startswith('#'): continue
			values = line.split()
			if not values: continue
			if values[0] == 'v':
				v = list(map(float, values[1:4]))
				if swapyz:
					v = v[0], v[2], v[1]
				self.vertices.append(v)
			elif values[0] == 'vn':
				v = list(map(float, values[1:4]))
				if swapyz:
					v = v[0], v[2], v[1]
				self.normals.append(v)
			elif values[0] == 'vt':
				self.texcoords.append(map(float, values[1:3]))
			#elif values[0] in ('usemtl', 'usemat'):
				#material = values[1]
			#elif values[0] == 'mtllib':
				#self.mtl = MTL(values[1])
			elif values[0] == 'f':
				face = []
				texcoords = []
				norms = []
				for v in values[1:]:
					w = v.split('/')
					face.append(int(w[0]))
					if len(w) >= 2 and len(w[1]) > 0:
						texcoords.append(int(w[1]))
					else:
						texcoords.append(0)
					if len(w) >= 3 and len(w[2]) > 0:
						norms.append(int(w[2]))
					else:
						norms.append(0)
				#self.faces.append((face, norms, texcoords, material))
				self.faces.append((face, norms, texcoords))

