import numpy as np
# import scipy.stats as scstats

class OnlineGMM():
	def __init__(self, args):
		self.k = args.num_gaussians
		self.alpha = args.alpha
		self.step = 0
		self.c_f = 0.4
		self.seed = args.seed

		# wts = np.ones((N,self.k))/self.k


	def step_frame(self, X):
		'''
		omega 	-> (prior) === contribution of each gaussian to the mixture, [k]
		wts 	-> weights of [N,k]
		means 	-> means of the k gaussians
		covs	-> stds/covariances of the k gaussians
		'''
		X_f = X.reshape(-1, X.shape[-1])
		# print(np.isnan(X_f).any())
		N,c = X_f.shape # c=3 for rgb
		if self.step > 0:
			assert (self.N == N) and (self.c == c), "New image not of same size."
		else:
			self.N, self.c = N, c
			# self.omega = np.zeros((self.k, N))
			# self.omega[0] = 1
			self.omega = np.ones((self.k, N))/self.k
			# init the initial means and sds
			rng = np.random.default_rng(seed=self.seed)
			# print('SEED = ', seed)
			# random_pixels = rng.choice(self.N, size=self.k)
			# mean will be [k, c], covs will be [k, c, c]
			self.means = np.clip( np.stack([X_f]*self.k) + rng.integers(-122, 122, (self.k, N, c)), 0, 255).astype('uint8')
			# self.stds = np.stack([np.std(X_f, axis=0)]*self.k)
			self.stds = np.ones((self.k, N, c))*50
			# self.stds = np.eye(self.k)*np.std(X_f, axis=0)
			# return

		likelihood, wts = self.update_weights(self.means, self.stds, X_f)
		# self.omega = wts.mean(axis=0)
		labels = self.match(X_f, likelihood, wts)
		# means, covs = self.update_means_covs(wts, X_f)
		# print(f"{self.step}th step completed.")
		# print('w', self.omega)
		# print('m', self.means)
		# print('s', self.stds)
		if np.isnan(self.omega).any():
			print("NAN FOUND IN OMEGAS", self.omega)
		if np.isnan(self.means).any():
			print("NAN FOUND IN MEANS", self.means)
		if np.isnan(self.stds).any():
			print("NAN FOUND IN STDS", self.stds)
		if np.isnan(likelihood).any():
			print("NAN FOUND IN likelihood", likelihood)
		self.step += 1

		return labels.reshape(X.shape[:-1])

	def match(self, X_f, likelihood, wts):
		label = np.zeros(self.N)

		diff = np.abs(X_f - self.means) # [k, N, c]
		# match = diff.sum(axis = 2).argmin(axis = 0) # [N] -> distribution with lowest difference |I-mu_k|

		# mask = (diff < self.stds[:, None, :]).all(axis = 2) # [k, N]
		# bin_mask = mask.any(axis = 0) # [N]
		# # if a pixel has bin_mask[i] = False then no match was found
		# match = mask.argmax(axis = 0) # [N] -> the matched gaussian distribution

		# match_unique = np.unique(match)
		M = (diff < 2.5*self.stds).all(axis=2) # [k, N]
		match = M.argmax(axis=0) # [N]
		matched = M.any(axis=0)
		# import ipdb; ipdb.set_trace()
		# M[match_unique] = 1
		# # print(M)
		# match[~bin_mask] = -1
		# # replace distrs with max diff (summed channels); where bin_mask is False
		# to_replace = diff[:,~bin_mask,:].sum(axis=2).argmax(axis=0)
		indicator = np.zeros((self.k, self.N))
		np.put_along_axis(indicator, match[None,:], 1, axis=0)

		self.omega = (1 - self.alpha)*self.omega + self.alpha*indicator
		self.omega /= self.omega.sum(axis=0) # normalize
		# print(self.omega)
		for i in range(self.k):
			i_mask = ( (match == i) & M[i]) # [N]
			if not i_mask.any():
				continue
			# if M[i] == 0:
			# 	# print("\ni_mask is 0 for", i, i_mask, '\n')
			# 	continue
			# print(label[i_mask].shape, end='\t')

			rho = self.alpha*(likelihood[i, i_mask])

			# self.omega[i, i_mask] = wts[i, i_mask]
			self.means[i, i_mask] = ( (1 - rho)[:,None]*self.means[i, i_mask] + rho[:,None]*(X_f[i_mask]) )
			self.stds[i, i_mask] = (1 - rho)[:, None]*self.stds[i, i_mask] + rho[:,None]*(diff[i, i_mask]**2)
			label[i_mask] = i
			# self.means[i] = (wts[i_mask,i,None]*X_f[i_mask]).sum(axis=0)/wts[i_mask,i].sum() 
			# tdiff = X_f[i_mask] - self.means[i]
			# tdiff = diff[i, i_mask]
			# t = tdiff[:,None,:].T.transpose(2,0,1) @ tdiff[:,None,:]
			# self.stds[i] = np.diag( (wts[i_mask, i, None, None]*t).sum(axis=0)/wts[i_mask, i].sum() ) 
			# print(self.stds[i, i_mask].shape, t.shape)
			# print((self.stds[i, None, i_mask] * np.eye(self.c)).shape)

			# stds[0,:,None] * np.eye(self.c)

			# s = ((1 - rho)[:,None,None]*np.eye(self.c)*self.stds[i, None, i_mask] + rho[:,None,None]*t)
			# s = np.sqrt(np.diag(s))
			# if not np.isnan(s).any():
			# 	self.stds[i, i_mask] = s

		self.means[-1, ~matched] = X_f[~matched]
		self.stds[-1, ~matched] = 25
		label[~matched] = self.k - 1
		# print(label[~matched].shape)

		# cv2.imshow("means all i", np.concatenate([self.means[i].reshape(240,320,-1) for i in range(self.k)], 1))
		# cv2.waitKey(0)

		# # omegas sorted in descending order, first B omegas to sum > C are BG 
		sort_mask = np.argsort(self.omega, axis=0)[::-1]
		# print(sort_mask.shape)
		# print(self.omega.shape)
		self.omega = np.sort(self.omega, axis=0)[::-1]
		self.means = np.take_along_axis(self.means, sort_mask[:,:,None], axis=1)
		self.stds = np.take_along_axis(self.stds, sort_mask[:,:,None], axis=1)
		self.B = np.argmax(np.cumsum(self.omega, axis=0)>(1 - self.c_f), axis=0)
		# label = (label > self.B).astype(int)
		# import ipdb; ipdb.set_trace()

		return label

		# If mindiffs[k, i] == 3, then ith pixel has diff < kth gaussian in all 3 channels -> match   
		# diff_allchannels = (diff<2.5*self.stds[:,None,:]).sum(axis=2)
		# mask = (diff_allchannels == 3)
		# # Finds the matched gaussian with lowest
		# match = (diff<self.stds[:,None,:]).sum(axis=2).argmax(axis=0)

	# def loop_once(self, X_f, means, covs):

	def update_weights(self, means, stds, X_f):
		# def norm_pdf(X, mean, s):
		# 	ndev = (x-mean)
		#     return (1 / (np.sqrt(2 * np.pi) * s)) * (np.exp(-0.5 * (((x - mean) / s) ** 2)))
		lpi = (2*np.pi)**3
		likelihood = np.zeros((self.k, self.N))
		for i in range(self.k):
			# likelihood[:, i] = scstats.multivariate_normal(mean=means[i], cov=covs[i]).pdf(X_f)
			ndev = (X_f - means[i])/stds[i]
			exp = (ndev[:,None,:] @ (X_f - means[i])[:,:,None]).squeeze()
			likelihood[i] = np.exp(-0.5*exp)/np.sqrt(lpi*stds[i].prod(axis=1))

		weighted_likelihood = likelihood*self.omega
		return likelihood, weighted_likelihood/weighted_likelihood.sum(axis=1)[:, None]



def NMS_vectorized(boxes, overlapThresh=0.4):
	if len(boxes) == 0:
		return np.array([])
	x1 = boxes[:, 0]  # x coordinate of the top-left corner [N, 4]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2] + x1
	y2 = boxes[:, 3] + y1
	areas = (x2 - x1 + 1) * (y2 - y1 + 1) # [N]

	ix1 = np.maximum(x1[:,None] , x1[None,:]) # [N, N]
	iy1 = np.maximum(y1[:,None] , y1[None,:])
	ix2 = np.minimum(x2[:,None] , x2[None,:])
	iy2 = np.minimum(y2[:,None] , y2[None,:])

	iw = np.maximum(0, ix2-ix1+1) # [N, N]
	ih = np.maximum(0, iy2-iy1+1)
		
	ux1 = np.minimum(x1[:,None] , x1[None,:])
	uy1 = np.minimum(y1[:,None] , y1[None,:])
	ux2 = np.minimum(x2[:,None] , x2[None,:])
	uy2 = np.minimum(y2[:,None] , y2[None,:])
	
	uw = np.maximum(0, ux2-ux1+1)
	uh = np.maximum(0, uy2-uy1+1)

	ious = (iw*ih)/(uw*uh)

	# overlap = (w * h) / areas[:, None] # [N, N] Dividing row wise
	for i in range(areas.shape[0]):
		ious[i,i] = 0 # dont count self-overlap (as it is 1)

	# import ipdb; ipdb.set_trace()

	indices = np.where( ~((ious > overlapThresh).any(axis=0)) )
	return boxes[indices].astype(int)

def plot(k, means, covs, X):
	bins = np.linspace(np.min(X, axis=0), np.max(X, axis=0), 100)

	plt.figure(figsize=(10,7))
	plt.xlabel("$x$")
	plt.ylabel("pdf")
	# plt.scatter(X, [0.005] * len(X), color='navy', s=30, marker=2, label="Train data")

	for i in range(k):
		plt.plot(bins.mean(axis=1), multivariate_normal(means[i], covs[i]).pdf(bins), label="True pdf")
	
	plt.legend()
	plt.show()
