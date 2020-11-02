'''
@ Yuru Song, Oct-30-2020
'''
import numpy as np 
import datetime
import matplotlib.pyplot as plt 
from cvxopt import matrix, solvers
import pickle
TOL = 1e-20
# SEED = 0
# np.random.seed(SEED)

class offline_NonLinearPerceptron:
	def __init__(self, N = 10, P = 20, epoch = 200, l2scale = 0):	
		self.N = N
		self.P = P
		self.epoch = epoch
		self.l2scale = l2scale
		self.I = np.zeros((self.N, self.P), dtype = float)
		self.rhosq = np.random.rand(self.N, self.P)
		self.X = np.random.rand(N, P) 
		self.q = (np.random.rand(P, 1) > .5) * 2 - 1
		self.exp_id = 'offline_lp_N_'+ str(self.N) + '_P_' + str(self.P) + '_' + '_norm_{:.2f}'.format(self.l2scale)  +datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
		ind_X = np.zeros((self.N, self.P), dtype = np.int64)
		self.loss = []
		self.T = 0
		self.worked = True
		for syn in range(N):
			ind_X[syn, :] = np.argsort(self.X[syn, :])
		self.rev_ind = ind_X#np.argsort(ind_X)	
		
	def _update_current(self):
		self.I = np.cumsum(self.rhosq, axis = 1) 

	def _predict(self):
		self.pred = np.zeros(self.P, dtype = int)
		self.pred = 2 * (self.total_I > self.T) - 1
		self.pred = np.reshape(self.pred, np.shape(self.q))
		self.loss.append(np.abs(pred - np.sign(self.q)).mean())

	def _update_weight(self):
		for i in range(self.P):
			if np.sign(self.q[i]) != self.pred[i]:
				self.q[i] += np.sign(self.q[i])

	def _solve_linprog(self):
		ind_pos = np.argwhere(self.q > 0)
		ind_neg = np.argwhere(self.q < 0)
		C_obj = np.zeros((self.N, self.P), dtype = float)
		A_ub = np.zeros((self.P + self.N * self.P , self.N * self.P ))
		for i in range(self.N):
			for j in range(len(ind_pos)):
				C_obj[i, 0: self.rev_ind[i, ind_pos[j, 0]]+1] += self.q[ind_pos[j, 0]]
			for j in range(len(ind_neg)):
				C_obj[i, 0: self.rev_ind[i, ind_neg[j, 0]]+1] += self.q[ind_neg[j, 0]]
		C_obj = C_obj.reshape((self.N * self.P, -1))
		C_obj -= 1 * self.l2scale #/ (self.N *self.P)
		C_obj *= -1
		for j in range(self.P):
			A_ub_tmp = np.zeros((self.N, self.P), dtype = float)
			for i in range(self.N):	
				A_ub_tmp[i, 0: self.rev_ind[i, j]+1] += 1
			A_ub[j,:] = np.reshape(A_ub_tmp, (self.N * self.P, ))
			
			A_ub[j, :] *= -np.sign(self.q[j])
		for j in range(self.N * self.P ):
			A_ub[self.P + j, j] = -1
		b_ub = np.sign(self.q) * self.T
		b_ub = np.append(b_ub, np.zeros((self.N * self.P), dtype = float))
		b_ub = b_ub.astype('float')
		solvers.options['show_progress'] = False
		try:
			res = solvers.lp(matrix(C_obj),matrix(A_ub), matrix(b_ub))
		except:
			self.worked = False
			return
		self.rhosq = np.reshape(np.array(res['x']), (self.N,self.P))
		self.rhosq[self.rhosq < 0] = 0
		self._update_current() 
		self.total_I = np.zeros((self.N, self.P), dtype = float)
		for i in range(self.N):
			for j in range(self.P):
				self.total_I[i, j] += self.I[i, self.rev_ind[i, j]]
		self.total_I = np.sum(self.total_I, axis = 0)
		self.T = (np.min(self.total_I[ind_pos[:,0]]) + np.max(self.total_I[ind_neg[:,0]]))/2.

	def train(self):
		for iter_epoch in range(self.epoch):
			self._solve_linprog()
			if self.worked == False:
				print('failed linprog')
				break
			self._predict()
			# self._update_weight()

			if self.loss[-1] < TOL and iter_epoch > 1:
				print('well done')
				break
		if iter_epoch == self.epoch:
			self.worked = False
			print('max iter reached')

class online_NonLinearPerceptron:
	"""docstring for online_NonLinearPerceptron"""
	def __init__(self, N = 100, epoch = 100, lr = .1):
		self.N = N
		self.epoch = epoch
		self.lr = lr
		self.P = 2
		self.rhosq = np.random.randn((N, 2), dtype = float) # this in sorted as X increases
		self.X = np.hstack((np.random.rand(N, 1), np.random.rand(N, 1)))
		self.q = np.array([-1, 1])
		self.loss = []
		self.T = 0
		self.worked = True
		# self._sort_input()
		self._update_current()
		
	def _sort_input(self):
		# sort current value based on the order of X
		self.ind_X = np.zeros((self.N, self.P), dtype = np.int64)
		for syn in range(self.N):
			self.ind_X[syn, :] = np.argsort(self.X[syn, :])

	def _update_current(self):
		for syn in range(self.N):
			self.I[syn, :] = np.cumsum(self.rhosq[syn, self.ind_X[syn, :]]) # correct

	def _adjust_synapse(self, syn):
		# learn the syn-th synaptic nonlinear function, can parallel
		sorted_I = self.I[syn, self.ind_X[syn, :]]

		q_sorted = self.q[self.ind_X[syn, :]]
		pred_sorted = self.pred[self.ind_X[syn, :]]
		for i in range(1, self.P):
			if np.sign(q_sorted[i]) > np.sign(pred_sorted[i]):
				sort_I[i] += self.lr
				ind_r = np.argwhere(sorted_I < sorted_I[i])
				ind_r = ind_r[ind_r > i]
				if len(ind_r) > 0:
					sorted_I[ind_r] = sorted_I[i]
			elif np.sign(q_sorted[i]) < np.sign(pred_sorted[i]):
				sorted_I[i] -= self.lr
				if sorted_I[i] < 0:
					sorted_I[i] = 0
				ind_l = np.argwhere(sorted_I > sorted_I[i])
				ind_l = ind_l[ind_l < i]
				if len(ind_l) > 0:
					sorted_I[ind_l] = sorted_I[i]
		self.I[syn, self.ind_X[syn, :]] = sorted_I
		self.rhosq[syn, self.ind_X[syn, :]] = np.hstack((np.array([0]), np.diff(sort_I)))

	def _add_sample(self):
		self.X = np.hstack((self.X, np.random.rand(N, 1)))
		self.q = np.hstack((self.q, (np.random.rand() > .5) * 2 - 1))
		self.P = self.q.size
		self.rhosq = np.hstack((self.I, np.random.rand(N, 1))) # set it as the previous value
		self._sort_input()
		self._update_current()
		self.exp_id = 'online_N_'+ str(self.N) + '_P_' + str(self.P) + '_' + '_norm_{:.2f}'.format()  +datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
		
	def _predict(self):
		self.T = self.q.sum()/(2 * self.P)
		total_I = self.I.sum(axis = 0)
		self.pred = (total_I > self.T) * 2 - 1
		self.pred = np.reshape(self.pred, np.shape(self.q))
		self.loss.append(np.abs(self.pred - self.q).mean())


	def train(self):
		while self.worked:
			for iter_epoch in range(self.epoch):
				self._add_sample()
				self._sort_input()
				self._predict()
				if self.loss[-1] < TOL:
					self.worked = True
					break
				for syn in range(self.N):
					self._adjust_synapse(syn)
				
			self.worked = False
		self.exp_id = 'online_N_'+ str(self.N) + '_P_' + str(self.P) + '_' + '_lr_{:.2f}'.format(self.lr)  +datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

if __name__ == '__main__':
	for N in range(100, 1000, 100):
		for P in range(N, 3*N, 10):
			for repeat in range(10):
				print([N, P, repeat])
				neuron = offline_NonLinearPerceptron(N = N, P = P, epoch = 10, l2scale = 0)
				neuron.train()
				
				with open(neuron.exp_id + ".pickle", "wb") as file_:
					pickle.dump(neuron1, file_)
				


