'''
@ Yuru Song, Oct-15-2020
'''
from utils import *
import argparse
import matplotlib.pyplot as plt 

if __name__ == '__main__':
	# multiple trial
	for N in range(100, 1000, 100):
		for P in range(int(1.5*N), 3*N, 20):
			for repeat in range(10):
				neuron = offline_NonLinearPerceptron(N = N, P = P, epoch = 3, l2scale = 0)
				print(neuron.exp_id)
				print(neuron.worked)
				neuron.train()
				
				with open(neuron.exp_id + ".pickle", "wb") as file_:
					pickle.dump(neuron, file_)
	# single trial
	# parser = argparse.ArgumentParser()
	# parser.add_argument('--N', type = int, default = 10)
	# parser.add_argument('--P', type = int, default = 100)
	# # parser.add_argument('--algorithm', type = str, default = 'max_margin')
	# parser.add_argument('--epoch', type = int, default = 10)
	# parser.add_argument('--l2scale', type = float, default = .1)
	# parser.add_argument('--lr', type = float, default = 0.1)
	# # parser.add_argument('--save', type = bool, default = True)
	# # parser.add_argument('--verbose', type = bool, default = True)
	# args = parser.parse_args()

	# neuron1 = offline_NonLinearPerceptron(N = args.N, P = args.P, epoch = args.epoch, l2scale = args.l2scale)
	# neuron1.train()
	# print(neuron1.loss)
	# plt.plot(neuron1.I.T)
	# plt.show()
	# with open(neuron1.exp_id + ".pickle", "wb") as file_:
	# 	pickle.dump(neuron1, file_)

	# neuron2 = online_NonLinearPerceptron(N = args.N, lr = args.lr)
	# neuron2.train()
	# print(neuron2.loss)
	# plt.plot(neuron2.I.T)
	# plt.show()

	# neuron = pickle.load(open("offline_lp_N_50_P_50__norm_0.0003:26AM on November 03, 2020.pickle", "rb"))
	# print(neuron.worked)
	# plt.plot(neuron.I.T)
	# plt.show()
	