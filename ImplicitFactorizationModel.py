import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim

import torch_utils; 
from torch_utils import gpu, minibatch, shuffle, hinge_loss, sample_items
from models import DotModel

class ImplicitFactorizationModel(object):
	def __init__(self,
		embedding_dim=32,
		n_iter=10,
		batch_size=256,
		l2=0.0,
		learning_rate=1e-3,
		use_cuda=False,
		net=None,
		num_users=None,
		num_items=None,
		random_state=None,
		loss=None):
		self._embedding_dim = embedding_dim
		self._n_iter = n_iter
		self._learning_rate = learning_rate
		self._batch_size = batch_size
		self._l2 = l2
		self._use_cuda = use_cuda
		self._num_users = num_users
		self._num_items = num_items
		self._net = net
		self._optimizer = None
		self._loss_func = loss
		self._random_state = random_state or np.random.RandomState()
			 
		
	def _initialize(self):
		if self._net is None:
			self._net = gpu(DotModel(self._num_users, self._num_items, self._embedding_dim),self._use_cuda)
		
		self._optimizer = optim.Adam(
				self._net.parameters(),
				lr=self._learning_rate,
				weight_decay=self._l2
			)
		
		if self._loss_func is None:
			self._loss_func = hinge_loss
	
	@property
	def _initialized(self):
		return self._optimizer is not None

	def fit(self, user_ids, item_ids, verbose=True):
		
		# Your code here
		# hint: copy and paste well-chosen pieces of the fit method in the ExlicitFactorization Model class
		#		use the _get_negative_prediction method below

	def _get_negative_prediction(self, user_ids):

		negative_items = sample_items(
			self._num_items,
			len(user_ids),
			random_state=self._random_state)
		negative_var = Variable(
			gpu(torch.from_numpy(negative_items), self._use_cuda)
		)
		negative_prediction = self._net(user_ids, negative_var)

		return negative_prediction

	def predict(self, user_ids, item_ids):
		
		# Your code here
			