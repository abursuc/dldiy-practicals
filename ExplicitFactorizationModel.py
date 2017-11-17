import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim

import torch_utils; 
from torch_utils import gpu, minibatch, shuffle, regression_loss
from models import DotModel

class ExplicitFactorizationModel(object):
		
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
						self._loss_func = regression_loss
		
		@property
		def _initialized(self):
				return self._optimizer is not None
		
				
		def fit(self, user_ids, item_ids, ratings, user_ids_test, item_ids_test, ratings_test, verbose=True):
				
				user_ids = user_ids.astype(np.int64)
				item_ids = item_ids.astype(np.int64)
				user_ids_test = user_ids_test.astype(np.int64)
				item_ids_test = item_ids_test.astype(np.int64)
				
				if not self._initialized:
						self._initialize()
						
				for epoch_num in range(self._n_iter):
						users, items, ratingss = shuffle(user_ids,
																						item_ids,
																						ratings)

						user_ids_tensor = gpu(torch.from_numpy(users),
																	self._use_cuda)
						item_ids_tensor = gpu(torch.from_numpy(items),
																	self._use_cuda)
						ratings_tensor = gpu(torch.from_numpy(ratingss),
																 self._use_cuda)
						epoch_loss = 0.0

						for (minibatch_num,
								 (batch_user,
									batch_item,
									batch_ratings)) in enumerate(minibatch(self._batch_size,
																												 user_ids_tensor,
																												 item_ids_tensor,
																												 ratings_tensor)):
								user_var = Variable(batch_user)
								item_var = Variable(batch_item)
								ratings_var = Variable(batch_ratings)
								
				
								predictions = self._net(user_var, item_var)

								self._optimizer.zero_grad()
								
								loss = self._loss_func(ratings_var, predictions)
								
								epoch_loss = epoch_loss + loss.data[0]
								
								loss.backward()
								self._optimizer.step()
								
						
						epoch_loss = epoch_loss / (minibatch_num + 1)

						if verbose:
								val_loss = self.test(user_ids_test, item_ids_test, ratings_test)
								print('Epoch {}: train loss {}'.format(epoch_num, epoch_loss), 'validation loss', val_loss)
								self._net.train(True)
												
						if np.isnan(epoch_loss) or epoch_loss == 0.0:
								raise ValueError('Degenerate epoch loss: {}'
																 .format(epoch_loss))
		
		
		def test(self,user_ids, item_ids, ratings):
				self._net.train(False)
				user_ids = user_ids.astype(np.int64)
				item_ids = item_ids.astype(np.int64)
				
				user_ids_tensor = gpu(torch.from_numpy(user_ids),
																	self._use_cuda)
				item_ids_tensor = gpu(torch.from_numpy(item_ids),
																	self._use_cuda)
				ratings_tensor = gpu(torch.from_numpy(ratings),
																 self._use_cuda)
				
				user_var = Variable(user_ids_tensor)
				item_var = Variable(item_ids_tensor)
				ratings_var = Variable(ratings_tensor)
				
				predictions = self._net(user_var, item_var)
				
				loss = self._loss_func(ratings_var, predictions)
				return loss.data[0]


		def predict(self, user_ids, item_ids):
			self._net.train(False)
			user_ids = user_ids.astype(np.int64)
			item_ids = item_ids.astype(np.int64)
			user_ids_tensor = gpu(torch.from_numpy(user_ids),
																	self._use_cuda)
			item_ids_tensor = gpu(torch.from_numpy(item_ids),
																	self._use_cuda)
			user_var = Variable(user_ids_tensor)
			item_var = Variable(item_ids_tensor)
			return self._net(user_var, item_var).data.numpy()
