import numpy as np
import pandas as pd

class DataReader:
	def read_movies(self, filename) :
		movies = pd.read_csv(filename , sep = '::')
		return movies

	def read_users(self, filename) :
		users = pd.read_csv(filename , sep = '::')
		return users

	def read_train(self, filename) :
		train = pd.read_csv(filename , sep = ',')	

		Users = train['UserID'].as_matrix()
		Movies = train['MovieID'].as_matrix()
		Ratings = train['Rating'].as_matrix()
		max_userid = train['UserID'].drop_duplicates().max()
		max_movieid = train['MovieID'].drop_duplicates().max()
		
		print (Users.shape)
		idx = np.arange(Users.shape[0])
		np.random.shuffle(idx)	
		print (idx)

		X_train = [Users[idx], Movies[idx]]
		Y_train = Ratings[idx]

		return X_train , Y_train , max_userid , max_movieid

	def read_test(self, filename) :
		test = pd.read_csv(filename , sep = ',')

		Users = test['UserID'].as_matrix()
		Movies = test['MovieID'].as_matrix()

		X_test = [Users, Movies]
		
		return X_test