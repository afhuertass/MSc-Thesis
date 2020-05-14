import numpy as np 


from frankfurt_evn import FrankfurtEnv 



def softmax(x, axis):
    x -= np.max(x, axis=axis, keepdims=True)
    return np.exp(x) / np.exp(x).sum(axis=axis, keepdims=True)


def main2():

	config_path = "/home/afhuertas/data-science/master/thesis/finance/scripts/frankfurt_gym/envs/frankfurt_env_dir/config_test.json"

	env_test = FrankfurtEnv( )
	env_test.update_config_path( config_path )
	env_test.reset(True)
	env_test.reward_sharpe(False)
	rws = [] 
	while True:
		a = env_test.action_space.sample()
		observation , reward , done , info = env_test.step( a )
		#print(  env_test.compute_portfolio()  )
		#print( reward  )
		rws.append( reward )
		if done:
			#print( reward )
			break
	#print( len( env_test.date_index ))
	#print( len( rws ))
	#print( env_test.get_returns_df() )
	#print( env_test.compute_portfolio() )
	dd = env_test.get_returns_df()
	#print( dd.head() )
	#print( env_test.port_serie )
	#print( dd.head() )
	#print( dd.sum() )
	#print( dd.max() )
	#print( dd.min() )
	print("Done")

def main():

	env = FrankfurtEnv( serial = False  )

	print( env.observation_space.shape )
	print( env.action_space.shape )
	M = env.action_space.shape[0 ]
	porfs = [ ]
	rws = []
	for i_episode in range(200):
		observation = env.reset(  )
		#rws = [ ]
		for t in range(1000):
		#env.render()
			#print(observation)
			action =  env.action_space.sample()#np.random.uniform( size = (M) )
			#ALL CASH 
			#action = -1*np.ones( ( M) ) 
			#action[0] = 1 
			#print( action )
			

			observation, reward, done, info = env.step(  action )
			rws.append( reward )
			#rws.append( reward )
			#print( reward )
			if done:

				print( "avg reward"  , np.array( rws ).mean()  )
				#print( "min reward"  , np.array( rws ).min()  )
				#print( "max reward"  , np.array( rws ).max()  )
				#print( "porfolio max" , env.portfolio_prev)
				#print(  np.array( rws).min() )
				p =  env.compute_portfolio()
				#print( p )
				porfs.append( p )
				#print( rws )
				#print( env.step_count )
				#print( env.portfolio_prev )

				#print("Episode finished after {} timesteps".format(t+1))

				break

	print( "Initial portfolio value" , env.initial_worth )
	print( "Portfolio avg " , np.array( porfs).mean() )
	print( "Portfolio std " , np.array( porfs).std() )


	print("Nice")

if __name__ == "__main__":
	main2()