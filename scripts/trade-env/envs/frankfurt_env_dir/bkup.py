
  def compute_reward( self , action , current_logreturn , returns  ):
    # action is portfolio weight [ M , 1 ]
    # current_logreturn [ M , 1 ] 
    # COMPUTE SHARPE RATIO    WT 
    if action[0] >= 1.0 :
        return 5.0
    else: 
        return 0.0 
    mean_return = returns.mean( axis = 1 )
    portfolio_mean_return = np.dot( action , mean_return )

    covariance_matrix = np.cov( returns )

    portfolio_std = np.matmul( action.T  , np.matmul( covariance_matrix , action ) ) 

    # compute sharpe ratio portoflio over risk
    #print( portfolio_mean_return )

    sharpe_ratio =  portfolio_mean_return/np.sqrt( portfolio_std ) 

    return mean_return.mean()


  def compute_reward2( self, action , time_step  ):


    val = 0 
    for   hi , pi in zip(  self.previous_holdings , self.prices[ : , time_step ] ):
        val += pi*hi

    reward = val - self.portfolio_prev  

    self.previous_holdings = self.compute_holdings( action , val , self.prices[: , time_step ] )
    self.portfolio_prev = val


    #print( "")
    return reward