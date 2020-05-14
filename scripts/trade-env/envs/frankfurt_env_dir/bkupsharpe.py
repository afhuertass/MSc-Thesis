    """
    daily_returns = np.exp( obs[ : , : , 0] ) - 2 
    # compute sharpe ratio 
    mean_return = ( daily_returns ) .mean( axis = 1 )
    #print("meaaaan returrrn")
    #print( mean_return )
    portfolio_mean_return = np.dot( self.current_weights , mean_return )
    covariance_matrix = np.cov( daily_returns   )
    #print( "covariance shape")
    #print( covariance_matrix.shape )
    portfolio_std = np.dot( self.current_weights.T  , np.dot( covariance_matrix ,  self.current_weights ) ) 
    #print("porfolio std")
    #print( np.sqrt( portfolio_std )  )
    #print( "porfolio mean return")
    #print( portfolio_mean_return )
    sharpe_ratio = 0.0 

    if portfolio_std == 0.0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio =  portfolio_mean_return/np.sqrt( portfolio_std )

    if np.isnan( sharpe_ratio ):
        sharpe_ratio = 0.0

        print("shit went bad ")
        #print( type( sharpe_ratio ) ) 
        #print( sharpe_ratio )
        print( portfolio_mean_return )
        print( portfolio_std )
        print( self.current_weights )
        #print( self.current_weights[ self.current_weights == np.nan ]) 
        #print( action[ action == np.nan ]) 
        #print( '#'*20)
    """
    # compute transaction cost
    #returns_mean = np.exp( obs[ : , : , 0 ] ) - 1   # pt/pt_1
    #returns_mean = returns_mean.mean( axis = 1 )
    #cost_t = self.cost_transaction( action )
    #risk_t = self.compute_risk( returns_mean , action )