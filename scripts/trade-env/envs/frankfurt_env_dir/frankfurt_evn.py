import pandas as pd
import numpy as np 
import gym
from gym import spaces
import pickle 
import json 
import random
import math

config_path = "/PATH_TO_CONFIG_FILE/config.json"


def softmax(x, axis):
    x -= np.max(x, axis=axis, keepdims=True)
    return np.exp(x) / np.exp(x).sum(axis=axis, keepdims=True)


class FrankfurtEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}



  def __init__(self , serial = False  ):


    super(FrankfurtEnv, self).__init__()    # Define action and observation space

    self.port_serie = [ ]
    self.initial_worth = 10000.0
    self.config_path = config_path
    self.load_config( )
    self.load_data(  self.start_date , self.end_date , self.stocks_names  )

    self.M = len( self.stocks_names )
    # the action space is the portfolio vector,
    self.step_count = self.window_size

    low = -1*np.ones( ( self.M) )
    high = np.ones(( self.M))
    self.action_space = spaces.Box( low = low , high = high , dtype=np.float64 )

    high = np.ones( ( self.M,self.window_size,3    ) )*self.data.max()
    low = np.ones( ( self.M,self.window_size, 3   ) )*self.data.min()

    self.observation_space = spaces.Box(low= low ,  high=high, dtype=np.float64 )

    self.action_type = "continuous"
    # PRICES AT TIME ZERO, CAN BE CHANGED TO START AT ANY TIME IN THE FUTURE. 
    self.current_weights = -1*np.zeros( ( self.M ) )
    self.current_weights[0] = 1.0 
    #self.current_weights[0] = 1.0 
    #self.previous_weights[0] = 1.0 # everything is invested on cash in the first stage 
    
    # serial = FALSE is for training, takes random slices from the code to create 
    # serial = True is for training, serially transverse the data set. 
    self.serial = serial
    self.MAX_DAYS = len( self.data[0] )
    #print( self.MAX_DAYS )
    # reset steps 
    self.rewards = []
    self.sharpe_rw = True

    self.alpha = 1.0
    self.beta = 0.002 
    self.reset()

  def reward_sharpe( self , sharpe_rw):

    self.sharpe_rw = sharpe_rw 

  def step(self, action):
    # Execute one time step within the environment
    # NORMALIZE ACTION  and assert is valid 
    #action = softmax( action  , axis = -1 )
    #print( action )
    #print( action.sum() )
    action = ( action + 1.0 )/2.0
    action = action/action.sum()
    #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

    # the observation data is the log returns for the previous self.window_size days + the current weights 

    observation = self.get_observation( self.step_count , self.window_size )

    #a = action.reshape( (-1 , 1 ))
    #observation = np.hstack( (observation , a)  )
    #assert self.observation_space.contains(observation), "%r (%s) invalid observation"%(observation, type(observation))
    reward = self.compute_reward2( action , self.step_count , observation  )
        

    # the current weights are the given action
    if self.step_count >= self.max_step:

      done = True 
    else:
      done = False
    #print( self.step_count )
    self.step_count += 1

    info = {  }
    return  observation , reward , done , info
    

  def compute_reward2( self , action , step_count , obs   ):
    # The sharpe 
    #obs = np.log( obs )
    #print( obs.shape )
    #print("jiaaaaoooo")
    rate_return = np.exp( obs[ : , self.window_size - 1 , 0 ] ) - 1   # pt/pt_1

    #simple_return = rate_return - 1  # pt/pt_1 - 1 

    rt = np.dot( rate_return , self.current_weights  ) # pt/pt_1*w 
    rt = math.log( rt )
    #simple_return_port = np.dot( simple_return , self.current_weights )

    self.current_weights = action
    self.rewards.append(  rt )
    
    #print( "Sharpe ratio")
    #print( sharpe_ratio )

    p = self.compute_portfolio()
    self.port_serie.append( p )

    #return final_p 
    #reward = simple_return_port*100 #- cost_t*100  #- risk_t*100
    #reward2 = simple_return_port/risk_t 

    return  rt*100 #*100#reward #simple_return_port*100 #math.log( rt )*100  #(final_p/init_p  - 1 )*100

  def compute_portfolio( self ):
    #np.exp( np.array( rws ).sum() )
    if len( self.rewards ) == 0:
        portfolio_value = self.initial_worth
    else:
        #portfolio_value = self.initial_worth*np.array( self.rewards ).prod()
        portfolio_value = self.initial_worth*np.exp(  np.array( self.rewards ).sum() )  
    #print( portfolio_value , "pf" , self.rewards )
    return portfolio_value

  def reset(self , serial = False ):
# Reset the state of the environment to an initial state

    
    #self.step_count = self.window_size
    # duration of the simularion
    self.serial = serial 
    self.rewards = [ ]
    self.port_serie = [ ]

    if self.serial:
        self.step_count = self.window_size
        self.MAX_DAYS = self.data.shape[1] 
        self.max_step = self.MAX_DAYS -1 

    else: 
        TRADING_DURATION = 90
        init_step = np.random .randint( self.window_size,  self.MAX_DAYS -   TRADING_DURATION )
        #max_step = np.random.randint( init_step ,  self.MAX_DAYS )

        #total_steps = max_step - init_step
        self.TRADING_DURATION = TRADING_DURATION
        self.step_count = init_step
        self.max_step = self.step_count + TRADING_DURATION


    obs = self.get_observation( self.step_count , self.window_size )
   
    #
    self.current_weights = -1*np.zeros( ( self.M ) )
    self.current_weights[0] = 1.0 

    return obs # .flatten()

  def get_observation( self,  time_step , window_size ):

    return self.data[ : , time_step - window_size:  time_step, :  ]

  def render(self, mode='human', close=False):
# Render the environment to the screen
    return None

  def update_config_path( self, new_config ):
    self.config_path = new_config
    self.load_config()
    self.load_data(  self.start_date , self.end_date , self.stocks_names  )
    self.reset()

  def load_config( self ):
    
    with open( self.config_path , "r") as f:
        #x = f.replace("'", '"')
        s = f.read()
        s = s.replace('\'','\"')
        s = s.replace("'", '"')
        #print(s)
        data = json.loads(s)

    self.data_path = data["data_path"]
    self.stocks_names = data["stocks"]
    self.start_date = data["start_date"]
    self.end_date = data["end_date"]
    self.window_size = data["window_size"]

## HELPER  FUNCTIONS

  def load_data(self  , start_date , end_date , stocks   ):

    # open dictorninary with the full data 
    self.path_bundle = self.data_path
    self.data = self.build_logreturns_matrix( start_date , end_date , stocks  )

    self.data = np.log( self.data + 1  )
    # load
    return None 


  def build_logreturns_stock( self, df , start_date , end_date   ):
    # df [ "date" , "Close"]
    #print( start_date , end_date )
    new_index = pd.date_range( start = start_date ,  end= end_date , freq = "D" )
    self.date_index = new_index
    df["date"] = pd.to_datetime(df['date'])
    df = df.set_index( ["date"])
    #print( df.index )
    df_p = df["close"].reindex( new_index , method = "ffill")
    P1 = df_p.values/df_p.shift(1).values

    df_p = df["high"].reindex( new_index , method = "ffill")
    P2 = df_p.values/df_p.shift(1).values 

    df_p = df["low"].reindex( new_index , method = "ffill")
    P3 = df_p.values/df_p.shift(1).values 


    obs = np.stack( [ P1 , P2 , P3] , axis = -1 )
    #log_r = np.log(df_p.values ) - np.log(df_p.shift(1).values  )
    
    # returns the closing price and the obs
    return df["close"].reindex(new_index , method = "ffill").values , obs 

  def build_logreturns_matrix( self , start_date , end_date , stocks = [ 'CBK_X','DTE_X','EON_X'] ):
      
    self.stocks = [ ]

    #closing prices
    self.close_prices = [ ]
    self.returns = [ ]
    self.price_tensor = [ ]
    for stock in stocks:
        path = "{}/{}.csv".format( self.path_bundle , stock )
        #print( path )
        df = pd.read_csv( path)
        
        prices , obs  = self.build_logreturns_stock( df , start_date , end_date )

        self.close_prices.append( prices )
        self.stocks.append( stock )
        self.price_tensor.append( obs )
        
        
    self.close_prices = np.array( self.close_prices )
    self.close_prices = np.nan_to_num( self.close_prices )

    self.price_tensor = np.array( self.price_tensor )
    self.price_tensor = np.nan_to_num( self.price_tensor )

    return self.price_tensor

  def cost_transaction( self , action ) :

    return self.beta*np.abs(  action - self.current_weights ).sum()

  def compute_risk( self , returns , action ):

    covariance_matrix = np.cov( returns   )
    #print( "covariance shape")
    #print( covariance_matrix.shape )
    portfolio_std = np.dot( action.T  , np.dot( covariance_matrix ,  action ) )

    return self.alpha*math.sqrt( portfolio_std )

  def get_returns_df(self ):

    df = pd.DataFrame( data = self.port_serie , columns = ["P"] )
    ret = df["P"]/df["P"].shift(1) - 1
    ret = ret.fillna(0.0)
    return pd.Series( data = ret.values , index = self.date_index[self.window_size:] )


