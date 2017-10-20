########## ICTP-SAIFR Minicourse on Machine Learning for Many-Body Physics ##########
### Roger Melko, Juan Carrasquilla, Lauren Hayward Sierens and Giacomo Torlai
### Tutorial 4: Sampling a Restricted Boltzmann Machine (RBM)
#####################################################################################

from __future__ import print_function
import tensorflow as tf
from rbm import RBM
import numpy as np
import os

#Input parameters:
L           = 4    #linear size of the system
num_visible = L*L  #number of visible nodes
num_hidden  = 4    #number of hidden nodes

#Temperature list for which there are trained RBM parameters stored in data_ising2d/RBM_parameters_solutions
T_list = [0.5,1.0,1.5,2.5,3.5,5.0]

#Read in nearest neighbours for the lattice:
path_to_lattice = 'data_ising2d/lattice2d_L'+str(L)+'.txt'
nn=np.loadtxt(path_to_lattice)

#Sampling parameters:
num_samples  = 500  # how many independent chains will be sampled
gibb_updates = 2    # how many gibbs updates per call to the gibbs sampler
nbins        = 100  # number of calls to the RBM sampler

observables_dir = 'data_ising2d/RBM_observables'
if not(os.path.isdir(observables_dir)):
  os.mkdir(observables_dir)
bins_filePaths = [] #file paths where bins for each T will be stored

#Initialize the RBM for each temperature in T_list:
rbms           = []
rbm_samples    = []
for i in range(len(T_list)):
  T = T_list[i]
  
  observables_filePath =  '%s/bins_nH%d_L%d' %(observables_dir,num_hidden,L)
  observables_filePath += '_T' + str(T) + '.txt'
  bins_filePaths.append(observables_filePath)
  fout = open(observables_filePath,'w')
  fout.write('# E          M           C            S\n')
  fout.close()
  
  #Read in the trained RBM parameters:
  path_to_params =  'data_ising2d/RBM_parameters_solutions/parameters_nH%d_L%d' %(num_hidden,L)
  path_to_params += '_T'+str(T)+'.npz'
  params         =  np.load(path_to_params)
  weights        =  params['weights']
  visible_bias   =  params['visible_bias']
  hidden_bias    =  params['hidden_bias']
  hidden_bias    =  np.reshape(hidden_bias,(hidden_bias.shape[0],1))
  visible_bias   =  np.reshape(visible_bias,(visible_bias.shape[0],1))

  # Initialize RBM class
  rbms.append(RBM(
    num_hidden=num_hidden, num_visible=num_visible,
    weights=weights, visible_bias=visible_bias,hidden_bias=hidden_bias,
    num_samples=num_samples
  ))
  rbm_samples.append(rbms[i].stochastic_maximum_likelihood(gibb_updates))
#end of loop over temperatures

# Initialize tensorflow
init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

# Sample thermodynamic observables:
N = num_visible
with tf.Session() as sess:
  sess.run(init)
  
  for i in range(nbins):
    print ('bin %d' %i)

    for t in range(len(T_list)):
      fout = open(bins_filePaths[t],'a')
      
      _,samples=sess.run(rbm_samples[t])
      spins = np.asarray((2*samples-1))

      #Calculate the averages of E and E^2:
      e = np.zeros((num_samples))
      e2= np.zeros((num_samples))
      for k in range(num_samples):
        for i in range(N):
          e[k] += -spins[k,i]*(spins[k,int(nn[i,0])]+spins[k,int(nn[i,1])])
        e2[k] = e[k]*e[k]
      e_avg  = np.mean(e)
      e2_avg = np.mean(e2)
      
      #Calculate the averages of |M| and M^2:
      m_avg  = np.mean(np.absolute(np.sum(spins,axis=1)))
      m2_avg = np.mean(np.multiply(np.sum(spins,axis=1),np.sum(spins,axis=1)))

      #Calculate the specific heat and susceptibility:
      c = (e2_avg-e_avg*e_avg)/float(N*T_list[t]**2)
      s = (m2_avg-m_avg*m_avg)/float(N*T_list[t])
      
      fout.write('%.8f  %.8f  %.8f  %.8f\n' % (e_avg/float(N), m_avg/float(N), c, s))
      fout.close()
