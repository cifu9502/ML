########## ICTP-SAIFR Minicourse on Machine Learning for Many-Body Physics ##########
### Roger Melko, Juan Carrasquilla and Lauren Hayward Sierens
### Tutorial 1: Monte Carlo for the Ising lattice gauge theory
#####################################################################################

import numpy as np
import random

### Input parameters: ###
T_list = np.linspace(5.0,0.5,19) #temperature list
L = 16            #linear size of the lattice
N_sites = L**2  #total number of lattice sites
N_spins = 2*L**2 #total number of spins (one spin on each link)
J = 1            #coupling parameter

### Monte Carlo parameters: ###
n_eqSweeps = 10000   #number of equilibration sweeps
n_bins = 500    #total number of measurement bins
n_sweepsPerBin=50 #number of sweeps performed in one bin

### Initially, the spins are in a random state (a high-T phase): ###
spins = np.zeros(N_spins,dtype=np.int)
for i in range(N_spins):
  spins[i] = 2*random.randint(0,1) - 1 #either +1 or -1

### Store each lattice site's four nearest neighbours in a neighbours array (using periodic boundary conditions): ###
neighbours = np.zeros((N_sites,4),dtype=np.int)
for i in range(N_sites):
  #neighbour to the right:
  neighbours[i,0]=i+1
  if i%L==(L-1):
    neighbours[i,0]=i+1-L
  
  #upwards neighbour:
  neighbours[i,1]=i+L
  if i >= (N_sites-L):
    neighbours[i,1]=i+L-N_sites
  
  #neighbour to the left:
  neighbours[i,2]=i-1
  if i%L==0:
    neighbours[i,2]=i-1+L
  
  #downwards neighbour:
  neighbours[i,3]=i-L
  if i <= (L-1):
    neighbours[i,3]=i-L+N_sites
#end of for loop

### Function to calculate the total energy: ###
def getEnergy():
  currEnergy = 0
  for i in range(N_sites):
    currEnergy += -J*getPlaquetteProduct(i)
  return currEnergy
#end of getEnergy() function

### Function to calculate the product of spins on plaquette i: ###

def getPlaquetteProduct(i):
  s1= spins[2*i]
  s2= spins[2*i+1]
  s3= spins[neighbours[i,0]*2+1]
  s4= spins[neighbours[i,1]*2]
  # ************************************************************** #
  # *** FILL IN: CALCULATE THE PRODUCT OF SPINS ON PLAQUETTE i *** #
  # ************************************************************** #
  return s1*s2*s3*s4

### Function to perform one Monte Carlo sweep ###
def sweep():
  #do one sweep (N_spins local updates):
  for i in range(N_spins):
    #randomly choose which spin to consider flipping:
    spinLoc = random.randint(0,N_spins-1)
    
    # ************************************************************ #
    # *** FILL IN: CALCULATE deltaE FOR THE PROPOSED SPIN FLIP *** #
    # ************************************************************ #
    deltaE = 0
    if i%2 == 0: 
        deltaE += 2*J*getPlaquetteProduct(spinLoc/2)
        deltaE += 2*J*getPlaquetteProduct(neighbours[spinLoc/2,3])
    else:    
        deltaE += 2*J*getPlaquetteProduct((spinLoc-1)/2)
        deltaE += 2*J*getPlaquetteProduct(neighbours[(spinLoc-1)/2,2])
  
    if (deltaE <= 0) or (random.random() < np.exp(-deltaE/T)):
      #flip the spin:
      spins[spinLoc] = -spins[spinLoc]
  #end loop over i
#end of sweep() function

#################################################################################
########## Loop over all temperatures and perform Monte Carlo updates: ##########
#################################################################################

def flipSpin():
  i = random.randint(0,N_sites-1)
  spins[2*i] = -spins[2*i]
  spins[2*i+1] = -spins[2*i+1]
  spins[neighbours[i,0]*2+1] = -spins[neighbours[i,0]*2+1] 
  spins[neighbours[i,1]*2] = -spins[neighbours[i,1]*2]


numdata = 10000  
T= 0.000001
fileName         = 'txt/Xtrain.txt' 
file_observables = open(fileName, 'w')
fileName2         = 'txt/ytrain.txt' 
file_test = open(fileName2, 'w')
for i in range(n_eqSweeps):
    sweep()
for n in range(numdata):
  N_sites
  flipSpin()
  #fileName         = 'T=0/gaugeTheory2d_L%d_T%.4f_n%d.txt' %(L,T,n)
  #file_observables = open(fileName, 'w')
  for s in range(N_spins):
    file_observables.write('%d ' %(((spins[s]+1)/2)))
  file_observables.write('\n')
  file_test.write('0\n')  
T= 9999999
#fileName         = 'Train_T=infty.txt' %(L,T)
#file_observables = open(fileName, 'w')
for i in range(n_eqSweeps):
    sweep()
for n in range(numdata):
  N_sites
  flipSpin()
  for s in range(N_spins):
    file_observables.write('%d ' %(((spins[s]+1)/2)))
  file_observables.write('\n')
  file_test.write('1\n')  
file_observables.close()  
file_test.close() 
    
numdata = 200  
T= 0.000001
fileName         = 'txt/Xtest.txt' 
file_observables2 = open(fileName, 'w')
fileName2         = 'txt/ytest.txt' 
file_test2 = open(fileName2, 'w')
for i in range(n_eqSweeps):
    sweep()
for n in range(numdata):
  N_sites
  flipSpin()
  #fileName         = 'T=0/gaugeTheory2d_L%d_T%.4f_n%d.txt' %(L,T,n)
  #file_observables = open(fileName, 'w')
  for s in range(N_spins):
    file_observables2.write('%d ' %(((spins[s]+1)/2)))
  file_test2.write('0\n')  
  file_observables2.write('\n')  

T= 9999999
#fileName         = 'Train_T=infty.txt' %(L,T)
#file_observables = open(fileName, 'w')
for i in range(n_eqSweeps):
    sweep()
for n in range(numdata):
  N_sites
  flipSpin()
  for s in range(N_spins):
    file_observables2.write('%d ' %(((spins[s]+1)/2)))
  file_test2.write('1\n')
  file_observables2.write('\n')   
file_observables2.close()
file_test2.close() 
