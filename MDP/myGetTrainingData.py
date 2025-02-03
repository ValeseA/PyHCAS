import h5py
import numpy as np

###### OPTIONS #######
tableFile = './Qtables/HCAS_oneSpeed_v6_diff_dists_good_w.h5'
tableFile_original = './Qtables/HCAS_oneSpeed_v6.h5'
trainingDataFiles = './TrainingData/my_HCAS_rect_TrainingData_v6_pra%d_tau%02d.h5'
useRect = False      # Set to true to use rectangular coordinates (x,y) instead of polar coordinates(rho,theta) 
                    # as inputs to neural network. Changing to rectangular coordinates was used to ease reachability equations.
                    # The two coordinate systems are related using: x = rho * cos(theta), y = rho * sin(theta)
                    # 
oneSpeed = True     # If the speed dimension has only one value, set to True to remove the speed dimensions from network input
                    # If true, the neural network will have only three inputs: x, y, psi if useRect, or rho, theta, psi otherwise.
                    # If false, neural network will five inputs: x, y, psi, vown, vint if useRect, or rho, theta, psi, vown, vint otherwise
saveTaus = [0,5,10,15,20,30,40,60] # Tau values at which neural networks will be trained
######################
# Load Q table
f = h5py.File(tableFile, 'r')
fo = h5py.File(tableFile_original, 'r')

Q_original = np.array(fo['q'])
Q = np.array(f['q'])
ranges = np.array(f['ranges'])
thetas = np.array(f['thetas'])
psis = np.array(f['psis'])
f.close()
###############################
# Define state space. Make sure this matches up with the constants used to generate the MDP table!
acts = [0,1,2,3,4]

vowns = [200.0]
vints = [200.0] 
taus  = np.linspace(0,60,61)

ranges_original = np.array([0.0,25.0,50.0,75.0,100.0,150.0,200.0,300.0,400.0,500.0,510.0,750.0,1000.0,1500.0,2000.0,3000.0,4000.0,5000.0,7000.0,9000.0,11000.0,13000.0,15000.0,17000.0,19000.0,21000.0,25000.0,30000.0,35000.0,40000.0,48000.0,56000.0])
thetas_original = np.linspace(-np.pi,np.pi,41)
psis_original  = np.linspace(-np.pi,np.pi,41)

###############################

# Get table cutpoints depending on useRect and oneSpeed settings
if useRect:
    if oneSpeed:
        X = np.array([[r*np.cos(t),r*np.sin(t),p,] for p in psis for t in thetas for r in ranges])
    else:
        X = np.array([[r*np.cos(t),r*np.sin(t),p,vo,vi] for vi in vints for vo in vowns for p in psis for t in thetas for r in ranges])
else:
    if oneSpeed:
        X = np.array([[r,t,p] for p in psis for t in thetas for r in ranges])
        X_original = np.array([[r,t,p] for p in psis_original for t in thetas_original for r in ranges_original])

    else:
        X = np.array([[r,t,p,vo,vi] for vi in vints for vo in vowns for p in psis for t in thetas for r in ranges])

# Compute means, ranges, mins and maxes of inputs
means = np.mean(X, axis=0)
rnges = np.max(X, axis=0) - np.min(X, axis=0)
min_inputs = np.min(X, axis=0)
max_inputs = np.max(X, axis=0)


means_original = np.mean(X_original, axis=0)
rnges_original = np.max(X_original, axis=0) - np.min(X_original, axis=0)
min_inputs_original = np.min(X_original, axis=0)
max_inputs_original = np.max(X_original, axis=0)

# Normalize each dimension of inputs to have 0 mean, unit range
# If only one value, then range is 0. Just divide by 1 instead of range
rnges = np.where(rnges==0.0, 1.0, rnges)
#X  = (X - means) / rnges

rnges_original = np.where(rnges_original==0.0, 1.0, rnges_original)
X_original  = (X_original - means_original) / rnges_original
X = (X - means_original) / rnges_original

print(means_original, rnges_original)

Q = Q.T
Q_original = Q_original.T

# Normalize entire output data to have 0 mean, unit range
# Add output normalization to the inputs means and rnges vector
meanQ = np.mean(Q)
meanQ_original = np.mean(Q_original)

rangeQ = np.max(Q) - np.min(Q)
rangeQ_original = np.max(Q_original) - np.min(Q_original)

Q = (Q - meanQ) / rangeQ
Q_original = (Q_original - meanQ_original) / rangeQ_original

means = np.concatenate((means,[meanQ]))
means_original = np.concatenate((means_original,[meanQ_original]))

rnges = np.concatenate((rnges,[rangeQ]))
ranges_original = np.concatenate((rnges_original,[rangeQ_original]))



# Sizes to help slice the table to create subtables used for training separate networks
ns2 = len(ranges) * len(thetas) * len(psis) * len(vowns) * len(vints) * len(acts)
ns3 = len(ranges) * len(thetas) * len(psis) * len(vowns) * len(vints)

#Save the Training Data
for tau in saveTaus:
    Qsub = Q[tau*ns2:(tau+1)*ns2]
    for pra in acts:
        Qsubsub = Qsub[pra*ns3:(pra+1)*ns3]
        with h5py.File(trainingDataFiles%(pra,tau),'w') as H:
            H.create_dataset('X', data=X)
            H.create_dataset('y', data=Qsubsub)
            H.create_dataset('means', data=means)
            H.create_dataset('ranges', data=rnges)
            H.create_dataset('min_inputs', data=min_inputs)
            H.create_dataset('max_inputs', data=max_inputs)