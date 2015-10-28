from rl import animate_rollout, ValueFunction, pathlength
from atari import AtariMDP
import ppo
import numpy as np
from tabulate import tabulate
from prepare_h5_file import prepare_h5_file
import argparse
from atari_ram_policy import AtariRAMPolicy
import ipdb
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
import matplotlib.pyplot as plt

class AtariRamLinearValueFunction(ValueFunction):
    coeffs = None
    def _features(self, path):
        o = path["observations"].astype('float64')/256.0 - .5
        #ipdb.set_trace()
        l = pathlength(path)
        al = np.arange(l).reshape(-1,1) / 50.0
        return np.concatenate([o, al, al**2, np.ones((l,1))], axis=1)
    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        # ipdb.set_trace()
        self.coeffs = np.linalg.lstsq(featmat, returns)[0]
    def predict(self, path):
        return np.zeros(pathlength(path)) if self.coeffs is None else \
                self._features(path).dot(self.coeffs)

class AtariRamForestValueFunction(ValueFunction):
    coeffs = None
    def _features(self, path):
        o = path["observations"].astype('float64')/256.0 - .5
        l = pathlength(path)
        al = np.arange(l).reshape(-1,1) / 50.0
        return np.concatenate([o, al, al**2, np.ones((l,1))], axis=1)
    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        self.clf = RandomForestRegressor(n_estimators = 100)
        self.clf.fit(featmat, returns)
    def predict(self, path):
        return np.zeros(pathlength(path)) if self.coeffs is None else \
                self.clf.predict(self._features(path))

class AtariRamNeuralValueFunction(ValueFunction):
    coeffs = None
    def _features(self, path):
        o = path["observations"].astype('float64')/256.0 - .5
        l = pathlength(path)
        al = np.arange(l).reshape(-1,1) / 50.0
        return np.concatenate([o, al, al**2, np.ones((l,1))], axis=1)
    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        model = Sequential()
        model.add(Dense(output_dim = 256, input_dim = featmat.shape[1], 
                        init="glorot_uniform"))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim = 256, init = "glorot_uniform"))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim = 1, init="glorot_uniform"))
        model.add(Activation("linear"))

        model.compile(loss='mean_squared_error', optimizer='adagrad')
        model.fit(featmat, returns, batch_size=128, nb_epoch=5)
        self.clf = model

    def predict(self, path):
        return np.zeros(pathlength(path)) if self.coeffs is None else \
                self.clf.predict(self._features(path))

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--outfile")
    parser.add_argument("--metadata")
    parser.add_argument("--plot",type=int,default=0)
    parser.add_argument("--game",type=str,choices=["pong","breakout","enduro","beam_rider","space_invaders","seaquest","qbert"],default='pong')

    # Parameters
    parser.add_argument("--n_iter",type=int,default=500)
    parser.add_argument("--gamma",type=float,default=.99)
    parser.add_argument("--lam",type=float,default=1.00)
    parser.add_argument("--timesteps_per_batch",type=int,default=100000)
    parser.add_argument("--penalty_coeff",type=float,default=0.5)
    parser.add_argument("--max_pathlength",type=int,default=10000)
    parser.add_argument("--max_kl",type=float,default=.01)

    args = parser.parse_args()

    np.random.seed(args.seed)

    mdp = AtariMDP('atari_roms/%s.bin'%args.game)
    policy = AtariRAMPolicy(mdp.n_actions)
    # vf = AtariRamLinearValueFunction()
    # vf = AtariRamForestValueFunction()
    vf = AtariRamNeuralValueFunction()

    hdf, diagnostics = prepare_h5_file(args, {"policy" : policy, "mdp" : mdp})


    for (iteration,stats) in enumerate(ppo.run_ppo(
            mdp, policy, 
            vf=vf,
            gamma=args.gamma,
            lam=args.lam,
            max_pathlength = args.max_pathlength,
            timesteps_per_batch = args.timesteps_per_batch,
            n_iter = args.n_iter,
            parallel=True,
            max_kl = 0.04,
            penalty_coeff=args.penalty_coeff)):

        print tabulate(stats.items())

        for (statname, statval) in stats.items():
            diagnostics[statname].append(statval)

        if args.plot:
            animate_rollout(mdp,policy,delay=.001,horizon=100)

        grp = hdf.create_group("snapshots/%.4i"%(iteration))
        policy.pc.to_h5(grp)
        
        if False:
            plt.figure()
            plt.title("Episode Reward")
            EpRewMean = np.array(diagnostics["EpRewMean"])
            EpRewStd = np.array(diagnostics["EpRewSEM"])
            plt.errorbar(np.arange(len(EpRewMean)), EpRewMean, yerr=EpRewStd, 
                         errorevery=5, linewidth=1)
            plt.savefig('./Output/%s %s Episode Reward.pdf' %(vf.__class__.__name__,
                                                           args.game ))
            plt.figure()
            plt.title("Mean Episode Length")
            plt.plot(diagnostics["EpLenMean"])
            plt.savefig('./Output/%s %s Mean Episode Length.pdf' %(
                            vf.__class__.__name__, args.game ))

            plt.figure()
            plt.title("Perplexity")
            plt.plot(diagnostics["Perplexity"])
            plt.savefig('./Output/%s %s Perplexity.pdf' %(vf.__class__.__name__,
                                                           args.game ))

            plt.figure()
            plt.title("Mean KL Divergence Between Old & New Policies")
            plt.plot(diagnostics["KLOldNew"]);
            plt.savefig('./Output/%s %s KL Old New.pdf' %(vf.__class__.__name__,
                                                           args.game ))
            
            plt.figure()
            plt.title("Reward wrt Running Time")
            plt.plot(diagnostics['TimeElapsed'], EpRewMean)
            plt.savefig('./Output/%s %s Reward vs RunTime.pdf' %(vf.__class__.__name__,
                                                           args.game ))


if __name__ == "__main__":
    main()

