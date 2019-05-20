import os, inspect, glob, time

import numpy as np

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

class DataSet(object):

    def __init__(self, key_tr):

        print("\n** Prepare the Dataset")

        self.data_path = os.path.join(PACK_PATH, "dataset")
        self.symbols = ['_V_', '_F_', '_O_', '_!_', '_e_', '_j_', '_E_', '_P_', '_f_', '_p_', '_Q_']

        self.subdir = glob.glob(os.path.join(self.data_path, "*"))
        self.subdir.sort() # sorting the subdir list is optional.
        for idx, sd in enumerate(self.subdir):
            self.subdir[idx] = sd.split('/')[-1]

        # List
        self.key_tr = key_tr
        self.key_tot = self.subdir
        # Dictionary
        self.list_total = {}

        for ktot in self.key_tot:
            self.list_total["%s" %(ktot)] = glob.glob(os.path.join(self.data_path, "%s" %(ktot), "*.npy"))
            self.list_total["%s" %(ktot)].sort() # Must be sorted.

        # Information of dataset
        self.am_tot = len(self.subdir)
        self.am_tr = len(self.key_tr)
        self.am_each = len(self.list_total[self.subdir[0]])
        self.data_dim = np.load(self.list_total[self.subdir[0]][0]).shape[0]
        # Variable for using dataset
        self.kidx_tr = 0
        self.kidx_tot = 0
        self.didx_tr = 0
        self.didx_tot = 0

        print("Total Record : %d" %(self.am_tot))
        print("Trining Set  : %d" %(self.am_tr))
        print("Each record was parsed to %d sequence." %(self.am_each))
        print("Each data has %d dimension." %(self.data_dim))

    def next_batch(self, batch_size, sequence_length, v_key=None):

        data_bat = np.zeros((0, sequence_length, self.data_dim), float)
        data_bunch = np.zeros((0, self.data_dim), float)

        if(v_key is None): # training batch
            index_bank = self.didx_tr
            while(True): # collect mini batch set

                while(True): # collect sequence set
                    list_from_key = self.list_total[self.key_tr[self.kidx_tr]]
                    np_data = np.load(list_from_key[self.didx_tr])
                    self.didx_tr = self.didx_tr + 1
                    if(self.didx_tr > (self.am_each - sequence_length)):
                        self.kidx_tr = (self.kidx_tr + 1) % (self.am_tr)
                        self.didx_tr = 0
                        data_bunch = np.zeros((0, self.data_dim), float)

                    if(data_bunch.shape[0] >= sequence_length): # break the loop when sequences are collected
                        break

                    data_tmp = np_data.reshape((1, self.data_dim))
                    data_bunch = np.append(data_bunch, data_tmp, axis=0)

                data_tmp = data_bunch.reshape((1, sequence_length, self.data_dim))
                data_bat = np.append(data_bat, data_tmp, axis=0)

                if(data_bat.shape[0] >= batch_size):  # break the loop when mini batch is collected
                    break
            self.didx_tr = (index_bank + 1) % (self.am_each - sequence_length + 1)
            return np.nan_to_num(data_bat) # replace nan to zero using np.nan_to_num

        else: # Usually used with 1 of the batch size.
            list_seqname = [] # it used for confirm the anomaly.

            index_bank = self.didx_tot
            cnt_ansymb = 0

            while(True): # collect sequence set
                list_from_key = self.list_total[v_key]

                list_seqname.append(list_from_key[self.didx_tot])

                if(data_bunch.shape[0] >= sequence_length): # break the loop when sequences are collected
                    break

                for symb in self.symbols: # count anomaly symbol (maximum 1 at one file).
                    if(symb in list_from_key[self.didx_tot]):
                        cnt_ansymb += 1
                        break
                np_data = np.load(list_from_key[self.didx_tot])
                self.didx_tot = self.didx_tot + 1
                if(self.didx_tot > (self.am_each - sequence_length)):
                    print("Cannot make bunch anymore. (length %d at %d)" %(sequence_length, self.didx_tot))
                    self.didx_tot = 0
                    return None, None

                data_tmp = np_data.reshape((1, self.data_dim))
                data_bunch = np.append(data_bunch, data_tmp, axis=0)

            data_tmp = data_bunch.reshape((1, sequence_length, self.data_dim))
            data_bat = np.append(data_bat, data_tmp, axis=0)

            self.didx_tot = (index_bank + 1) % (self.am_each - sequence_length + 1)

            return np.nan_to_num(data_bat), list_seqname # replace nan to zero using np.nan_to_num
