######################################################
#                    LIBRAIRIES                      #
######################################################

import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Custom imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from optimization import MM_LS_SlIPL, MM_LS_IPL
from estimation import phase_only

######################################################
#                    Functions                       #
######################################################

def mainfunc(parameters, stepx, stepy, data, i):
    """
    3D-Dataset has 3 dimension of Date x Azimuth x Range.
    To boost the computation time, the dataset is divided in Azimuth direction 
    so that we can take advantage of the cluster
    
    Parameters
    ----------
    data : Input data with dimension of Date x Azimuth x Range.
    Lx : window size in Range direction
    Ly : window size in Azimuth direction
    i : step in azimuth direction

    Returns
    -------
    Phase difference and coherence matrix
    """
    k, p, n_images, lamda, maxIterMM, Lx, Ly = parameters
    # initialization
    delta_theta_tot = []
    coherence_list = []
    n_block = 1 + (n_images - p) / k # number of temporal stacks
    newarr = data[:, stepy[i]:stepy[i] + Ly, :]

    for j in range(len(stepx)):
        win = newarr[:, :, stepx[j]:stepx[j] + Lx]
        winar = win.reshape(win.shape[0], win.shape[1] * win.shape[2])

        # Offline approach on the first temporal stack
        z0 = winar[0:p, :]
        Sigma_tilde1 = phase_only(z0)
        theta_temp1 = MM_LS_IPL(Sigma_tilde1, maxIterMM)
        theta_temp_0 = np.angle(theta_temp1[0])

        w_overlap_past = theta_temp1[k:p]

        for i in range(1, round(n_block)):
            zi = winar[i:i + p, :]
            Sigma_tilde_zi = phase_only(zi)
            w_new = MM_LS_SlIPL(w_overlap_past, lamda, Sigma_tilde_zi, maxIterMM)
            w_overlap_past = w_new[k:p]

        delta_theta_MM_i = np.angle(w_new) - theta_temp_0
        delta_theta_MM_i = (delta_theta_MM_i + np.pi) % (2 * np.pi) - np.pi
        delta_theta_tot.append(delta_theta_MM_i)
        coherence_list.append(abs(Sigma_tilde_zi))

    return delta_theta_tot, coherence_list

######################################################
#                USER PARAMETERS                     #
######################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Process 3D-Dataset with MM algorithm")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the datacube file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save results')
    parser.add_argument('--p', type=int, default=5, help='Size of the temporal stack')
    parser.add_argument('--k', type=int, default=1, help='Stride')
    parser.add_argument('--Lx', type=int, default=8, help='Spatial window size in Range direction')
    parser.add_argument('--Ly', type=int, default=8, help='Spatial window size in Azimuth direction')
    parser.add_argument('--maxIterMM', type=int, default=100, help='Number of iterations of the MM algorithm')
    parser.add_argument('--lamda', type=float, default=1.5, help='Regularization parameter lambda')
    return parser.parse_args()


def main():
    args = parse_args()

    datacube = np.load(args.data_path)
    datacube = datacube.transpose(2, 0, 1)
    datacube = datacube + 0.0000001

    n_images = datacube.shape[0]
    parameters = args.k, args.p, n_images, args.lamda, args.maxIterMM, args.Lx, args.Ly

    stepy = np.arange(0, datacube.shape[1] - args.Ly + 1)
    stepx = np.arange(0, datacube.shape[2] - args.Lx + 1)

    folder = f'D_S_COFI_PL_nimages={n_images}_n={args.Lx*args.Ly}_p={args.p}_k={args.k}_iterMM{args.maxIterMM}_lambda={args.lamda}_PO'
    pathout = os.path.join(args.output_path, folder)
    os.makedirs(pathout, exist_ok=True)

    pathout_npy = os.path.join(pathout, 'npy_files')
    os.makedirs(pathout_npy, exist_ok=True)

    pathout_pdf = os.path.join(pathout, 'pdf_files')
    os.makedirs(pathout_pdf, exist_ok=True)

    result = Parallel(n_jobs=-1, backend='loky', verbose=100)(
        delayed(mainfunc)(parameters, stepx, stepy, datacube, i) for i in tqdm(range(len(stepy)))
    )

    phases_list = [item[0] for item in result]
    coherence_list = [item[1] for item in result]

    PLarray = np.array(phases_list)
    coherence_array = np.array(coherence_list)

    # save npy files
    np.save(os.path.join(pathout_npy, "PLarray.npy"), PLarray)
    np.save(os.path.join(pathout_npy, "coherence_array.npy"), coherence_array)

    # save pdf files
    for i in range(args.p):
        plt.figure()
        im = plt.imshow(PLarray[:, :, i], interpolation='nearest', cmap='jet', vmin=-np.pi, vmax=np.pi)
        plt.colorbar(im)
        plt.savefig(os.path.join(pathout_pdf, f"{folder}_date_{n_images - args.p + i + 1}_phase.pdf"), dpi=400)
        plt.show()

if __name__ == "__main__":
    main()
