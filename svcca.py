from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import Representations

########################################################################################################################
#                                               TWEAKED GOOGLE CODE                                                    #
########################################################################################################################

# Copyright 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

num_cca_trials = 5


def positivedef_matrix_sqrt(array):
    """Stable method for computing matrix square roots, supports complex matrices.

    Args:
              array: A numpy 2d array, can be complex valued that is a positive
                     definite symmetric (or hermitian) matrix

    Returns:
              sqrtarray: The matrix square root of array
    """
    w, v = np.linalg.eigh(array)
    #  A - np.dot(v, np.dot(np.diag(w), v.T))
    wsqrt = np.sqrt(w)
    sqrtarray = np.dot(v, np.dot(np.diag(wsqrt), np.conj(v).T))
    return sqrtarray


def remove_small(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon):
    """Takes covariance between X, Y, and removes values of small magnitude.

    Args:
              sigma_xx: 2d numpy array, variance matrix for x
              sigma_xy: 2d numpy array, crossvariance matrix for x,y
              sigma_yx: 2d numpy array, crossvariance matrixy for x,y,
                        (conjugate) transpose of sigma_xy
              sigma_yy: 2d numpy array, variance matrix for y
              epsilon : cutoff value for norm below which directions are thrown
                         away

    Returns:
              sigma_xx_crop: 2d array with low x norm directions removed
              sigma_xy_crop: 2d array with low x and y norm directions removed
              sigma_yx_crop: 2d array with low x and y norm directiosn removed
              sigma_yy_crop: 2d array with low y norm directions removed
              x_idxs: indexes of sigma_xx that were removed
              y_idxs: indexes of sigma_yy that were removed
    """

    x_diag = np.abs(np.diagonal(sigma_xx))
    y_diag = np.abs(np.diagonal(sigma_yy))
    x_idxs = (x_diag >= epsilon)
    y_idxs = (y_diag >= epsilon)

    sigma_xx_crop = sigma_xx[x_idxs][:, x_idxs]
    sigma_xy_crop = sigma_xy[x_idxs][:, y_idxs]
    sigma_yx_crop = sigma_yx[y_idxs][:, x_idxs]
    sigma_yy_crop = sigma_yy[y_idxs][:, y_idxs]

    return (sigma_xx_crop, sigma_xy_crop, sigma_yx_crop, sigma_yy_crop,
            x_idxs, y_idxs)


def compute_ccas(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon,
                 verbose=True):
    """Main cca computation function, takes in variances and crossvariances.

    This function takes in the covariances and cross covariances of X, Y,
    preprocesses them (removing small magnitudes) and outputs the raw results of
    the cca computation, including cca directions in a rotated space, and the
    cca correlation coefficient values.

    Args:
              sigma_xx: 2d numpy array, (num_neurons_x, num_neurons_x)
                        variance matrix for x
              sigma_xy: 2d numpy array, (num_neurons_x, num_neurons_y)
                        crossvariance matrix for x,y
              sigma_yx: 2d numpy array, (num_neurons_y, num_neurons_x)
                        crossvariance matrix for x,y (conj) transpose of sigma_xy
              sigma_yy: 2d numpy array, (num_neurons_y, num_neurons_y)
                        variance matrix for y
              epsilon:  small float to help with stabilizing computations
              verbose:  boolean on whether to print intermediate outputs

    Returns:
              [ux, sx, vx]: [numpy 2d array, numpy 1d array, numpy 2d array]
                            ux and vx are (conj) transposes of each other, being
                            the canonical directions in the X subspace.
                            sx is the set of canonical correlation coefficients-
                            how well corresponding directions in vx, Vy correlate
                            with each other.
              [uy, sy, vy]: Same as above, but for Y space
              invsqrt_xx:   Inverse square root of sigma_xx to transform canonical
                            directions back to original space
              invsqrt_yy:   Same as above but for sigma_yy
              x_idxs:       The indexes of the input sigma_xx that were pruned
                            by remove_small
              y_idxs:       Same as above but for sigma_yy
    """

    (sigma_xx, sigma_xy, sigma_yx, sigma_yy,
     x_idxs, y_idxs) = remove_small(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon)

    numx = sigma_xx.shape[0]
    numy = sigma_yy.shape[0]

    if numx == 0 or numy == 0:
        return ([0, 0, 0], [0, 0, 0], np.zeros_like(sigma_xx),
                np.zeros_like(sigma_yy), x_idxs, y_idxs)

    if verbose:
        print("adding eps to diagonal and taking inverse")
    sigma_xx += epsilon * np.eye(numx)
    sigma_yy += epsilon * np.eye(numy)
    inv_xx = np.linalg.pinv(sigma_xx)
    inv_yy = np.linalg.pinv(sigma_yy)

    if verbose:
        print("taking square root")
    invsqrt_xx = positivedef_matrix_sqrt(inv_xx)
    invsqrt_yy = positivedef_matrix_sqrt(inv_yy)

    if verbose:
        print("dot products...")
    arr = np.dot(invsqrt_xx, np.dot(sigma_xy, invsqrt_yy))

    if verbose:
        print("trying to take final svd")
    u, s, v = np.linalg.svd(arr)

    if verbose:
        print("computed everything!")

    return [u, np.abs(s), v], invsqrt_xx, invsqrt_yy, x_idxs, y_idxs


def sum_threshold(array, threshold):
    """Computes threshold index of decreasing nonnegative array by summing.

    This function takes in a decreasing array nonnegative floats, and a
    threshold between 0 and 1. It returns the index i at which the sum of the
    array up to i is threshold*total mass of the array.

    Args:
              array: a 1d numpy array of decreasing, nonnegative floats
              threshold: a number between 0 and 1

    Returns:
              i: index at which np.sum(array[:i]) >= threshold
    """
    assert (threshold >= 0) and (threshold <= 1), "print incorrect threshold"

    for i in range(len(array)):
        if np.sum(array[:i]) / np.sum(array) >= threshold:
            return i


def create_zero_dict(compute_dirns, dimension):
    """Outputs a zero dict when neuron activation norms too small.

    This function creates a return_dict with appropriately shaped zero entries
    when all neuron activations are very small.

    Args:
              compute_dirns: boolean, whether to have zero vectors for directions
              dimension: int, defines shape of directions

    Returns:
              return_dict: a dict of appropriately shaped zero entries
    """
    return_dict = {}
    return_dict["mean"] = (np.asarray(0), np.asarray(0))
    return_dict["sum"] = (np.asarray(0), np.asarray(0))
    return_dict["cca_coef1"] = np.asarray(0)
    return_dict["cca_coef2"] = np.asarray(0)
    return_dict["idx1"] = 0
    return_dict["idx2"] = 0

    if compute_dirns:
        return_dict["cca_dirns1"] = np.zeros((1, dimension))
        return_dict["cca_dirns2"] = np.zeros((1, dimension))

    return return_dict


def get_cca_similarity(acts1, acts2, epsilon=0., threshold=0.98,
                       compute_coefs=False,
                       compute_dirns=False,
                       verbose=False):
    """The main function for computing cca similarities.

    This function computes the cca similarity between two sets of activations,
    returning a dict with the cca coefficients, a few statistics of the cca
    coefficients, and (optionally) the actual directions.

    Args:
              acts1: (num_neurons1, data_points) a 2d numpy array of neurons by
                     datapoints where entry (i,j) is the output of neuron i on
                     datapoint j.
              acts2: (num_neurons2, data_points) same as above, but (potentially)
                     for a different set of neurons. Note that acts1 and acts2
                     can have different numbers of neurons, but must agree on the
                     number of datapoints

              epsilon: small float to help stabilize computations

              threshold: float between 0, 1 used to get rid of trailing zeros in
                         the cca correlation coefficients to output more accurate
                         summary statistics of correlations.


              compute_coefs: boolean value determining whether coefficients
                             over neurons are computed. Needed for computing
                             directions

              compute_dirns: boolean value determining whether actual cca
                             directions are computed. (For very large neurons and
                             datasets, may be better to compute these on the fly
                             instead of store in memory.)

              verbose: Boolean, whether intermediate outputs are printed

    Returns:
              return_dict: A dictionary with outputs from the cca computations.
                           Contains neuron coefficients (combinations of neurons
                           that correspond to cca directions), the cca correlation
                           coefficients (how well aligned directions correlate),
                           x and y idxs (for computing cca directions on the fly
                           if compute_dirns=False), and summary statistics. If
                           compute_dirns=True, the cca directions are also
                           computed.
    """
    #

    # assert dimensionality equal
    assert acts1.shape[1] == acts2.shape[1], "dimensions don't match"
    # check that acts1, acts2 are transposition
    # assert acts1.shape[0] < acts1.shape[1], ("input must be number of neurons by datapoints")
    return_dict = {}

    # compute covariance with numpy function for extra stability
    numx = acts1.shape[0]
    numy = acts2.shape[0]

    covariance = np.cov(acts1, acts2)
    sigmaxx = covariance[:numx, :numx]
    sigmaxy = covariance[:numx, numx:]
    sigmayx = covariance[numx:, :numx]
    sigmayy = covariance[numx:, numx:]

    # rescale covariance to make cca computation more stable
    xmax = np.max(np.abs(sigmaxx))
    ymax = np.max(np.abs(sigmayy))
    sigmaxx /= xmax
    sigmayy /= ymax
    sigmaxy /= np.sqrt(xmax * ymax)
    sigmayx /= np.sqrt(xmax * ymax)

    ([u, s, v], invsqrt_xx, invsqrt_yy,
     x_idxs, y_idxs) = compute_ccas(sigmaxx, sigmaxy, sigmayx, sigmayy,
                                    epsilon=epsilon,
                                    verbose=verbose)

    # if x_idxs or y_idxs is all false, return_dict has zero entries
    if (not np.any(x_idxs)) or (not np.any(y_idxs)):
        return create_zero_dict(compute_dirns, acts1.shape[1])

    if compute_coefs:
        # also compute full coefficients over all neurons
        x_mask = np.dot(x_idxs.reshape((-1, 1)), x_idxs.reshape((1, -1)))
        y_mask = np.dot(y_idxs.reshape((-1, 1)), y_idxs.reshape((1, -1)))

        return_dict["coef_x"] = u.T
        return_dict["invsqrt_xx"] = invsqrt_xx
        return_dict["full_coef_x"] = np.zeros((numx, numx))
        np.place(return_dict["full_coef_x"], x_mask,
                 return_dict["coef_x"])
        return_dict["full_invsqrt_xx"] = np.zeros((numx, numx))
        np.place(return_dict["full_invsqrt_xx"], x_mask,
                 return_dict["invsqrt_xx"])

        return_dict["coef_y"] = v
        return_dict["invsqrt_yy"] = invsqrt_yy
        return_dict["full_coef_y"] = np.zeros((numy, numy))
        np.place(return_dict["full_coef_y"], y_mask,
                 return_dict["coef_y"])
        return_dict["full_invsqrt_yy"] = np.zeros((numy, numy))
        np.place(return_dict["full_invsqrt_yy"], y_mask,
                 return_dict["invsqrt_yy"])

        # compute means
        neuron_means1 = np.mean(acts1, axis=1, keepdims=True)
        neuron_means2 = np.mean(acts2, axis=1, keepdims=True)
        return_dict["neuron_means1"] = neuron_means1
        return_dict["neuron_means2"] = neuron_means2

    if compute_dirns:
        # orthonormal directions that are CCA directions
        cca_dirns1 = np.dot(np.dot(return_dict["full_coef_x"],
                                   return_dict["full_invsqrt_xx"]),
                            (acts1 - neuron_means1)) + neuron_means1
        cca_dirns2 = np.dot(np.dot(return_dict["full_coef_y"],
                                   return_dict["full_invsqrt_yy"]),
                            (acts2 - neuron_means2)) + neuron_means2

    # get rid of trailing zeros in the cca coefficients
    idx1 = sum_threshold(s, threshold)
    idx2 = sum_threshold(s, threshold)

    return_dict["cca_coef1"] = s
    return_dict["cca_coef2"] = s
    return_dict["x_idxs"] = x_idxs
    return_dict["y_idxs"] = y_idxs
    # summary statistics
    return_dict["mean"] = np.mean(s[:idx1])
    return_dict["sum"] = np.sum(s)

    if compute_dirns:
        return_dict["cca_dirns1"] = cca_dirns1
        return_dict["cca_dirns2"] = cca_dirns2

    return return_dict


def fft_resize(images, new_size=None):
    """Function for applying DFT and resizing.

    This function takes in an array of images, applies the 2-d fourier transform
    and resizes them according to new_size, keeping the frequencies that overlap
    between the two sizes.

    Args:
              images: a numpy array with shape
                      [batch_size, height, width, num_channels]
              new_size: a tuple (size, size), with height and width the same

    Returns:
              im_fft_downsampled: a numpy array with shape
                           [batch_size, (new) height, (new) width, num_channels]
    """
    assert len(images.shape) == 4, ("expecting images to be"
                                    "[batch_size, height, width, num_channels]")

    im_complex = images.astype("complex64")
    im_fft = np.fft.fft2(im_complex, axes=(1, 2))

    # resizing images
    if new_size:
        # get fourier frequencies to threshold
        assert (im_fft.shape[1] == im_fft.shape[2]), ("Need images to have same"
                                                      "height and width")
        # downsample by threshold
        width = im_fft.shape[2]
        new_width = new_size[0]
        freqs = np.fft.fftfreq(width, d=1.0 / width)
        idxs = np.flatnonzero((freqs >= -new_width / 2.0) & (freqs <
                                                             new_width / 2.0))
        im_fft_downsampled = im_fft[:, :, idxs, :][:, idxs, :, :]

    else:
        im_fft_downsampled = im_fft

    return im_fft_downsampled


def fourier_ccas(conv_acts1, conv_acts2, return_coefs=False,
                 compute_dirns=False, verbose=False, epsilon=0.):
    """Computes cca similarity between two conv layers with DFT.

    This function takes in two sets of convolutional activations, conv_acts1,
    conv_acts2 After resizing the spatial dimensions to be the same, applies fft
    and then computes the ccas.

    Finally, it applies the inverse fourier transform to get the CCA directions
    and neuron coefficients.

    Args:
              conv_acts1: numpy array with shape
                          [batch_size, height1, width1, num_channels1]
              conv_acts2: numpy array with shape
                          [batch_size, height2, width2, num_channels2]
              compute_dirns: boolean, used to determine whether results also
                             contain actual cca directions.

    Returns:
              all_results: a pandas dataframe, with cca results for every spatial
                           location. Columns are neuron coefficients (combinations
                           of neurons that correspond to cca directions), the cca
                           correlation coefficients (how well aligned directions
                           correlate) x and y idxs (for computing cca directions
                           on the fly if compute_dirns=False), and summary
                           statistics. If compute_dirns=True, the cca directions
                           are also computed.
    """

    height1, width1 = conv_acts1.shape[1], conv_acts1.shape[2]
    height2, width2 = conv_acts2.shape[1], conv_acts2.shape[2]
    if height1 != height2 or width1 != width2:
        height = min(height1, height2)
        width = min(width1, width2)
        new_size = [height, width]
    else:
        height = height1
        width = width1
        new_size = None

    # resize and preprocess with fft
    fft_acts1 = fft_resize(conv_acts1, new_size=new_size)
    fft_acts2 = fft_resize(conv_acts2, new_size=new_size)

    # loop over spatial dimensions and get cca coefficients
    all_results = pd.DataFrame()
    for i in range(height):
        for j in range(width):
            results_dict = get_cca_similarity(fft_acts1[:, i, j, :].T, fft_acts2[:, i, j, :].T,
                                              compute_dirns=compute_dirns, verbose=verbose, epsilon=epsilon)

            # apply inverse FFT to get coefficients and directions if specified
            if return_coefs:
                results_dict["cca_coef1"] = np.fft.ifft2(results_dict["cca_coef1"])
                results_dict["cca_coef2"] = np.fft.ifft2(results_dict["cca_coef2"])
            else:
                plt.plot(results_dict['cca_coef1'])
                plt.show()
                del results_dict["cca_coef1"]
                del results_dict["cca_coef2"]

            if compute_dirns:
                results_dict["cca_dirns1"] = np.fft.ifft2(results_dict["cca_dirns1"])
                results_dict["cca_dirns2"] = np.fft.ifft2(results_dict["cca_dirns2"])

            # accumulate results
            results_dict["location"] = (i, j)
            all_results = all_results.append(results_dict, ignore_index=True)

    return all_results

########################################################################################################################
#                                          NOT GOOGLE CODE FROM THIS POINT                                             #
########################################################################################################################


def svd_prep(acts, percentage, sub):
    acts = acts[:, ::sub]
    acts = acts -acts.mean(axis=0, keepdims=True)
    u, s, vh = np.linalg.svd(acts)

    # keep specified percentage of variance
    var_percent = 0
    i = 1
    while var_percent < percentage:
        var_percent = s[:i].sum() / s.sum()
        i += 1

    return np.dot(u.T[:i], acts)


def get_activations(base, dataset, size, channels, mode='train'):
    dataset = Representations(f'{base}/0/{dataset}/{size}/{channels}/codes_{mode}.pt', fraction=1)
    batch_size = len(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    samples = next(iter(loader))
    return samples[0].permute(0,2,3,1).numpy()


def full_prep(base, dataset, size, channels, i, j, mode, new_size, svd, sub, prepped, min_size, dft=True):
    try:
        acts = prepped[mode][dataset][min_size][size][channels][i][j] if dft else prepped[mode][dataset][min_size][size][channels]
    except KeyError:
        acts = get_activations(base, dataset, size, channels, mode)
        if min_size not in prepped[mode][dataset].keys():
            prepped[mode][dataset][min_size] = defaultdict(dict)
        try:
            prepped[mode][dataset][min_size][size][channels]
        except KeyError:
            prepped[mode][dataset][min_size][size][channels] = defaultdict(dict)
        if dft:
            acts = fft_resize(acts, new_size)
            acts = svd_prep(acts.T[:, i, j, :], svd, sub)
            prepped[mode][dataset][min_size][size][channels][i][j] = acts
        else:
            acts = acts.reshape(len(acts), -1).astype(float)
            acts = svd_prep(acts.T, svd, sub)
            prepped[mode][dataset][min_size][size][channels] = acts
        print(i, j, acts.shape)

    return acts


def compute_svcca(base, dataset, size1, channels1, size2, channels2, mode, svd, sub, prepped, dft=True):
    results = pd.DataFrame()
    min_size = min(size1, size2)
    new_size = None if size1 != size2 else (min_size, min_size)
    if dft:
        for i in range(min_size):
            for j in range(min_size):
                acts1 = full_prep(base, dataset, size1, channels1, i, j, mode, new_size, svd, sub, prepped, min_size, dft)
                acts2 = full_prep(base, dataset, size2, channels2, i, j, mode, new_size, svd, sub, prepped, min_size, dft)
                result = get_cca_similarity(acts1, acts2, 10e-6)
                del result["cca_coef1"]
                del result["cca_coef2"]
                result["location"] = (i, j)
                results = results.append(result, ignore_index=True)
    else:
        acts1 = full_prep(base, dataset, size1, channels1, 0, 0, mode, new_size, svd, sub, prepped, min_size, dft)
        acts2 = full_prep(base, dataset, size2, channels2, 0, 0, mode, new_size, svd, sub, prepped, min_size, dft)
        result = get_cca_similarity(acts1, acts2, 10e-6)
        del result["cca_coef1"]
        del result["cca_coef2"]
        results = results.append(result, ignore_index=True)
    return results



