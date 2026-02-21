
import sys
import os.path as path
from numpy import linalg as la
from numpy import dot, trace
from numpy.linalg import cholesky, eig, inv, norm
import numpy as np
import json
import h5py
import mne

__file__ = path.realpath(__file__)    # expand potentially relative path to a full one
sys.path.append(path.dirname(path.dirname(__file__))+ "/beam-python")

_LEFT_HEMI_ZERO = -1_111_111

def construct_mcmv_weights(fs, r_inv, n_cov = None, beam = "mpz", c_avg = None):

	MIN_RELATIVE_E_VAL = 0.001
	beam_types = {"mpz":k_mpz, "mai":k_mai, "mer":k_mer, "rmer":k_rmer}

	if len(fs.shape) == 3:
		n_src, n_chan, tmp = fs.shape
		if tmp != 3:
			raise ValueError("fs should be a (n_src x n_chan x 3) numpy array")
		if r_inv.shape != (n_chan,n_chan):
			raise ValueError("r_inv should be a ({} x {}) numpy array".
					format(n_chan, n_chan))
		if not is_pos_def(r_inv):
			raise ValueError("r_inv should be a symmetric positively defined matrix")
		if n_cov is None:
			n_cov = 0.01 * trace(r_inv) * np.identity(n_chan)
		elif n_cov.shape != (n_chan,n_chan):
			raise ValueError("n_cov should be a ({} x {}) numpy array".
					format(n_chan, n_chan))
		elif not is_pos_def(n_cov):
			raise ValueError("n_cov should be a symmetric positively defined matrix")
		beam = beam.lower()
		if not (beam in beam_types):
			raise ValueError("beam should be one of: {}".format([*beam_types]))
		n_inv = inv(n_cov) if beam == 'mai' else None
		is_evoked = False
		if (beam in ("mer", "rmer")):
			if c_avg is None:
				raise ValueError("An evoked beamformer '{}' is requested, but c_avg is not specified"
						.format(beam))
			elif c_avg.shape != (n_chan,n_chan):
				raise ValueError("c_avg should be a ({} x {}) numpy array".
					format(n_chan, n_chan))
			is_evoked = True
		l_mtx = np.concatenate(fs, axis = 1)
		k_mtx = beam_types[beam](l_mtx, r_inv, n_cov, n_inv, c_avg)
		e_vals, e_vecs = eig(k_mtx)
		e_vals = np.real(e_vals)
		e_vecs = np.real(e_vecs)
		idx = np.argsort(-e_vals)[:n_src]
		e_vals = e_vals[idx]
		e_vecs = e_vecs[:,idx]
		tmp = 0. if is_evoked else 1.
		tmp = (e_vals - tmp)
		idx = (tmp / tmp[0] >= MIN_RELATIVE_E_VAL)
		e_vals = e_vals[idx]
		e_vecs = e_vecs[:,idx]
		if np.any(np.iscomplex(e_vecs)):
			raise ValueError("Got complex-valued eigenvectors of the K-matrix")
		u = parse_eigenvectors(e_vecs)
		h_mtx = np.zeros((n_chan, n_src))
		for i_src in range(n_src):
			h_mtx[:,i_src] = dot(fs[i_src], u[:,i_src])
	else:
		h_mtx = fs
		u = None

	s_inv = inv(dot(h_mtx.T, dot(r_inv, h_mtx)))
	w_mtx = dot(r_inv, dot(h_mtx, s_inv))
	return (w_mtx, u)

def is_pos_def(a):
    try:
        cholesky(a)
        return True
    except:
        return False

def k_mpz(l_mtx, r_inv, n_cov, n_inv, c_avg):
	lt_rm1 = dot(l_mtx.T, r_inv)
	s = dot(lt_rm1, l_mtx)
	t = dot(dot(lt_rm1, n_cov), lt_rm1.T)
	return dot(inv(t), s)

def k_mai(l_mtx, r_inv, n_cov, n_inv, c_avg):
	lt_rm1 = dot(l_mtx.T, r_inv)
	lt_rn1 = dot(l_mtx.T, n_inv)
	s = dot(lt_rm1, l_mtx)
	g = dot(lt_rn1, l_mtx)
	return dot(inv(s), g)

def k_mer(l_mtx, r_inv, n_cov, n_inv, c_avg):
	lt_rm1 = dot(l_mtx.T, r_inv)
	t = dot(dot(lt_rm1, n_cov), lt_rm1.T)
	e = dot(dot(lt_rm1, c_avg), lt_rm1.T)
	return dot(inv(t), e)

def k_rmer(l_mtx, r_inv, n_cov, n_inv, c_avg):
	lt_rm1 = dot(l_mtx.T, r_inv)
	s = dot(lt_rm1, l_mtx)
	e = dot(dot(lt_rm1, c_avg), lt_rm1.T)
	return dot(inv(s), e)

def parse_eigenvectors(v):
	n_src = int(v.shape[0]/3)
	n_vec = v.shape[1]
	powers = np.zeros((n_src, n_vec))
	for i_src in range(n_src):
		i1 = i_src * 3
		i2 = i1 + 3
		for i_vec in range(n_vec):
			a = v[i1:i2, i_vec]
			powers[i_src, i_vec] = dot(a, a)
	idx_vec = np.argmax(powers, axis = 1)
	u = np.zeros((3, n_src))
	for i_src in range(n_src):
		i1 = i_src * 3
		i2 = i1 + 3
		t = v[i1:i2, idx_vec[i_src]]
		for i_vec in range(n_vec):
			a = v[i1:i2, i_vec]
			if dot(a, t) > 0:
				u[:,i_src] = u[:,i_src] + powers[i_src, i_vec]*a
			else:
				u[:,i_src] = u[:,i_src] - powers[i_src, i_vec]*a
		u[:,i_src] = u[:,i_src]/norm(u[:,i_src])

	return u

def construct_single_source_weights(fs, r_inv, n_cov = None, beam = "mpz", c_avg = None):
	if len(fs.shape) == 3:
		n_src, n_chan, _ = fs.shape
		w_mtx = np.zeros((n_chan, n_src))
		u = np.zeros((3, n_src))

		for i in range(n_src):
			f = fs[i,:,:][np.newaxis,:]
			w, u1 = construct_mcmv_weights(f, r_inv, n_cov, beam, c_avg)
			w_mtx[:,i] = w[:,0]
			u[:,i] = u1[:,0]
	else:
		n_chan, n_src = fs.shape
		w_mtx = np.zeros((n_chan, n_src))
		u = None

		for i in range(n_src):
			f = fs[:,i]
			w, _ = construct_mcmv_weights(f[:, np.newaxis], r_inv, n_cov, beam, c_avg)
			w_mtx[:,i] = w[:,0]
	return (w_mtx, u)

def nearestPD(A):
    B = (A + A.T) / 2
    _, s, V = la.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if isPD(A3):
        return A3
    spacing = np.spacing(la.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3

def isPD(B):
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

def fwd_file_name(scan_id, src_file):
    f = src_file.replace('fsaverage-', '{}-'.format(scan_id))
    f = f.replace('-src.fif', '-fwd.fif')
    return f

def get_voxel_coords(src, vertices):
    ihemi = lambda i: 0 if i < 0 else 1                     # The hemisphere num
    lh_vtx = lambda i: 0 if i == _LEFT_HEMI_ZERO else -i    # The actual vtx # in left hemi

    rr = list()

    for i in vertices:
        hemi = ihemi(i)
        vtx = i if hemi else lh_vtx(i)
        rr.append(src[hemi]['rr'][vtx,:])

    return np.array(rr)

def construct_noise_and_inv_cov(fwd, data_cov, *, tol = 1e-2, rcond = 1e-10):
    U, S, Vh = np.linalg.svd(data_cov, full_matrices=False, hermitian=True)
    t = rcond * S[0]
    U = U[:, S > t]
    rank = U.shape[1]
    H = fwd['sol']['data']
    uH = U.T @ H
    unoise_cov = uH @ uH.T
    udata_cov = np.diag(S[:rank])
    inv_cov = nearestPD(U @ np.diag(1./S[:rank]) @ U.T)
    pwr = np.trace(udata_cov)
    unoise_cov = (pwr / np.trace(unoise_cov)) * unoise_cov
    upper = pwr
    lower = 0.
    tr = pwr
    tr0 = tr

    while True:
        if is_pos_def(udata_cov - unoise_cov):
            lower = tr
            tr = lower + (upper - lower)/2
            is_pd = True
        else:
            upper = tr
            tr = upper - (upper - lower)/2
            is_pd = False

        assert upper > lower
        ratio = tr/tr0
        unoise_cov = ratio * unoise_cov
        tr0 = tr

        if is_pd and (np.abs(ratio - 1) < tol):
            break

    noise_cov = nearestPD(U @ unoise_cov @ U.T)
    pz = pwr / tr
    return noise_cov, inv_cov, rank, pz

def get_label_src_idx(fwd, label):
    if label.hemi == 'lh':
        ihemi = 0
    elif label.hemi == 'rh':
        ihemi = 1
    else:
        raise ValueError("Only single hemisphere labels are allowed")
    all_vertices = fwd["src"][ihemi]["vertno"]
    idx = np.nonzero(np.in1d(all_vertices, label.vertices))[0]
    if ihemi == 1:
        idx += len(fwd["src"][0]["vertno"])

    return idx

def get_label_fwd(fwd, label):
    idx = get_label_src_idx(fwd, label)
    l0 = 3*idx
    idx3 = (np.array([l0, l0 + 1, l0 + 2]).T).flatten()
    H = fwd['sol']['data']      # nchans x (3*nsrc)
    assert 3*(len(fwd["src"][0]["vertno"]) + len(fwd["src"][1]["vertno"])) == H.shape[1]
    return H[:,idx3]

def get_label_wts(fwd, W, label):
    assert fwd['sol']['data'].shape[0] == W.shape[0]
    assert int(fwd['sol']['data'].shape[1]/3) == W.shape[1]
    idx = get_label_src_idx(fwd, label)
    return W[:,idx]

def get_label_pca_weight(R, fwd, W, label):
    W_label = get_label_wts(fwd, W, label)
    R_label = W_label.T @ R @ W_label
    E, U = np.linalg.eigh(R_label)
    e0 = E[-1]
    U0 = U[:,-1]
    scale = np.sqrt(np.trace(R_label)/e0/W_label.shape[1])
    flip = mne.label_sign_flip(label, fwd['src'])
    sign = np.sign(U0.T @ flip)
    w_pca = sign * scale * (W_label @ U0)
    return w_pca

def get_beam_weights(H, inv_cov, noise_cov, units):
    if units == "source":
        normalize = False
    elif units == "pz":
        normalize = True
    else:
        raise ValueError("The 'units' parameter value should be either 'source' or 'pz'")
    nchan, nsrc3 = H.shape
    nsrc = int(nsrc3 / 3)
    fs = np.reshape(H.T, (nsrc, 3, nchan))
    fs = np.transpose(fs, axes = (0, 2, 1))	# fs is nsrc x nchan x 3
    W, U = construct_single_source_weights(fs, inv_cov, noise_cov, beam = "mpz", c_avg = None)
    if not normalize:
        return W
    scales = np.sqrt(np.einsum('ns,nm,ms->s', W, noise_cov, W))	# A vector of nsrc values = sqrt(diag(W'N W))
    return W / scales, U

def compute_beamformer_stc(raw, fwd, *, return_stc = True, beam_type = 'pz', units = 'pz',
                           tol = 1e-2, rcond = 1e-10, verbose = None):
    eeg_data = raw.get_data(    # eeg_data is nchannels x ntimes
        pzpicks = 'eeg',          # bads are already dropped
        start=0,                # starting time sample number (int)
        stop=None,
        reject_by_annotation=None,
        return_times=False,
        units=None,             # return SI units
        verbose=verbose)
    nchan = eeg_data.shape[0]
    data_cov = nearestPD(np.cov(eeg_data, rowvar=True, bias=False))
    noise_cov, inv_cov, rank, pz = construct_noise_and_inv_cov(fwd, data_cov, tol = tol, rcond = rcond)
    if verbose == 'INFO':
        print('Data covarince: nchan = {}, rank = {}'.format(nchan, rank))
    W, U = get_beam_weights(fwd['sol']['data'], inv_cov, noise_cov, units = units)
    assert fwd['src'][0]['type'] == 'surf'
    assert fwd['src'][0]['coord_frame'] == mne.io.constants.FIFF.FIFFV_COORD_HEAD
    Usurf = list()
    vertices = list()
    for ihemi in (0, 1):
        src = fwd['src'][ihemi]
        vts = src['vertno']
        Usurf.append(src['nn'][vts,:])
        vertices.append(vts)
    Usurf = np.concatenate(Usurf, axis = 0)	# This is nsrc x 3 array
    assert Usurf.shape[0] == U.shape[1]
    signs = np.sign(np.einsum('ij,ji->i', Usurf, U))	# A vector of nsrc 1s, -1s
    W *= signs
    U *= signs
    W /= np.sqrt(pz)
    if return_stc:
        src_data = (W.T @ eeg_data)
        stc = mne.SourceEstimate(
            src_data,
            vertices, 
            tmin = raw.times[0],
            tstep = raw.times[1] - raw.times[0],
            subject='fsaverage',
            verbose=verbose)
    else:
        stc = None
    return stc, data_cov, W, U, 

def compute_source_timecourses(raw, fwd, *, method = "beam", return_stc = True, **kwargs): 
    if method == 'beam':
        return compute_beamformer_stc(raw, fwd, return_stc = return_stc, **kwargs) 
    raise ValueError('Method {} is unknown or not implemented'.format(method))

def ltc_file_name(scan_id, src_file):
    f = src_file.replace('fsaverage-', '{}-'.format(scan_id))
    f = f.replace('-src.fif', '-ltc.hdf5')
    return f

def read_roi_time_courses(ltc_file):
    with h5py.File(ltc_file, 'r') as f:
        label_tcs = f['label_tcs'][:,:]
        label_names = f['label_names'].asstr()[:]
        if 'vertno' in f:
            vertno = f['vertno'][:]
        else:
            vertno = None

        if 'rr' in f:
            rr = f['rr'][:,:]
        else:
            rr = None

        if 'W' in f:
            W = f['W'][:,:]
        else:
            W = None

        if 'pz' in f:
            pz = f['pz'][()]
        else:
            pz = None

        if 'ps_events' in f:
            ps_events = f['ps_events'][:,:]
        else:
            ps_events = None

        if 'ps_id_dict' in f:
            ps_id_dict = json.loads(f['ps_id_dict'][()])
        else:
            ps_id_dict = None

    return (label_tcs, label_names, vertno, rr, W, pz, ps_events, ps_id_dict)  

def write_roi_time_courses(ltc_file, label_tcs, label_names, vertno = None, rr = None, W = None,
                           pz = None, ps_events = None, ps_id_dict = None):
    with h5py.File(ltc_file, 'w') as f:
        f.create_dataset('label_tcs', data=label_tcs)
        f.create_dataset('label_names', data=label_names)

        if not (vertno is None):
            f.create_dataset('vertno', data=vertno)

        if not (rr is None):
            f.create_dataset('rr', data=rr)

        if not (W is None):
            f.create_dataset('W', data=W)

        if not (pz is None):
            f.create_dataset('pz', data=pz)

        if ps_events is not None:
            f.create_dataset('ps_events', data=ps_events)

        if ps_id_dict is not None:
            sjson = json.dumps(ps_id_dict)
            f.create_dataset('ps_id_dict', data=sjson)

def beam_extract_label_time_course(sensor_data, cov, labels, fwd, W, mode = 'pca_flip',
        verbose = None):
    roi_modes = {'pca_flip': get_label_pca_weight}

    if not mode in roi_modes:
        raise ValueError('Mode {} is unknown or not supported'.format(mode))

    if verbose == 'INFO':
        print('Reconstructing ROI time courses using beamformer weights, mode = {}'.format(mode))

    func = roi_modes[mode]
    nlabels = len(labels)
    nchans, ntimes = sensor_data.shape
    label_wts = np.zeros((nchans, nlabels))

    for i,label in enumerate(labels):
        label_wts[:, i] = func(cov, fwd, W, label)

    label_tcs = label_wts.T @ sensor_data

    return label_tcs, label_wts

def compute_roi_time_courses(inv_method, labels, fwd, mode = 'pca_flip',
        stc = None, sensor_data = None, cov = None, W = None, verbose = None):
    beam_modes = ['pca_flip']    # A list of modes supported by beam_extract_label_time_course() 
    if (inv_method == "beam") and (mode in beam_modes):
        label_tcs, label_wts = beam_extract_label_time_course(sensor_data, cov, labels,
                                   fwd, W, mode = mode, verbose = verbose)
    else: 
        label_tcs = mne.extract_label_time_course(stc, labels, fwd['src'],
            mode=mode,                 # How to extract a time course for ROI
            allow_empty=False,         # Raise exception for empty ROI 
            return_generator=False,    # Return nRoi x nTimes matrix, not a generator
            mri_resolution=False,      # Do not upsample source space
            verbose=verbose)

        label_wts = None

    return label_tcs, label_wts

def get_label_coms(labels, fs_dir):
    lcoms = list()

    for l in labels:
        if not len(l.vertices):
            raise ValueError('Label {} has an empty vertices list; COM cannot be computed.'.\
                format(l.name))     # Happens for labels restricted to a coarse surface

        icom = l.center_of_mass(subject="fsaverage",
            restrict_vertices=True,    # Assign COM to one of label's vertices
            subjects_dir=fs_dir, surf='sphere')

        # Make left hemi voxels negative, replace zero with _LEFT_HEMI_ZERO
        if l.hemi == 'lh':
            icom = -icom if icom else _LEFT_HEMI_ZERO

        lcoms.append(icom)

    return np.array(lcoms)

def parse_vertex_list(vertno):
    lh_idx = vertno < 0
    rh_idx = np.logical_not(lh_idx)

    lh = -vertno[lh_idx]
    rh = vertno[rh_idx]

    # Special treatment of _LEFT_HEMI_ZERO value
    izero = np.flatnonzero(lh == -_LEFT_HEMI_ZERO)

    if len(izero):
        lh[izero[0]] = 0

    return lh, rh, lh_idx, rh_idx

def encode_vertex_list(vertno, is_left):
    if not is_left:
        return vertno

    l = -vertno
    lz = (l == 0)
    l[lz] = _LEFT_HEMI_ZERO

    return l

