"""
@author: Colley Jean-Marc CNRS/IN2P3/LPNHE
"""

from rshower.basis.efield_event import *


#
# OTHERS METHOD
#


def fit_vec_linear_polar_with_max(trace):
    """

    :param trace:
    :type trace: float (3, n_s)
    :param threshold:
    :type threshold:
    """
    assert trace.shape[0] == 3
    n_elec = np.linalg.norm(trace, axis=0)
    idx_max = np.argmax(n_elec)
    v_pol = trace[:, idx_max] / n_elec[idx_max]
    logger.debug(v_pol)
    return v_pol


def fit_vec_linear_polar_l2_2(trace, threshold=20, plot=False):
    """Fit the unit linear polarization vec with samples out of noise (>threshold)

    We used weighted estimation with square l2 norm

    :param trace:
    :type trace: float (3, n_s)
    :param threshold:
    :type threshold:
    """
    n_elec = np.linalg.norm(trace, axis=0)
    idx_hb = np.where(n_elec > threshold)[0]
    logger.debug(f"{len(idx_hb)} samples out noise :\n{idx_hb}")
    # to unit vector for samples out noise
    # (3,ns)/(ns) => OK
    sple_ok = trace[:, idx_hb] / n_elec[idx_hb]
    logger.debug(sple_ok)
    n_elec_2 = n_elec[idx_hb] * n_elec[idx_hb]
    # weighted estimation with norm^2
    # temp is (3,n_s)=(3,ns)*(ns)
    temp = sple_ok * n_elec_2
    pol_est = np.sum(temp, axis=1) / np.sum(n_elec_2)
    logger.info(pol_est)
    # unit vect
    pol_est /= np.linalg.norm(pol_est)
    logger.info(f"pol_est: {pol_est}, {np.linalg.norm(pol_est)}")
    return pol_est, idx_hb


def fit_vec_linear_polar_l2(trace, threshold=20):
    """Fit the unit linear polarization vec with samples out of noise (>threshold)

    We used weighted estimation with norm l2

    :param trace:
    :type trace: float (3, n_s)
    :param threshold:
    :type threshold:
    """
    n_elec = np.linalg.norm(trace, axis=0)
    idx_hb = np.where(n_elec > threshold)[0]
    if len(idx_hb) == 0:
        # set to nan to be excluded by plot
        return np.array([[np.nan, np.nan, np.nan]]), np.array([])
    # logger.debug(f"{len(idx_hb)} samples out noise :\n{idx_hb}")
    # to unit vector for samples out noise
    # (3,ns)/(ns) => OK
    n_elec_hb = n_elec[idx_hb]
    sple_ok = trace[:, idx_hb]
    # logger.debug(sple_ok)
    # weighted estimation with norm
    pol_est = np.sum(sple_ok, axis=1) / np.sum(n_elec_hb)
    # unit vect
    pol_est /= np.linalg.norm(pol_est)
    logger.debug(f"pol_est: {pol_est} with {len(idx_hb)} values out of noise.")
    return pol_est, idx_hb


def fit_vec_linear_polar_hls(trace):
    """Fit the unit linear pola vec with homogenous linear system


    :param trace:
    :type trace: float (3, n_s)

    """
    # TODO: add weigth
    n_sple = trace.shape[1]
    m_a = np.zeros((3 * n_sple, 3), dtype=np.float32)
    # p_x coeff
    m_a[:n_sple, 0] = -trace[1]
    m_a[:n_sple, 1] = trace[0]
    # p_y coeff
    m_a[n_sple : 2 * n_sple, 0] = -trace[2]
    m_a[n_sple : 2 * n_sple, 2] = trace[0]
    # p_z coeff
    m_a[2 * n_sple : 3 * n_sple, 1] = -trace[2]
    m_a[2 * n_sple : 3 * n_sple, 2] = trace[1]
    # solve
    m_ata = np.matmul(m_a.T, m_a)
    assert m_ata.shape == (3, 3)
    w_p, vec_p = np.linalg.eig(m_ata)
    # logger.debug(f"{w_p}")
    # logger.debug(f"{vec_p}")
    vec_pol = vec_p[:, 0]
    logger.debug(f"vec_pol eigen : {vec_pol}")
    res = np.matmul(m_a, vec_pol)
    # logger.debug(f"{res} ")
    # logger.debug(f"{np.linalg.norm(res)} {res.min()} {res.max()}")
    # assert np.allclose(np.linalg.norm(np.matmul(m_ata, vec_pol)), 0)
    return vec_pol


def efield_in_polar_frame(efield3d, threshold=40):
    """Return E field in linear polarization direction

    :param efield3d: [uV/m] Efield 3D
    :type efield3d: float (3, n_s)
    :param threshold: [uV/m] used to select sample to fit direction
    :type threshold: float (n_s,)
    """
    pol_est, idx_on = fit_vec_linear_polar_l2(efield3d, threshold)
    check_vec_linear_polar_l2(efield3d, idx_on, pol_est)
    efield1d = np.dot(efield3d.T, pol_est)
    return efield1d, pol_est


#
# CHECK
#


def check_vec_linear_polar_l2(trace, idx_on, vec_pol):
    """

    :param trace_on: sample of trace out noise
    :type trace_on: float (3,n) n number of sample
    :param vec_pol:
    :type vec_pol:float (3,)
    """
    if idx_on is not None:
        if len(idx_on) == 0:
            # set to nan to be excluded by plot
            return np.nan, np.nan
        trace_on = trace[:, idx_on]
    else:
        trace_on = trace
        idx_on = np.arange(trace.shape[1])
    norm_tr = np.linalg.norm(trace_on, axis=0)
    # logger.info(norm_tr)
    tr_u = trace_on / norm_tr
    # logger.info(tr_u)
    cos_angle = np.dot(vec_pol, tr_u)
    idx_pb = np.argwhere(cos_angle > 1)
    cos_angle[idx_pb] = 1.0
    # logger.info(cos_angle)
    assert cos_angle.shape[0] == idx_on.shape[0]
    angles = np.rad2deg(np.arccos(cos_angle))
    # logger.info(angles)
    # for measures in opposite direction
    idx_neg = np.argwhere(angles > 180)
    angles[idx_neg] -= 180
    idx_neg = np.argwhere(angles > 90)
    angles[idx_neg] = 180 - angles[idx_neg]
    # logger.info(angles)
    assert np.alltrue(angles >= 0)
    prob = norm_tr / np.sum(norm_tr)
    assert np.isclose(prob.sum(), 1)
    mean_l2 = np.sum(angles * prob)
    diff = angles - mean_l2
    std_l2 = np.sqrt(np.sum(prob * diff * diff))
    logger.debug(f"Angle error l2: {mean_l2} {std_l2}")
    return mean_l2, std_l2


def check_vec_linear_polar_proto(trace, idx_on, vec_pol):
    """

    :param trace_on: sample of trace out noise
    :type trace_on: float (3,n) n number of sample
    :param vec_pol:
    :type vec_pol:float (3,)
    """
    if idx_on is not None:
        if len(idx_on) == 0:
            # set to nan to be excluded by plot
            return np.nan, np.nan
        trace_on = trace[:, idx_on]
    else:
        trace_on = trace
        idx_on = np.arange(trace.shape[1])
    # plt.figure()
    # plt.plot(idx_on, trace_on[0], label="x")
    # plt.plot(idx_on, trace_on[1], label="y")
    # plt.plot(idx_on, trace_on[2], label="z")
    norm_tr = np.linalg.norm(trace_on, axis=0)
    # logger.info(norm_tr)
    tr_u = trace_on / norm_tr
    # logger.info(tr_u)
    cos_angle = np.dot(vec_pol, tr_u)
    idx_pb = np.argwhere(cos_angle > 1)
    cos_angle[idx_pb] = 1.0
    # logger.info(cos_angle)
    assert cos_angle.shape[0] == idx_on.shape[0]
    angles = np.rad2deg(np.arccos(cos_angle))
    # logger.info(angles)
    # for measures in opposite direction
    idx_neg = np.where(angles > 180)[0]
    angles[idx_neg] -= 180
    idx_neg = np.where(angles > 90)[0]
    angles[idx_neg] = 180 - angles[idx_neg]
    # logger.info(angles)
    assert np.alltrue(angles >= 0)
    mean, std = angles.mean(), angles.std()
    # weighted estimation
    norm2_tr = np.sum(trace_on * trace_on, axis=0)
    prob = norm2_tr / np.sum(norm2_tr)
    mean_w = np.sum(angles * norm_tr * norm_tr) / np.sum(norm_tr * norm_tr)
    mean_w2 = np.sum(angles * prob)
    diff = angles - mean_w2
    std_w2 = np.sqrt(np.sum(prob * diff * diff))
    logger.debug(f"Angle error: {mean}, sigma {std} ")
    logger.debug(f"Angle error w2: {mean_w2} {std_w2}")
    prob = norm_tr / np.sum(norm_tr)
    assert np.isclose(prob.sum(), 1)
    mean_w1 = np.sum(angles * prob)
    diff = angles - mean_w2
    std_w1 = np.sqrt(np.sum(prob * diff * diff))
    logger.debug(f"Angle error w1: {mean_w1} {std_w1}")
    # plt.figure()
    # plt.hist(angles)
    return mean_w2, std_w2


def check_xmax_line_pyramid():
    """
    xmax  [0. 0. 1.]
    """
    du_pos = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]], dtype=np.float64)
    v_dir_src = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64)
    print(f"v_dir_src:\n{v_dir_src}")
    xmax, res = estimate_xmax_line(v_dir_src, du_pos)
    print(f"xmax 4pts: {xmax}\n")
    xmax, res = estimate_xmax_line(v_dir_src[:3], du_pos[:3])
    print(f"xmax 3pts: {xmax}\n")
    print(v_dir_src[[0, 2]])
    xmax, res = estimate_xmax_line(v_dir_src[:2], du_pos[:2])
    print(f"xmax 2pts: {xmax}\n")
    xmax, res = estimate_xmax_line(v_dir_src[[0, 2]], du_pos[[0, 2]])
    print(f"xmax 2pts: {xmax}\n")
    # with noise 1%
    noise_dir = np.random.normal(0, 1e-2, v_dir_src.shape)
    v_dir_src += noise_dir
    print(f"v_dir_src: {v_dir_src}")
    xmax, res = estimate_xmax_line(v_dir_src, du_pos)
    print(f"xmax 4pts: {xmax}\n")

def check_xmax_line_pyramid_rec():
    """
    xmax  [0. 0. 1.]
    """
    du_pos = np.array([[2, 1, 0], [-2, 1, 0], [-2, -1, 0], [2, -1, 0]], dtype=np.float64)
    v_dir_src = np.array([[-2, -1, 1], [2, -1, 1], [2, 1, 1], [-2, 1, 1]], dtype=np.float64)
    print(f"v_dir_src:\n{v_dir_src}")
    xmax, res = estimate_xmax_line(v_dir_src, du_pos)
    print(f"xmax 4pts: {xmax}\n")
    xmax, res = estimate_xmax_line(v_dir_src[:3], du_pos[:3])
    print(f"xmax 3pts: {xmax}\n")
    print(v_dir_src[[0, 2]])
    xmax, res = estimate_xmax_line(v_dir_src[:2], du_pos[:2])
    print(f"xmax 2pts: {xmax}\n")
    xmax, res = estimate_xmax_line(v_dir_src[[0, 2]], du_pos[[0, 2]])
    print(f"xmax 2pts: {xmax}\n")
    # with noise 1%
    noise_dir = np.random.normal(0, 1e-2, v_dir_src.shape)
    v_dir_src += noise_dir
    print(f"v_dir_src: {v_dir_src}")
    xmax, res = estimate_xmax_line(v_dir_src, du_pos)
    print(f"xmax 4pts: {xmax}\n")



def check_xmax_line_pyramid_offset():
    """
    xmax  [3. 3. 1.]
    """
    du_pos = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]], dtype=np.float64)
    du_pos += np.array([3, 3, 0])
    v_dir_src = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64)
    print(f"v_dir_src:\n{v_dir_src}")
    xmax, res = estimate_xmax_line(v_dir_src, du_pos)
    print(f"xmax 4pts: {xmax}\n")
    xmax, res = estimate_xmax_line(v_dir_src[:3], du_pos[:3])
    print(f"xmax 3pts: {xmax}\n")
    print(v_dir_src[[0, 2]])
    xmax, res = estimate_xmax_line(v_dir_src[:2], du_pos[:2])
    print(f"xmax 2pts: {xmax}\n")
    xmax, res = estimate_xmax_line(v_dir_src[[0, 2]], du_pos[[0, 2]])
    print(f"xmax 2pts: {xmax}\n")
    # with noise 1%
    noise_dir = np.random.normal(0, 1e-2, v_dir_src.shape)
    v_dir_src += noise_dir
    print(f"v_dir_src: {v_dir_src}")
    xmax, res = estimate_xmax_line(v_dir_src, du_pos)
    print(f"xmax 4pts: {xmax}\n")

def check_xmax_line_2pts_opposite():
    """
    xmax  [0. 0. 1.]
    """
    du_pos = np.array([[0, 1, 0], [0, -1, 0]], dtype=np.float64)
    v_dir_src = np.array([[0, -1, 1], [0, 1, 1]], dtype=np.float64)
    print(f"v_dir_src:\n{v_dir_src}")
    xmax, res = estimate_xmax_line(v_dir_src, du_pos)
    print(f"xmax 2pts: {xmax}\n")


def check_xmax_line_pyramid_outliner():
    """
    xmax  [0. 0. 1.]
    """
    du_pos = np.array([[2, 1, 0], [-2, 1, 0], [-2, -1, 0], [2, -1, 0]], dtype=np.float64)
    v_dir_src = np.array([[-2, -1, 1], [2, -1, 1], [2, 1, 1], [-2, 1, 1]], dtype=np.float64)    
    print(f"v_dir_src:\n{v_dir_src}")
    # exact
    xmax, res = estimate_xmax_line(v_dir_src, du_pos)
    print(f"xmax 4pts: {xmax}")
    print(f"residu:\n{res}\n")
    # noise
    noise = np.random.normal(0, 1e-3, v_dir_src.shape)
    xmax, res = estimate_xmax_line(v_dir_src+noise, du_pos)
    print(f"xmax 4pts noise: {xmax}")
    print(f"residu:\n{res}\n")
    # outlier
    v_dir_src[0, 0] += 2
    print(f"v_dir_src:\n{v_dir_src}")
    xmax, res = estimate_xmax_line(v_dir_src, du_pos)
    print(f"xmax 4pts: {xmax}")
    print(f"residu:\n{res}\n")
    # sans outlier
    xmax, res = estimate_xmax_line(v_dir_src[1:], du_pos[1:])
    print(f"xmax 4pts: {xmax}")
    print(f"residu:\n{res}\n")

if __name__ == "__main__":
    # check_xmax_line_pyramid()
    # check_xmax_line_2pts_opposite()
    #check_xmax_line_pyramid_offset()
    check_xmax_line_pyramid_outliner()
    #check_xmax_line_pyramid_rec()