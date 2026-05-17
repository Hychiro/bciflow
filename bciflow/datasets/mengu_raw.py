import numpy as np
import h5py
from typing import List, Optional, Dict, Any


def mengu_raw(subject: int = 1,
              session_list: Optional[List[str]] = None,
              labels: Optional[List[str]] = None,
              depth: Optional[List[str]] = None,
              path='data/mengu/') -> Dict[str, Any]:
    """
    Description
    -----------

    Load MenGu dataset as continuous raw EEG data with
    sample-wise labels.

    Parameters
    ----------
    subject : int
        subject index.
    session_list : list
        sessions to use.
    labels : list
        frequency labels to use.
    depth : list
        stimulus depth selection.
    path : str
        dataset path.

    Returns
    -------
    dict
        Raw EEG dictionary.
    """

    if type(subject) != int:
        raise ValueError("subject has to be a int type value")
    if subject < 1 or subject > 30:
        raise ValueError("subject has to be between 1 and 30")
    _available_sessions = ['s%02d' % i for i in range(1, 13)]
    if session_list is None:
        session_list = _available_sessions
    elif type(session_list) != list:
        raise ValueError("session_list has to be a list type value")
    else:
        for i in session_list:
            if i not in _available_sessions:
                raise ValueError(
                    "invalid session_list value"
                )
    _available_labels = ['f%02d' % i for i in range(1, 61)]
    if labels is None:
        labels = _available_labels
    elif type(labels) != list:
        raise ValueError("labels has to be a list type value")
    else:
        for i in labels:
            if i not in _available_labels:
                raise ValueError(
                    "invalid labels value"
                )
    _available_depths = ['low', 'high']
    if depth is None:
        depth = _available_depths
    elif type(depth) != list:
        raise ValueError("depth has to be a list type value")
    else:
        for i in depth:
            if i not in _available_depths:
                raise ValueError(
                    "invalid depth value"
                )
    if type(path) != str:
        raise ValueError("path has to be a str type value")
    if path[-1] != '/':
        path += '/'

    sfreq = 1000.
    tmin = 0.

    ch_names = np.array([
        "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5",
        "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
        "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
        "FC4", "FC6", "FT8", "T7", "C5", "C3",
        "C1", "CZ", "C2", "C4", "C6", "T8",
        "M1", "TP7", "CP5", "CP3", "CP1", "CPZ",
        "CP2", "CP4", "CP6", "TP8", "M2", "P7",
        "P5", "P3", "P1", "PZ", "P2", "P4",
        "P6", "P8", "PO7", "PO5", "PO3", "POZ",
        "PO4", "PO6", "PO8", "CB1", "O1", "OZ",
        "O2", "CB2"
    ])

    with h5py.File(path + 'data_s%d_64.mat' % subject, "r") as f:
        data = np.asarray(f["datas"])
    session_id = np.where(
        np.isin(_available_sessions, session_list)
    )[0]
    labels_id = np.where(
        np.isin(_available_labels, labels)
    )[0]
    depth_id = np.where(
        np.isin(_available_depths, depth)
    )[0]
    data = data[session_id]
    data = data[:, labels_id]
    data = data[:, :, :, :, depth_id]

    raw_segments = []
    raw_labels = []

    y_dict = {
        i + 1: label
        for i, label in enumerate(labels)
    }

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[4]):
                epoch = data[i, j, :, :, k].T
                raw_segments.append(epoch)
                class_id = j + 1
                raw_labels.append(
                    np.ones(epoch.shape[1]) * class_id
                )
    X = np.concatenate(raw_segments, axis=1)
    y = np.concatenate(raw_labels).astype(int)

    return {
        'data_type': "raw",
        'X': X,
        'y': y,
        'sfreq': sfreq,
        'y_dict': y_dict,
        'ch_names': ch_names,
        'tmin': tmin
    }
