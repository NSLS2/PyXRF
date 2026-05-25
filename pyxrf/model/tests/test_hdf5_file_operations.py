import copy
import os

import numpy as np
import numpy.testing as npt
import pytest

from pyxrf.api_dev import read_data_from_hdf5, save_data_to_hdf5


def _prepare_raw_dataset(N=5, M=10, K=4096):
    det1 = np.ones(shape=[N, M, K]) * 100
    det2 = np.ones(shape=[N, M, K]) * 150
    det3 = np.ones(shape=[N, M, K]) * 200
    det_sum = det1 + det2 + det3

    scaler_names = ["i0", "scaler1", "scaler2"]
    scaler_data = np.ones(shape=[N, M, len(scaler_names)])
    for n in range(len(scaler_names)):
        scaler_data[:, :, n] *= n

    pos_names = ["x_pos", "y_pos"]
    pos_data = np.zeros(shape=[2, N, M])
    pos_data[0, :, :] = np.broadcast_to(np.linspace(1, 1 + (M - 1) * 0.1, M), shape=[N, M])
    pos_data[1, :, :] = np.broadcast_to(
        np.reshape(np.linspace(5, 5 + (N - 1) * 0.2, N), shape=[N, 1]), shape=[N, M]
    )

    data = {
        "det_sum": det_sum,
        "det1": det1,
        "det2": det2,
        "det3": det3,
        "scaler_names": scaler_names,
        "scaler_data": scaler_data,
        "pos_names": pos_names,
        "pos_data": pos_data,
    }

    metadata = {"scan_id": 1000, "scan_uid": "SOME-RANDOM-UID-STRING", "name": "Some more metadata"}

    return data, metadata


def test_save_data_to_hdf5_1(tmp_path):
    """
    Basic test for ``save_data_to_hdf5``
    """
    fln = "test.h5"
    fpath = os.path.join(tmp_path, fln)
    data, metadata = _prepare_raw_dataset(N=5, M=10, K=4096)

    save_data_to_hdf5(fpath, data, metadata=metadata)

    # Check if the file exists
    assert os.path.isfile(fpath)

    # Check that the exception occurs in case of attempt to overwrite the file
    with pytest.raises(IOError, match="File .* already exists"):
        save_data_to_hdf5(fpath, data, metadata=metadata)

    # File should be overwritten
    save_data_to_hdf5(fpath, data, metadata=metadata, file_overwrite_existing=True)

    # Different version of the file should be created
    save_data_to_hdf5(fpath, data, metadata=metadata, fname_add_version=True)
    save_data_to_hdf5(fpath, data, metadata=metadata, fname_add_version=True)

    # Check if the file exists
    assert os.path.isfile(fpath)
    assert os.path.isfile(os.path.join(tmp_path, "test_v1.h5"))
    assert os.path.isfile(os.path.join(tmp_path, "test_v2.h5"))


# fmt: off
@pytest.mark.parametrize("sets_to_select, sets_to_save", [
    ("all", "default"),
    ("all", "all"),
    ("channels", "all"),
    ("all", "sum"),
    ("channels", "sum"),
    ("sum", "sum"),
    ("sum", "default"),
    ("sum", "all"),
])
# fmt: on
def test_save_data_to_hdf5_2(tmp_path, sets_to_select, sets_to_save):
    """
    Save data to hdf5 and then read it.
    """
    fln = "test.h5"
    fpath = os.path.join(tmp_path, fln)
    data, metadata = _prepare_raw_dataset(N=5, M=10, K=4096)

    data_to_save = copy.deepcopy(data)
    if sets_to_select == "sum":
        # Leave only
        del data_to_save["det1"]
        del data_to_save["det2"]
        del data_to_save["det3"]
    elif sets_to_select == "channels":
        # Leave only sum
        del data_to_save["det_sum"]

    kwargs = {}
    if sets_to_save == "all":
        kwargs = {"create_each_det": True}
    elif sets_to_save == "sum":
        kwargs = {"create_each_det": False}

    save_data_to_hdf5(fpath, data_to_save, metadata=metadata, **kwargs)
    data_loaded, metadata_loaded = read_data_from_hdf5(fpath)

    metadata_selected = {_: metadata_loaded[_] for _ in metadata}
    assert metadata_selected == metadata

    assert data_loaded["pos_names"] == data["pos_names"]
    npt.assert_array_almost_equal(data_loaded["pos_data"], data["pos_data"])

    assert isinstance(data_loaded["det_sum"], np.ndarray)
    npt.assert_array_almost_equal(data_loaded["det_sum"], data["det_sum"])
    if (sets_to_select == "sum") or (sets_to_save == "sum"):
        # Only 'sum' of all channels will be saved
        keys = set(data.keys())
        keys.remove("det1")
        keys.remove("det2")
        keys.remove("det3")
        assert "det1" not in data_loaded
        assert "det2" not in data_loaded
        assert "det3" not in data_loaded
        assert set(data_loaded.keys()) == keys
    else:
        assert set(data_loaded.keys()) == set(data.keys())
        npt.assert_array_almost_equal(data_loaded["det1"], data["det1"])
        npt.assert_array_almost_equal(data_loaded["det2"], data["det2"])
        npt.assert_array_almost_equal(data_loaded["det3"], data["det3"])

    assert data_loaded["pos_names"] == data["pos_names"]
    npt.assert_array_almost_equal(data_loaded["pos_data"], data["pos_data"])
    assert data_loaded["scaler_names"] == data["scaler_names"]
    npt.assert_array_almost_equal(data_loaded["scaler_data"], data["scaler_data"])


def test_save_data_to_hdf5_3(tmp_path):
    """
    Verify that default metadata is automatically created.
    """
    fln = "test.h5"
    fpath = os.path.join(tmp_path, fln)
    data, metadata = _prepare_raw_dataset(N=5, M=10, K=4096)

    save_data_to_hdf5(fpath, data)
    data_loaded, metadata_loaded = read_data_from_hdf5(fpath)

    metadata_keys = [
        "file_type",
        "file_format",
        "file_format_version",
        "file_created_time",
        "file_software",
        "file_software_version",
    ]
    for k in metadata_keys:
        assert k in metadata_loaded


def test_save_data_to_hdf5_4(tmp_path):
    """
    Verify that metadata ``file_software`` and ``file_software_version`` can be set by the user.
    """
    fln = "test.h5"
    fpath = os.path.join(tmp_path, fln)
    data, metadata = _prepare_raw_dataset(N=5, M=10, K=4096)

    application = "Testing Application"
    version = "v4.5.6"

    metadata = {"file_software": application, "file_software_version": version}

    save_data_to_hdf5(fpath, data, metadata=metadata)
    data_loaded, metadata_loaded = read_data_from_hdf5(fpath)

    assert metadata_loaded["file_software"] == application
    assert metadata_loaded["file_software_version"] == version
