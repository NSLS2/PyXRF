import logging
import os
import uuid

import dask.array as da
import h5py
import numpy as np
import numpy.testing as npt
import pytest
from skbeam.core.fitting.background import snip_method

from pyxrf.core.fitting import fit_spectrum
from pyxrf.core.map_processing import (
    RawHDF5Dataset,
    TerminalProgressBar,
    _array_numpy_to_dask,
    _chunk_numpy_array,
    _compute_optimal_chunk_size,
    _compute_roi,
    _fit_xrf_block,
    _prepare_xrf_mask,
    compute_selected_rois,
    compute_total_spectrum,
    compute_total_spectrum_and_count,
    dask_client_create,
    fit_xrf_map,
    prepare_xrf_map,
    snip_method_numba,
    wait_and_display_progress,
)
from pyxrf.core.tests.test_fitting import DataForFittingTest

logger = logging.getLogger(__name__)


def test_dask_client_create(tmpdir):
    """
    `dask_client_create` is a trivial function that instantiates Dask client
    in uniform way throughout the program. We test that we can pass addition
    kwargs to the object constructor or override default parameters.
    """
    # Set current directory. This is not required to create Dask client, but
    #   we will check to make sure that the directory for temporary files is
    #   not created in the current directory.
    os.chdir(tmpdir)
    dask_worker_space_path = os.path.join(tmpdir, "dask-worker-space")

    # Set the number of workers to some strange number (11 is unusual)
    #   (pass another kwarg in addition to default)
    n_workers_requested = 3
    client = dask_client_create(n_workers=n_workers_requested)
    n_workers = len(client.scheduler_info()["workers"])
    assert n_workers == n_workers_requested, "The number of workers was set incorrectly"

    client.close()

    assert not os.path.exists(dask_worker_space_path), "Temporary directory was created in the current directory"

    # Disable multiprocessing: client is expected to have a single worker
    #   (replace the default value of the parameter)
    client = dask_client_create(processes=False)
    n_workers = len(client.scheduler_info()["workers"])
    assert n_workers == 1, "Dask client is expected to have one worker"
    client.close()

    assert not os.path.exists(dask_worker_space_path), "Temporary directory was created in the current directory"


def test_TerminalProgressBar():
    """Basic functionality of `TerminalProgressBar`"""

    # We just run the sequence of instructions. I didn't find a way to check the output,
    #   since 'progress' package requires TTY terminal. Otherwise output is blocked
    title = "Monitor progress"
    pbar = TerminalProgressBar(title)
    pbar.start()
    for n in range(10):
        pbar(n * 10)  # Go from 0 to 90%
    pbar.finish()  # This should set it to 100%


class _SampleProgressBar:
    """
    Progress bar class convenient for testing of the functions that
    display progress bar.
    """

    def __init__(self, title):
        self.title = title

        # The list of strings, which we expect to see in the output (capsys out)
        #   Used only for testing purposes only.
        self._expected_output = [
            f"Starting progress bar: {self.title}",
            "Percent completed: 100.0",
            f"Finished: {self.title}",
        ]

    def start(self):
        print(f"Starting progress bar: {self.title}")

    def __call__(self, percent_completed):
        print(f"Percent completed: {percent_completed}")

    def finish(self):
        print(f"Finished: {self.title}")

    def _check_output(self, capsys_out):
        # For testing purposes only, not needed for actual progress bar implementation
        for s in self._expected_output:
            assert (
                s in capsys_out
            ), f"Expected string {s} is missing in the progress bar object output:\n{capsys_out}"


# fmt: off
@pytest.mark.parametrize("progress_bar", [
    _SampleProgressBar("Monitoring progress"),
    TerminalProgressBar("Monitoring progress: "),
    None])
# fmt: on
def test_wait_and_display_progress(progress_bar, capsys):
    """Basic test for the function `wait_and_display_progress`"""

    # There is no way to monitor the output (no TTY device -> no output is generated)
    # So we just run a typical sequence of commands and make sure it doesn't crash

    client = dask_client_create()
    data = da.random.random(size=(100, 100), chunks=(10, 10))
    sm_fut = da.sum(data, axis=0).persist(scheduler=client)

    # Call the progress monitor
    wait_and_display_progress(sm_fut, progress_bar)

    sm = sm_fut.compute(scheduler=client)
    client.close()

    # Just in case check that the computations were done correctly.
    sm_expected = np.sum(data.compute(), axis=0)
    npt.assert_array_almost_equal(sm, sm_expected, err_msg="Computations are incorrect")

    # Make sure that the output contains all necessary components
    if isinstance(progress_bar, _SampleProgressBar):
        captured = capsys.readouterr()
        progress_bar._check_output(captured.out)


def test_RawHDF5Dataset(tmpdir):
    """Class RawHDF5Dataset"""
    dir_list, fln, dset = ("dir1", "dir2"), "tmp.txt", "/some/dataset"
    file_dir = os.path.join(tmpdir, *dir_list)
    os.makedirs(file_dir, exist_ok=True)
    os.chdir(file_dir)
    # Create file
    with open(fln, "w"):
        pass

    # 'fln' should be converted to absolute
    # 'shape' is set to some arbitrary valid tuple
    rds = RawHDF5Dataset(fln, dset, shape=(10, 50))

    assert os.path.isabs(rds.abs_path), "Path was not successfully converted to absolute path"
    # If file exists, then the file name was correctly set
    assert os.path.exists(rds.abs_path), "Path was not correctly set"
    assert rds.dset_name == dset, "Dataset name was not currectly set"
    assert rds.shape == (10, 50), "Attribute 'shape' was not set correctly"


# fmt: off
@pytest.mark.parametrize("data_chunksize, chunk_optimal, n_pixels, data_shape", [
    ((3, 5), (12, 10), 100, (20, 20)),
    ((5, 3), (10, 12), 100, (20, 20)),
    ((2, 5), (10, 10), 100, (20, 20)),
    ((11, 13), (11, 13), 100, (20, 20)),
    # The case of the stretched array
    ((2, 2), (6, 2), 10, (50, 2)),  # Stretched vertically
    ((2, 2), (2, 6), 10, (2, 50)),  # Stretched horizontally
    # Chunk size exceeds the size of the array: processed as one big chunk
    ((25, 25), (20, 20), 400, (20, 20)),
    # The number of pixels in the chunk exceeds the size of the array: same
    ((25, 25), (20, 20), 500, (20, 20)),
    # The original chunk size allows to divide the array by 4 chunks
    #   in order to parallelize processing (4 is the default value, but
    #   may be set)
    ((5, 2), (10, 10), 400, (20, 20)),
    ((15, 15), (15, 15), 400, (20, 20)),
])
# fmt: on
def test_compute_optimal_chunk_size(data_chunksize, chunk_optimal, n_pixels, data_shape):
    """Basic functionality of the '_compute_optimal_chunk_size'"""

    # Call with kwargs
    res = _compute_optimal_chunk_size(
        chunk_pixels=n_pixels, data_chunksize=data_chunksize, data_shape=data_shape, n_chunks_min=4
    )
    assert res == chunk_optimal, "Computed optimal chunks size doesn't match the expected"

    # Call with args
    res = _compute_optimal_chunk_size(n_pixels, data_chunksize, data_shape, 4)
    assert res == chunk_optimal, "Computed optimal chunks size doesn't match the expected"


# fmt: off
@pytest.mark.parametrize("data_chunksize, data_shape", [
    ((3,), (20, 20)),
    ((3, 5, 7), (20, 20)),
    (5, (20, 20)),
    ((5, 3), (20,)),
    ((5, 3), (20, 20, 20)),
    ((5, 3), 20),
])
# fmt: on
def test_compute_optimal_chunk_size_fail(data_chunksize, data_shape):
    """Failing cases for `compute_optimal_chunk_size`"""
    with pytest.raises(ValueError, match="Unsupported value of parameter"):
        _compute_optimal_chunk_size(10, data_chunksize, data_shape, 4)


# fmt: off
@pytest.mark.parametrize("chunk_target, data_shape", [
    ((2, 2), (10, 10, 3)),  # 3D array (primary use case)
    ((2, 3), (9, 11, 3)),
    ((3, 2), (11, 9, 3)),
    ((3, 3), (3, 3, 2)),
    ((1, 1), (1, 1, 3)),
    ((1, 5), (1, 10, 3)),
    ((5, 1), (10, 1, 3)),
    ((3, 4), (2, 3, 2)),
    ((2, 2), (9, 11, 5)),
    ((2, 2), (9, 11)),  # 2D array
    ((2, 2), (9, 11, 5, 3)),  # 4D array
    ((2, 2), (9, 11, 5, 3, 4)),  # 5D array

])
# fmt: on
def test_chunk_numpy_array(chunk_target, data_shape):
    """Basic functionailty tests for '_chunk_xrf_map_numpy"""

    data = np.random.random(data_shape)
    data_dask = _chunk_numpy_array(data, chunk_target)

    chunksize_expected = tuple(
        [min(chunk_target[0], data_shape[0]), min(chunk_target[1], data_shape[1]), *data_shape[2:]]
    )

    assert data_dask.shape == data.shape, "The shape of the original and chunked array don't match"
    assert (
        data_dask.chunksize == chunksize_expected
    ), "The chunk size of the Dask array doesn't match the desired chunk size"
    npt.assert_array_equal(
        data_dask.compute(), data, err_msg="The chunked array is different from the original array"
    )


# fmt: off
@pytest.mark.parametrize("chunk_pixels, data_shape, res_chunk_size", [
    (4, (10, 10, 3), (2, 2, 3)),
    (5, (10, 10, 3), (2, 3, 3)),
    (5, (50, 1, 3), (5, 1, 3)),
    (5, (1, 50, 3), (1, 5, 3)),
    (4, (9, 11), (2, 2)),
    (4, (9, 11, 3, 4), (2, 2, 3, 4)),
    (4, (9, 11, 3, 4, 2), (2, 2, 3, 4, 2)),
])
# fmt: on
def test_array_numpy_to_dask1(chunk_pixels, data_shape, res_chunk_size):
    """Basic functionality of `_xrf_map_numpy_to_dask`"""

    data = np.random.random(size=data_shape)

    res = _array_numpy_to_dask(data, chunk_pixels)

    assert res.shape == data.shape, "The shape of the original and chunked array don't match"
    assert res.chunksize == res_chunk_size, "The chunk size of the Dask array doesn't match the desired chunk size"


def test_array_numpy_to_dask2():
    """Test if `chunk_pixels` is properly adjusted if it is large compared to the XRF map
    size. Explicitely set the minimum number of chunks."""

    data_shape = (8, 8, 2)
    chunk_pixels = 100  # Some arbitrary large number
    data = np.random.random(size=data_shape)
    res = _array_numpy_to_dask(data, chunk_pixels, n_chunks_min=16)
    res_chunk_size = (2, 2, 2)

    assert res.shape == data.shape, "The shape of the original and chunked array don't match"
    assert res.chunksize == res_chunk_size, "The chunk size of the Dask array doesn't match the desired chunk size"


# fmt: off
@pytest.mark.parametrize("data", [
    da.random.random((5, 3, 5)),  # Dask array is not accepted
    10.6,  # Wrong data type
    (5, 6, 7),
    "test",
    np.random.random((3,)),  # Wrong number of dimensions
])
# fmt: on
def test_array_numpy_to_dask_fail(data):
    """Wrong type of `data` array"""
    with pytest.raises(ValueError, match="Parameter 'data' must numpy array with at least 2 dimensions"):
        _array_numpy_to_dask(data, (5, 2))


def _create_xrf_data(data_dask, data_representation, tmpdir, *, chunked_HDF5=True):
    """Prepare data represented as numpy array, dask array (no change) or HDF5 dataset"""
    if data_representation == "numpy_array":
        data = data_dask.compute()
    elif data_representation == "dask_array":
        data = data_dask
    elif data_representation == "hdf5_file_dset":
        os.chdir(tmpdir)
        fln = f"test-{uuid.uuid4()}.h5"  # Include UUID in the file name
        dset_name = "level1/level2"
        with h5py.File(fln, "w") as f:
            # In this test all computations are performed using 'float64' precision,
            #   so we create the dataset with dtype="float64" for consistency.
            kwargs = {"shape": data_dask.shape, "dtype": "float64"}
            if chunked_HDF5:
                kwargs.update({"chunks": data_dask.chunksize})
            dset = f.create_dataset(dset_name, **kwargs)
            dset[:, :, :] = data_dask.compute()
        data = RawHDF5Dataset(fln, dset_name, shape=data_dask.shape)
    else:
        raise RuntimeError(
            f"Error in test parameter: unknown value of 'data_representation' = {data_representation}"
        )

    return data


def _create_xrf_mask(data_shape, apply_mask, select_area):
    """
    Generate a mask for testing of XRF dataset processing functions.

    Parameters
    ----------
    data_shape: tuple or list
        (ny, nx, ...) - the shape of XRF dataset. All dimensions except 0 and 1 are ignored
    apply_mask: bool
        True: generate random mask,
    select_area: bool
        True: select area of the XRF map for processing.

    Returns
    -------
    mask: ndarray(int) or None
        mask is ndarray with shape (ny, nx). Integer values: 0 - pixel inactive,
        1, 2 - pixel active. Note: any positive integer marks pixel as active, but only
        values of 1 and 2 are generated.
    selection: tuple or None
        selected area is (y0, x0, ny_sel, nx_sel)

    If `select_area==True`, then all pixels in `mask` outside the selected area
    are disabled.
    """
    if apply_mask:
        mask = np.random.randint(0, 2, data_shape[0:2])
        mask[3, 4] = 0  # Make sure that at least 1 element is zero
        mask[2, 4] = 1  # Make sure that at least 1 element is non-zero
    else:
        mask = None

    # Selected area (between two points, the second point is not included)
    if select_area:
        selection = (2, 3, 4, 2)  # (y0, x0, ny, nx)
    else:
        selection = None

    return mask, selection


# fmt: off
@pytest.mark.parametrize("data_representation", ["numpy_array", "dask_array", "hdf5_file_dset"])
# fmt: on
def test_prepare_xrf_data(tmpdir, data_representation):
    # Start with dask array
    data_shape = (7, 12, 20)
    data_dask = da.random.random(data_shape, chunks=(2, 3, 4))
    data_numpy = data_dask.compute()

    data = _create_xrf_data(data_dask, data_representation, tmpdir)
    data, file_obj = prepare_xrf_map(data, chunk_pixels=12, n_chunks_min=4)

    assert (
        data.chunksize[0] * data.chunksize[1] == 12
    ), f"Dataset was not properly chunked: data.chunksize={data.chunksize}"

    npt.assert_array_almost_equal(
        data.compute(), data_numpy, err_msg="Prepared dataset is different from the original"
    )


def test_prepare_xrf_data_hdf5_not_chunked(tmpdir):
    # Start with dask array
    data_shape = (7, 12, 20)
    data_dask = da.random.random(data_shape, chunks=(2, 3, 4))
    data_numpy = data_dask.compute()

    data = _create_xrf_data(data_dask, "hdf5_file_dset", tmpdir, chunked_HDF5=False)

    data, file_obj = prepare_xrf_map(data, chunk_pixels=12, n_chunks_min=4)

    assert (data.chunksize[0] == 7) and (
        data.chunksize[1] == 12
    ), f"Dataset was not properly chunked: data.chunksize={data.chunksize}"

    npt.assert_array_almost_equal(
        data.compute(), data_numpy, err_msg="Prepared dataset is different from the original"
    )


def test_prepare_xrf_data_fail():
    """Failing test for `_prepare_xrf_mask_fail`"""
    data = 50.0  # Just a number
    with pytest.raises(TypeError, match="Type of parameter 'data' is not supported"):
        prepare_xrf_map(data, chunk_pixels=12, n_chunks_min=4)


@pytest.mark.parametrize("apply_mask", [False, True])
@pytest.mark.parametrize("select_area", [False, True])
def test_prepare_xrf_mask(apply_mask, select_area):
    """Basic functionality of `_prepare_xrf_mask`"""
    data_shape = (7, 12, 20)
    data_dask = da.random.random(data_shape, chunks=(2, 3, 4))
    mask, selection = _create_xrf_mask(data_shape, apply_mask, select_area)

    # Apply selection to mask
    mask_expected = None
    if mask is not None:
        mask_expected = mask
    if selection:
        if mask_expected is None:
            mask_expected = np.ones(shape=data_shape[0:2])
        ny, nx = mask_expected.shape
        y0, x0 = selection[0], selection[1]
        y1, x1 = y0 + selection[2], x0 + selection[3]
        for y in range(ny):
            if (y < y0) or (y >= y1):
                mask_expected[y, :] = 0
        for x in range(nx):
            if (x < x0) or (x >= x1):
                mask_expected[:, x] = 0

    mask_prepared = _prepare_xrf_mask(data_dask, mask=mask, selection=selection)

    npt.assert_array_equal(mask_prepared, mask_expected, err_msg="The prepared mask is not equal to expected")


# fmt: off
@pytest.mark.parametrize("data_dask, mask, selection, msg", [
    (np.random.random((5, 5, 10)), np.random.randint(0, 3, size=(5, 5)), None,
     "Parameter 'data' must be a Dask array"),
    (da.random.random((5,)), np.random.randint(0, 3, size=(5, 5)), None,
     "Parameter 'data' must have at least 2 dimensions"),
    (da.random.random((5, 4, 10)), np.random.randint(0, 3, size=(5, 5)), None,
     "Dimensions 0 and 1 of parameters 'data' and 'mask' do not match"),
    (da.random.random((5, 5, 10)), np.random.randint(0, 3, size=(5, 5)), (1, 3, 4),
     "Parameter 'selection' must be iterable with 4 elements"),
])
# fmt: on
def test_prepare_xrf_mask_fail(data_dask, mask, selection, msg):
    """Failing cases of `_prepare_xrf_mask`"""
    with pytest.raises(TypeError, match=msg):
        _prepare_xrf_mask(data_dask, mask=mask, selection=selection)


@pytest.fixture(scope="class")
def _start_dask_client(request):
    # Setup
    client = dask_client_create()

    # Set class variables
    request.cls.client = client
    yield

    # Tear down (after all tests in the class are completed)
    client.close()


@pytest.mark.usefixtures("_start_dask_client")
class TestComputeTotalSpectrum:
    @pytest.mark.parametrize("data_representation", ["numpy_array", "dask_array", "hdf5_file_dset"])
    @pytest.mark.parametrize("apply_mask", [False, True])
    @pytest.mark.parametrize("select_area", [False, True])
    def test_compute_total_spectrum1(self, data_representation, apply_mask, select_area, tmpdir):
        """Using 'global' dask client to save time"""

        global_client = self.client

        # Start with dask array
        data_shape = (7, 12, 20)
        data_dask = da.random.random(data_shape, chunks=(2, 3, 4))
        # Using the same distributed scheduler doesn't work as expected
        data_numpy = data_dask.compute(scheduler="synchronous")

        mask, selection = _create_xrf_mask(data_shape, apply_mask, select_area)

        # Compute the expected result
        data_tmp = data_numpy
        if mask is not None:
            mask_conv = (mask > 0).astype(dtype=int)
            mask_conv = np.broadcast_to(np.expand_dims(mask_conv, axis=2), data_tmp.shape)
            data_tmp = data_tmp * mask_conv
        if selection is not None:
            y0, x0, ny, nx = selection
            data_tmp = data_tmp[y0 : y0 + ny, x0 : x0 + nx, :]
        total_spectrum_expected = np.sum(np.sum(data_tmp, axis=0), axis=0)

        data = _create_xrf_data(data_dask, data_representation, tmpdir)

        total_spectrum = compute_total_spectrum(
            data,
            selection=selection,
            mask=mask,
            chunk_pixels=12,
            # Also run all computations with the progress bar
            progress_bar=TerminalProgressBar("Monitoring progress: "),
            client=global_client,
        )

        npt.assert_array_almost_equal(
            total_spectrum, total_spectrum_expected, err_msg="Total spectrum was computed incorrectly"
        )


def test_compute_total_spectrum2(tmpdir):
    """Create an instance of Dask client in the 'compute_total_spectrum' function to test if it works"""

    # Start with dask array
    data_shape = (7, 12, 20)
    data_dask = da.random.random(data_shape, chunks=(2, 3, 4))

    data_numpy = data_dask.compute()
    total_spectrum_expected = np.sum(np.sum(data_numpy, axis=0), axis=0)

    data = _create_xrf_data(data_dask, "dask_array", tmpdir=tmpdir)

    # Run computations without the progress bar
    total_spectrum = compute_total_spectrum(data, chunk_pixels=12)

    npt.assert_array_almost_equal(
        total_spectrum, total_spectrum_expected, err_msg="Total spectrum was computed incorrectly"
    )


# fmt: off
@pytest.mark.parametrize("mask", [da.random.random((7, 12)), (7, 12), 20, "abcde"])
# fmt: on
def test_compute_total_spectrum_fail(mask):
    """Failing cases: incorrect type for the mask ndarray"""
    data_dask = da.random.random((7, 12, 20), chunks=(2, 3, 4))
    with pytest.raises(TypeError, match="Parameter 'mask' must be a numpy array or None"):
        compute_total_spectrum(data_dask, mask=mask, chunk_pixels=12)


@pytest.mark.usefixtures("_start_dask_client")
class TestComputeTotalSpectrumAndCount:
    # fmt: off
    @pytest.mark.parametrize("data_representation", ["numpy_array", "dask_array", "hdf5_file_dset"])
    @pytest.mark.parametrize("apply_mask", [False, True])
    @pytest.mark.parametrize("select_area", [False, True])
    # fmt: on
    def test_compute_total_spectrum_and_count1(self, data_representation, apply_mask, select_area, tmpdir):
        """Using 'global' dask client to save time"""

        global_client = self.client

        # Start with dask array
        data_shape = (7, 12, 20)
        data_dask = da.random.random(data_shape, chunks=(2, 3, 4))
        # Using the same distributed scheduler doesn't work as expected
        data_numpy = data_dask.compute(scheduler="synchronous")

        mask, selection = _create_xrf_mask(data_shape, apply_mask, select_area)

        # Compute the expected result
        data_tmp = data_numpy
        if mask is not None:
            mask_conv = (mask > 0).astype(dtype=int)
            mask_conv = np.broadcast_to(np.expand_dims(mask_conv, axis=2), data_tmp.shape)
            data_tmp = data_tmp * mask_conv
        if selection is not None:
            y0, x0, ny, nx = selection
            # Primitive simple algorithm to apply the selection (we need to keep all the pixels)
            for y in range(data_tmp.shape[0]):
                for x in range(data_tmp.shape[1]):
                    if (y < y0) or (y >= y0 + ny) or (x < x0) or (x >= x0 + nx):
                        data_tmp[y, x, :] = 0.0
        total_spectrum_expected = np.sum(np.sum(data_tmp, axis=0), axis=0)
        total_count_expected = np.sum(data_tmp, axis=2)

        data = _create_xrf_data(data_dask, data_representation, tmpdir)

        total_spectrum, total_count = compute_total_spectrum_and_count(
            data,
            selection=selection,
            mask=mask,
            chunk_pixels=12,
            # Also run all computations with the progress bar
            progress_bar=TerminalProgressBar("Monitoring progress: "),
            client=global_client,
        )

        npt.assert_array_almost_equal(
            total_spectrum, total_spectrum_expected, err_msg="Total spectrum was computed incorrectly"
        )

        npt.assert_array_almost_equal(
            total_count, total_count_expected, err_msg="Total count (map) was computed incorrectly"
        )


def test_compute_total_spectrum_and_count2(tmpdir):
    """Create an instance of Dask client in the 'compute_total_spectrum' function to test if it works"""

    # Start with dask array
    data_shape = (7, 12, 20)
    data_dask = da.random.random(data_shape, chunks=(2, 3, 4))

    data_numpy = data_dask.compute()
    total_spectrum_expected = np.sum(np.sum(data_numpy, axis=0), axis=0)
    total_count_expected = np.sum(data_numpy, axis=2)

    data = _create_xrf_data(data_dask, "dask_array", tmpdir=tmpdir)

    # Run computations without the progress bar
    total_spectrum, total_count = compute_total_spectrum_and_count(data, chunk_pixels=12)

    npt.assert_array_almost_equal(
        total_spectrum, total_spectrum_expected, err_msg="Total spectrum was computed incorrectly"
    )

    npt.assert_array_almost_equal(
        total_count, total_count_expected, err_msg="Total count (map) was computed incorrectly"
    )


# fmt: off
@pytest.mark.parametrize("mask", [da.random.random((7, 12)), (7, 12), 20, "abcde"])
# fmt: on
def test_compute_total_spectrum_and_count_fail(mask):
    """Failing cases: incorrect type for the mask ndarray"""
    data_dask = da.random.random((7, 12, 20), chunks=(2, 3, 4))
    with pytest.raises(TypeError, match="Parameter 'mask' must be a numpy array or None"):
        compute_total_spectrum_and_count(data_dask, mask=mask, chunk_pixels=12)


class _FitXRFMapTesting:
    """
    The class implements methods for testing of XRF fitting algorithm.
    Used for testing `_fit_xrf_block` and `fit_xrf_map` functions.
    See the respective tests for examples
    """

    def __init__(self, *, dataset_params, use_snip, add_pts_before, add_pts_after):
        self.use_snip = use_snip
        self.add_pts_before = add_pts_before
        self.add_pts_after = add_pts_after

        self.fitting_data = DataForFittingTest(**dataset_params)

        # 'spectra' has dimensions (n_spec_points, n_lines), which is correct
        self.spectra = self.fitting_data.spectra
        self.n_spectrum_points, self.n_lines = self.spectra.shape

        # 'data_tmp' has dimensions (n_spec_points, ny, nx), so it needs to be rearranged
        self.data_tmp = self.fitting_data.data_input

        # Add some small background if snip is used
        if self.use_snip:
            self.data_tmp += 1.0

        # We want to also add points at the beginning and the end of the spectra
        # The original data is filled with random values. Those values should be either
        #   overwritten by actual spectral data or ignored during fitting.
        self.data_input = np.random.random(
            size=(
                self.data_tmp.shape[1],
                self.data_tmp.shape[2],
                self.data_tmp.shape[0] + self.add_pts_before + self.add_pts_after,
            )
        )

        # Range of indices of the experimental spectrum that should be used for fitting
        #   (it contains the spectrum data). The rest of the points should be ignored
        ne_start = self.add_pts_before
        ne_stop = self.add_pts_before + self.data_tmp.shape[0]
        self.data_sel_indices = (ne_start, ne_stop)

        for ny in range(self.data_tmp.shape[1]):
            for nx in range(self.data_tmp.shape[2]):
                ne_start = self.add_pts_before
                ne_stop = self.add_pts_before + self.data_tmp.shape[0]
                self.data_input[ny, nx, ne_start:ne_stop] = self.data_tmp[:, ny, nx]

        # The snip parameters are set so that the snip width is about 10 points
        self.snip_param = {"e_offset": 0.0, "e_linear": 0.1, "e_quadratic": 0.0, "b_width": 1}

    def verify_fit_output(self, *, data_out, snip_param=None):
        assert data_out.shape == (
            self.data_tmp.shape[1],
            self.data_tmp.shape[2],
            self.n_lines + 4,
        ), f"The shape of 'data_out' is incorrect: data_out.shape={data_out.shape}"

        weights_estimated = data_out[:, :, 0 : self.n_lines]
        bg_sum = data_out[:, :, self.n_lines]

        if not self.use_snip:
            # No background
            self.fitting_data.validate_output_weights(np.moveaxis(weights_estimated, 2, 0), decimal=10)

            assert (bg_sum == 0.0).all(), (
                f"Baseline estimate is non-zero when snip method is disabled: \n"
                f"bg_sum = {bg_sum}\nmin(bg_sum) = {np.min(bg_sum)}\nmax(bg_sum) = {np.max(bg_sum)}"
            )
        else:
            # Background is present in the data. Here we repreat the procedure
            #   of background subtraction and fitting. Unfortunately, the test code
            #   is very similar to the computational code used
            assert snip_param is not None, "Test parameter `snip_param` must be provided if 'use_snip' is enabled"
            _data = np.moveaxis(self.data_tmp, 0, 2)
            bg_sel = np.zeros(shape=_data.shape)
            for ny in range(bg_sel.shape[0]):
                for nx in range(bg_sel.shape[1]):
                    bg = snip_method_numba(
                        _data[ny, nx, :],
                        snip_param["e_offset"],
                        snip_param["e_linear"],
                        snip_param["e_quadratic"],
                        width=snip_param["b_width"],
                    )
                    bg_sel[ny, nx, :] = bg

            _data_no_bg = _data - bg_sel
            weights_expected, rfactor, _ = fit_spectrum(_data_no_bg, self.spectra, axis=2, method="nnls")
            npt.assert_array_almost_equal(
                weights_estimated,
                weights_expected,
                err_msg="Estimated weights are not equal to expected (use_snip==True)",
            )

            bg_sum_expected = np.sum(bg_sel, axis=2)
            npt.assert_array_almost_equal(
                bg_sum, bg_sum_expected, err_msg="Baseline is estimated incorrectly (use_snip==True)"
            )

        # Let's trust, that R-factor is correctly inserted into
        #   the 'data_out' array. Computation of R-factor is tested elsewhere,

        # Verify if the total count for the whole spectra and the selected region
        #   was computed correctly
        sm_total = np.sum(self.data_input, axis=2)
        sm_sel = np.sum(self.data_tmp, axis=0)  # Sum over the original dataset
        npt.assert_array_almost_equal(
            data_out[:, :, self.n_lines + 2],
            sm_sel,
            err_msg="Total count for the selected region is computed incorrectly",
        )
        npt.assert_array_almost_equal(
            data_out[:, :, self.n_lines + 3], sm_total, err_msg="Total count is computed incorrectly"
        )

    def verify_roi_output(self, *, data_out, roi_dict, snip_param=None):
        """
        Verify computated ROI. Computation of ROI is repeated in this function.
        Call with the same value of `snip_param` as the one sent to ROI computing function.
        It may be different from autogenerated `self.snip_param`.

        Parameters
        ----------
        data_out: dict
            Dictionary: key - emission line, value - 2D array with ROI values
        roi_dict: dict
            Dictionary: key - emission line, value - tuple (left_val, right_val)
            Energy values are in keV.
        snip_param: dict
            Parameters for SNIP algorithm for background subtraction. See `self.snip_param`
            defined in the constructor for the example. The values are used for subtracting
            baseline and for finding ranges of indices to define bands.
        """

        e_offset = snip_param["e_offset"]
        e_linear = snip_param["e_linear"]

        # Already truncated data (containing only selected region
        _data = np.moveaxis(self.data_tmp, 0, 2)

        if self.use_snip:
            bg_sel = np.zeros(shape=_data.shape)
            for ny in range(bg_sel.shape[0]):
                for nx in range(bg_sel.shape[1]):
                    bg = snip_method_numba(
                        _data[ny, nx, :],
                        snip_param["e_offset"],
                        snip_param["e_linear"],
                        snip_param["e_quadratic"],
                        width=snip_param["b_width"],
                    )
                    bg_sel[ny, nx, :] = bg
            _data = _data - bg_sel

        data_expected = {}
        for eline in roi_dict.keys():
            vleft, vright = roi_dict[eline]
            n_left = int(round((vleft - e_offset) / e_linear)) - self.add_pts_before
            n_right = int(round((vright - e_offset) / e_linear)) - self.add_pts_before
            n_left = int(np.clip(n_left, a_min=0, a_max=_data.shape[2] - 1))
            n_right = int(np.clip(n_right, a_min=0, a_max=_data.shape[2] - 1))
            if n_right > n_left:
                data_expected[eline] = np.sum(_data[:, :, n_left:n_right], axis=2)
            else:
                data_expected[eline] = np.zeros(shape=_data.shape[0:2])

        assert list(data_out.keys()) == list(
            data_expected.keys()
        ), "The list of output data keys is different from expected"

        for key in data_out.keys():
            npt.assert_array_almost_equal(
                data_out[key],
                data_expected[key],
                err_msg=f"Output ROI count for the key '{key}' is different from expected",
            )


# fmt: off
@pytest.mark.parametrize("dataset_params", [
    {"n_data_dimensions": (8, 1)},
    {"n_data_dimensions": (1, 8)},
    {"n_data_dimensions": (4, 4)},
    {"n_data_dimensions": (3, 5)},
])
@pytest.mark.parametrize("add_pts_before, add_pts_after", [(0, 0), (50, 100)])
@pytest.mark.parametrize("use_snip", [False, True])
# fmt: on
def test_fit_xrf_block(dataset_params, add_pts_before, add_pts_after, use_snip):
    ft = _FitXRFMapTesting(
        dataset_params=dataset_params,
        use_snip=use_snip,
        add_pts_before=add_pts_before,
        add_pts_after=add_pts_after,
    )

    data_out = _fit_xrf_block(
        ft.data_input,
        data_sel_indices=ft.data_sel_indices,
        matv=ft.spectra,
        snip_param=ft.snip_param,
        use_snip=use_snip,
    )

    ft.verify_fit_output(data_out=data_out, snip_param=ft.snip_param)


@pytest.mark.usefixtures("_start_dask_client")
class TestFitXRFMap:
    # fmt: off
    @pytest.mark.parametrize("data_representation", ["numpy_array", "dask_array", "hdf5_file_dset"])
    @pytest.mark.parametrize("dataset_params", [
        {"n_data_dimensions": (10, 10)},
        {"n_data_dimensions": (9, 11)},
        {"n_data_dimensions": (1, 100)},
        {"n_data_dimensions": (100, 1)},
    ])
    @pytest.mark.parametrize("add_pts", [(0, 0), (50, 100)])
    @pytest.mark.parametrize("use_snip", [False, True])
    # fmt: on
    def test_fit_xrf_map1(self, data_representation, dataset_params, add_pts, use_snip, tmpdir):
        """
        Basic functionality of `fit_xrf_map`.
        Tests are run using global Dask clients to inprove testing speed.
        """

        # Dask client object is used for multiple tests to save execution time
        global_client = self.client

        add_pts_before, add_pts_after = add_pts

        ft = _FitXRFMapTesting(
            dataset_params=dataset_params,
            use_snip=use_snip,
            add_pts_before=add_pts_before,
            add_pts_after=add_pts_after,
        )

        # Unfortunately 'data_input' is ndarray, but we need dask array to work with
        #   Select very small chunk size. This is not efficient, but works fine for testing.
        data_dask = _array_numpy_to_dask(ft.data_input, chunk_pixels=4, n_chunks_min=1)

        # Now create the dataset we need
        data = _create_xrf_data(data_dask, data_representation, tmpdir)

        # Run fitting
        data_out = fit_xrf_map(
            data,
            data_sel_indices=ft.data_sel_indices,
            matv=ft.spectra,
            snip_param=ft.snip_param,
            use_snip=use_snip,
            chunk_pixels=10,
            n_chunks_min=4,
            progress_bar=None,
            client=global_client,
        )

        ft.verify_fit_output(data_out=data_out, snip_param=ft.snip_param)


def test_fit_xrf_map2():
    """
    Basic functionality of `fit_xrf_map`.
    Tests are run using global Dask clients to inprove testing speed.
    """

    dataset_params = {"n_data_dimensions": (20, 20)}
    add_pts_before, add_pts_after = 15, 10
    use_snip = False

    ft = _FitXRFMapTesting(
        dataset_params=dataset_params,
        use_snip=use_snip,
        add_pts_before=add_pts_before,
        add_pts_after=add_pts_after,
    )

    # Just run the test with input data represented as numpy array
    data = ft.data_input

    # Run fitting
    data_out = fit_xrf_map(
        data,
        data_sel_indices=ft.data_sel_indices,
        matv=ft.spectra,
        snip_param=ft.snip_param,
        use_snip=use_snip,
        chunk_pixels=10,
        n_chunks_min=4,
        progress_bar=None,
        client=None,
    )

    ft.verify_fit_output(data_out=data_out, snip_param=ft.snip_param)


# fmt: off
@pytest.mark.parametrize("params, except_type, err_msg", [
    ({"data_sel_indices": 50}, TypeError,
     "Parameter 'data_sel_indices' must be tuple or list"),
    ({"data_sel_indices": (3, 10, 5)}, TypeError,
     "Parameter 'data_sel_indices' must contain two elements"),
    ({"data_sel_indices": [3]}, TypeError,
     "Parameter 'data_sel_indices' must contain two elements"),
    ({"data_sel_indices": (-1, 10)}, ValueError,
     "Some of the indices in 'data_sel_indices' are negative"),
    ({"data_sel_indices": [0, -10]}, ValueError,
     "Some of the indices in 'data_sel_indices' are negative"),
    ({"data_sel_indices": [3, 3]}, ValueError,
     "Parameter 'data_sel_indices' must select at least 1 element"),
    ({"data_sel_indices": [3, 2]}, ValueError,
     "Parameter 'data_sel_indices' must select at least 1 element"),
    ({"matv": np.zeros(shape=(2, 3, 5))}, TypeError,
     "Parameter 'matv' must be 2D ndarray"),
    ({"matv": np.zeros(shape=(50, 3)), "data_sel_indices": (5, 5 + 51)}, ValueError,
     "The number of selected points .* is not equal"),
    ({"snip_param": [1, 2, 3]}, TypeError,
     "Parameter 'snip_param' must be a dictionary"),
    ({"snip_param": {}, "use_snip": True}, TypeError,
     "Parameter 'snip_param' must a dictionary with keys"),
    ({"snip_param": {"e_offset": 0, "e_linear": 0.1, "e_quadratic": 0},
      "use_snip": True}, TypeError,
     "Parameter 'snip_param' must a dictionary with keys"),
    ({"data": np.zeros(shape=(10, 15, 100)), "matv": np.zeros(shape=(50, 3)),
      "data_sel_indices": (70, 70 + 50)}, ValueError,
     "Selection indices .* are outside the allowed range"),

])
# fmt: on
def test_fit_xrf_map_fail(params, except_type, err_msg):
    """Failing cases of `fit_xrf_map` (wrong input parameters)"""

    # Create a dataset first
    dataset_params = {"n_data_dimensions": (20, 20)}
    add_pts_before, add_pts_after = 15, 10
    use_snip = False

    ft = _FitXRFMapTesting(
        dataset_params=dataset_params,
        use_snip=use_snip,
        add_pts_before=add_pts_before,
        add_pts_after=add_pts_after,
    )

    kwargs = {
        "data": ft.data_input,
        "data_sel_indices": ft.data_sel_indices,
        "matv": ft.spectra,
        "snip_param": ft.snip_param,
        "use_snip": use_snip,
        "chunk_pixels": 10,
        "n_chunks_min": 4,
        "progress_bar": None,
        "client": None,
    }
    kwargs.update(params)

    # Run fitting
    with pytest.raises(except_type, match=err_msg):
        fit_xrf_map(**kwargs)


# fmt: off
@pytest.mark.parametrize("dataset_params", [
    {"n_data_dimensions": (8, 1)},
    {"n_data_dimensions": (1, 8)},
    {"n_data_dimensions": (4, 4)},
    {"n_data_dimensions": (3, 5)},
])
@pytest.mark.parametrize("add_pts_before, add_pts_after", [(0, 0), (50, 100)])
@pytest.mark.parametrize("use_snip", [False, True])
# fmt: on
def test_compute_roi(dataset_params, add_pts_before, add_pts_after, use_snip):
    """Basic functionality of `_compute_roi` function"""

    ft = _FitXRFMapTesting(
        dataset_params=dataset_params,
        use_snip=use_snip,
        add_pts_before=add_pts_before,
        add_pts_after=add_pts_after,
    )

    n_pts = ft.n_spectrum_points
    energy_min, energy_max = 3.1, 12.2  # Energy range for the 'selected' data, keV
    energy_step = (energy_max - energy_min) / (n_pts - 1)

    # 'e_offset' - the offset for the 0th element of the array, not the selected range
    snip_param = {
        "e_offset": energy_min - energy_step * add_pts_before,
        "e_linear": energy_step,
        "e_quadratic": 0,
        "b_width": 2.0,
    }

    roi_dict = {
        "roi-1": (2.5, 3.5),  # Outside the range
        "roi-2": (3.5, 4.8),
        "roi-3": (5.2, 7.4),
        "roi-4": (6.0, 6.0),  # No data points
        "roi-5": (6.0, 5.5),  # No data points
        "roi-6": (10.1, 15.0),  # Outside the range
    }

    roi_bands = [_ for _ in roi_dict.values()]

    # We don't use autogenerated 'ft.snip_param', because we want to use specific
    #   values to make sure that computations are done correctly.

    data_out = _compute_roi(
        ft.data_input,
        data_sel_indices=ft.data_sel_indices,
        roi_bands=roi_bands,
        snip_param=snip_param,
        use_snip=use_snip,
    )

    roi_keys = list(roi_dict.keys())
    assert data_out.shape == (*ft.data_input.shape[0:2], len(roi_keys)), "Output data has unexpected shape"

    # Convert data from numpy array to dictionary
    data_out = {roi_keys[_]: data_out[:, :, _] for _ in range(len(roi_keys))}
    # Verify the dictionary
    ft.verify_roi_output(data_out=data_out, roi_dict=roi_dict, snip_param=snip_param)


@pytest.mark.usefixtures("_start_dask_client")
class TestComputeSelectedROIs:
    # fmt: off
    @pytest.mark.parametrize("data_representation", ["numpy_array", "dask_array", "hdf5_file_dset"])
    @pytest.mark.parametrize("dataset_params", [
        {"n_data_dimensions": (10, 10)},
        {"n_data_dimensions": (9, 11)},
        {"n_data_dimensions": (1, 100)},
        {"n_data_dimensions": (100, 1)},
    ])
    @pytest.mark.parametrize("add_pts", [(0, 0), (50, 100)])
    @pytest.mark.parametrize("use_snip", [False, True])
    # fmt: on
    def test_compute_selected_rois1(self, data_representation, dataset_params, add_pts, use_snip, tmpdir):
        """
        Basic functionality of `compute_selected_rois`.
        Tests are run using global Dask clients to inprove testing speed.
        """

        # Dask client object is used for multiple tests to save execution time
        global_client = self.client

        add_pts_before, add_pts_after = add_pts

        ft = _FitXRFMapTesting(
            dataset_params=dataset_params,
            use_snip=use_snip,
            add_pts_before=add_pts_before,
            add_pts_after=add_pts_after,
        )

        # Unfortunately 'data_input' is ndarray, but we need dask array to work with
        #   Select very small chunk size. This is not efficient, but works fine for testing.
        data_dask = _array_numpy_to_dask(ft.data_input, chunk_pixels=4, n_chunks_min=1)
        # Now create the dataset we need
        data = _create_xrf_data(data_dask, data_representation, tmpdir)

        n_pts = ft.n_spectrum_points
        energy_min, energy_max = 3.1, 12.2  # Energy range for the 'selected' data, keV
        energy_step = (energy_max - energy_min) / (n_pts - 1)

        roi_dict = {
            "roi-1": (2.5, 3.5),  # Outside the range
            "roi-2": (3.5, 4.8),
            "roi-3": (5.2, 7.4),
            "roi-4": (6.0, 6.0),  # No data points
            "roi-5": (6.0, 5.5),  # No data points
            "roi-6": (10.1, 15.0),  # Outside the range
        }

        # 'e_offset' - the offset for the 0th element of the array, not the selected range
        snip_param = {
            "e_offset": energy_min - energy_step * add_pts_before,
            "e_linear": energy_step,
            "e_quadratic": 0,
            "b_width": 2.0,
        }

        # We don't use autogenerated 'ft.snip_param', because we want to use specific
        #   values to make sure that computations are done correctly.

        data_out = compute_selected_rois(
            data,
            data_sel_indices=ft.data_sel_indices,
            roi_dict=roi_dict,
            snip_param=snip_param,
            use_snip=use_snip,
            client=global_client,
        )

        # Verify the dictionary
        ft.verify_roi_output(data_out=data_out, roi_dict=roi_dict, snip_param=snip_param)


def test_compute_selected_rois2():
    """
    Basic functionality of `compute_selected_rois`.
    Tests are run using global Dask clients to inprove testing speed.
    """

    dataset_params = {"n_data_dimensions": (20, 20)}
    add_pts_before, add_pts_after = 15, 10
    use_snip = False

    ft = _FitXRFMapTesting(
        dataset_params=dataset_params,
        use_snip=use_snip,
        add_pts_before=add_pts_before,
        add_pts_after=add_pts_after,
    )

    # Just run the test with input data represented as numpy array
    data = ft.data_input

    n_pts = ft.n_spectrum_points
    energy_min, energy_max = 3.1, 12.2  # Energy range for the 'selected' data, keV
    energy_step = (energy_max - energy_min) / (n_pts - 1)

    roi_dict = {
        "roi-1": (2.5, 3.5),  # Outside the range
        "roi-2": (3.5, 4.8),
        "roi-3": (5.2, 7.4),
        "roi-4": (6.0, 6.0),  # No data points
        "roi-5": (6.0, 5.5),  # No data points
        "roi-6": (10.1, 15.0),  # Outside the range
    }

    # 'e_offset' - the offset for the 0th element of the array, not the selected range
    snip_param = {
        "e_offset": energy_min - energy_step * add_pts_before,
        "e_linear": energy_step,
        "e_quadratic": 0,
        "b_width": 2.0,
    }

    # We don't use autogenerated 'ft.snip_param', because we want to use specific
    #   values to make sure that computations are done correctly.

    data_out = compute_selected_rois(
        data, data_sel_indices=ft.data_sel_indices, roi_dict=roi_dict, snip_param=snip_param, use_snip=use_snip
    )

    # Verify the dictionary
    ft.verify_roi_output(data_out=data_out, roi_dict=roi_dict, snip_param=snip_param)


# fmt: off
@pytest.mark.parametrize("params, except_type, err_msg", [
    ({"data_sel_indices": 50}, TypeError,
     "Parameter 'data_sel_indices' must be tuple or list"),
    ({"data_sel_indices": (3, 10, 5)}, TypeError,
     "Parameter 'data_sel_indices' must contain two elements"),
    ({"data_sel_indices": [3]}, TypeError,
     "Parameter 'data_sel_indices' must contain two elements"),
    ({"data_sel_indices": (-1, 10)}, ValueError,
     "Some of the indices in 'data_sel_indices' are negative"),
    ({"data_sel_indices": [0, -10]}, ValueError,
     "Some of the indices in 'data_sel_indices' are negative"),
    ({"data_sel_indices": [3, 3]}, ValueError,
     "Parameter 'data_sel_indices' must select at least 1 element"),
    ({"data_sel_indices": [3, 2]}, ValueError,
     "Parameter 'data_sel_indices' must select at least 1 element"),
    ({"snip_param": [1, 2, 3]}, TypeError,
     "Parameter 'snip_param' must be a dictionary"),
    ({"snip_param": None}, TypeError,
     "Parameter 'snip_param' must be a dictionary"),
    ({"snip_param": {}}, TypeError,
     "Parameter 'snip_param' must a dictionary with keys"),
    ({"snip_param": {"e_offset": 0, "e_linear": 0.1, "e_quadratic": 0},
      "use_snip": True}, TypeError,
     "Parameter 'snip_param' must a dictionary with keys"),
    ({"data": np.zeros(shape=(10, 15, 100)),
      "data_sel_indices": (70, 70 + 50)}, ValueError,
     "Selection indices .* are outside the allowed range"),
])
# fmt: on
def test_compute_selected_rois_fail(params, except_type, err_msg):
    """Failing cases of `compute_selected_rois` (wrong input parameters)"""

    # Create a dataset first
    dataset_params = {"n_data_dimensions": (20, 20)}
    add_pts_before, add_pts_after = 15, 10
    use_snip = False

    ft = _FitXRFMapTesting(
        dataset_params=dataset_params,
        use_snip=use_snip,
        add_pts_before=add_pts_before,
        add_pts_after=add_pts_after,
    )

    # Just run the test with input data represented as numpy array
    data = ft.data_input

    # The particular values 'snip_param' are unimportant. Instead, the general
    #   structure of the parameter is tested. So we just use the autogenerated
    #   `ft.snip_param`, which is sometimes overriddent by the test parameter

    roi_dict = {
        "roi-1": (2.5, 3.5),  # Outside the range
        "roi-2": (3.5, 4.8),
        "roi-3": (5.2, 7.4),
        "roi-4": (6.0, 6.0),  # No data points
        "roi-5": (6.0, 5.5),  # No data points
        "roi-6": (10.1, 15.0),  # Outside the range
    }

    kwargs = {
        "data": data,
        "data_sel_indices": ft.data_sel_indices,
        "roi_dict": roi_dict,
        "snip_param": ft.snip_param,
        "use_snip": use_snip,
        "chunk_pixels": 10,
        "n_chunks_min": 4,
        "progress_bar": None,
        "client": None,
    }
    kwargs.update(params)

    # Run fitting
    with pytest.raises(except_type, match=err_msg):
        compute_selected_rois(**kwargs)


def test_snip_method_numba():
    """
    Compare the output of `snip_method_numba` with the output produced
    by `snip_method` to make sure the functions are equivalent.
    """

    dataset_params = {"n_data_dimensions": (20, 20)}
    add_pts_before, add_pts_after = 0, 0
    use_snip = True

    ft = _FitXRFMapTesting(
        dataset_params=dataset_params,
        use_snip=use_snip,
        add_pts_before=add_pts_before,
        add_pts_after=add_pts_after,
    )

    # Just run the test with input data represented as numpy array
    data = ft.data_input

    width = 0.5

    for ny in range(data.shape[0]):
        for nx in range(data.shape[1]):
            spec_sel = data[ny, nx, :]

            bg = snip_method_numba(spec_sel, 0, 0.01, 0, width=width)
            bg_expected = snip_method(spec_sel, 0, 0.01, 0, width=width)

            npt.assert_array_almost_equal(
                bg, bg_expected, err_msg=f"Background estimates don't match for the pixel ({ny}, {nx})"
            )
