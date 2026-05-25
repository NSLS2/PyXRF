import copy
import os
import re
import time as ttime

import jsonschema
import numpy as np
import numpy.testing as npt
import pytest

from pyxrf.core.quant_analysis import (
    ParamQuantEstimation,
    ParamQuantitativeAnalysis,
    _xrf_quant_fluor_schema,
    _xrf_standard_schema,
    compute_standard_element_densities,
    fill_quant_fluor_data_dict,
    get_quant_fluor_data_dict,
    load_included_xrf_standard_yaml_file,
    load_xrf_quant_fluor_json_file,
    load_xrf_standard_yaml_file,
    prune_quant_fluor_data_dict,
    save_xrf_quant_fluor_json_file,
    save_xrf_standard_yaml_file,
    set_quant_fluor_data_dict_optional,
    set_quant_fluor_data_dict_time,
)
from pyxrf.core.utils import convert_time_from_nexus_string
from pyxrf.core.xrf_utils import generate_eline_list, split_compound_mass, validate_element_str

# Short example of XRF standard data
_standard_data_sample = [
    {
        "name": "Test Micromatter 411570",
        "serial": "411570",
        "description": "InSx 22.4 (In=16.0 S=6.4) / SrF2 20.6 / Cr 19.8 / Ni 19.2 / GaAs 19.1 (Ga=8.3 As=10.8)",
        "compounds": {"In": 16.0, "S": 6.4, "SrF2": 20.6, "Cr": 19.8, "Ni": 19.2, "Ga": 8.3, "As": 10.8},
        "density": 101.1,
    },
    {
        "name": "Test Micromatter 411640",
        "serial": "411640",
        "description": "CeF3 21.1 / Au 20.6",
        "compounds": {"CeF3": 21.1, "Au": 20.6},
        # Missing optional 'density' field
    },
]


# fmt: off
@pytest.mark.parametrize("standard_data", [
    _standard_data_sample,
    []  # The function should also work if the list is empty
])
# fmt: on
def test_save_xrf_standard_yaml_file1(tmp_path, standard_data):
    r"""Basic test of function 'save_xrf_standard_yaml_file' and 'load_xrf_standard_yaml_file'"""

    yaml_path = ["yaml", "param", "file"]
    file_name = "standard.yaml"
    yaml_path = os.path.join(tmp_path, *yaml_path, file_name)

    # Sample data
    save_xrf_standard_yaml_file(yaml_path, standard_data)

    data_loaded = load_xrf_standard_yaml_file(yaml_path)

    assert data_loaded == standard_data, "Loaded data is not equal to the original data"


def test_save_xrf_standard_yaml_file2(tmp_path):
    r"""Test the case of of no match between total density and sum of densities of individual compounds"""

    yaml_path = ["yaml", "param", "file"]
    file_name = "standard.yaml"
    yaml_path = os.path.join(tmp_path, *yaml_path, file_name)

    standard_data = []  # No records, but the file will still be saved

    # Sample data
    save_xrf_standard_yaml_file(yaml_path, standard_data)

    # Try to overwrite the file ('re.escape' is required when tests are run on Windows)
    with pytest.raises(IOError, match=f"File '{re.escape(yaml_path)}' already exists"):
        save_xrf_standard_yaml_file(yaml_path, standard_data)

    # The following should work
    save_xrf_standard_yaml_file(yaml_path, standard_data, overwrite_existing=True)


def test_save_xrf_standard_yaml_file3(tmp_path):
    r"""Test the case of existing file"""

    yaml_path = ["yaml", "param", "file"]
    file_name = "standard.yaml"
    yaml_path = os.path.join(tmp_path, *yaml_path, file_name)

    standard_data = copy.deepcopy(_standard_data_sample)
    standard_data[0]["density"] = 50.0  # Wrong value of total density

    # Save incorrect data
    save_xrf_standard_yaml_file(yaml_path, standard_data)
    #    and now try to read it
    with pytest.raises(RuntimeError, match="Sum of areal densities does not match total density"):
        load_xrf_standard_yaml_file(yaml_path)


def test_load_xrf_standard_yaml_file1(tmp_path):
    r"""Try loading non-existent YAML file"""

    file_name = "standard.yaml"

    # Try loading from the existing directory
    # 're.escape' is necessary when test are run on Windows
    yaml_path = os.path.join(tmp_path, file_name)
    with pytest.raises(IOError, match=f"File '{re.escape(yaml_path)}' does not exist"):
        load_xrf_standard_yaml_file(yaml_path)

    # Try loading from the non-existing directory
    yaml_path = ["yaml", "param", "file"]
    yaml_path = os.path.join(tmp_path, *yaml_path, file_name)
    with pytest.raises(IOError, match=f"File '{re.escape(yaml_path)}' does not exist"):
        load_xrf_standard_yaml_file(yaml_path)


def test_load_xrf_standard_yaml_file2(tmp_path):
    r"""Test for reporting schema violation"""

    yaml_path = ["yaml", "param", "file"]
    file_name = "standard.yaml"
    yaml_path = os.path.join(tmp_path, *yaml_path, file_name)

    standard_data = copy.deepcopy(_standard_data_sample)
    # Change name from string to number (this will not satisfy the built-in schema)
    for v in standard_data:
        v["name"] = 50.36

    # Save the changed dataset
    save_xrf_standard_yaml_file(yaml_path, standard_data)

    with pytest.raises(jsonschema.ValidationError):
        load_xrf_standard_yaml_file(yaml_path)

    # Disable schema validation by setting 'schema=None'
    load_xrf_standard_yaml_file(yaml_path, schema=None)

    # Change the schema to match data and validate with changed schema
    schema = copy.deepcopy(_xrf_standard_schema)
    schema["properties"]["name"] = {"type": ["string", "number"]}
    standard_data_out = load_xrf_standard_yaml_file(yaml_path, schema=schema)
    assert standard_data_out == standard_data, "Loaded data is different from the original"


def test_load_included_xrf_standard_yaml_file():
    data = load_included_xrf_standard_yaml_file()
    assert len(data) > 1, "Standard YAML file can not be read"


def test_compute_standard_element_densities():
    standard_data = _standard_data_sample

    for data in standard_data:
        # Find total density
        if "density" in data:
            total_density = data["density"]
        else:
            total_density = np.sum(list(data["compounds"].values()))

        element_densities = compute_standard_element_densities(data["compounds"])

        # Validate that all the keys of the returned dictionary represent elements
        assert all(
            [validate_element_str(_) for _ in element_densities.keys()]
        ), f"Some of the elements in the list {list(element_densities.keys())} have invalid representation"

        # Check if the sum of all return densities equals total density
        npt.assert_almost_equal(
            np.sum(list(element_densities.values())),
            total_density,
            err_msg="The sum of element densities and the total sum don't match",
        )


# -----------------------------------------------------------------------------------------------------


# Short example of XRF standard quantitative data record
#   The data has no physical meaning, used for testing of saving/loading of JSON file
_xrf_standard_fluor_sample = {
    "name": "Hypothetical sample #41157",
    "serial": "41157",
    "description": "Name of hypothetical sample",
    "element_lines": {
        "In": {"density": 16.0, "fluorescence": 1.5453452},
        "S_K": {"density": 6.4, "fluorescence": 2.0344345},
        "Sr_L": {"density": 20.6, "fluorescence": 0.93452365},
        "Au_M": {"density": 10.4, "fluorescence": 0.734234},
        "Cr_Ka": {"density": 19.8, "fluorescence": 0.7435234},
        "Ni_Kb": {"density": 19.2, "fluorescence": 0.7435234},
        "Ga_Ka1": {"density": 8.3, "fluorescence": 0.7435234},
        "Mg_Ka2": {"density": 9.6, "fluorescence": None},
    },
    "incident_energy": 10.5,
    "detector_channel": "sum",
    "scaler_name": "i0",
    "distance_to_sample": 50.6,
    "creation_time_local": "2020-01-10T11:50:39+00:00",
    "source_scan_id": 92276,
    "source_scan_uid": "f07e3065-ab92-4b20-a702-ef61ed164dbc",
}


def _get_data_and_json_path(tmp_path):
    # Create some complicated path
    json_path = ["json", "param", "file"]
    file_name = "standard.yaml"
    json_path = os.path.join(tmp_path, *json_path, file_name)

    data = _xrf_standard_fluor_sample

    return data, json_path


def test_save_xrf_quant_fluor_json_file1(tmp_path):
    r"""Basic test of function 'save_xrf_standard_yaml_file' and 'load_xrf_standard_yaml_file'"""

    data, json_path = _get_data_and_json_path(tmp_path)

    # Sample data
    save_xrf_quant_fluor_json_file(json_path, data)

    data_loaded = load_xrf_quant_fluor_json_file(json_path)

    assert data_loaded == data, "Loaded data is not equal to the original data"


def test_save_xrf_quant_fluor_json_file2(tmp_path):
    # 'save_xrf_quant_fluor_json_file' - overwrite existing file

    data, json_path = _get_data_and_json_path(tmp_path)

    # Create file
    save_xrf_quant_fluor_json_file(json_path, data)

    # Attempt to overwrite (note: 're.escape' is necessary for Windows paths)
    with pytest.raises(IOError, match=f"File '{re.escape(json_path)}' already exists"):
        save_xrf_quant_fluor_json_file(json_path, data)

    # Now overwrite the file by setting the flag 'overwrite_existing=True'
    save_xrf_quant_fluor_json_file(json_path, data, overwrite_existing=True)


def test_save_xrf_quant_fluor_json_file3(tmp_path):
    # 'save_xrf_quant_fluor_json_file' - invalid schema

    data, json_path = _get_data_and_json_path(tmp_path)

    data = copy.deepcopy(data)
    # Change the data so that it doesn't satisfy the schema
    data["incident_energy"] = "incident_energy"  # Supposed to be a number

    with pytest.raises(jsonschema.ValidationError):
        save_xrf_quant_fluor_json_file(json_path, data)


def test_save_xrf_quant_fluor_json_file4(tmp_path):
    r"""Schema allows some data fields to hold value of ``None``. Test if this works."""
    data, json_path = _get_data_and_json_path(tmp_path)

    # Modify some elements of the dictionary
    data = copy.deepcopy(data)
    data["detector_channel"] = None
    data["scaler_name"] = None
    data["distance_to_sample"] = None

    # Sample data
    save_xrf_quant_fluor_json_file(json_path, data)

    data_loaded = load_xrf_quant_fluor_json_file(json_path)

    assert data_loaded == data, "Loaded data is not equal to the original data"


def test_load_xrf_quant_fluor_json_file1(tmp_path):
    # 'load_xrf_quant_fluor_json_file' - non-existing file

    _, json_path = _get_data_and_json_path(tmp_path)

    # 're.escape' is necessary if test is run on Windows
    with pytest.raises(IOError, match=f"File '{re.escape(json_path)}' does not exist"):
        load_xrf_quant_fluor_json_file(json_path)


def test_load_xrf_quant_fluor_json_file2(tmp_path):
    # 'load_xrf_quant_fluor_json_file' - schema is not matched

    data, json_path = _get_data_and_json_path(tmp_path)

    # Create file
    save_xrf_quant_fluor_json_file(json_path, data)

    schema = copy.deepcopy(_xrf_quant_fluor_schema)
    # Modify schema, so that it is incorrect
    schema["properties"]["scaler_name"] = {"type": "number"}  # Supposed to be a string

    with pytest.raises(jsonschema.ValidationError):
        load_xrf_quant_fluor_json_file(json_path, schema=schema)


def test_get_quant_fluor_data_dict():
    """Tests for 'get_quant_fluor_data_dict': basic tests for consistensy of the returned dictionary"""

    for standard_data in _standard_data_sample:
        quant_fluor_data_dict = get_quant_fluor_data_dict(standard_data, incident_energy=12.0)
        # Will raise exception is schema is not satisfied
        jsonschema.validate(instance=quant_fluor_data_dict, schema=_xrf_quant_fluor_schema)

        assert quant_fluor_data_dict["name"] == standard_data["name"], "Dictionary element 'name' is incorrect"

        assert (
            quant_fluor_data_dict["serial"] == standard_data["serial"]
        ), "Dictionary element 'serial' is incorrect"

        assert (
            quant_fluor_data_dict["description"] == standard_data["description"]
        ), "Dictionary element 'description' is incorrect"

        eline_set = set()
        # The 'mass' is not actual mass. If elements has multiple emission lines activated, then
        #   the mass (density) of the element will be counted multiple times. There is no
        #   physical meaning in the computed value: it is used to verify the sum of densities in
        #   the 'element_lines' dictionary of emission lines in 'quant_fluor_data_dict'
        mass_sum_expected = 0
        for cmpd, mass in standard_data["compounds"].items():
            em_dict = split_compound_mass(cmpd, mass)
            for el, ms in em_dict.items():
                elines = generate_eline_list([el], incident_energy=12.0)
                n_elines = len(elines)
                if n_elines:
                    mass_sum_expected += n_elines * ms
                    eline_set.update(elines)

        eline_out_list = list(quant_fluor_data_dict["element_lines"].keys())
        assert len(eline_out_list) == len(eline_set), "The number of emission lines is not as expected"
        assert (
            set(eline_out_list) == eline_set
        ), "Generated object contains emission lines that are different from expected"

        mass_sum = sum([_["density"] for _ in quant_fluor_data_dict["element_lines"].values()])
        npt.assert_almost_equal(
            mass_sum,
            mass_sum_expected,
            err_msg="The total mass (density) of the components is different from expected",
        )


def gen_xrf_map_dict(nx=10, ny=5, elines=["S_K", "Au_M", "Fe_K"]):
    r"""
    Create dictionary with the given set of emission lines and scaler ``sclr``
    """
    img = {}
    for e in elines:
        map = np.random.rand(ny, nx) * np.random.rand() * 10
        img[e] = map

    # Scaler
    map_sclr = np.ones(shape=(ny, nx), dtype=float) * np.random.rand() * 2
    img["sclr"] = map_sclr

    return img


# fmt: off
@pytest.mark.parametrize("map_dims", [
    {"nx": 10, "ny": 5},
    {"nx": 1, "ny": 5},
    {"nx": 2, "ny": 5},
    {"nx": 3, "ny": 5},
    {"nx": 10, "ny": 1},
    {"nx": 10, "ny": 2},
    {"nx": 10, "ny": 3},
])
# fmt: on
def test_fill_quant_fluor_data_dict(map_dims):
    r"""Test for 'fill_quant_fluor_data_dict': testing basic functionality"""
    # Create copy, because the dictionary will be modified
    fluor_standard = copy.deepcopy(_xrf_standard_fluor_sample)

    nx = map_dims["nx"]
    ny = map_dims["ny"]

    img = gen_xrf_map_dict(nx=nx, ny=ny)

    if nx < 3:
        nx_min, nx_max = 0, nx
    else:
        nx_min, nx_max = 1, -1

    if ny < 3:
        ny_min, ny_max = 0, nx
    else:
        ny_min, ny_max = 1, -1

    map_S_K_fluor = np.mean(img["S_K"][ny_min:ny_max, nx_min:nx_max])
    map_Au_M_fluor = np.mean(img["Au_M"][ny_min:ny_max, nx_min:nx_max])
    v_sclr = np.mean(img["sclr"][ny_min:ny_max, nx_min:nx_max])

    # Fill the dictionary, use existing scaler 'sclr'
    fill_quant_fluor_data_dict(fluor_standard, xrf_map_dict=img, scaler_name="sclr")

    npt.assert_almost_equal(
        fluor_standard["element_lines"]["S_K"]["fluorescence"],
        map_S_K_fluor / v_sclr,
        err_msg="Fluorescence of 'S_K' is estimated incorrectly",
    )
    npt.assert_almost_equal(
        fluor_standard["element_lines"]["Au_M"]["fluorescence"],
        map_Au_M_fluor / v_sclr,
        err_msg="Fluorescence of 'Au_M' is estimated incorrectly",
    )
    for eline, param in fluor_standard["element_lines"].items():
        assert (eline in img) or (
            param["fluorescence"] is None
        ), f"Fluorescence line {eline} is not present in the dataset and it was not reset to None"

    # Fill the dictionary, use non-existing scaler 'abc'
    fill_quant_fluor_data_dict(fluor_standard, xrf_map_dict=img, scaler_name="abc")
    npt.assert_almost_equal(
        fluor_standard["element_lines"]["S_K"]["fluorescence"],
        map_S_K_fluor,
        err_msg="Fluorescence of 'S_K' is estimated incorrectly",
    )

    # Fill the dictionary, don't use a scaler (set to None)
    fill_quant_fluor_data_dict(fluor_standard, xrf_map_dict=img, scaler_name=None)
    npt.assert_almost_equal(
        fluor_standard["element_lines"]["S_K"]["fluorescence"],
        map_S_K_fluor,
        err_msg="Fluorescence of 'S_K' is estimated incorrectly",
    )


def test_prune_quant_fluor_data_dict():
    r"""Test for ``prune_quant_fluor_data_dict``: basic test of functionality"""

    # Create copy, because the dictionary will be modified
    fluor_standard = copy.deepcopy(_xrf_standard_fluor_sample)

    elines_not_none = []
    for eline, info in fluor_standard["element_lines"].items():
        set_to_none = np.random.rand() < 0.5  # 50% chance
        if set_to_none:
            info["fluorescence"] = None
        # For some emission lines, fluorescence could already be set to None
        if info["fluorescence"] is not None:
            elines_not_none.append(eline)

    fluor_standard_pruned = prune_quant_fluor_data_dict(fluor_standard)
    for eline, info in fluor_standard_pruned["element_lines"].items():
        assert eline in elines_not_none, f"Emission line {eline} should have been removed from the dictionary"
        assert info["fluorescence"] is not None, f"Emission line {eline} has 'fluorescence' set to None"


def test_set_quant_fluor_data_dict_optional_1():
    r"""
    Tests for 'set_quant_fluor_data_dict_optional': basic test of functionality.
    Successful tests.
    """

    # Create copy, because the dictionary will be modified
    fluor_standard = copy.deepcopy(_xrf_standard_fluor_sample)

    scan_id = 45378  # scan_id has to be int or a string representing int
    scan_uid = "abc-12345"  # Just some string, format is not checked

    set_quant_fluor_data_dict_optional(fluor_standard, scan_id=scan_id, scan_uid=scan_uid)

    # Check if scan_id and scan_uid are set
    assert fluor_standard["source_scan_id"] == scan_id, "Scan ID is set incorrectly"
    assert fluor_standard["source_scan_uid"] == scan_uid, "Scan UID is set incorrectly"

    # Check if time is set correctly
    t = fluor_standard["creation_time_local"]
    assert t is not None, "Time is not set"

    t = convert_time_from_nexus_string(t)
    t = ttime.mktime(t)
    t_current = ttime.mktime(ttime.localtime())

    # 5 seconds is more than sufficient to complete this test
    assert abs(t_current - t) < 5.0, "Time is set incorrectly"

    # Test if sending Scan ID as a string works
    scan_id2 = 45346
    scan_id2_str = f"{scan_id2}"
    set_quant_fluor_data_dict_optional(fluor_standard, scan_id=scan_id2_str)
    assert fluor_standard["source_scan_id"] == scan_id2, "Scan ID is set incorrectly"


def test_set_quant_fluor_data_dict_optional_2():
    r"""
    Tests for 'set_quant_fluor_data_dict_optional'.
    Failing tests.
    """

    # Create copy, because the dictionary will be modified
    fluor_standard = copy.deepcopy(_xrf_standard_fluor_sample)

    # Scan ID is a string, which can not be interpreted as int
    with pytest.raises(RuntimeError, match="Parameter 'scan_id' must be integer or a string representing integer"):
        set_quant_fluor_data_dict_optional(fluor_standard, scan_id="abc_34g")

    # Scan ID is of wrong type
    with pytest.raises(RuntimeError, match="Parameter 'scan_id' must be integer or a string representing integer"):
        set_quant_fluor_data_dict_optional(fluor_standard, scan_id=[1, 5, 14])

    # Scan UID is of wrong type
    with pytest.raises(RuntimeError, match="Parameter 'scan_uid' must be a string representing scan UID"):
        set_quant_fluor_data_dict_optional(fluor_standard, scan_uid=[1, 5, 14])


def test_set_quant_fluor_data_dict_time():
    # Create copy, because the dictionary will be modified
    fluor_standard = copy.deepcopy(_xrf_standard_fluor_sample)

    set_quant_fluor_data_dict_time(fluor_standard)

    # Check if time is set correctly
    t = fluor_standard["creation_time_local"]
    assert t is not None, "Time is not set"

    t = convert_time_from_nexus_string(t)
    t = ttime.mktime(t)
    t_current = ttime.mktime(ttime.localtime())

    # 5 seconds is more than sufficient to complete this test
    assert abs(t_current - t) < 5.0, "Time is set incorrectly"


# --------------------------------------------------------------------------------------------------------


def test_ParamQuantEstimation_1(tmp_path):
    r"""
    Create the object of the 'ParamQuantEstimation' class. Load and then clear reference data.
    """

    # 'home_dir' is typically '~', but for testing it is set to 'tmp_dir'
    home_dir = tmp_path
    config_dir = ".pyxrf"
    standards_fln = "quantitative_standards.yaml"

    pqe = ParamQuantEstimation(home_dir=home_dir)

    # Verify that the cofig file was created
    file_path = os.path.join(home_dir, config_dir, standards_fln)
    assert os.path.isfile(
        file_path
    ), f"Empty file for user-defined reference standards '{file_path}' was not created"

    # Load reference standards
    pqe.load_standards()
    assert pqe.standards_built_in is not None, "Failed to load built-in standards"
    assert pqe.standards_custom is not None, "Failed to load user-defined standards"

    # Clear loaded standards
    pqe.clear_standards()
    assert pqe.standards_built_in is None, "Failed to clear loaded built-in standards"
    assert pqe.standards_custom is None, "Failed to clear loaded user-defined standards"


def test_ParamQuantEstimation_2(tmp_path):
    standard_data = _standard_data_sample

    # 'home_dir' is typically '~', but for testing it is set to 'tmp_dir'
    home_dir = tmp_path
    config_dir = ".pyxrf"
    standards_fln = "quantitative_standards.yaml"

    # Create the file with user-defined reference definitions
    file_path = os.path.join(home_dir, config_dir, standards_fln)
    save_xrf_standard_yaml_file(file_path, standard_data)

    # Create the object and load references
    pqe = ParamQuantEstimation(home_dir=home_dir)
    pqe.load_standards()
    assert pqe.standards_built_in is not None, "Failed to load built-in standards"
    assert len(pqe.standards_built_in) > 0, "The number of loaded built-in standards is ZERO"
    assert pqe.standards_custom is not None, "Failed to load user-defined standards"
    assert len(pqe.standards_custom) > 0, "The number of loaded user-defined standards is ZERO"

    # Verify if the functions for searching the user-defined standards
    for st in pqe.standards_custom:
        serial = st["serial"]
        assert pqe._find_standard_custom(st), f"Standard {serial} was not found in user-defined list"
        assert not pqe._find_standard_built_in(st), f"Standard {serial} was found in built-in list"
        assert pqe.find_standard(st), f"Standard {serial} was not found"
        assert pqe.find_standard(st["name"], key="name"), f"Failed to find standard {serial} by name"
        assert pqe.find_standard(st["serial"], key="serial"), f"Failed to find standard {serial} by serial number"
        assert pqe.is_standard_custom(st), f"Standard {serial} was not identified as user-defined"
        pqe.set_selected_standard(st)
        assert pqe.standard_selected == st, f"Can't select standard {serial}"

    # Verify if the functions for searching the user-defined standards
    for st in pqe.standards_built_in:
        serial = st["serial"]
        assert not pqe._find_standard_custom(st), f"Standard {serial} was found in user-defined list"
        assert pqe._find_standard_built_in(st), f"Standard {serial} was not found in built-in list"
        assert pqe.find_standard(st), f"Standard {serial} was not found"
        assert pqe.find_standard(st["name"], key="name"), f"Failed to find standard {serial} by name"
        assert pqe.find_standard(st["serial"], key="serial"), f"Failed to find standard {serial} by serial number"
        assert not pqe.is_standard_custom(st), f"Standard {serial} was identified as user-defined"
        pqe.set_selected_standard(st)
        assert pqe.standard_selected == st, f"Can't select standard {serial}"

    # Selecting non-existing standard
    st = {"serial": "09876", "name": "Some name"}  # Some arbitatry dictionary
    st_selected = pqe.set_selected_standard(st)
    assert st_selected == pqe.standard_selected, "Return value of 'set_selected_standard' is incorrect"
    assert st_selected == pqe.standards_custom[0], "Incorrect standard is selected"
    # No argument (standard is None)
    pqe.set_selected_standard()
    assert st_selected == pqe.standards_custom[0], "Incorrect standard is selected"
    # Delete the user-defined standards (this is not normal operation, but it's OK for testing)
    pqe.standards_custom = None
    pqe.set_selected_standard(st)
    assert pqe.standard_selected == pqe.standards_built_in[0], "Incorrect standard is selected"
    pqe.standards_built_in = None
    pqe.set_selected_standard(st)
    assert pqe.standard_selected is None, "Incorrect standard is selected"


def test_ParamQuantEstimation_3(tmp_path):
    standard_data = _standard_data_sample

    # 'home_dir' is typically '~', but for testing it is set to 'tmp_dir'
    home_dir = tmp_path
    config_dir = ".pyxrf"
    standards_fln = "quantitative_standards.yaml"

    # Create the file with user-defined reference definitions
    file_path = os.path.join(home_dir, config_dir, standards_fln)
    save_xrf_standard_yaml_file(file_path, standard_data)

    # Create the object and load references
    pqe = ParamQuantEstimation(home_dir=home_dir)
    pqe.load_standards()

    # Select first 'user-defined' sample (from '_standard_data_sample' list)
    incident_energy = 12.0
    img = gen_xrf_map_dict()

    # Generate and fill fluorescence data dictionary
    pqe.set_selected_standard()
    pqe.gen_fluorescence_data_dict(incident_energy=incident_energy)
    scaler_name = "sclr"
    pqe.fill_fluorescence_data_dict(xrf_map_dict=img, scaler_name=scaler_name)

    # Equivalent transformations using functions that are already tested. The sequence
    #   must give the same result (same functions are called).
    qfdd = get_quant_fluor_data_dict(standard_data[0], incident_energy)
    fill_quant_fluor_data_dict(qfdd, xrf_map_dict=img, scaler_name=scaler_name)

    assert (
        pqe.fluorescence_data_dict == qfdd
    ), "The filled fluorescence data dictionary does not match the expected"

    # Test generation of preview (superficial)
    pview, msg_warnings = pqe.get_fluorescence_data_dict_text_preview()
    # It is expected that the preview will contain 'WARNING:'
    assert len(msg_warnings), "Warning is not found in preview"
    # Disable warnings
    pview, msg_warnings = pqe.get_fluorescence_data_dict_text_preview(enable_warnings=False)
    assert not len(msg_warnings), "Warnings are disabled, but still generated"

    # Test function for setting detector channel name
    pqe.set_detector_channel_in_data_dict(detector_channel="sum")
    assert pqe.fluorescence_data_dict["detector_channel"] == "sum", "Detector channel was not set correctly"

    # Test function for setting distance to sample
    distance_to_sample = 2.5
    pqe.set_distance_to_sample_in_data_dict(distance_to_sample=distance_to_sample)
    assert (
        pqe.fluorescence_data_dict["distance_to_sample"] == distance_to_sample
    ), "Distance-to-sample was not set correctly"

    # Function for setting Scan ID and Scan UID
    scan_id = 65476
    scan_uid = "abcdef-12345678"  # Some string

    qfdd = copy.deepcopy(pqe.fluorescence_data_dict)
    pqe.set_optional_parameters(scan_id=scan_id, scan_uid=scan_uid)
    set_quant_fluor_data_dict_optional(qfdd, scan_id=scan_id, scan_uid=scan_uid)
    # We don't check if time is computed correctly (this was checked at different place)
    #   Time computed at different function calls may be different
    qfdd["creation_time_local"] = pqe.fluorescence_data_dict["creation_time_local"]
    assert pqe.fluorescence_data_dict == qfdd, "Optional parameters are not set correctly"

    # Try generating preview again: there should be no warnings
    pview, msg_warnings = pqe.get_fluorescence_data_dict_text_preview()
    # There should be no warnings in preview (all parameters are set)
    assert not len(msg_warnings), "Preview is expected to contain no warnings"

    #  Test the method 'get_suggested_json_fln'
    fln_suggested = pqe.get_suggested_json_fln()
    assert (
        f"_{pqe.fluorescence_data_dict['serial']}." in fln_suggested
    ), f"Serial of the reference is not found in the suggested file name {fln_suggested}"

    # Test saving calibration data
    file_path = os.path.join(tmp_path, fln_suggested)
    pqe.save_fluorescence_data_dict(file_path=file_path)

    qfdd = load_xrf_quant_fluor_json_file(file_path=file_path)
    assert qfdd == prune_quant_fluor_data_dict(
        pqe.fluorescence_data_dict
    ), "Error occurred while saving and loading calibration data dictionary. Dictionaries don't match"


# ---------------------------------------------------------------------------------------------


def _create_file_with_ref_standards(*, wd):
    r"""
    Create a file with user-defined standards based on ``_standard_data_sample[0]``.
    The file contains the descriptions of 2 standards with identical sets of elements/compounds
    with slightly different densities.

    The created file is placed at the standard default location ``<wd>/.pyxrf/quantiative_standards.yaml``.
    Working directory (typically ``~``) is set to temporary directory for using with PyTest.

    It is expected that the dataset will be processed using incident energy value 12.0 keV.
    """
    # Create artificial dataset based on the same standard
    sd = _standard_data_sample[0]
    file_path = os.path.join(wd, ".pyxrf", "quantitative_standards.yaml")

    standard_data = []
    for n in range(2):
        sd_copy = copy.deepcopy(sd)
        # Make sure that the datasets have different serial numbers
        sd_copy["serial"] += f"{n}"
        sd_copy["name"] = f"Test reference standard #{sd_copy['serial']}"
        d = sd_copy["compounds"]
        # Also introduce some small disturbance to the density values
        total_density = 0
        for c in d:
            d[c] += np.random.rand() * 2 - 1
            total_density += d[c]
        sd_copy["density"] = total_density  # Total density also changes
        standard_data.append(sd_copy)

    save_xrf_standard_yaml_file(file_path=file_path, standard_data=standard_data)

    # Create sets of XRF maps for each simulated 'standard'. The images
    #   will have random fluorescence
    img_list = []
    # Pick two overlapping sets of emission lines
    eline_lists = [("Ni_K", "Ga_K", "S_K"), ("S_K", "Cr_K", "Ni_K")]
    for elist in eline_lists:
        img_list.append(gen_xrf_map_dict(elines=elist))

    return file_path, img_list


def _create_files_with_ref_calib(*, wd, img_list, incident_energy=12.0):
    r"""
    Create reference calibration files based on the reference standards read from
    user-defined configuration file at standard location ``<wd>/.pyxrf/quantiative_standards.yaml``.
    The files are created in the working directory <wd>. The list of file paths is returned
    by the function.

    Parameter ``img_list`` is the list of XRF map dictionaries. The number of elements in the list
    must match the number of reference standards. Incident energy is used to determine active
    emission lines, so it has to be set properly.
    """
    pqe = ParamQuantEstimation(home_dir=wd)
    pqe.load_standards()
    pqe.standards_built_in = None  # Remove all 'built-in' standards
    # Now only the 'user-defined' standards are loaded, which were generated by
    #   'create_file_with_ref_standards' function

    file_paths = []
    for n, st in enumerate(pqe.standards_custom):
        pqe.set_selected_standard(st)
        pqe.gen_fluorescence_data_dict(incident_energy=incident_energy)
        pqe.fill_fluorescence_data_dict(xrf_map_dict=img_list[n], scaler_name="sclr")
        pqe.set_detector_channel_in_data_dict(detector_channel="sum")
        pqe.set_distance_to_sample_in_data_dict(distance_to_sample=2.0)
        pqe.set_optional_parameters(scan_id=f"12345{n}", scan_uid="some-uid-string-{n}")
        fln = pqe.get_suggested_json_fln()
        f_path = os.path.join(wd, fln)
        pqe.save_fluorescence_data_dict(file_path=f_path)
        file_paths.append(f_path)

    return file_paths


def create_ref_calib_data(tmp_path, incident_energy=12.0):
    r"""
    Create the dataset, which contains 2 QA reference calibration files (located at ``tmp_path``)
    and XRF map dictionaries with simulated experimental data for each calibration scan. Those
    dictionaries were used to compute fluorescence values for the emission lines during calibration.

    Returns: ``file_paths`` - list of paths to calibration files,
    ``img_list`` - list of simulated XRF map dictionaries, one dictionary per calibration file.
    """
    # Create file with refernce standards
    _, img_list = _create_file_with_ref_standards(wd=tmp_path)

    # Create files with reference calibration data
    file_paths = _create_files_with_ref_calib(wd=tmp_path, img_list=img_list, incident_energy=incident_energy)

    return file_paths, img_list


def test_ParamQuantitativeAnalysis(tmp_path):
    incident_energy = 12.0
    file_paths, img_list = create_ref_calib_data(tmp_path, incident_energy=incident_energy)

    n_entries = len(file_paths)
    assert n_entries >= 2, "The number of calibration data entries must be >= 2. Test can not be performed."
    fln0 = file_paths[0]

    # Load all QA calibration files
    pqa = ParamQuantitativeAnalysis()
    for fpath in file_paths:
        pqa.load_entry(file_path=fpath)  # Use kwarg

    # Check if the number of calibration entries is equal to the number of loaded files
    #   Also check if the function 'get_n_entries' works
    assert pqa.get_n_entries() == n_entries, "Incorrect number of loaded calibration data entries"

    # Check 'get_file_path_list' function
    assert (
        pqa.get_file_path_list() == file_paths
    ), "The returned list of file paths is not the same as the expected list"

    # Attempt to load calibration file that was already loaded
    #   Should not be loaded, the number of entries should remain unchanged
    pqa.load_entry(fln0)  # Send 'fpath' as arg, not kwarg
    assert len(pqa.calibration_data) == n_entries, "Incorrect number of loaded calibration data entries"
    assert len(pqa.calibration_settings) == n_entries, "Unexpected number of 'calibration_settings' elements"

    # Try removing the calibration data entry (the first)
    pqa.remove_entry(file_path=fln0)
    assert len(pqa.calibration_data) == n_entries - 1, "The result of calibration entry removal is not as expected"
    assert len(pqa.calibration_settings) == n_entries - 1, "Unexpected number of 'calibration_settings' elements"

    # Reload the deleted calibration data entry
    pqa.load_entry(fln0)  # Send 'fpath' as arg, not kwarg
    assert len(pqa.calibration_data) == n_entries, "Incorrect number of loaded calibration data entries"
    assert len(pqa.calibration_settings) == n_entries, "Unexpected number of 'calibration_settings' elements"
    # Make sure that the last element contains the last loaded calibration data entry
    assert (
        pqa.calibration_settings[-1]["file_path"] == fln0
    ), "The last element of 'calibration_settings' list contains wrong calibration data entry"

    # Find the index of calibration data entry (by file name)
    for n, fp in enumerate(file_paths):
        n_expected = n - 1
        if n_expected < 0:
            n_expected += n_entries  # n_entries - the number of configuration entries (files)
        assert (
            pqa.find_entry_index(fp) == n_expected
        ), f"Index of the calibration entry '{fp}' is determined incorrectly"

    # Test preview function
    for fp in file_paths:
        assert pqa.get_entry_text_preview(fp), f"Failed to generate text preview for the calibration entry '{fp}'"

    # Get the emission line list and verify if it matches the expected list (including the order of the lines)
    eline_list_expected = []
    for settings in pqa.calibration_settings:
        for ln in settings["element_lines"].keys():
            if ln not in eline_list_expected:
                eline_list_expected.append(ln)
    pqa.update_emission_line_list()
    assert (
        pqa.active_emission_lines == eline_list_expected
    ), "Generated emission line list does not match the expected list"

    # -----------------------------------------------
    # Check how management of selections of calibration entries for individual emission lines
    #   Note, that the calibration entries have the common set of emission lines

    # We will use only the first 2 entries. Find the common set of lines
    elist0 = list(pqa.calibration_settings[0]["element_lines"].keys())
    elist1 = list(pqa.calibration_settings[1]["element_lines"].keys())
    elist = list(set(elist0).intersection(set(elist1)))

    def set_selection(pqa, elist, selection):
        for ln, sel in zip(elist, selection):
            el_info = pqa.get_eline_info_complete(ln)
            assert len(el_info) >= 2, f"The emission line {ln} must exist in at least 2 entries"
            el_info[0]["eline_settings"]["selected"] = sel
            el_info[1]["eline_settings"]["selected"] = not sel

    def check_selection(pqa, elist, selection):
        # We check the selection status directly
        #   (in part to verify if 'get_eline_info_complete' works correctly)
        for ln, sel in zip(elist, selection):
            selected = pqa.calibration_settings[0]["element_lines"][ln]["selected"]
            selected1 = pqa.calibration_settings[1]["element_lines"][ln]["selected"]
            assert selected == sel, f"Entry 0: selection state of element line '{ln}' incorrect"
            assert selected1 != sel, f"Entry 1: selection state of element line '{ln}' incorrect"

    # This should be run at least once to select all entries of the 0th set
    pqa.update_emission_line_list()

    # All elines of group #0 must be selected by default (we didn't change selections)
    sel_list = [True] * len(elist)
    check_selection(pqa, elist, sel_list)

    # Select all elements of group #1
    sel_list = [False] * len(elist)
    set_selection(pqa, elist, sel_list)
    pqa.update_emission_line_list()
    check_selection(pqa, elist, sel_list)

    # Alternatively check selections using 'is_eline_selected'
    for eline in elist1:
        assert pqa.is_eline_selected(
            eline, file_paths[0]
        ), f"Emission line '{eline}' in the set '{file_paths[0]}' was not selected"
        if eline in elist:
            assert not pqa.is_eline_selected(
                eline, file_paths[1]
            ), f"Emission line '{eline}' in the set '{file_paths[1]}' was selected"
    for eline in elist0:
        if eline not in elist:
            assert pqa.is_eline_selected(
                eline, file_paths[1]
            ), f"Emission line '{eline}' in the set '{file_paths[1]}' was not selected"

    # Change selection of one emission line (test for 'select_eline' function)
    eline = elist[0]  # 'eline' is present in both sets
    pqa.select_eline(eline, file_paths[1])
    assert pqa.is_eline_selected(eline, file_paths[1]), "Emission line is not selected"
    assert not pqa.is_eline_selected(eline, file_paths[0]), "Emission line is selected"
    pqa.select_eline(eline, file_paths[0])
    assert pqa.is_eline_selected(eline, file_paths[0]), "Emission line is not selected"
    assert not pqa.is_eline_selected(eline, file_paths[1]), "Emission line is selected"

    # Select one element in both entries (it is already selected in entry #1
    pqa.calibration_settings[0]["element_lines"][elist[0]]["selected"] = True
    pqa.update_emission_line_list()
    # Now the element line with index [0] is selected in entry #0, but the rest are selected in entry #1
    sel_list[0] = True
    pqa.update_emission_line_list()
    check_selection(pqa, elist, sel_list)

    # Delete current entry #0 (entries are arranged in the order 'fln1', 'fln0')
    fln1 = pqa.calibration_settings[0]["file_path"]
    pqa.remove_entry(fln1)
    # Check only the entry #0 (all lines must be selected
    for ln in elist:
        assert pqa.calibration_settings[0]["element_lines"][ln][
            "selected"
        ], f"Element line {ln} is not selected in calibration entry #0"

    # Reload the entry (now the entries are arranged in the order 'fln0', 'fln1')
    #   Entry #0 should still remain selected for all emission lines
    pqa.load_entry(fln1)  # Send 'fpath' as arg, not kwarg
    sel_list = [True] * len(elist)
    check_selection(pqa, elist, sel_list)

    # -----------------------------------------------------

    # Check function 'get_eline_info_complete'
    el_info = pqa.get_eline_info_complete(elist[0])
    assert (
        el_info[0]["eline_data"] == pqa.calibration_data[0]["element_lines"][elist[0]]
    ), "Returned calibration data does not match the expected data"
    assert (
        el_info[0]["eline_settings"] == pqa.calibration_settings[0]["element_lines"][elist[0]]
    ), "Returned calibration settings do not match the settings dictionary"
    assert (
        el_info[0]["standard_data"] == pqa.calibration_data[0]
    ), "Returned standard data does not match the expected data"
    assert (
        el_info[0]["standard_settings"] == pqa.calibration_settings[0]
    ), "Returned standard settings does not match the settings dictionary"

    # Check function 'get_eline_calibration'
    el_calib = pqa.get_eline_calibration(elist[0])
    assert (
        el_calib["fluorescence"] == pqa.calibration_data[0]["element_lines"][elist[0]]["fluorescence"]
    ), "The returned selected emission line calibration is incorrect"

    pqa.set_experiment_detector_channel("sum")
    assert pqa.experiment_detector_channel == "sum", "Detector channel is set incorrectly"

    pqa.set_experiment_distance_to_sample(2.0)
    assert pqa.experiment_distance_to_sample == 2.0, "Distance-to-sample is set incorrectly"

    pqa.set_experiment_incident_energy(12.0)
    assert pqa.experiment_incident_energy == 12.0, "Incident energy is set incorrectly"

    # -------------------------------------------------------------------------------
    # Check processing function 'apply_quantitative_normalization'. Most of the possible
    #   combinations of parameters are checked for cases when quantitative normalization
    #   is performed, q.n. is skipped and regular normalization is performed and no
    #   normalization is performed.

    # Select the emission line, which is definitely exists in the dataset
    img_dict = img_list[0]
    eline = elist[0]  # The dataset is 'img_dict[eline]'

    # Create a separate dictionary of scalers
    scaler, scaler2 = "sclr", "sclr2"
    scaler_dict = {scaler: img_dict[scaler], scaler2: img_dict[scaler] * 2}

    data_mean, scaler_mean = np.mean(img_dict[eline]), np.mean(scaler_dict[scaler])

    # Calibration parameters (we use it to predict the output
    el_calib = pqa.get_eline_calibration(elist[0])
    calib_fluor, calib_density = el_calib["fluorescence"], el_calib["density"]

    # Successfully perform quantitative normalization
    data_out, is_applied = pqa.apply_quantitative_normalization(
        img_dict[eline],
        scaler_dict=scaler_dict,
        scaler_name_default=scaler2,
        data_name=eline,
        name_not_scalable=None,
    )
    npt.assert_almost_equal(
        np.mean(data_out),
        data_mean / scaler_mean * calib_density / calib_fluor,
        err_msg="The value of normalized map is different from expected",
    )
    assert is_applied, "Quantitative normalization was not applied"

    # Ignore correction for the distance to sample
    pqa.set_experiment_distance_to_sample(0.0)
    # Successfully perform quantitative normalization
    data_out, is_applied = pqa.apply_quantitative_normalization(
        img_dict[eline],
        scaler_dict=scaler_dict,
        scaler_name_default=scaler2,
        data_name=eline,
        name_not_scalable=None,
    )
    npt.assert_almost_equal(
        np.mean(data_out),
        data_mean / scaler_mean * calib_density / calib_fluor,
        err_msg="The value of normalized map is different from expected",
    )
    assert is_applied, "Quantitative normalization was not applied"

    # Increase distance-to-sample by the factor of 2
    pqa.set_experiment_distance_to_sample(4.0)
    # Successfully perform quantitative normalization
    data_out, is_applied = pqa.apply_quantitative_normalization(
        img_dict[eline],
        scaler_dict=scaler_dict,
        scaler_name_default=scaler2,
        data_name=eline,
        name_not_scalable=None,
    )
    npt.assert_almost_equal(
        np.mean(data_out),
        data_mean / scaler_mean * calib_density / calib_fluor * 4.0,
        err_msg="The value of normalized map is different from expected",
    )
    assert is_applied, "Quantitative normalization was not applied"

    # Skip quantitative normalization (normalize by 'scaler2') because eline 'data_name'
    #   does not have loaded calibration data associated with it
    data_out, is_applied = pqa.apply_quantitative_normalization(
        img_dict[eline],
        scaler_dict=scaler_dict,
        scaler_name_default=scaler2,
        data_name="Cu_K",
        name_not_scalable=None,
    )
    npt.assert_almost_equal(
        np.mean(data_out),
        data_mean / (scaler_mean * 2),
        err_msg="The value of normalized map is different from expected",
    )
    assert not is_applied, "Quantitative normalization was applied"

    # Skip quantitative normalization (normalize by 'scaler2') because scaler is not
    #   in the scaler dictionary
    scaler_dict2 = scaler_dict.copy()  # The new scaler dict contains only scaler 'sclr2'
    del scaler_dict2["sclr"]
    data_out, is_applied = pqa.apply_quantitative_normalization(
        img_dict[eline],
        scaler_dict=scaler_dict2,
        scaler_name_default=scaler2,
        data_name=eline,
        name_not_scalable=None,
    )
    npt.assert_almost_equal(
        np.mean(data_out),
        data_mean / (scaler_mean * 2),
        err_msg="The value of normalized map is different from expected",
    )
    assert not is_applied, "Quantitative normalization was applied"

    # Skip normalization completely because the scaler name is invalid (non-existing scaler)
    data_out, is_applied = pqa.apply_quantitative_normalization(
        img_dict[eline],
        scaler_dict=scaler_dict,
        scaler_name_default="sclr3",
        data_name="Cu_K",  # Non-existing scaler
        name_not_scalable=["sclr2", "sclr"],
    )
    assert data_out is img_dict[eline], "Function output is not a reference to the input data"
    assert not is_applied, "Quantitative normalization was applied"

    # Skip normalization completely because eline 'data_name'
    #   is in the list of non-scalable names
    data_out, is_applied = pqa.apply_quantitative_normalization(
        img_dict[eline],
        scaler_dict=scaler_dict,
        scaler_name_default=scaler2,
        data_name=eline,
        name_not_scalable=["sclr2", eline, "sclr"],
    )
    assert data_out is img_dict[eline], "Function output is not a reference to the input data"
    assert not is_applied, "Quantitative normalization was applied"

    # ----------------------------------------------------------------------
