import os
import yaml
import jsonschema
import numpy as np
import math
import json
import copy
from .xrf_utils import split_compound_mass, generate_eline_list
from ..model.utils import normalize_data_by_scaler
import logging
logger = logging.getLogger()

# ==========================================================================================
#    Functions for operations with YAML files used for keeping descriptions of XRF standards

_xrf_standard_schema = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "serial", "description", "compounds"],
    "properties": {
        "name": {"type": "string"},
        "serial": {"type": "string"},
        "description": {"type": "string"},
        "compounds": {"type": "object",
                      # Chemical formula should always start with a captial letter (Fe2O3)
                      "patternProperties": {"^[A-Z][A-Za-z0-9]*$": {"type": "number"}},
                      "additionalProperties": False,
                      "minProperties": 1
                      },
        "density": {"type": "number"}  # Total density is an optional parameter
    }
}

_xrf_standard_schema_instructions = """
# The file was automatically generated.
#
# Instructions for editing this file:
#
# Description of each standard starts with '- name: ...'. Every following line
#   must be indented by 4 spaces. Each description contains the following items:
#   'name' (name of the standard, arbitrary string), 'serial' (serial number, of
#   the standard, but can be arbitrary string, 'description' (string that contains
#   description of the standard). Those fields may be filled with arbitrary information,
#   best suited to distinguish the standard later. If string consists of only digits
#   (in case of serial number) it must be enclosed in quotes.
#
# The field 'compounds' lists all compounds in the standard. The compounds are
#   presented in the form <compound_formula>: <concentration>.
#   <compound_formula> has to be a valid chemical formula, representing a pure
#   element (C, Fe, Ga, etc.) or compound (Fe2O3, GaAs, etc). Element names
#   must start with a capital letter followed by a lowercase letter (if present).
#   No characters except 'A-Z', 'a-z' and '0-1' are allowed. Lines containing
#   compound specifications must be indented by extra 4 spaces.
#
# The optional field 'density' specifies total density of the sample and used
#   to check integrity of the data (the sum of densities of all compounds
#   must be equaly to 'density' value.
#
# All density values (for compounds and total density) are specified in ug/cm^2
#
# Example (the lines contain extra '#' character, which is not part of YAML file):
#
#-   name: Micromatter 41164
#    serial: '41164'
#    description: CeF3 21.1 / Au 20.6
#    compounds:
#        CeF3: 21.1
#        Au: 20.6
#    density: 41.7
#
# The easiest way to start creating the list of custom standards is to uncomment
#   and edit the following example. To create extra records, duplicate and
#   edit the example or any existing record.

#-    name: Name of the Standard
#     serial: '32654'
#     description: CeF3 21.1 / Au 20.6 (any convenient description)
#     compounds:
#         CeF3: 21.1
#         Au: 20.6

"""


def save_xrf_standard_yaml_file(file_path, standard_data, *, overwrite_existing=False):
    r"""
    Save descriptions of of XRF standards to YAML file

    Parameters
    ----------

    file_path: str
        absolute or relative path to the saved YAML file. If the path does not exist, then
        it is created.

    standard_data: list(dict)
        list of dictionaries, each dictionary is representing the description of one
        XRF standard. Sending ``[]`` will create YAML file, which contains only instructions
        for manual editing of records. Such file can be read by the function
        ``load_xrf_standard_yaml_file``, which returns ``[]``.

    overwrite_existing: bool
        indicates if existing file should be overwritten. Default is False, since
        overwriting of an existing parameter file will lead to loss of data.

    Returns
    -------

        no value is returned

    Raises
    ------

    IOError if the YAML file already exists and ``overwrite_existing`` is not enabled.
    """

    # Make sure that the directory exists
    file_path = os.path.expanduser(file_path)
    file_path = os.path.abspath(file_path)
    flp, _ = os.path.split(file_path)
    os.makedirs(flp, exist_ok=True)

    if not overwrite_existing and os.path.isfile(file_path):
        raise IOError(f"File '{file_path}' already exists")

    s_output = _xrf_standard_schema_instructions
    if standard_data:
        s_output += yaml.dump(standard_data, default_flow_style=False, sort_keys=False, indent=4)
    with open(file_path, "w") as f:
        f.write(s_output)


def load_xrf_standard_yaml_file(file_path, *, schema=_xrf_standard_schema):
    r"""
    Load the list of XRF standard descriptions from YAML file and verify the schema.

    Parameters
    ----------

    file_path: str
        absolute or relative path to YAML file. If file does not exist then IOError is raised.

    schema: dict
        reference to schema used for validation of the descriptions. If ``schema`` is ``None``,
        then validation is disabled (this is not the default behavior).

    Returns
    -------

        list of dictionaries, each dictionary is representing the description of one XRF
        standard samples. Empty dictionary is returned if the file contains no data.

    Raises
    ------

    IOError is raised if the YAML file does not exist.

    jsonschema.ValidationError is raised if schema validation fails.

    RuntimeError if the sum of areal densities of all compounds does not match the
    total density of the sample for at least one sample. The list of all sample
    records for which the data integrity is not confirmed is returned in the
    error message. For records that do not contain 'density' field the integrity
    check is not performed.
    """

    file_path = os.path.expanduser(file_path)
    file_path = os.path.abspath(file_path)

    if not os.path.isfile(file_path):
        raise IOError(f"File '{file_path}' does not exist")

    with open(file_path, 'r') as f:
        standard_data = yaml.load(f, Loader=yaml.FullLoader)

    if standard_data is None:
        standard_data = []

    if schema is not None:
        for data in standard_data:
            jsonschema.validate(instance=data, schema=schema)

    # Now check if all densities of compounds sums to total density in every record
    msg = []
    for data in standard_data:
        if "density" in data:
            # The sum of all densities must be equal to total density
            sm = np.sum(list(data["compounds"].values()))
            if not math.isclose(sm, data["density"], abs_tol=1e-6):
                msg.append(f"Record #{data['serial']} ({data['name']}): "
                           f"computed {sm} vs total {data['density']}")
    if msg:
        msg = [f"    {_}" for _ in msg]
        msg = '\n'.join(msg)
        msg = "Sum of areal densities does not match total density:\n" + msg
        raise RuntimeError(msg)

    return standard_data


def load_included_xrf_standard_yaml_file():
    r"""
    Load YAML file with descriptions of XRF standards that is part of the
    package.

    Returns
    -------

    List of dictionaries, each dictionary represents description of one XRF standard.

    Raises
    ------

    Exceptions may be raised by ``load_xrf_standard_yaml_file`` function
    """

    # Generate file name (assuming that YAML file is in the same directory)
    file_name = "xrf_quant_standards.yaml"
    file_path = os.path.realpath(__file__)
    file_path, _ = os.path.split(file_path)
    file_path = os.path.join(file_path, file_name)

    return load_xrf_standard_yaml_file(file_path)


def compute_standard_element_densities(compounds):
    r"""
    Computes areal density of each element in the mix of compounds.
    Some compounds in the mix may contain the same elements.

    Parameters
    ----------

    compounds: dict

        dictionary of compound densities: key - compound formula,
        value - density (typically ug/cm^2)

    Returns
    -------

    Dictionary of element densities: key - element name (symbolic),
    value - elmenet density.
    """

    element_densities = {}

    for key, value in compounds.items():
        el_dens = split_compound_mass(key, value)
        for el, dens in el_dens.items():
            if el in element_densities:
                element_densities[el] += dens
            else:
                element_densities[el] = dens

    return element_densities


# ==========================================================================================
#    Functions for operations with JSON files used for keeping quantitative data obtained
#      after processing of XRF standard samples. The data is saved after processing
#      XRF scan of standard samples and later used for quantitative analysis of
#      experimental samples.

_xrf_quant_fluor_schema = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "serial", "description", "element_lines",
                 "incident_energy", "scaler_name", "distance_to_sample"],
    "properties": {
        # 'name', 'serial' and 'description' (optional) are copied
        #   from the structure used for description of XRF standard samples
        "name": {"type": "string"},
        "serial": {"type": "string"},
        "description": {"type": "string"},
        # The list of element lines. The list is not expected to be comprehensive:
        #   it includes only the lines selected for processing of standard samples.
        "element_lines": {
            "type": "object",
            "additionalProperties": False,
            "minProperties": 1,
            # Symbolic expression representing an element line:
            # Fe - represents all lines, Fe_K - K-lines, Fe_Ka - K alpha lines,
            # Fe_Ka1 - K alpha 1 line. Currently only selections that contain
            # all K, L or M lines is supported.
            "patternProperties": {
                r"^[A-Z][a-z]?(_[KLM]([ab]\d?)?)?$": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["density", "fluorescence"],
                    "properties": {
                        "density": {"type": "number"},
                        "fluorescence": {"type": ["number", "null"]}
                    }
                }
            },
        },
        # Incident energy used in the processing experiment
        "incident_energy": {"type": "number"},
        # Selected channel, expected values are 'sum', 'det1', 'det2', 'det3' etc.
        "detector_channel": {"type": ["string", "null"]},
        # Name of the valid scaler name (specific for data recorded on the beamline
        "scaler_name": {"type": ["string", "null"]},
        # Distance to the sample (number or null)
        "distance_to_sample": {"type": ["number", "null"]},
    }
}


def save_xrf_quant_fluor_json_file(file_path, fluor_data, *, overwrite_existing=False):
    r"""
    Save the results of processing of a scan data for XRF standard sample to a JSON file.
    The saved data will be used later for quantitative analysis of experimental samples.

    Parameters
    ----------

    file_path: str
        absolute or relative path to the saved JSON file. If the path does not exist, then
        it is created.

    fluor_data: dict
        dictionary, which contains the results of processing of a scan of an XRF standard.
        The dictionary should conform to ``_xrf_quantitative_fluorescence_schema``.
        The schema is verified before saving to ensure that the data can be successfully read.

    overwrite_existing: bool
        indicates if existing file should be overwritten. Default is False, since
        overwriting of an existing parameter file will lead to loss of data.

    Returns
    -------

        no value is returned

    Raises
    ------

    IOError if the JSON file already exists and ``overwrite_existing`` is not enabled.

    jsonschema.ValidationError if schema validation fails
    """

    # Note: the schema is fixed (not passed as a parameter). If data format is changed,
    #   then the built-in schema must be changed. The same schema is always used
    #   both for reading and writing of data.
    jsonschema.validate(instance=fluor_data, schema=_xrf_quant_fluor_schema)

    # Make sure that the directory exists
    file_path = os.path.expanduser(file_path)
    file_path = os.path.abspath(file_path)
    flp, _ = os.path.split(file_path)
    os.makedirs(flp, exist_ok=True)

    if not overwrite_existing and os.path.isfile(file_path):
        raise IOError(f"File '{file_path}' already exists")

    s_output = json.dumps(fluor_data, sort_keys=False, indent=4)
    with open(file_path, "w") as f:
        f.write(s_output)


def load_xrf_quant_fluor_json_file(file_path, *, schema=_xrf_quant_fluor_schema):
    r"""
    Load the quantitative data for XRF standard sample from JSON file and verify the schema.

    Parameters
    ----------

    file_path: str
        absolute or relative path to JSON file. If file does not exist then IOError is raised.

    schema: dict
        reference to schema used for validation of the descriptions. If ``schema`` is ``None``,
        then validation is disabled (this is not the default behavior).

    Returns
    -------

        dictionary containing quantitative fluorescence data on XRF sample.

    Raises
    ------

    IOError is raised if the YAML file does not exist.

    jsonschema.ValidationError is raised if schema validation fails.
    """

    file_path = os.path.expanduser(file_path)
    file_path = os.path.abspath(file_path)

    if not os.path.isfile(file_path):
        raise IOError(f"File '{file_path}' does not exist")

    with open(file_path, 'r') as f:
        fluor_data = json.load(f)

    if schema is not None:
        jsonschema.validate(instance=fluor_data, schema=schema)

    return fluor_data


def get_quant_fluor_data_dict(quant_param_dict, incident_energy):
    # TODO: Documentation and tests

    quant_fluor_data_dict = {}
    quant_fluor_data_dict["name"] = quant_param_dict["name"]
    quant_fluor_data_dict["serial"] = quant_param_dict["serial"]
    quant_fluor_data_dict["description"] = quant_param_dict["description"]

    element_dict = {}
    for compound, mass in quant_param_dict["compounds"].items():
        el_and_mass = split_compound_mass(compound, mass)
        for el, ms in el_and_mass.items():
            if el in element_dict:
                element_dict[el] += ms
            else:
                element_dict[el] = ms

    element_lines = {}
    for el, ms in element_dict.items():
        lines = generate_eline_list([el], incident_energy=incident_energy)
        e = {_: {"density": ms, "fluorescence": None} for _ in lines}
        element_lines.update(e)

    quant_fluor_data_dict["element_lines"] = element_lines

    quant_fluor_data_dict["incident_energy"] = incident_energy

    quant_fluor_data_dict["detector_channel"] = None
    quant_fluor_data_dict["scaler_name"] = None
    quant_fluor_data_dict["distance_to_sample"] = None

    return quant_fluor_data_dict


def fill_quant_fluor_data_dict(quant_fluor_data_dict, *, xrf_map_dict, scaler_name):
    # TODO: Documentation and tests

    if not scaler_name:
        logger.warning(f"No scaler is selected for computing quantitative coefficients. Data is not normalized.")
    elif scaler_name not in xrf_map_dict:
        logger.warning(f"Scaler '{scaler_name}' is not in XRF map dictionary. Normalization can not be performed.")
        scaler_name = None

    # Clear ALL fluorescence values. Don't touch any other data
    for eline, info in quant_fluor_data_dict["element_lines"].items():
        info["fluorescence"] = None

    # Save the scaler name
    quant_fluor_data_dict["scaler_name"] = scaler_name

    # Compute fluorescence of the emission lines
    eline_list = tuple(quant_fluor_data_dict["element_lines"].keys())
    for eline, map in xrf_map_dict.items():
        if eline in eline_list:
            # Normalize the map if scaler is selected. (Typically scaler IS selected.)
            if scaler_name:
                norm_map = normalize_data_by_scaler(xrf_map_dict[eline], xrf_map_dict[scaler_name])
            else:
                norm_map = xrf_map_dict[eline]
            mean_fluor = np.mean(norm_map)
            # Note: numpy 'float64' is explicitely converted to 'float'
            #     (yaml package does not seem to support 'float64')
            quant_fluor_data_dict["element_lines"][eline]["fluorescence"] = float(mean_fluor)


def prune_quant_fluor_data_dict(quant_fluor_data_dict):
    r"""
    Prunes the fluorescence data dictionary by removing the element lines that are not
    present (fluorescence is None) or have fluorescence <= 0. 'Pruning' is performed before
    saving calibration data, so that only meaningful information is saved.
    The function does not modify the original data structure. Instead it returns the
    copy of the original dictionary with non-existent emission line removed.
    """
    # TODO: Documentation and tests

    quant_fluor_data_dict = copy.deepcopy(quant_fluor_data_dict)
    for key, val in quant_fluor_data_dict["element_lines"].copy().items():
        if (val["fluorescence"] is None) or (val["fluorescence"] <= 0):
            del quant_fluor_data_dict["element_lines"][key]

    return quant_fluor_data_dict

# -------------------------------------------------------------------------------------------------


class ParamQuantEstimation:
    # TODO: Documentation and tests

    def __init__(self):

        custom_path = ("~", ".pyxrf", "quantitative_standards.yaml")
        self.custom_standards_file_path = os.path.join(*custom_path)

        # If file with custom set of standards does not exist, create one
        if not os.path.isfile(self.custom_standards_file_path):
            try:
                save_xrf_standard_yaml_file(self.custom_standards_file_path, [])
            except Exception as ex:
                logger.error(f"Failed to create empty file for custom set of quantitative standards: {ex}")

        # Lists of standard descriptions
        self.standards_built_in = None
        self.standards_custom = None

        # Reference to the selected standard description (in custom or built-in list)
        self.standard_selected = None

        # List of emission lines for the selected incident energy
        self.incident_energy = 0.0
        self.emission_line_list = None

        # Dictionary with fluorescence data for the selected standard. Filled as data is processed
        self.fluorescence_data_dict = None

    def load_standards(self):

        try:
            self.standards_built_in = load_included_xrf_standard_yaml_file()
        except Exception as ex:
            logger.error(f"Failed to load built-in set of quantitative standards: {ex}")
        try:
            self.standards_custom = load_xrf_standard_yaml_file(self.custom_standards_file_path)
        except Exception as ex:
            logger.error(f"Failed to load custom set of quantitative standards: {ex}")

    def clear_standards(self):

        self.standards_built_in = None
        self.standards_custom = None
        self.standard_selected = None
        self.emission_line_list = None

    def _find_standard_custom(self, standard):

        standard_ref = None
        if self.standards_custom:
            for st in self.standards_custom:
                if st == standard:
                    standard_ref = st
                    break
        return standard_ref

    def _find_standard_built_in(self, standard):

        standard_ref = None
        if self.standards_built_in:
            for st in self.standards_built_in:
                if st == standard:
                    standard_ref = st
                    break
        return standard_ref

    def find_standard(self, standard):

        standard_ref = self._find_standard_custom(standard)
        if not standard_ref:
            standard_ref = self._find_standard_built_in(standard)
        return standard_ref

    def set_selected_standard(self, standard):

        standard_ref = self.find_standard(standard)

        if not standard_ref:
            # Set reference pointing to the first available standard description
            if self.standards_custom:
                self.standard_selected = self.standards_custom[0]
            elif self.standards_built_in:
                self.standard_selected = self.standards_built_in[0]
            else:
                self.standard_selected = None

        else:
            # The reference was found in one of the arrays
            self.standard_selected = standard_ref

        return self.standard_selected

    def is_standard_custom(self, standard):

        return bool(self._find_standard_custom(standard))

    def gen_fluorescence_data_dict(self, incident_energy):

        if incident_energy:
            self.incident_energy = incident_energy
        np.clip(self.incident_energy, a_min=0.0, a_max=None)

        if incident_energy == 0.0:
            logger.warning("Attempting to compute the list of emission lines with incident energy set to 0")

        self.fluorescence_data_dict = get_quant_fluor_data_dict(self.standard_selected, incident_energy)

    def fill_fluorescence_data_dict(self, *, xrf_map_dict, scaler_name):

        fill_quant_fluor_data_dict(self.fluorescence_data_dict,
                                   xrf_map_dict=xrf_map_dict,
                                   scaler_name=scaler_name)

    def set_detector_channel_in_data_dict(self, *, detector_channel=None):
        self.fluorescence_data_dict["detector_channel"] = detector_channel

    def set_distance_to_sample_in_data_dict(self, *, distance_to_sample=None):
        self.fluorescence_data_dict["distance_to_sample"] = distance_to_sample

    def get_suggested_json_fln(self):
        r"""Requires that the fluorescence data dict is filled"""
        fln = f"standard_{self.fluorescence_data_dict['serial']}.json"
        return fln

    def get_fluorescence_data_dict_text_preview(self, distance_to_sample=None,
                                                enable_warnings=True):

        pruned_dict = prune_quant_fluor_data_dict(self.fluorescence_data_dict)
        # This will not modify the original dictionary
        pruned_dict["distance_to_sample"] = distance_to_sample
        # Print preview in YAML format (easier to read)
        s = yaml.dump(pruned_dict, default_flow_style=False, sort_keys=False, indent=4)
        if enable_warnings:
            if (pruned_dict["scaler_name"] is None) or (pruned_dict["scaler_name"] == ""):
                s = "WARNING: Scaler is not selected, data is not normalized.\n\n" + s
        return s

    def save_fluorescence_data_dict(self, file_path, *, overwrite_existing=False):
        pruned_dict = prune_quant_fluor_data_dict(self.fluorescence_data_dict)
        save_xrf_quant_fluor_json_file(file_path, pruned_dict, overwrite_existing=overwrite_existing)


class ParamQuantitativeAnalysis:
    # TODO: documentation and tests

    def __init__(self):

        # List of opened calibration standards
        self.calibration_data = []
        self.calibration_settings = []
        self.active_emission_lines = []

    def load_calibration_data(self, file_path):

        # Do not load duplicates (distinguished by file name). Different file names may contain
        #   the same data, but this is totally up to the user. Loading duplicates should not
        #   disrupt the processing, since the user may choose the source of calibration data
        #   for each emission line.
        if not any(file_path == _["file_path"] for _ in self.calibration_settings):
            # May raise exception if not successful
            data_new = load_xrf_quant_fluor_json_file(file_path)

            self.calibration_data.append(data_new)

            settings = {}
            settings['file_path'] = file_path
            settings['element_lines'] = {}
            for l in data_new['element_lines']:
                settings["element_lines"][l] = {}
                # Do not select the emission line
                settings["element_lines"][l]["selected"] = False

            self.calibration_settings.append(settings)
            # This will also select the emission line if it occurs the first time
            self.gen_emission_line_list()
        else:
            logger.info(f"Calibration data file '{file_path}' is already loaded")

    def remove_calibration_data(self, file_path):
        n_item = self.find_calibration_data(file_path)
        if n_item is not None:
            self.calibration_data.pop(n_item)
            self.calibration_settings.pop(n_item)
            self.gen_emission_line_list()
        else:
            raise RuntimeError(f"Calibration data from source '{file_path}' is not found.")

    def find_calibration_data(self, file_path):
        n_item = None
        for n, v in enumerate(self.calibration_settings):
            if v["file_path"] == file_path:
                n_item = n
                break
        return n_item

    def get_calibration_data_text_preview(self, file_path):
        n_item = self.find_calibration_data(file_path)
        if n_item is not None:
            s = yaml.dump(self.calibration_data[n_item], default_flow_style=False,
                          sort_keys=False, indent=4)
        else:
            s = ""
        return s

    def gen_emission_line_list(self):
        # The emission lines are arrange in the order in which they appear
        #   Changing the order of the list will change the order in which
        #   the lines are presented in the preview tab
        elines_all = set()
        self.active_emission_lines = []  # Clear the list, it is recreated
        for data in self.calibration_data:
            for l in data["element_lines"].keys():
                if l not in elines_all:
                    elines_all.add(l)
                    self.active_emission_lines.append(l)

        # Make sure that emission line is selected only once
        elines_selected = set()  # We make sure that each emission line is selected
        for settings in self.calibration_settings:
            for k, v in settings["element_lines"].items():
                if v["selected"]:
                    if k in elines_selected:
                        # If eline is in the set, then deselect it (it is selected twice)
                        v["selected"] = False
                    else:
                        # Add to the list
                        elines_selected.add(k)

        elines_not_selected = elines_all - elines_selected

        if elines_not_selected:
            for settings in self.calibration_settings:
                for k, v in settings["element_lines"].items():
                    if k in elines_not_selected:
                        v["selected"] = True
                        elines_not_selected.remove(k)
