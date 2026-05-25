import os
import sys

import jsonschema
import numpy as np
import pytest

from pyxrf.core.yaml_param_files import (
    _parse_docstring_parameters,
    _verify_parsed_docstring,
    create_yaml_parameter_file,
    read_yaml_parameter_file,
)


def _generate_parameter_set():
    """
    Generates sample parameter dictionary for testing parameter manipulation functions
    """
    param_dict = {
        "param_none": None,
        "param_int": 265,
        "param_float": 345.834,
        "param_str1": "This_is_string_1",
        "param_str2": "This is string 2",
        "param_list_int": [2, 4, 6, 8, 10],
        "param_list_float": [2.2, 4.4, 6.6, 8.8, 10.01, 345623.453654762342, 3.453456e-15],
        "param_list_string": ["str1", "str2", "str3", "str4", "str5"],
        "param_list_misc": [2, 3.6, "str2", None, 3.45e10],
        "param_dictionary": {"p1": 65, "p2": 3.45, "p3": None, "p4": "some_string", "p5": [1, 2, 3.5]},
    }
    return param_dict


def _generate_parameter_set_schema():
    """
    Generates schema for the parameter set and validates the parameter set
    """

    param_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "param_none",
            "param_int",
            "param_float",
            "param_str1",
            "param_str2",
            "param_list_int",
            "param_list_float",
            "param_list_string",
            "param_list_misc",
            "param_dictionary",
        ],
        "properties": {
            "param_none": {"type": "null"},
            "param_int": {"type": "integer"},
            "param_float": {"type": "number"},
            "param_str1": {"type": "string"},
            "param_str2": {"type": "string"},
            "param_list_int": {"type": "array", "items": {"type": "integer"}},
            "param_list_float": {"type": "array", "items": {"type": "number"}},
            "param_list_string": {"type": "array", "items": {"type": "string"}},
            "param_list_misc": {
                "type": "array",
                "additionalItems": False,
                "uniqueItems": True,
                "prefixItems": [
                    {"type": "integer"},
                    {"type": "number"},
                    {"type": "string"},
                    {"type": "null"},
                    {"type": "number"},
                ],
            },
            "param_dictionary": {
                "type": "object",
                "properties": {
                    "p1": {"type": "integer"},
                    "p2": {"type": "number"},
                    "p3": {"type": "null"},
                    "p4": {"type": "string"},
                    "p5": {"type": "array", "items": {"type": "number"}},
                },
            },
        },
    }

    # Make sure that the schema matches the parameter set
    param_dict = _generate_parameter_set()
    jsonschema.validate(instance=param_dict, schema=param_schema)

    return param_schema


def _generate_sample_docstring(param_dict, include_section_titles=True):
    """
    Generates sample docstring based on the supplied parameter dictionary,
    returns docstring and the list of parameter/description pairs.
    For testing of ``_parse_docstring_parameters`` function.
    If ``include_section_titles`` is ``False``, then the titles
    ``Parameters`` and ``Returns`` are not included in the output string,
    i.e. the string contains only parameter descriptions.
    """
    parameters = []
    for p_name in param_dict.keys():
        # Create tuple (param_name, param_description), param_description - array of strings,
        #   each string will be printed in the separate line
        p = (
            f"{p_name}",
            [
                f"{p_name} : {type(param_dict[p_name])}",
                f"Description of parameter {p_name}",
                "",
                f"The end of the description of {p_name}",
            ],
        )
        parameters.append(p)

    n_empty_lines_before, n_empty_lines_after = 5, 5
    d_str = [""] * n_empty_lines_before

    if include_section_titles:
        d_str.append("    Parameters")
        d_str.append("    ----------")

    for p in parameters:
        # Indentation by 4 spaces
        st = [f"    {s}" if s else s for s in p[1]]
        s = "\n".join(st)
        s += "\n" * np.random.choice(4)  # Insert 0 .. 3 empty strings
        d_str.append(s)

    if include_section_titles:
        d_str.append("    Returns")
        d_str.append("    -------")

    d_str.extend([""] * n_empty_lines_after)
    d_str = "\n".join(d_str)  # Convert the list to a single string

    # Remove initial 4 spaces from all lines to mimick behavior of func.__doc__ in Python 3.13
    # and later (in Python 3.13, the initial spaces are removed automatically,
    # but in earlier versions they are not)
    is_py313 = sys.version_info >= (3, 13)
    if is_py313:
        d_str = d_str.split("\n")
        d_str = [_[4:] if len(_) > 4 else "" for _ in d_str]
        d_str = "\n".join(d_str)

    return d_str, parameters


def test_parse_docstring_parameters_01():
    # Simple test for the successfully parsed docstring. It seems sufficient, since all error cases are trivial.

    param_dict = _generate_parameter_set()
    d_str, parameters = _generate_sample_docstring(param_dict)
    parameters_output = _parse_docstring_parameters(d_str)
    assert parameters == parameters_output, "Parsed parameters or parameter descriptions are invalid"

    # Basically repeats the same test, but enables section titles explicitely
    param_dict = _generate_parameter_set()
    d_str, parameters = _generate_sample_docstring(param_dict, include_section_titles=True)
    parameters_output = _parse_docstring_parameters(d_str, search_param_section=True)
    assert parameters == parameters_output, "Parsed parameters or parameter descriptions are invalid"

    # Disable section titles and see if parsing still works
    param_dict = _generate_parameter_set()
    d_str, parameters = _generate_sample_docstring(param_dict, include_section_titles=False)
    parameters_output = _parse_docstring_parameters(d_str, search_param_section=False)
    assert parameters == parameters_output, "Parsed parameters or parameter descriptions are invalid"

    # Check for exception if the section titles are required, but don't exist
    param_dict = _generate_parameter_set()
    d_str, parameters = _generate_sample_docstring(param_dict, include_section_titles=False)
    with pytest.raises(AssertionError, match="'Parameters' or 'Returns' statement was not found in the docstring"):
        _parse_docstring_parameters(d_str, search_param_section=True)


def test_verify_parsed_docstring():
    # Generate the set of parameters (we don't use docstring in this test)
    param_dict = _generate_parameter_set()
    _, parameters = _generate_sample_docstring(param_dict)

    # This verification should be successful
    parameters_copy = parameters.copy()
    param_dict_copy = param_dict.copy()
    _verify_parsed_docstring(parameters, param_dict)  # This may raise an exception
    assert parameters == parameters_copy, "'parameters' was unintentionally changed by the function"
    assert param_dict == param_dict_copy, "'param_dict' was unintentionally changed by the function"

    # This test should fail (2 extra parameters)
    param_dict2 = param_dict.copy()
    param_dict2["extra_parameter1"] = 0
    param_dict2["extra_parameter2"] = 0

    parameters_copy = parameters.copy()
    param_dict_copy = param_dict2.copy()
    with pytest.raises(AssertionError, match="not found in the docstring.+extra_parameter1.+extra_parameter2"):
        _verify_parsed_docstring(parameters, param_dict2)
    assert parameters == parameters_copy, "'parameters' was unintentionally changed by the function"
    assert param_dict2 == param_dict_copy, "'param_dict' was unintentionally changed by the function"

    # This test should fail (1 parameter is removed)
    param_dict2 = param_dict.copy()
    # Select random key for removal
    key_to_remove = list(param_dict2.keys())[np.random.choice(len(param_dict2))]
    del param_dict2[key_to_remove]

    parameters_copy = parameters.copy()
    param_dict_copy = param_dict2.copy()
    with pytest.raises(AssertionError, match=f"not in the dictionary.+{key_to_remove}"):
        _verify_parsed_docstring(parameters, param_dict2)
    assert parameters == parameters_copy, "'parameters' was unintentionally changed by the function"
    assert param_dict2 == param_dict_copy, "'param_dict' was unintentionally changed by the function"


def test_create_read_yaml_parameter_file(tmp_path):
    # Some directory
    yaml_dirs = ["yaml", "file", "dirs"]
    yaml_fln = "parameter.yaml"
    file_path = os.path.join(tmp_path, *yaml_dirs, yaml_fln)

    # Generate the set of parameters (we don't use docstring in this test)
    param_dict = _generate_parameter_set()
    doc_string, parameters = _generate_sample_docstring(param_dict)

    create_yaml_parameter_file(
        file_path=file_path, function_docstring=doc_string, param_value_dict=param_dict, dir_create=True
    )

    param_dict_recovered = read_yaml_parameter_file(file_path=file_path)

    # Validate the schema of the recovered data (this will be part of the procedure of reading real data)
    param_schema = _generate_parameter_set_schema()
    jsonschema.validate(instance=param_dict_recovered, schema=param_schema)

    assert (
        param_dict == param_dict_recovered
    ), "Parameter dictionary read from YAML file is different from the original parameter dictionary"

    # Test: creating parameter file without instructions (and read it)
    yaml_fln2 = "parameter2.yaml"
    file_path2 = os.path.join(tmp_path, *yaml_dirs, yaml_fln2)
    create_yaml_parameter_file(
        file_path=file_path2,
        function_docstring=doc_string,
        param_value_dict=param_dict,
        dir_create=False,
        user_editing_instructions=False,
    )
    param_dict_recovered2 = read_yaml_parameter_file(file_path=file_path2)
    jsonschema.validate(instance=param_dict_recovered2, schema=param_schema)
    assert (
        param_dict == param_dict_recovered2
    ), "Parameter dictionary read from YAML file is different from the original parameter dictionary"

    # Test: creating parameter file with custom instructions (and read it)
    yaml_fln3 = "parameter3.yaml"
    file_path3 = os.path.join(tmp_path, *yaml_dirs, yaml_fln3)
    instructions = "    Those are custom instructions\n        in the correct format"
    create_yaml_parameter_file(
        file_path=file_path3,
        function_docstring=doc_string,
        param_value_dict=param_dict,
        dir_create=False,
        user_editing_instructions=instructions,
    )
    param_dict_recovered3 = read_yaml_parameter_file(file_path=file_path3)
    jsonschema.validate(instance=param_dict_recovered3, schema=param_schema)
    assert (
        param_dict == param_dict_recovered3
    ), "Parameter dictionary read from YAML file is different from the original parameter dictionary"


# fmt: off
@pytest.mark.parametrize("test_params", [
    # Writing to existing directory (absolute and relative paths)
    {"path_create": ["yaml", "file", "dirs"], "path_current": ["yaml", "file"],
     "path_write": ["."], "path_read": ["."], "dir_create": False,
     "succeed_write": True, "succeed_read": True},
    {"path_create": ["yaml", "file", "dirs"], "path_current": ["yaml", "file"],
     "path_write": [".", "dirs"], "path_read": ["yaml", "file", "dirs"],
     "dir_create": False, "succeed_write": True, "succeed_read": True},
    {"path_create": ["yaml", "file", "dirs"], "path_current": ["yaml", "file"],
     "path_write": ["yaml", "file", "dirs"], "path_read": [".", "dirs"],
     "dir_create": False, "succeed_write": True, "succeed_read": True},
    {"path_create": ["yaml", "file", "dirs"], "path_current": ["yaml", "file"],
     "path_write": [".."], "path_read": [".."], "dir_create": False,
     "succeed_write": True, "succeed_read": True},
    {"path_create": ["yaml", "file", "dirs"], "path_current": ["yaml", "file"],
     "path_write": [".."], "path_read": ["yaml"], "dir_create": False,
     "succeed_write": True, "succeed_read": True},
    {"path_create": ["yaml", "file", "dirs"], "path_current": ["yaml", "file"],
     "path_write": ["yaml"], "path_read": [".."], "dir_create": False,
     "succeed_write": True, "succeed_read": True},

    # Writing to non-existing directory
    {"path_create": ["yaml", "file"], "path_current": ["yaml", "file"],
     "path_write": [".", "dirs"], "path_read": [".", "dirs"], "dir_create": False,
     "succeed_write": False, "succeed_read": False},
    {"path_create": ["yaml", "file"], "path_current": ["yaml", "file"],
     "path_write": [".", "dirs"], "path_read": [".", "dirs"], "dir_create": True,
     "succeed_write": True, "succeed_read": True},

    # Reading non-existing file
    {"path_create": ["yaml", "file"], "path_current": ["yaml", "file"],
     "path_write": ["yaml"], "path_read": ["yaml", "file"], "dir_create": False,
     "succeed_write": True, "succeed_read": False},
])
# fmt: on
def test_create_yaml_parameter_file1(tmp_path, test_params):
    """Test functions ``create_yaml_parameter_file`` and ``read_yaml_parameter_file``
    for different combinations of input/output directories"""

    path_create = test_params["path_create"]
    path_current = test_params["path_current"]
    path_write = test_params["path_write"]
    path_read = test_params["path_read"]
    dir_create = test_params["dir_create"]
    succeed_write = test_params["succeed_write"]
    succeed_read = test_params["succeed_read"]

    def assemble_path(tmp_path, components):
        path = tmp_path
        if components:
            if components[0] in (".", ".."):
                # Relative path
                if components[0] == ".":
                    components.pop(0)
                if components:
                    path = os.path.join(*components)
                else:
                    path = ""
            else:
                # Absolute path
                components = [tmp_path] + components
                path = os.path.join(*components)
        return path

    path_create = assemble_path(tmp_path, path_create)
    path_current = assemble_path(tmp_path, path_current)
    path_write = assemble_path(tmp_path, path_write)
    path_read = assemble_path(tmp_path, path_read)

    # Generate the set of parameters (we don't use docstring in this test)
    param_dict = _generate_parameter_set()
    doc_string, parameters = _generate_sample_docstring(param_dict)

    os.makedirs(path_create, exist_ok=True)
    os.chdir(path_current)

    yaml_fln = "parameters.yaml"
    if succeed_write:
        create_yaml_parameter_file(
            file_path=os.path.join(path_write, yaml_fln),
            function_docstring=doc_string,
            param_value_dict=param_dict,
            dir_create=dir_create,
        )
    else:
        with pytest.raises(IOError):
            create_yaml_parameter_file(
                file_path=os.path.join(path_write, yaml_fln),
                function_docstring=doc_string,
                param_value_dict=param_dict,
                dir_create=dir_create,
            )

    if succeed_read:
        param_dict_recovered = read_yaml_parameter_file(file_path=os.path.join(path_read, yaml_fln))
    else:
        with pytest.raises(IOError):
            param_dict_recovered = read_yaml_parameter_file(file_path=os.path.join(path_read, yaml_fln))

    if succeed_write and succeed_read:
        assert (
            param_dict == param_dict_recovered
        ), "Parameter dictionary read from YAML file is different from the original parameter dictionary"


def test_create_yaml_parameter_file2(tmp_path):
    """Test if file overwriting protection works"""

    fln = os.path.join(tmp_path, "parameters.yaml")

    # Generate the set of parameters (we don't use docstring in this test)
    param_dict = _generate_parameter_set()
    doc_string, parameters = _generate_sample_docstring(param_dict)

    # Write the file the first time
    create_yaml_parameter_file(file_path=fln, function_docstring=doc_string, param_value_dict=param_dict)
    # Try to write it again
    with pytest.raises(IOError):
        create_yaml_parameter_file(file_path=fln, function_docstring=doc_string, param_value_dict=param_dict)
    # Try again, but allow overwriting the file
    create_yaml_parameter_file(
        file_path=fln, function_docstring=doc_string, param_value_dict=param_dict, file_overwrite=True
    )
