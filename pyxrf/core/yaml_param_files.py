import os
import re
import sys

import yaml


def _parse_docstring_parameters(doc_string, search_param_section=True):
    r"""
    Parses Google-style docstring and returns the list of (``parameter_name``, ``parameter_description``)
    pairs. The properly composed parameter section of the docstring should start with the line
    containing the word ``    Parameters`` followed by the line containing ``    ----------`` and
    end with the line containing the word ``    Returns`` followed by the line containing ``    -------``.
    The parameter descriptions must start with the line containing parameter name and type:
    ``   parameter_name : type``. Parameter name may contain symbols '_', 'A-Z', 'a-z', '0'-'9' and
    start with a letter or '_'. Every line in parameter description section must be empty or
    start with FOUR spaces. The lines that contain description text may contain additional spaces.

    This function:

    - finds the parameter description section;

    - removes 4 spaces from the beginning of each line;

    - finds parameter names and descriptions for each parameter.

    Parameters
    ----------

    doc_string : str
        doc string as return by ``some_function.__doc__``. Every line of the docstring must be
        indented at least by 4 spaces.

    search_param_section : bool
        tells the function to separate the parameter section of the docstring before parsing it.
        If set ``False``, the function assumes that ``doc_string`` is contains only descriptions
        of parameters (already separated parameter section of the docstring). This option may
        be useful if the ``doc_string`` is not actual function docstring, but just the prepared
        description of parameters to be saved in the YAML file. Such description should not
        contain ``Parameters`` and ``Returns`` section titles. Note, that every line of the
        description must be indented at least by 4 spaces (as in docstrings).

    Returns
    -------

        A list of tuples ``(parameter_name, parameter_description)``:

        - ``parameter_name`` is a string

        - ``parameter_description`` is a list of strings
    """

    str_list = doc_string.split("\n")

    is_py313 = sys.version_info >= (3, 13)
    if is_py313:
        # Add initial 4 spaces to all lines
        str_list = [f"    {_}" for _ in str_list]

    # Remove all spaces at the end of the strings (the should be no spaces there, but still)
    str_list = [s.rstrip() for s in str_list]

    if search_param_section:
        # We are interested only in the part of the docstring that contains description of parameters
        #   Google-style docstrings are expected
        n_first, n_last = None, None
        for n in range(1, len(str_list) - 1):
            if (str_list[n - 1] == "    Parameters") and re.search(r"^    -+$", str_list[n]):
                n_first = n + 1
            if (str_list[n] == "    Returns") and re.search(r"^    -+$", str_list[n + 1]):
                n_last = n - 1
                break
    else:
        # Not search for the parameter section. All the lines of the list should be parsed
        n_first, n_last = 0, len(str_list) - 1

    assert (n_first is not None) or (
        n_last is not None
    ), "Incorrect docstring format: 'Parameters' or 'Returns' statement was not found in the docstring"

    # The list of strings contains parameter descriptions
    str_list = str_list[n_first : n_last + 1]
    # Each line must start with 4 spaces or be empty. Verify this
    assert all(
        [(not s) or re.search(r"^    ", s) for s in str_list]
    ), "Incorrect docstring format: parameter descriptions should be indented by at least FOUR spaces"

    # Now remove the spaces from nonempty lines
    str_list = [s[4:] if s else s for s in str_list]

    param_pos = [n for n, s in enumerate(str_list) if re.search(r"^[_A-Za-z][_A-Za-z0-9]* :", s)]
    n_parameters = len(param_pos)
    assert n_parameters, "Incorrect docstring format: no parameters were found"

    param_pos.append(len(str_list))  # Having the last index will be helpful

    param_names = [re.search("[_A-Za-z][_A-Za-z0-9]*", str_list[param_pos[n]])[0] for n in range(n_parameters)]
    param_descriptions = [str_list[param_pos[n] : param_pos[n + 1]] for n in range(n_parameters)]

    # Remove empty strings from the end of each description (if any)
    for pd in param_descriptions:
        while pd and (not pd[-1]):
            pd.pop(-1)

    # Check if some of the parameter has no descriptions (the number of line must be > 1)
    #   The fist line of the description is actually the
    assert all(
        [len(s) > 1 for s in param_descriptions]
    ), "Incomplete docstring: some parameters have no descriptions"

    params = list(zip(param_names, param_descriptions))

    return params


def _verify_parsed_docstring(parameters, param_dict):
    """
    Verification of consistency of the parameters extracted from docstring and
    dictionary of parameters (probably default parameters). There must be one-to-one
    match between the entries of the list of the parameters extracted from the docstring
    and the parameter dictionary.

    Parameters
    ----------

    parameters : list(tuple)
        The list of ``(parameter_name, parameter_description)`` tuples.
        Descriptions are not analyzed by this function.

    param_dict : dict
        The dictionary of parameters: key - parameter name, value - default value.
        Values are not analyized by this function.

    Returns
    -------

    Raises exception if there is mismatch between the parameter sets.
    """

    pd = param_dict.copy()
    missing_param_list = []
    for p_name, _ in parameters:
        if p_name in pd:
            del pd[p_name]
        else:
            missing_param_list.append(p_name)

    extra_param_list = pd.keys()

    err_msg_list = []
    if missing_param_list:
        err_msg_list.append(f"parameters that are in docstring, but not in the dictionary {missing_param_list}")
    if extra_param_list:
        err_msg_list.append(f"dictionary parameters that are not found in the docstring: {extra_param_list}")

    if err_msg_list:
        err_msg = "Parsed parameter verification failed: parameter set is inconsistent:\n"
        err_msg += ", ".join(err_msg_list)
        assert False, err_msg


_user_instructions_on_editing_yaml = """
--------------------------------------------------
Brief instructions on editing YAML parameter files
--------------------------------------------------

Parameters are presented in the form:

parameter_name: parameter_value

Parameter values may be numbers (integers or floating point numbers), strings, lists and
  dictionaries. STRINGS are presented without quotes:

parameter_name: some_string_value

or

parameter_name: Some string value

LISTS are presented in the form:

parameter_name:
- item1   # Items have the same indentation as 'parameter_name'
- item2
- item3

DICTIONARIES are presented in the form:

parameter_name:
  dict_key1: dict_value1  # Indented by 2 spaces
  dict_key2: dict_value2
  dict_key3:    # The value for this dictionary pair is an array with 3 items
  - list_item1  # Same indentation as 'dict_key3'
  - list_item2
  - list_item3

Some parameters may be set to the value of 'None'. Python value 'None' is represented
as 'null' in YAML files:

parameter_name: null  # The parameter has a value of None
--------------------------------------------------
"""


def create_yaml_parameter_file(
    *,
    file_path,
    function_docstring,
    param_value_dict,
    dir_create=False,
    file_overwrite=False,
    search_param_section=True,
    user_editing_instructions=_user_instructions_on_editing_yaml,
):
    """
    Creates YAML parameter file based on parameter names and descriptions from ``parameters``
    and default values from ``param_value_dict``. The file is supposed to have simple
    human-readable and editable structure, so the descriptions from the docstring are
    inserted above each parameter entry in the YAML file. Values from the ``param_value_dict``
    are used as the default values of the parameters. In addition, a set of instructions for the
    user may be inserted at the beginning of the YAML file (as comments). By default, the
    basic instructions on editing YAML file are inserted.

    Instead of Google-style docstring (which contains Parameters and Returns) statements,
    a string that contains only parameter descriptions may be supplied as ``function_docstring``.
    The parameter descriptions must still be formatted according to Google style and have
    indentation of at least 4 spaces. In this case search of the parameter section should
    be disabled by setting ``search_param_section=False``.

    The function should be used to create YAML file with default parameter values that are later
    modified by users according to their needs.

    Parameters
    ----------

    file_path : str
        absolute or relative path of the new YAML parameter file. The file may have any
        extension. The function may create the nonexisting directory or overwrite existing
        files if ``dir_create`` and/or ``file_overwrite`` are set ``True``.

    function_docstring : str
        doc string as return by ``some_function.__doc__``. Every line of the docstring must be
        indented at least by 4 spaces.

    param_value_dict : dict
        the dictionary that contains (param_name: param_value) pairs. The values of the parameters
        are saved in the YAML file (may be considered default values). ``function_docstring`` and
        ``param_value_dict`` must contain the same parameters (one-to-one match). Exception will
        be raised if there is mismatch between the number or names of parameters.

    dir_create : bool
        if set True, then the directory will be automatically created if it does not exist,
        if set False, then the exception will be raised if the directory does not exist

    file_overwrite : bool
        if set True, then existing parameter file may be overwritten with the new file if it exists,
        otherwise the exception ``IOError`` is be raised if file with the name ``file_path``
        already exists.

    search_param_section : bool
        tells the function to separate the parameter section of the docstring before parsing it.
        If set ``False``, the function assumes that ``doc_string`` is contains only descriptions
        of parameters (already separated parameter section of the docstring). This option may
        be useful if the ``doc_string`` is not actual function docstring, but just the prepared
        description of parameters to be saved in the YAML file. Such description should not
        contain ``Parameters`` and ``Returns`` section titles. Note, that every line of the
        description must be indented at least by 4 spaces (as in docstrings).

    user_editing_instructions : str
        text that is added to the beginning of the YAML file. Typically the text will contain
        the instruction for the user. The default set of instructions contain brief information
        on editing the YAML file. Before the text is added to the YAML file, the "# "
        is added to the beginning of each line, so that the text appears as a comment in
        the YAML file. Set to ``None`` if no instructions need to be added.

    Returns
    -------

    No value is returned

    The function will raise ``IOError`` if the directory in the ``file_path`` does not exist or
    the file already exists, unless the parameters ``dir_create`` and/or ``file_overwrite`` are
    set ``True``. The function will raise ``AssertionError`` if parsing of the docstring is
    not successful or there is a mismatch between parameter set found in docstring and
    the dictionary ``param_value_dict``.
    """

    # Convert to absolute path
    file_path = os.path.expanduser(file_path)
    file_path = os.path.abspath(file_path)

    # Parse docstring (returns the list of param_name/param_description pairs, may raise AssertionError)
    parameters = _parse_docstring_parameters(function_docstring, search_param_section=search_param_section)

    # Check if entries in 'parameters' and 'param_value_dict' match (may raise AssertionError).
    _verify_parsed_docstring(parameters, param_value_dict)

    # Check if file already exists
    file_exists = False
    if not file_overwrite:
        file_exists = os.path.exists(file_path)
    else:
        # We would like to overwrite only files, not directories etc.
        file_exists = os.path.exists(file_path) and not os.path.isfile(file_path)

    if file_exists:
        raise IOError(f"File '{file_path}' already exists")

    # Create the directory if necessary
    dir_path, _ = os.path.split(file_path)
    if not dir_create and not os.path.isdir(dir_path):
        raise IOError(f"Directory '{dir_path}' does not exist")

    # Attempt to create the directory
    os.makedirs(dir_path, exist_ok=True)

    # Create the file output
    s_output = ""
    s_output += (
        "# This file is autogenerated with the default parameters and expected to be modified by the user\n"
    )

    # Insert the user instructions if provided
    if user_editing_instructions:
        # Remove the last '\n'
        if user_editing_instructions[-1] == "\n":
            user_editing_instructions = user_editing_instructions[:-1]
        # Now add "# " to the beginning of each non-empty line,
        #   and replace empty lines with "#"
        list_lines = user_editing_instructions.split("\n")
        list_lines = [f"# {s}" if s else "#" for s in list_lines]
        user_editing_instructions = "\n".join(list_lines)
        s_output += user_editing_instructions + "\n\n"

    for p_name, p_desc in parameters:
        desc = [f"#  {p}" for p in p_desc]
        s = "\n".join(desc)
        s += "\n\n"

        # Print the dictionary entry itself: yaml.dump performs necessary formatting
        d = {p_name: param_value_dict[p_name]}
        s += yaml.dump(d, indent=4)
        s += "\n\n"
        s_output += s

    with open(file_path, "w") as f:
        f.write(s_output)


def read_yaml_parameter_file(*, file_path):
    """
    Reads YAML parameter file. It is assumed that the file is created by the function
    ``create_yaml_parameter_file`` and subsequent editing did not change the overall file
    layout. The function does not check the schema, so any compatible parameter file
    will be loaded as a dictionary.

    Parameters
    ----------

    file_path : str
        relative or absolute path to the YAML file. The exception ``IOError`` will be
        raised if the file does not exist.

    Returns
    -------

    The dictionary that contains (param_name, param_value) pairs
    """

    # Convert to absolute path
    file_path = os.path.expanduser(file_path)
    file_path = os.path.abspath(file_path)

    if not os.path.isfile(file_path):
        raise IOError(f"File '{file_path}' does not exist")

    with open(file_path, "r") as f:
        param_dict = yaml.load(f, Loader=yaml.FullLoader)

    return param_dict
