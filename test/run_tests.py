import unittest
import subprocess
import shutil
import os
import numpy as np
import sys

TEST_ROOT = os.path.dirname(os.path.abspath(__file__))
MEGO_ROOT = os.path.abspath(os.path.join(TEST_ROOT, os.pardir))

sys.path.append(MEGO_ROOT)
from src.multiego import io


def read_infile(path):
    """
    Reads a test-case input file and parses the system name the multi-eGO
    command line parameters.

    Parameters
    ----------
    path : str
        The path to the test-case text file

    Returns
    -------
    input_list : list of list
        A list of the commands split at each whitespace
    test_systems : list
        A list containing the system names
    """
    input_list = []
    test_systems = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.replace("\n", "").split(" ")
            system_index = line.index("--system") + 1

            input_list.append(line)
            test_systems.append(line[system_index])

    return input_list, test_systems


def read_outfile(path):
    """
    Reads multi-eGO output files ignoring the comments

    Parameters
    ----------
    path : string
        A path to the multi-eGO output file be it ffnonbonded or topology

    Returns
    -------
    out_string : str
        The file contents
    """
    out_string = ""
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.split(";")[0] if ";" in line else line
            out_string += line

    return out_string


def prep_system_data(name, egos, test_case):
    """
    Prepares system data to be compared and tested by reading all necessary files.

    Parameters
    ----------
    name : str
        Represents the system name (--system parameter in multi-eGO)
    egos : str or list
        Can take two types of values:
         - a string in the case of random coil
         - a list in case of production
        When egos is a list the list contains the two epsilon values for intra and inter.

    Returns
    -------
    topol_ref : str
        The contents of the reference topology which needs to be matched
    topol_test : str
        The contents of the newly created topology which needs to match topol_ref
    ffnonbonded_ref : str
        The contents of the reference ffnonbonded which needs to be matched
    ffnonbonded_test : str
        The contents of the newly created ffnonbonded which needs to match ffnonbonded_ref
    """
    if egos == "rc":
        out_egos = egos
    elif "--epsilon" in test_case:
        out_egos = f"production_e{egos[0]}_{egos[1]}"
    elif "--multi_epsi_intra" in test_case:
        name_appo, multi_epsilon = io.read_intra_file(egos[0])
        name_appo, multi_epsilon_inter_domain = io.read_intra_file(egos[1])
        name_appo, multi_epsilon_inter = io.read_inter_file(egos[2])
        out_egos = (
            "production_epsis_intra"
            + "-".join(np.array(multi_epsilon, dtype=str))
            + "_interdomain"
            + "-".join(np.array(multi_epsilon_inter_domain, dtype=str))
            + "_inter"
            + "-".join(np.array(multi_epsilon_inter, dtype=str).flatten())
        )

    topol_ref = read_outfile(f"{TEST_ROOT}/test_outputs/{name}_{out_egos}/topol_GRETA.top")
    topol_test = read_outfile(f"{MEGO_ROOT}/outputs/{name}_{out_egos}/topol_GRETA.top")
    ffnonbonded_ref = read_outfile(f"{TEST_ROOT}/test_outputs/{name}_{out_egos}/ffnonbonded.itp")
    ffnonbonded_test = read_outfile(f"{MEGO_ROOT}/outputs/{name}_{out_egos}/ffnonbonded.itp")
    return topol_ref, topol_test, ffnonbonded_ref, ffnonbonded_test


def create_test_cases(test_case):
    """
    Creates a test function based on the parameters. The metafunctions can be used with TestOutputs
    to automatically generate test cases.

    Parameters
    ----------
    test_case : list
        Contains the multi-eGO command line flags followed by arguments.

    Returns
    -------
    function_name : str
        The name of the function in format 'test_<system_name>_<egos>'
    function_template : function(self)
        A function taking only self as a parameter intended to be used as a unittest test case.
    """
    # get system name
    system_index = test_case.index("--system") + 1
    system_name = test_case[system_index]

    # get egos type
    egos_index = test_case.index("--egos") + 1
    egos = test_case[egos_index]

    if egos == "rc":
        name_suffix = "rc"
    else:
        # figure out the name suffix and epsilons
        name_suffix = "production"
        if "--epsilon" in test_case:
            intra_epsilon_index = test_case.index("--epsilon") + 1
            intra_epsilon = test_case[intra_epsilon_index]
            if "--inter_epsilon" not in test_case:
                inter_epsilon = intra_epsilon
            else:
                inter_epsilon_index = (
                    egos_index if "--inter_epsilon" not in test_case else test_case.index("--inter_epsilon") + 1
                )
                inter_epsilon = test_case[inter_epsilon_index]
        elif "--multi_epsi_intra" in test_case:
            intra_epsilon_index = test_case.index("--multi_epsi_intra") + 1
            intra_epsilon = test_case[intra_epsilon_index]
            if "--multi_epsi_inter_domain" not in test_case:
                inter_domain_epsilon = intra_epsilon
            else:
                inter_domain_epsilon_index = (
                    egos_index
                    if "--multi_epsi_inter_domain" not in test_case
                    else test_case.index("--multi_epsi_inter_domain") + 1
                )
                inter_domain_epsilon = test_case[inter_domain_epsilon_index]

            if "--multi_epsi_inter" not in test_case:
                inter_epsilon = None
            else:
                inter_epsilon_index = (
                    egos_index if "--multi_epsi_inter" not in test_case else test_case.index("--multi_epsi_inter") + 1
                )
                inter_epsilon = test_case[inter_epsilon_index]

        else:
            print("ERROR: test cases are impossible: either --epsilon or --multi_epsi_intra ")
            exit()

    function_name = f"test_{system_name}_{name_suffix}"
    if egos == "rc":
        system_egos = "rc"
    elif "--epsilon" in test_case:
        system_egos = [intra_epsilon, inter_epsilon]

    elif "--multi_epsi_intra" in test_case:
        system_egos = [intra_epsilon, inter_domain_epsilon, inter_epsilon]

    def function_template(self):
        name = system_name
        egos = system_egos

        topol_ref, topol_test, ffnonbonded_ref, ffnonbonded_test = prep_system_data(name=name, egos=egos, test_case=test_case)
        self.assertEqual(topol_ref, topol_test, f"{name} :: {egos} topology not equal")
        self.assertEqual(ffnonbonded_ref, ffnonbonded_test, f"{name} :: {egos} nonbonded not equal")

    return function_name, function_template


class TestOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        test_commands, test_systems = read_infile(f"{TEST_ROOT}/test_cases.txt")
        for system in test_systems:
            inputs_path = f"{MEGO_ROOT}/inputs/{system}"
            if os.path.exists(inputs_path):
                shutil.rmtree(inputs_path)
            shutil.copytree(f"{TEST_ROOT}/test_inputs/{system}", inputs_path)

        error_codes = [subprocess.call(["python", f"{MEGO_ROOT}/multiego.py", *command]) for command in test_commands]
        for e in error_codes:
            assert e == 0, "Test setup exited with non-zero error code"


if __name__ == "__main__":
    test_commands, test_systems = read_infile(f"{TEST_ROOT}/test_cases.txt")
    for command in test_commands:
        function_name, new_method = create_test_cases(command)
        setattr(TestOutputs, function_name, new_method)

    unittest.main()
