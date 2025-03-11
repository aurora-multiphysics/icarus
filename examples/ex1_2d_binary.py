from icarus import (DatasetGenerator,
                    ModelBuilder,
                    MooseSetup,
                    UserInterface)

def main():
    """main: runs all of the other functions.

    Raises
    ------
    FileNotFoundError
        If the file submission window is closed (instead of submitting file paths)
    ValueError
        If the input file is improperly formatted and no parameters are found as expected
    ValueError
        If the parameter submission window is closed (instead of submitting parameter data)
    """
    input_file_path, output_file_path = UserInterface.accept_file("scripts/moose/plate_2d_thermal.i", 
                                                                  "examples/example_outputs/ex1_outputs")

    if input_file_path == None or output_file_path == None:
        raise FileNotFoundError(f"Specified input and/or output file path not found. Exiting.")

    moose_runner, moose_modifier = MooseSetup.setup_moose_runner(input_file_path)

    found_vars = moose_modifier.get_vars()
    if len(found_vars) == 0:
        raise ValueError(f"No parameters found in input file. Check input file and try again.")
    else:
        num_validation_values, parameters = UserInterface.accept_parameters(found_vars)

    if parameters == None:
        raise ValueError(f"Unacceptable parameters. Exiting.")

    DatasetGenerator.generate_datasets(output_file_path, parameters, num_validation_values, moose_runner, moose_modifier)

    ModelBuilder.model(output_file_path)


if __name__ == "__main__":
    main()


# Next steps:  
    # Refactor to OOP
    # Test suite using PyTest (develop as you go)
    # Improve binary classifier by:
        # Allow choice of modelling frameworks - modelling class
        # Incorporating errors - labelled_dataset function
        # Accepting multiple simultaneous perturbations - generate datasets class
        # Allowing user to define number of sensors + positions - labelled dataset function
    # Expand to multi-classifier
    # Examples + tutorials
    # Packaging for pip distribution
    # Stretch goal: more complex input files, e.g. 3D monoblock