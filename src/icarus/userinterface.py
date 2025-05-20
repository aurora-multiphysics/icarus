from tkinter import Tk, Label, Entry, Button, Frame, Checkbutton, BooleanVar, StringVar
from pathlib import Path

class UserInterface:
    """Provides a user interface using tkinter to allow the user to control aspects of the modelling process,
        including selecting parameters to be perturbed and defining their values and how many validation datasets
        to generate for each parameter.
    """
    def __init__(self):
        pass


    def accept_parameters(self, parameters: dict[str, float]) -> tuple[list[int], dict[str, list[float]]]:
        """accept_parameters: used to allow user to select which parameters to perturb, and the range,
            interval of perturbation values, and number of validation values to be used for each parameter.

        Parameters
        ----------
        parameters : dict[str, float]
            Dictionary of parameter names and their corresponding values in the default input file.
            Used to allow users to select which parameters to modify, and show them the default
            values so they don't include them again.

        Returns
        ----------
        num_val_values : list[int]
            Number of validation values to use for each parameter
        parameters : dict[str, list[float]]
            Dictionary of parameter names and their corresponding list of values for perturbation.
        """

        def submit_parameters():
            """submit_parameter: specifies what should happen when the submit button is pressed.
                The values within the Entry boxes for min_val, max_val and interval for the selected 
                parameters should be saved to their corresponding variables to allow creation of 
                param_values list, and the tkinter window should close.

            Raises 
            ----------
            ValueError
                If the range or interval specified are invalid
            Value Error
                If there are fewer than 2 values for perturbation 
            Value Error
                If there are fewer than 2 values for validation
            """
            nonlocal parameters, num_val_values
            parameters = {}
            num_val_values = []

            for row in rows:
                if row['checkbox_var'].get():  # Check if checkbox is ticked
                    param_name = row['param_name']
                    try:
                        param_class = str(row['param_class'].get())
                        default_val = float(row['default_val'])
                        min_val = float(row['min_val'].get())
                        max_val = float(row['max_val'].get())
                        interval = float(row['interval'].get())
                        num_validation_values = int(row['num_validation_values'].get())

                        if param_class.lower() not in ["geom", "bc", "mat_prop"]:
                            raise ValueError("Invalid parameter classification")

                        if min_val >= max_val or interval <= 0:
                            raise ValueError("Invalid range or interval")
                        
                        param_data = []
                        param_data.append(param_class)

                        param_values = []
                        current_val = min_val
                        while current_val <= max_val:
                            if current_val != default_val:
                                param_values.append(current_val)
                            current_val += interval

                        if len(param_values) < 2:
                            raise ValueError("Insufficient perturbation values")
                        
                        if num_validation_values < 2:
                            raise ValueError("Insufficient validation values")

                        param_data.append(param_values)
                        num_val_values.append(num_validation_values)
                        parameters[param_name] = param_data

                    except ValueError as e:
                        error_label.config(text=f"Error: {str(e)}", fg="red")
                        return  # Stop execution if an error occurs

            if len(parameters) == 0:
                error_label.config(text="No parameters submitted", fg="red")
                return
            
            param_root.destroy()

        param_root = Tk()

        table_frame = Frame(param_root)
        table_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        headers = ["Select", "Param Name", "Param Class", "Default Value", "Min Value", "Max Value", "Interval", "No. Validation Values"]
        for col, header in enumerate(headers):
            Label(table_frame, text=header).grid(row=0, column=col, padx=5, pady=5)

        params = []
        for parameter_name, parameter_value in parameters.items():
            try:
                if float(parameter_value) <= 10:
                    x = parameter_value
                elif float(parameter_value) > 10 and parameter_value < 100:
                    x = 10
                else:
                    x = parameter_value/2
                params.append({"param_name": parameter_name,"param_class": "geom/bc/mat_prop","default_val": parameter_value,
                               "min_val": parameter_value+x,"max_val": parameter_value+5*x,"interval": x,"num_validation_values":3})
            except ValueError:
                params.append({"param_name": parameter_name,"param_class": "geom/bc/mat_prop","default_val": parameter_value,
                               "min_val": "","max_val": "","interval": "","num_validation_values":""})

        rows = []
        for i, param in enumerate(params):
            row = {}

            row['checkbox_var'] = BooleanVar(value=False)

            checkbox = Checkbutton(table_frame, variable=row['checkbox_var'])
            checkbox.grid(row=i+1, column=0, padx=5, pady=5)

            Label(table_frame, text=param['param_name']).grid(row=i+1, column=1, padx=5, pady=5)
            row['param_name'] = param['param_name']

            row['param_class'] = StringVar(value=param['param_class'])
            param_class_entry = Entry(table_frame, textvariable=row['param_class'], width=15)
            param_class_entry.grid(row=i+1, column=2, padx=5, pady=5)

            Label(table_frame, text=param['default_val']).grid(row=i+1, column=3, padx=5, pady=5)
            row['default_val'] = param['default_val']

            row['min_val'] = StringVar(value=param['min_val'])
            min_val_entry = Entry(table_frame, textvariable=row['min_val'], width=10)
            min_val_entry.grid(row=i+1, column=4, padx=5, pady=5)

            row['max_val'] = StringVar(value=param['max_val'])
            max_val_entry = Entry(table_frame, textvariable=row['max_val'], width=10) 
            max_val_entry.grid(row=i+1, column=5, padx=5, pady=5)

            row['interval'] = StringVar(value=param['interval'])
            interval_entry = Entry(table_frame, textvariable=row['interval'], width=10)
            interval_entry.grid(row=i+1, column=6, padx=5, pady=5)

            row['num_validation_values'] = StringVar(value=param['num_validation_values'])
            num_validation_values_entry = Entry(table_frame, textvariable=row['num_validation_values'], width=10)
            num_validation_values_entry.grid(row=i+1, column=7, padx=5, pady=5)

            rows.append(row)

        parameters = None
        num_val_values = None

        submit_button = Button(param_root, text="Submit", command=submit_parameters)
        submit_button.grid(row=3, column=0, columnspan=2, pady=10)

        error_label = Label(param_root, text="", fg="red")
        error_label.grid(row=4, column=0, columnspan=2) 

        param_root.mainloop()

        return num_val_values, parameters