# All temperatures in the AS and OS are in celcius, if not, they are converted using Unit Conversions
# latest fridge temp is weird.

# When running the code using hpc, make sure you are in the ck_fridge_sim as you need to be one directory below datasets to be able to locate the csv file.


<!-- The variation in time steps and the simulation ending slightly before the expected final time occurs because the numerical solver you're using (likely an ODE solver) adjusts the step size dynamically based on error estimates and system dynamics. The solver's primary goal is to maintain accuracy and stability rather than hitting exact time points. This is why:

Adaptive Time-Stepping: The solver adjusts the time step based on the local error, which means the steps might not align perfectly with your expected final time (e.g., exactly 3 minutes).

Solver Termination: The solver might stop the simulation slightly before the final time if the next step would exceed the time boundary or if it determines that further refinement isn't necessary within the accuracy constraints.

This behavior is typical and is intended to ensure the simulation remains accurate and stable. If exact time alignment is critical, you might need to adjust the solver settings or interpolate the results to include the final time point. -->

# DEBUGGING: integrator erro: The same simulation_time is used for extracting the observation, meaning it doesn't advance to the next time step.

# Simulation failed to reach end time: 884.936, the total refrigeration power would be huge which would cause an error in the solver step. 

<!-- Yes, the total refrigeration power over a 15-minute time step is likely calculated regularly throughout the simulation, with updates at each internal time step of the simulation process. However, for practical calculations or further analysis, you can indeed take the final value of total_refrigeration_power__kw at the end of the 15-minute time step and assume it to be constant for that time period. -->

        <!-- if step_result.error_message:
            print(f"Solver error at time {time}: {step_result.error_message}")  # Just print the error
            error_message = f"Solver error at time {time}: {step_result.error_message}"  # Assign the message to the variable
            raise RuntimeError(error_message)  # Raise the exception with the message
Instead of breaking, we raise an error? -->

To plot set point graphs, had to add discharge_pressure list to the simualte or step() function to later plot it as it was not present in the data frame/process results. All other set points kept constant.