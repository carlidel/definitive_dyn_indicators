henon_square_configuration = {
    'samples': 500,
    
    'x_extents' : [0.0, 1.0],
    'y_extents' : [0.0, 1.0],
    'epsilon_list' : [0.0, 1.0, 2.0, 8.0, 16.0, 32.0, 64.0],

    'omega_x' : 0.168,
    'omega_y' : 0.201,

    'long_tracking' : 10000000,

    'frequency_tracking' : {
        'max_power_of_two' : 14,
        'min_power_of_two' : 5,
    }
}
