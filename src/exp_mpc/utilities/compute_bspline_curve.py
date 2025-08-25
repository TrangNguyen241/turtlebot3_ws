import numpy as np

def compute_bspline_curve(control_point, d):
    num_control_point = len(control_point)
    samples = num_control_point * 20
    #### First: create clamped knot vector ####
    def clamped_knot_vector(number_control_point: int, degree: int):
    
        n = number_control_point - 1
        # we will set t_d-1 = 0 and t_n+1 = 1 and between them have [(n+1)-(d-1)] = (n - d + 2) intervals
        pre_knot_vector = np.linspace(0, 1, (n - degree + 2 + 1)) # +1 represent the number of knot, not the intervals
        knot_vector = [0.0]*(degree - 1) + pre_knot_vector.tolist() + [1.0]*(degree-1) # repeat t_d-1 from t_0 and t_n+1 till end
        return knot_vector
    knot_vector = clamped_knot_vector(number_control_point=num_control_point, degree=d) 

    #### Second: compute the basic function ####
    def compute_basic_function(knot_vector, degree: int):
        ''' the variable b_i_d represents the basic function at "degree: d, 1 <= d <= degree_setting",
                                                and between 2 knot "t_i <= t < t_i+1" 

        '''
        # base on the paper, creat an array of basic function, which has 'Degree + 1' rows from 0 -> Degree, i will use row 1 -> Degree to make the same theory as the paper
        #                                                                'n + 2' columns from 0 -> n + 1
        row = degree + 1
        column = len(knot_vector)
        b = np.full((row, column), lambda t: 0, dtype=object).tolist()
        # set the basic function at degree 1
        for i in range(column - 1):
            b[1][i] = lambda t, i=i: (t >= knot_vector[i]) * (t < knot_vector[i+1])

        for k in range(2, row): 
            for i in range (0, column - k  ):
                if (knot_vector[i+k-1]-knot_vector[i]) != 0 and (knot_vector[i+k] - knot_vector[i+1]) != 0:
                    b[k][i] = lambda t, k=k, i=i:b[k-1][i](t) * (t - knot_vector[i])/(knot_vector[i+k-1]-knot_vector[i]) +\
                    b[k-1][i+1](t) * (knot_vector[i+k]-t)/(knot_vector[i+k]-knot_vector[i+1])
                elif (knot_vector[i+k-1] - knot_vector[i]) != 0:
                    b[k][i] = lambda t,k=k,i=i: b[k-1][i](t) * (t - knot_vector[i])/(knot_vector[i+k-1]-knot_vector[i])
                elif (knot_vector[i+k] - knot_vector[i+1]) != 0:
                    b[k][i] = lambda t,k=k,i=i: b[k-1][i+1](t) * (knot_vector[i+k]-t)/(knot_vector[i+k]-knot_vector[i+1])
                else:
                    b[k][i] = lambda t: 0

        return b[-1]
    
    basic_function = compute_basic_function(knot_vector=knot_vector, degree=d)

    #### Third: compute curve ####
    def compute_curve(ctrl_point, basic_funtion, knot_vector, degree: int, samples: int):
        """
        Computes points of an open uniform B-spline.
        IN:
            ctrl_points: control points, [(x0, y0), (x1, y1), ..., (xn, yn)]
            basis_functions: 2D array of basis functions
            knot_vector: uniform non-clamped knot vector, [0., .1, .2, ..., 1.]
            d: curve order
            samples: number of points to be computed
        OUT:
            points of the curve, [(x0, y0), (x1, y1), ..., (xn, yn)]

        Note: because at time t (t_i <= t < t_i+1, not include t_i+1), so if we seperate the time serie
        like normal, there will be a case where t = t_i+1. To solve this problem, i will devide the time
        between 0 and 0,999999 instead of 1.
        """
        time_series = np.linspace(0, 0.999999, samples)
        m = len(ctrl_point)
        result = []
        for u in time_series:
            point = (0,0)
            sum_bid = 0
            for i in range(m):
                b_i_d = basic_funtion[i](u)
                point = (b_i_d * ctrl_point[i][0] + point[0], b_i_d * ctrl_point[i][1] + point[1])
                sum_bid += b_i_d
            result.append(point)
        print(f"the sum of b_i_d is: {sum_bid}")
        return result
    
    #### result to plot curve and store the reference path ####
    result = compute_curve(ctrl_point=control_point, basic_funtion=basic_function, knot_vector=knot_vector, degree=d, samples=samples)
    return result
