import numpy as np

def create_si_and_uni_mapping(projection_distance = 0.05, angular_velocity_limit = np.pi):
    '''Create two functions for mapping from single integrator dynamics to unicycle dynamics and
    unicycle states to single states. 

    This mapping is done by placing a virtual control "point" in front of the unicycle.

    projection_distance: How far ahead to place the point. 
    angular_velocity_limit: The maximum angular velocity that can be provided
    
    -> (function, function)
    '''

    # Check user input types
    assert isinstance(projection_distance, (int, float)), "In the function create_uni_and_si_mapping, the projection distance must be an integer or float. Received type %r." % type(projection_distance).__name__
    assert isinstance(angular_velocity_limit, (int, float)), "In the function create_uni_and_si_mapping, the angular velocity limit must be an integer or float. Received type %r." % type(angular_velocity_limit).__name__

    # Check user input ranges/sizes
    assert projection_distance > 0, "In the function create_uni_and_si_mapping, the projection distance must be positive. Recieved %r." % projection_distance
    assert angular_velocity_limit >= 0, "In the function create_uni_and_si_mapping, the maximum angular velocity command (angular_velocity_limit) must be greater than or equal to zero. Recieved %r." % angular_velocity_limit

    def si_to_uni_dyn(dxi, poses):
        '''Takes single-integrator velocitiews and transforms them to unicycle control inputs. 
        dxi: 2xN numpy array of single-integrator control inputs
        poses:  3xN numpy array of unicycle poses
        
        -> 2xN numpy array of unicycle control inputs'''
        #Check user input types
        assert isinstance(dxi, np.ndarray), "In the si_to_uni_dyn function created by the create_si_and_uni_mapping function, the single integrator velocity inputs (dxi) must be a numpy array. Recieved type %r." % type(dxi).__name__
        assert isinstance(poses, np.ndarray), "In the si_to_uni_dyn function created by the create_si_and_uni_mapping function, the current robot poses (poses) must be a numpy array. Recieved type %r." % type(poses).__name__

        #Check user input ranges/sizes
        assert dxi.shape[0] == 2, "In the si_to_uni_dyn function created by the create_si_and_uni_mapping function, the dimension of the single integrator velocity inputs (dxi) must be 2 ([x_dot;y_dot]). Recieved dimension %r." % dxi.shape[0]
        assert poses.shape[0] == 3, "In the si_to_uni_dyn function created by the create_si_and_uni_mapping function, the dimension of the current pose of each robot must be 3 ([x;y;theta]). Recieved dimension %r." % poses.shape[0]
        assert dxi.shape[1] == poses.shape[1], "In the si_to_uni_dyn function created by the create_si_and_uni_mapping function, the number of single integrator velocity inputs must be equal to the number of current robot poses. Recieved a single integrator velocity input array of size %r x %r and current pose array of size %r x %r." % (dxi.shape[0], dxi.shape[1], poses.shape[0], poses.shape[1])

        M,N = np.shape(dxi)

        cs = np.cos(poses[2, :])
        ss = np.sin(poses[2, :])

        dxu = np.zeros((2, N))
        dxu[0, :] = (cs*dxi[0, :] + ss*dxi[1, :])
        dxu[1, :] = (1/projection_distance)*(-ss*dxi[0, :] + cs*dxi[1, :])

        #Impose angular velocity cap.
        # dxu[1,dxu[1,:]>angular_velocity_limit] = angular_velocity_limit
        # dxu[1,dxu[1,:]<-angular_velocity_limit] = -angular_velocity_limit 

        return dxu

    def uni_to_si_states(poses):
        '''Take unicycle states an returns single-integrator states
        poses: 3xN numpy array of unicycle states
        -> 2xN numpy array of single-integrator states
        '''
        _,N = np.shape(poses)

        si_states = np.zeros((2, N))
        si_states[0, :] = poses[0, :] + projection_distance*np.cos(poses[2, :])
        si_states[1, :] = poses[1, :] + projection_distance*np.sin(poses[2, :])

        return si_states
    return si_to_uni_dyn, uni_to_si_states
    

