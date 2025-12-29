"""
Krauss Car-Following Model Implementation

The Krauss model is a stochastic car-following model that considers:
- Safe speed based on distance to leading vehicle
- Random deceleration to simulate driver behavior
- Maximum acceleration and deceleration constraints
"""

import numpy as np
from typing import Optional


class KraussModel:
    """
    Implementation of the Krauss car-following model for traffic simulation.
    
    The model calculates the next speed of a vehicle based on:
    1. Safe speed (distance to leading vehicle)
    2. Desired speed (free flow speed)
    3. Random deceleration (driver behavior)
    4. Physical constraints (max acceleration/deceleration)
    """
    
    def __init__(self, 
                 max_speed: float = 30.0,  # m/s
                 max_acceleration: float = 2.0,  # m/s²
                 max_deceleration: float = -10.0,  # m/s²
                 reaction_time: float = 1.0,  # seconds
                 random_deceleration_prob: float = 0.1,
                 random_deceleration_max: float = 0.5, # m/s²
                 desired_stop_gap: float = 2.5 # m
                 ):  
        """
        Initialize the Krauss model parameters.
        
        Args:
            max_speed: Maximum speed of vehicles (m/s)
            max_acceleration: Maximum acceleration (m/s²)
            max_deceleration: Maximum deceleration (m/s², negative value)
            reaction_time: Driver reaction time (seconds)
            random_deceleration_prob: Probability of random deceleration
            random_deceleration_max: Maximum random deceleration (m/s²)
            desired_stop_gap (float): Desired gap when speed of leader and the car is 0.
        """
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration
        self.reaction_time = reaction_time
        self.random_deceleration_prob = random_deceleration_prob
        assert random_deceleration_max >= 0, f"Maximum random deceleration must >= 0 and {random_deceleration_max} was given"
        self.random_deceleration_max = random_deceleration_max
        self.desired_stop_gap = desired_stop_gap
    
    def calculate_safe_speed(self, 
                           current_speed: float,
                           distance_to_leader: float,
                           leader_speed: float) -> float:
        """
        Calculate the safe speed based on distance to leading vehicle.
        
        Args:
            current_speed: Current speed of the vehicle (m/s)
            distance_to_leader: Distance to leading vehicle (m)
            leader_speed: Speed of leading vehicle (m/s)
            
        Returns:
            Safe speed (m/s)
        """
        if leader_speed is None:
            return self.max_speed
        
        if distance_to_leader <= 0:
            return 0.0
        
        desired_gap = self._calculate_desired_gap(leader_speed, current_speed)
        
        safe_speed = leader_speed + (distance_to_leader - desired_gap) / self.reaction_time

        print(f"distance: {round(distance_to_leader,2)}; speeds(leader/follower): {round(leader_speed, 2)}/{round(current_speed, 2)}; deisred gap: {round(desired_gap, 2)}; safe speed: {round(safe_speed, 2)}")
        return max(0.0, safe_speed)
    
    def _calculate_desired_gap(self, leader_speed: float, current_speed: float) -> float:
        """Calculate desired gap at given leader speed.

        Args:
            leader_speed (float): Speed of leader at the moment.

        Returns:
            float: Desired speed in in conditions.
        """
        v_f_0 = current_speed
        v_l_0 = leader_speed
        a_max = abs(self.max_deceleration)
        t_r = self.reaction_time

        # 1.2 to ensure the gap is greater than needed to stop before leader
        # formula derived analitically to ensure it's enough to stop
        return self.desired_stop_gap + 1.2 * max(0, (v_f_0 * (v_f_0 / a_max + t_r) - v_l_0 * v_l_0 / a_max) / 2)
    
    def calculate_desired_speed(self, 
                              current_speed: float,
                              distance_to_leader: float,
                              dt: float,
                              leader_speed: Optional[float] = None) -> float:
        """
        Calculate the desired speed considering safe speed and free flow speed.
        
        Args:
            current_speed: Current speed of the vehicle (m/s)
            distance_to_leader: Distance to leading vehicle (m)
            dt (float): Time step of a traffic model (s)
            leader_speed: Speed of leading vehicle (m/s), None if no leader
            
        Returns:
            Desired speed (m/s)
        """

        safe_speed = self.calculate_safe_speed(
            current_speed,
            distance_to_leader,
            leader_speed
        )
        desired_speed = min(
            safe_speed,
            current_speed + self.max_acceleration * dt,
            self.max_speed
        )
        
        return desired_speed
    
    def apply_random_deceleration(self, speed: float, dt: float) -> float:
        """
        Apply random deceleration to simulate driver behavior.
        
        Args:
            speed: Current speed (m/s)
            dt: Time step of a traffic model (s)
            
        Returns:
            Speed after random deceleration (m/s)
        """
        if np.random.random() < self.random_deceleration_prob:
            # Apply random deceleration
            random_decel = np.random.uniform(0, self.random_deceleration_max)
            speed = max(0.0, speed - random_decel * dt)
        
        return speed
    
    def calculate_traffic_lights_based_next_speed(self, current_speed: float, distance_to_traffic_lights: float, dt: float) -> float:
        """Calculate next speed based on distance to traffic lights in state 'red'.

        Args:
            current_speed (float): Current car speed.
            distance_to_traffic_lights (float): Distance to traffic lights.
            dt (float): Time step in traffic model.

        Returns:
            float: Speed in the next step of simulation.
        """
        minimal_distance = current_speed ** 2 / abs(2*self.max_deceleration)
        print(f"minimal breaking distance: {minimal_distance}")

        if distance_to_traffic_lights < 50:
            print(f"breaking from: {current_speed} to: {current_speed + self.max_deceleration * dt}")
            return current_speed + self.max_deceleration * dt

        # if distance_to_traffic_lights < 1.2 * minimal_distance:
        #     # Start breaking
        #     return current_speed + self.max_deceleration * dt
        
        return current_speed + self.max_acceleration * dt
    


        # if distance_to_traffic_lights < 0.5:
        #     return 0.0
        
        # # if distance_to_traffic_lights > 150:
        # #     return current_speed

        # decel = -3*current_speed ** 2 / (2*distance_to_traffic_lights)
        # decel = max(self.max_deceleration, decel) # both are negative
        # result = current_speed + decel * dt


        # if result * dt >= distance_to_traffic_lights:
        #     return 0.0
        # return max(0, result)
    
    def calculate_next_speed(self, 
                           current_speed: float,
                           distance_to_leader: float,
                           dt: float,
                           leader_speed: Optional[float] = None) -> float:
        """
        Calculate the next speed using the Krauss model.
        
        Args:
            current_speed: Current speed of the vehicle (m/s)
            distance_to_leader: Distance to leading vehicle (m)
            dt (float): Time step of a traffic model (s)
            leader_speed: Speed of leading vehicle (m/s), None if no leader
            
        Returns:
            Next speed (m/s)
        """
        # Step 1: Calculate desired speed
        desired_speed = self.calculate_desired_speed(current_speed, distance_to_leader, dt, leader_speed)
        
        # Step 2: Apply random deceleration
        # speed_with_random = self.apply_random_deceleration(desired_speed, dt)
        print(f"next speed: {max(0, desired_speed)}")
        return max(0, desired_speed)
    
    
