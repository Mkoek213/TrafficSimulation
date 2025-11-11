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
                 desired_stop_gap: float = 2 # m
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
        if distance_to_leader <= 0:
            return 0.0
        
        # Safe speed calculation: v_safe = v_leader + (gap - desired_gap - v_leader * reaction_time) / reaction_time
        # This ensures the vehicle can stop safely if the leader stops suddenly
        # But also consider that we shouldn't exceed the leader's speed by too much
        desired_gap = self._calculate_desired_gap(leader_speed)
        safe_speed = leader_speed + (distance_to_leader - desired_gap - leader_speed * self.reaction_time) / self.reaction_time
        
        # Additional constraint: don't exceed leader speed by more than a reasonable amount
        max_speed_above_leader = leader_speed + 5.0  # Don't exceed leader by more than 5 m/s
        safe_speed = min(safe_speed, max_speed_above_leader)
        
        # More lenient safe distance - allow closer following
        min_safe_distance = 10.0  # meters - allow closer following (reduced from 15.0)
        # if distance_to_leader < min_safe_distance:
        #     # Gradual speed reduction instead of sudden cut
        #     speed_reduction_factor = max(0.8, distance_to_leader / min_safe_distance)
        #     safe_speed = min(safe_speed, leader_speed * speed_reduction_factor)
        
        # Ensure safe speed is not negative
        return max(0.0, safe_speed)
    
    def _calculate_desired_gap(self, leader_speed: float) -> float:
        """Calculate desired gap at given leader speed.

        Args:
            leader_speed (float): Speed of leader at the moment.

        Returns:
            float: Desired speed in in conditions.
        """
        return leader_speed * self.reaction_time * 1.2
    
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
        if leader_speed is None or distance_to_leader > 100:  # No leader or very far
            # Free flow: accelerate towards maximum speed
            desired_speed = min(self.max_speed, current_speed + self.max_acceleration * dt)
        else:
            # Car-following: consider safe speed
            safe_speed = self.calculate_safe_speed(current_speed, distance_to_leader, leader_speed)
            # Allow acceleration but don't exceed safe speed
            desired_speed = min(safe_speed, current_speed + self.max_acceleration * dt)
        
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
        
        decel = -3*current_speed ** 2 / (2*distance_to_traffic_lights)
        decel = max(self.max_deceleration, decel) # both are negative
        result = current_speed + decel * dt

        if result < 0.05 * self.max_speed:
            return 0.0
        return result
    
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
        speed_with_random = self.apply_random_deceleration(desired_speed, dt)
        
        # Step 3: Apply physical constraints
        # Ensure speed doesn't exceed maximum deceleration
        min_speed = max(0.0, current_speed + self.max_deceleration * dt)
        final_speed = max(min_speed, speed_with_random)
        
        # Ensure speed doesn't exceed maximum speed
        final_speed = min(self.max_speed, final_speed)
        
        return final_speed
    
    
