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
        
        This uses a smooth approach where vehicles:
        1. Maintain speed matching when following at desired gap
        2. Gradually adjust speed based on gap error
        3. Avoid sudden acceleration/deceleration
        
        Args:
            current_speed: Current speed of the vehicle (m/s)
            distance_to_leader: Distance to leading vehicle (m)
            leader_speed: Speed of leading vehicle (m/s)
            
        Returns:
            Safe speed (m/s)
        """
        if distance_to_leader <= 0:
            return 0.0
        
        # Calculate desired gap (space we want to maintain)
        desired_gap = self._calculate_desired_gap(leader_speed)
        
        # Gap error: positive = too far, negative = too close
        gap_error = distance_to_leader - desired_gap
        
        # Base safe speed: match leader's speed
        safe_speed = leader_speed
        
        # Adjust based on gap error with smooth proportional control
        # If gap is too large, we can go faster (but not too much faster)
        # If gap is too small, we need to slow down
        gap_control_gain = 0.3  # How aggressively to correct gap (lower = smoother, was 0.5)
        speed_adjustment = gap_control_gain * gap_error / self.reaction_time
        
        # Limit speed adjustment to avoid jerky behavior
        max_adjustment = 2.0  # m/s maximum adjustment per calculation (was 3.0)
        speed_adjustment = np.clip(speed_adjustment, -max_adjustment, max_adjustment)
        
        safe_speed = leader_speed + speed_adjustment
        
        # Safety limits:
        # 1. Don't go faster than leader + reasonable margin when close
        if distance_to_leader < desired_gap * 1.5:
            max_speed_above_leader = leader_speed + 2.0  # Only 2 m/s above when close
            safe_speed = min(safe_speed, max_speed_above_leader)
        else:
            # More room = can go faster
            max_speed_above_leader = leader_speed + 5.0
            safe_speed = min(safe_speed, max_speed_above_leader)
        
        # 2. Start decelerating earlier when approaching slower leader
        # Calculate "anticipation distance" - how far ahead to start adjusting speed
        anticipation_distance = desired_gap * 2.0  # Start reacting 2x the desired gap away
        
        if distance_to_leader < anticipation_distance:
            # We're getting close - adjust speed more aggressively
            # The closer we get, the more we match the leader's speed
            closeness_factor = 1.0 - (distance_to_leader / anticipation_distance)  # 0 = far, 1 = very close
            
            # Blend between proportional control and direct speed matching
            # When close, match leader speed more directly
            speed_blend = (1.0 - closeness_factor) * safe_speed + closeness_factor * leader_speed
            safe_speed = speed_blend
        
        # 3. Emergency braking if critically close
        critical_gap = self.desired_stop_gap + 5.0  # Critical distance
        if distance_to_leader < critical_gap:
            # Proportional braking based on how close we are
            emergency_factor = max(0.0, distance_to_leader / critical_gap)
            safe_speed = min(safe_speed, leader_speed * emergency_factor)
        
        # Ensure safe speed is not negative
        return max(0.0, safe_speed)
    
    def _calculate_desired_gap(self, leader_speed: float) -> float:
        """Calculate desired gap at given leader speed.
        
        Uses the safe distance formula: gap = stop_gap + speed * time_headway
        This ensures vehicles maintain a time-based following distance.

        Args:
            leader_speed (float): Speed of leader at the moment.

        Returns:
            float: Desired gap distance in meters.
        """
        # Time headway: how many seconds of travel distance to maintain
        # Increased to 7.0 seconds - 2X BIGGER GAPS
        time_headway = 7.0  # seconds (2x from 3.5)
        
        # Minimum stop gap - 2X BIGGER GAPS
        min_stop_gap = max(self.desired_stop_gap, 10)  # At least 10m when stopped
        
        # Desired gap = minimum stop gap + speed-dependent spacing
        desired_gap = min_stop_gap + leader_speed * time_headway
        return desired_gap
    
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
        
        Uses smooth deceleration profile to gradually stop at the light.

        Args:
            current_speed (float): Current car speed.
            distance_to_traffic_lights (float): Distance from front bumper to traffic lights.
            dt (float): Time step in traffic model.

        Returns:
            float: Speed in the next step of simulation.
        """
        # Start decelerating well before the light for SMOOTH, EARLY stopping
        # Scale with speed - faster vehicles need more distance
        deceleration_start_distance = max(200.0, current_speed * 4.0)  # At least 200m, or 4 seconds at current speed (was 120m/2s)
        
        if distance_to_traffic_lights > deceleration_start_distance:
            return current_speed  # Far enough, maintain speed
        
        # Already very close - stop completely
        if distance_to_traffic_lights < 5.0:
            return 0.0
        
        # Use GENTLE deceleration for smooth stopping
        # Calculate required deceleration to reach near-zero speed at the light
        # Using kinematic equation: v² = u² + 2as, solving for a: a = (v² - u²) / (2s)
        target_speed_at_light = 0.0  # Come to complete stop
        
        # Calculate required deceleration (negative value)
        if distance_to_traffic_lights > 0.1:
            required_decel = -(current_speed**2 - target_speed_at_light**2) / (2 * distance_to_traffic_lights)
        else:
            required_decel = self.max_deceleration  # Emergency stop
        
        # Use gentler deceleration for smoother stopping
        # Limit to comfortable deceleration instead of max
        comfortable_decel = -3.0  # m/s² - gentle braking (was using max_deceleration)
        
        # Use the gentler of: required or comfortable deceleration
        # (both are negative, so we want the less negative one = smoother)
        decel = max(required_decel, comfortable_decel)
        
        # Only use harder braking if we're getting too close
        if distance_to_traffic_lights < 30.0:
            # Emergency zone - use required deceleration
            decel = max(required_decel, self.max_deceleration)
        
        # Apply deceleration
        new_speed = current_speed + decel * dt
        
        return max(0.0, new_speed)
    
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
    
    
