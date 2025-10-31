"""
MOBIL (Minimizing Overall Braking Induced by Lane changes) Model Implementation

This module implements the MOBIL lane-changing model, similar to SUMO's approach.
MOBIL evaluates lane changes based on:
- Acceleration gain for the changing vehicle
- Acceleration impact on following vehicles
- Safety criteria
"""

import numpy as np
from typing import Optional, Tuple


class MOBILModel:
    """
    MOBIL lane-changing model for traffic simulation.
    
    The model evaluates whether a lane change is beneficial by comparing:
    1. Acceleration gain for the changing vehicle
    2. Deceleration impact on vehicles in the target lane
    3. Safety constraints
    """
    
    def __init__(self,
                 politeness_factor: float = 0.5,  # Similar to SUMO's politeness
                 acceleration_threshold: float = 0.2,  # m/s² - minimum gain to change lanes
                 safety_criterion: float = -2.0,  # m/s² - maximum safe deceleration
                 right_lane_bias: float = 0.1):  # Bias for staying in right lane
        """
        Initialize the MOBIL model.
        
        Args:
            politeness_factor: Factor for considering impact on other vehicles (0-1)
            acceleration_threshold: Minimum acceleration gain to justify lane change (m/s²)
            safety_criterion: Maximum safe deceleration for following vehicles (m/s²)
            right_lane_bias: Bias for staying in right lane (higher = prefer right)
        """
        self.politeness_factor = politeness_factor
        self.acceleration_threshold = acceleration_threshold
        self.safety_criterion = safety_criterion
        self.right_lane_bias = right_lane_bias
    
    def calculate_acceleration_gain(self,
                                   current_accel: float,
                                   new_accel: float) -> float:
        """
        Calculate the acceleration gain from a lane change.
        
        Args:
            current_accel: Current acceleration in current lane (m/s²)
            new_accel: Expected acceleration in target lane (m/s²)
            
        Returns:
            Acceleration gain (m/s²)
        """
        return new_accel - current_accel
    
    def evaluate_lane_change(self,
                            current_accel: float,
                            new_accel: float,
                            new_lane_follower_accel_after: float,
                            new_lane_follower_accel_before: float,
                            direction: str = 'left') -> Tuple[bool, float]:
        """
        Evaluate whether a lane change is beneficial using MOBIL criteria.
        
        Args:
            current_accel: Current acceleration in current lane (m/s²)
            new_accel: Expected acceleration in target lane (m/s²)
            new_lane_follower_accel_after: Acceleration of follower in target lane after change (m/s²)
            new_lane_follower_accel_before: Acceleration of follower in target lane before change (m/s²)
            direction: Direction of lane change ('left' or 'right')
            
        Returns:
            Tuple of (should_change, incentive_score)
        """
        # Calculate acceleration gain
        accel_gain = self.calculate_acceleration_gain(current_accel, new_accel)
        
        # Calculate impact on follower (negative = braking)
        follower_impact = new_lane_follower_accel_before - new_lane_follower_accel_after
        
        # Apply politeness factor
        # Positive impact means follower has to brake more
        total_incentive = accel_gain - self.politeness_factor * follower_impact
        
        # Apply right lane bias (prefer staying in right lane)
        if direction == 'right':
            total_incentive -= self.right_lane_bias
        elif direction == 'left':
            total_incentive += self.right_lane_bias
        
        # Safety criterion: follower shouldn't brake too hard
        is_safe = follower_impact >= self.safety_criterion
        
        # Decide if lane change is beneficial
        should_change = is_safe and total_incentive >= self.acceleration_threshold
        
        return should_change, total_incentive
    
    def estimate_new_acceleration(self,
                                 vehicle_speed: float,
                                 vehicle_max_speed: float,
                                 distance_to_new_leader: float,
                                 new_leader_speed: Optional[float] = None,
                                 new_leader_position: Optional[float] = None,
                                 vehicle_position: Optional[float] = None) -> float:
        """
        Estimate acceleration in the target lane.
        
        This is a simplified estimation based on car-following principles.
        In a full implementation, this would use the actual car-following model.
        
        Args:
            vehicle_speed: Current speed of the vehicle (m/s)
            vehicle_max_speed: Maximum speed of the vehicle (m/s)
            distance_to_new_leader: Distance to leading vehicle in target lane (m)
            new_leader_speed: Speed of leading vehicle in target lane (m/s)
            new_leader_position: Position of leading vehicle (m)
            vehicle_position: Current position of vehicle (m)
            
        Returns:
            Estimated acceleration (m/s²)
        """
        # Free flow acceleration
        if distance_to_new_leader > 100 or new_leader_speed is None:
            # No leader or very far - free flow
            desired_speed = min(vehicle_max_speed, vehicle_speed + 2.0)
            if desired_speed > vehicle_speed:
                return 2.0  # Max acceleration
            else:
                return 0.0
        
        # Car-following situation
        # Simple estimation: if there's more space, we can accelerate
        if distance_to_new_leader > 50:
            # Plenty of space - can accelerate
            speed_diff = new_leader_speed - vehicle_speed if new_leader_speed else 0
            if speed_diff > 0:
                return min(2.0, speed_diff * 0.1)  # Accelerate towards leader speed
            else:
                return -1.0  # Need to slow down
        
        # Close following - need to match leader speed
        if new_leader_speed < vehicle_speed:
            return -2.0  # Need to brake
        
        return 0.0  # Maintain speed

