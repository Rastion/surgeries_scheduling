import random
import math
from qubots.base_problem import BaseProblem
import os

class SurgeriesSchedulingProblem(BaseProblem):
    """
    Surgeries Scheduling Problem

    A hospital has a fixed number of operating rooms and nurses, and a set of surgeries to be scheduled.
    Each surgery s must be scheduled within a given time window [min_start[s], max_end[s]] and takes a fixed
    duration (duration[s]). It also requires a given number of nurses (needed_nurses[s]). Each nurse n has a
    shift defined by an earliest start (shift_earliest_start[n]), a latest end (shift_latest_end[n]), and a maximum
    shift duration (max_shift_duration). Furthermore, each surgery s has a set of incompatible rooms: if
    incompatible_rooms[s][r] is 1, then surgery s cannot be performed in room r.

    A candidate solution is represented as a dictionary with:
      - "surgery_room": a list of room assignments (one per surgery).
      - "surgery_start": a list of start times (in minutes) for each surgery.
      - "surgery_end": a list of end times (in minutes) for each surgery.
      - "nurse_assignment": a list (one per nurse) of lists of surgery indices (in the order the nurse works them).

    The objective is to minimize the makespan (the maximum end time among all surgeries). Infeasible
    solutions are penalized heavily.
    """
    
    def __init__(self, instance_file):
        # Read instance data from file.
        # File format:
        #   Line 1: num_rooms num_nurses num_surgeries
        #   Line 2: min_start for each surgery (in hours)
        #   Line 3: max_end for each surgery (in hours)
        #   Line 4: duration for each surgery (in minutes)
        #   Line 5: number of nurses needed for each surgery
        #   Line 6: earliest shift start for each nurse (in hours)
        #   Line 7: latest shift end for each nurse (in hours)
        #   Line 8: maximum shift duration (in hours)
        #   Next num_surgeries lines: for each surgery, num_rooms integers (0 for compatible, 1 for incompatible)

        # Resolve relative path with respect to this module’s directory.
        if not os.path.isabs(filename):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(base_dir, filename)

        with open(instance_file) as f:
            lines = [line.strip() for line in f if line.strip()]
        first_line = lines[0].split()
        self.num_rooms = int(first_line[0])
        self.num_nurses = int(first_line[1])
        self.num_surgeries = int(first_line[2])
        
        # Surgery time windows and durations (convert hours to minutes where applicable)
        self.min_start = [int(x)*60 for x in lines[1].split()[:self.num_surgeries]]
        self.max_end   = [int(x)*60 for x in lines[2].split()[:self.num_surgeries]]
        self.duration  = [int(x) for x in lines[3].split()[:self.num_surgeries]]
        self.needed_nurses = [int(x) for x in lines[4].split()[:self.num_surgeries]]
        
        # Nurse shift constraints
        self.shift_earliest_start = [int(x)*60 for x in lines[5].split()[:self.num_nurses]]
        self.shift_latest_end     = [int(x)*60 for x in lines[6].split()[:self.num_nurses]]
        self.max_shift_duration   = int(lines[7].split()[0])*60
        
        # Incompatible rooms for each surgery (next num_surgeries lines)
        self.incompatible_rooms = []
        for s in range(self.num_surgeries):
            incompat_line = lines[8+s].split()
            self.incompatible_rooms.append([int(x) for x in incompat_line[:self.num_rooms]])
    
    def evaluate_solution(self, candidate) -> float:
        penalty = 0
        # Unpack candidate solution.
        surgery_room = candidate.get("surgery_room", [])
        surgery_start = candidate.get("surgery_start", [])
        surgery_end = candidate.get("surgery_end", [])
        nurse_assignment = candidate.get("nurse_assignment", [])
        
        # Check basic lengths.
        if len(surgery_room) != self.num_surgeries or \
           len(surgery_start) != self.num_surgeries or \
           len(surgery_end) != self.num_surgeries:
            penalty += 1e6
        
        # Verify each surgery's time constraints.
        for s in range(self.num_surgeries):
            if s < len(surgery_start) and s < len(surgery_end):
                if surgery_start[s] < self.min_start[s]:
                    penalty += 1e6 * (self.min_start[s] - surgery_start[s])
                if surgery_end[s] > self.max_end[s]:
                    penalty += 1e6 * (surgery_end[s] - self.max_end[s])
                if surgery_end[s] - surgery_start[s] != self.duration[s]:
                    penalty += 1e6 * abs((surgery_end[s] - surgery_start[s]) - self.duration[s])
            else:
                penalty += 1e6
        
        # Check room assignment compatibility.
        room_assignments = {r: [] for r in range(self.num_rooms)}
        for s, r in enumerate(surgery_room):
            # If r is out-of-range or surgery s is incompatible with room r, add penalty.
            if r < 0 or r >= self.num_rooms or self.incompatible_rooms[s][r] == 1:
                penalty += 1e6
            room_assignments[r].append(s)
        
        # For each room, ensure surgeries do not overlap.
        for r in range(self.num_rooms):
            surgeries_in_room = room_assignments[r]
            surgeries_in_room.sort(key=lambda s: surgery_start[s])
            for i in range(len(surgeries_in_room) - 1):
                s1 = surgeries_in_room[i]
                s2 = surgeries_in_room[i+1]
                if surgery_end[s1] > surgery_start[s2]:
                    penalty += 1e6 * (surgery_end[s1] - surgery_start[s2])
        
        # Check nurse assignment.
        # Each nurse's list must be in non-decreasing order of start times and respect shift limits.
        for n in range(self.num_nurses):
            assigned = nurse_assignment[n] if n < len(nurse_assignment) else []
            if assigned:
                sorted_assigned = sorted(assigned, key=lambda s: surgery_start[s])
                if assigned != sorted_assigned:
                    penalty += 1e6
                first_start = surgery_start[assigned[0]]
                last_end = surgery_end[assigned[-1]]
                if first_start < self.shift_earliest_start[n]:
                    penalty += 1e6 * (self.shift_earliest_start[n] - first_start)
                if last_end > self.shift_latest_end[n]:
                    penalty += 1e6 * (last_end - self.shift_latest_end[n])
                if last_end - first_start > self.max_shift_duration:
                    penalty += 1e6 * ((last_end - first_start) - self.max_shift_duration)
        
        # Check that each surgery is assigned to enough nurses.
        nurse_count = [0] * self.num_surgeries
        for n in range(self.num_nurses):
            for s in nurse_assignment[n]:
                nurse_count[s] += 1
        for s in range(self.num_surgeries):
            if nurse_count[s] < self.needed_nurses[s]:
                penalty += 1e6 * (self.needed_nurses[s] - nurse_count[s])
        
        # Objective: makespan = maximum end time over all surgeries.
        if surgery_end:
            makespan = max(surgery_end)
        else:
            makespan = 1e6
        return makespan + penalty

    def random_solution(self):
        """
        Generates a random candidate solution.
        
        For each surgery:
          - Assign a random room from 0 to num_rooms-1 (ignoring incompatibility for randomness).
          - Choose a random start time between min_start[s] and (max_end[s] - duration[s]),
            and set end = start + duration.
        For nurse assignment:
          - For each surgery, randomly select a set of nurses (of size needed_nurses[s])
            and add surgery s to their assignments.
          - Then, for each nurse, sort their assigned surgeries by start time.
        """
        surgery_room = []
        surgery_start = []
        surgery_end = []
        for s in range(self.num_surgeries):
            # Random room selection.
            r = random.randint(0, self.num_rooms - 1)
            surgery_room.append(r)
            latest_start = self.max_end[s] - self.duration[s]
            if latest_start < self.min_start[s]:
                st = self.min_start[s]
            else:
                st = random.randint(self.min_start[s], latest_start)
            surgery_start.append(st)
            surgery_end.append(st + self.duration[s])
        
        # For each surgery, assign it to a random subset of nurses of size needed_nurses[s].
        nurse_assignment = [[] for _ in range(self.num_nurses)]
        for s in range(self.num_surgeries):
            if self.needed_nurses[s] > self.num_nurses:
                assigned = list(range(self.num_nurses))
            else:
                assigned = random.sample(range(self.num_nurses), self.needed_nurses[s])
            for n in assigned:
                nurse_assignment[n].append(s)
        # For each nurse, sort their assigned surgeries by start time.
        for n in range(self.num_nurses):
            nurse_assignment[n].sort(key=lambda s: surgery_start[s])
        
        return {
            "surgery_room": surgery_room,
            "surgery_start": surgery_start,
            "surgery_end": surgery_end,
            "nurse_assignment": nurse_assignment
        }
