# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 10:22:23 2023

@author: MARINAMU
"""

class FindMinMaxScore:
    def __init__(self, target_score_name):
        self.target_score_name = target_score_name
        self.min_y, self.max_y = self.calculate_min_max()

    def calculate_min_max(self):
        if self.target_score_name.casefold() == "ados":
            min_y = 0
            max_y = 26
        elif self.target_score_name.casefold() == "sa":
            min_y = 0
            max_y = 22
        elif self.target_score_name.casefold() == "rrb":
            min_y = 0
            max_y = 8
        else:
            raise ValueError(f"{self.target_score_name} doesn't exist. Please enter a correct target score name")
        return min_y, max_y



