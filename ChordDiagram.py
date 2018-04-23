"""
ChordDiagram.py
An implementation of a Chord Diagram in Python using Plotly. The dataset utilized for this visualization is described
in detail in the project README.

Attribute descriptions are as follows:
* Num_Acc: The ID number of the accident.
* catv: The category of the vehicle:
    01 - Bicycle 02 - Moped <50cm3 03 - Cart (Quadricycle with bodied motor) (formerly "cart or motor tricycle") 04 - Most used since 2006 (registered scooter) 05 - Most used since 2006 (motorcycle) 06 - Most used since 2006 (side-car) 07 - VL only 08 - Most used category (VL + caravan) 09 - Most used category (VL + trailer) 10 - VU only 1,5T <= GVW <= 3,5T with or without trailer (formerly VU only 1,5T <= GVW <= 3.5T) 11 - Most used since 2006 (VU (10) + caravan) 12 - Most used since 2006 (VU (10) + trailer) 13 - PL alone 3,5T <PTCA <= 7,5T 14 - PL only> 7,5T 15 - PL> 3,5T + trailer 16 - Tractor only 17 - Tractor + semi-trailer 18 - Most used since 2006 (public transit) 19 - Most used since 2006 (tramway) 20 - Special machinery 21 - Farm Tractor 30 - Scooter <50 cm3 31 - Motorcycle> 50 cm 3 and <= 125 cm 3 32 - Scooter> 50 cm 3 and <= 125 cm 3 33 - Motorcycle> 125 cm 3 34 - Scooter> 125 cm 3 35 - Lightweight Quad <= 50 cm 3 (Quadricycle with no bodied motor) 36 - Heavy Quad> 50 cm 3 (Quadricycle with no bodied motor) 37 - Bus 38 - Coach 39 - Train 40 - Tramway 99 - Other vehicle
"""

import pandas as pd
import plotly.plotly as py
import plotly.figure_factory as ff
from plotly.graph_objs import *


__author__ = "Chris Campell"
__created__ = "4/20/2018"


def main(input_dir):
    with open(input_dir, 'r') as fp:
        df_animals = pd.read_csv(fp, header=0)
    pass
    # for dir in input_dirs:
    #     with open(dir, 'r') as fp:
    #         df_vehicles = pd.read_csv(fp, header=0)
    # pass


if __name__ == '__main__':
    # input_dirs = ['data/zoo.data.csv']
    input_dir = 'data/zoo.data.csv'
    main(input_dir)
