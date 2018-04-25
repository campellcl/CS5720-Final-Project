# CS5720-Final-Project
Final two visualizations for CS 5720 (Scientific Computing with Visualization). 

## Project Proposal
* [Proposal](https://docs.google.com/document/d/1_HCKTHz8wSD9y48NpfAA3F-t9V3hiOMYgpq0Dt18YxY/edit?usp=sharing)

## Visualization One:

## Visualization Two:
### Learning Resources:
* [Visualizing what ConvNets Learn](http://cs231n.github.io/understanding-cnn/)
* [Feature Visualization](https://distill.pub/2017/feature-visualization/)
* [Understanding Neural Networks Through Deep Visualization](http://yosinski.com/deepvis)
* [PyTorch CNN Visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
* [Intel AI Academy: Visualising CNN Models Using PyTorch](https://software.intel.com/en-us/articles/visualising-cnn-models-using-pytorch)

### Run Configurations:
* `../data/ImageNet/Subsets/`


### Dataset:
* [Attributes (French)](https://www.data.gouv.fr/s/resources/base-de-donnees-accidents-corporels-de-la-circulation/20170915-155209/Description_des_bases_de_donnees_ONISR_-Annees_2005_a_2016.pdf)
### Dataset Attributes (English):
* Num_Acc (Integer):
    * Identifier of the accident identical to that of the file "rubric CHARACTERISTICS" taken for each vehicle described involved in the accident.
* senc
* catv (Integer): The category of the vehicle:
    * 01 - Bicycle
    * 02 - Moped < 50 cm^3
    * 03 - Cart (Quadricycle with bodied motor) (formerly "cart or motor tricycle")
    * 04 - Most used since 2006 (registered scooter)
    * 05 - Most used since 2006 (motorcycle)
    * 06 - Most used since 2006 (side-car)
    * 07 - VL only
    * 08 - Most used category (VL + caravan)
    * 09 - Most used category (VL + trailer)
    * 10 - VU only 1,5T <= GVW <= 3,5T with or without trailer (formerly VU only 1,5T <= GVW <= 3.5T)
    * 11 - Most used since 2006 (VU (10) + caravan)
    * 12 - Most used since 2006 (VU (10) + trailer)
    * 13 - PL alone 3,5T <PTCA <= 7,5T 14 - PL only> 7,5T 15 - PL> 3,5T + trailer
    * 16 - Tractor only
    * 17 - Tractor + semi-trailer
    * 18 - Most used since 2006 (public transit)
    * 19 - Most used since 2006 (tramway)
    * 20 - Special machinery
    * 21 - Farm Tractor
    * 30 - Scooter < 50 cm^3
    * 31 - Motorcycle > 50 cm^3 and <= 125 cm^3
    * 32 - Scooter > 50 cm^3 and <= 125 cm^3
    * 33 - Motorcycle > 125 cm^3
    * 34 - Scooter > 125 cm^3
    * 35 - Lightweight Quad <= 50 cm^3 (Quadricycle with no bodied motor)
    * 36 - Heavy Quad > 50 cm^3 (Quadricycle with no bodied motor)
    * 37 - Bus
    * 38 - Coach
    * 39 - Train
    * 40 - Tramway
    * 99 - Other vehicle
* occutc
* obs (Integer): Obstacle fixed hit:
    * 1 - Vehicle parked
    * 2 - Tree
    * 3 - Metal slide
    * 4 - Concrete slide
    * 5 - Other slide
    * 6 - Building, wall, bridge stack
    * 7 - Vertical sign support or emergency call station
    * 8 - Post
    * 9 - Urban furniture
* obsm
* choc
* manv
* num_veh
### Resources:
* [Helpful Tutorial](https://freakonometrics.hypotheses.org/20667)
