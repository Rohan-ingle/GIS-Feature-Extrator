# GIS Feature Extraction Model

This repository contains a GIS feature extraction model trained using the YOLOv8s and YOLOv8n architectures. The model is designed to identify and classify various types of objects and structures from geographical imagery. This project aims to provide an efficient and accurate method for feature extraction in geospatial analysis.

## Dataset

The dataset used for training and evaluation contains 60 distinct classes of features commonly found in geographical data. The classes include various types of vehicles, vessels, buildings, and other structures. Below is the list of all feature classes:

1. Fixed-wing Aircraft
2. Small Aircraft
3. Passenger/Cargo Plane
4. Helicopter
5. Passenger Vehicle
6. Small Car
7. Bus
8. Pickup Truck
9. Utility Truck
10. Truck
11. Cargo Truck
12. Truck Tractor w/ Box Trailer
13. Truck Tractor
14. Trailer
15. Truck Tractor w/ Flatbed Trailer
16. Truck Tractor w/ Liquid Tank
17. Crane Truck
18. Railway Vehicle
19. Passenger Car
20. Cargo/Container Car
21. Flat Car
22. Tank Car
23. Locomotive
24. Maritime Vessel
25. Motorboat
26. Sailboat
27. Tugboat
28. Barge
29. Fishing Vessel
30. Ferry
31. Yacht
32. Container Ship
33. Oil Tanker
34. Engineering Vehicle
35. Tower Crane
36. Container Crane
37. Reach Stacker
38. Straddle Carrier
39. Mobile Crane
40. Dump Truck
41. Haul Truck
42. Scraper/Tractor
43. Front Loader/Bulldozer
44. Excavator
45. Cement Mixer
46. Ground Grader
47. Hut/Tent
48. Shed
49. Building
50. Aircraft Hangar
51. Damaged Building
52. Facility
53. Construction Site
54. Vehicle Lot
55. Helipad
56. Storage Tank
57. Shipping Container Lot
58. Shipping Container
59. Pylon
60. Tower

## Model Architecture

The models are based on the YOLOv8 architecture, which is known for its speed and accuracy in object detection tasks. The `YOLOv8s` and `YOLOv8n` versions are specifically optimized for smaller sizes, making them suitable for deployment in resource-constrained environments.

### YOLOv8s and YOLOv8n

- **YOLOv8s**: A smaller, faster version of YOLOv8 designed for scenarios where computational efficiency is a priority.
- **YOLOv8n**: An even smaller and more efficient variant, suitable for very limited computational resources.

## Training

The models were trained on the dataset using the following parameters:

- **Number of epochs**: 3 (sample models for demonstration purposes)

## Usage

To use the model for feature extraction:

1. Clone the repository:
   ```bash
   git clone https://github.com/Rohan-ingle/GIS-Feature-Extrator.git
