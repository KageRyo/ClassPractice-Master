# National Chung Cheng University

## Introduction to Deep Learning  
**Mid-Term Programming Exam (Take-Home)**  
Period: 2026-04-23 ~ 2026-05-07 (deadline: 23:59:59) — 50% of total

---

## Overview
You are provided a time-series forecasting problem centered on restaurant visitors. The data comes from two separate systems:

- Hot Pepper Gourmet (hpg): similar to Yelp — users can search restaurants and make online reservations.  
- AirREGI / Restaurant Board (air): reservation control and cash register system.

Use reservations, visits, and other information to forecast future restaurant visitor totals on given dates. The dataset covers 2016 until April 2017. Split data into:
- Training: full year of 2016
- Testing: the data provided for 2017

Notes:
- Some days in the test set have restaurants closed (no visitors); these days are ignored in scoring.
- The training set omits days when restaurants were closed.

---

## File descriptions
Files are prefixed with `air_` or `hpg_` to indicate source. Each restaurant has a unique `air_store_id` and `hpg_store_id`. Not all restaurants appear in both systems. Latitude/longitude are approximate.

### Reservation files
- `air_reserve.csv`  
    Contains reservations made in the air system.  
    Fields:
    - `air_store_id` — restaurant id in air system
    - `visit_datetime` — datetime of the reservation (visit date/time)
    - `reserve_datetime` — datetime when reservation was created
    - `reserve_visitors` — number of visitors for that reservation

- `hpg_reserve.csv`  
    Contains reservations made in the hpg system.  
    Fields:
    - `hpg_store_id` — restaurant id in hpg system
    - `visit_datetime`
    - `reserve_datetime`
    - `reserve_visitors`

### Store info files
- `air_store_info.csv`  
    Information about select air restaurants. Columns:
    - `air_store_id`
    - `air_genre_name`
    - `air_area_name`
    - `latitude`
    - `longitude`  
    Note: latitude/longitude correspond to the area the store belongs to (approximate).

- `hpg_store_info.csv`  
    Information about select hpg restaurants. Columns:
    - `hpg_store_id`
    - `hpg_genre_name`
    - `hpg_area_name`
    - `latitude`
    - `longitude`  
    Note: latitude/longitude correspond to the area the store belongs to (approximate).

### Mapping and visit data
- `store_id_relation.csv`  
    Mapping between systems for stores that appear in both:
    - `hpg_store_id`
    - `air_store_id`

- `air_visit_data.csv`  
    Historical visit data for air restaurants. Fields:
    - `air_store_id`
    - `visit_date` — the date
    - `visitors` — number of visitors on that date

---

## Submission
Submit your model code and fill in the following information in your report:

Model Information:
- Number of layers: _________________
- Number of units in each layer: __________________________________
- Activation functions used: _____________________________________
- Loss function: __________________________________________
- Cost function: ____________________________________________
- Training epochs: _________________________________________
- Training accuracy: __________%
- Testing accuracy: __________%
- Optimization techniques employed: ________________________________
    ______________________________________________________________

Difference in accuracies after each optimization technique applied:
1) Optimization technique name: ____________________________  
     - Before optimization: Training/Testing Accuracies = _________/________  
     - After optimization: Training/Testing Accuracies = ________/________  
     - Any other changes: __________________________________________

2) Optimization technique name: ____________________________  
     - Before optimization: Training/Testing Accuracies = _________/________  
     - After optimization: Training/Testing Accuracies = ________/________  
     - Any other changes: __________________________________________

3) Optimization technique name: ____________________________  
     - Before optimization: Training/Testing Accuracies = _________/________  
     - After optimization: Training/Testing Accuracies = ________/________  
     - Any other changes: __________________________________________

Anything special about your model:  
____________________________________________________________________________________

Comments on the course:  
____________________________________________________________________________________

