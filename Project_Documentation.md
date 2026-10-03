# AI-Based Proactive Traffic Prediction & Fleet Routing System
## Comprehensive System Architecture & Flow Documentation

This document explains the end-to-end flow of the traffic routing project, detailing the mathematics, state management, and algorithmic decisions that power the application from the moment a user requests a route to the moment they arrive at their destination.

---

### 1. Overview & Data Structures
The system operates as a hybrid traffic routing engine designed to manage 3-5 concurrent vehicles in real-time. It uses a 30-node directed graph representing the Bengaluru road network (stored via `networkx`). 
The goal of the system is not just to find the fastest route for a single user, but to proactively manage the flow of the entire "fleet" of users so they do not inadvertently create new traffic jams by all taking the same road.

---

### 2. Route Prediction & Weight Assignment (`/api/predict_route`)
When a user selects a Source and Destination and requests a route, the backend must assign a "weight" (estimated travel time in minutes) to every single road (edge) in the Bengaluru graph. 

#### The 3-Tier Weighting Engine
The base speed of an edge is determined by one of three tiers:
1. **Live Mode (TomTom API):** The server queries the TomTom Traffic Flow API using the exact midpoint coordinates of the road segment. It extracts the `currentSpeed`.
2. **Predictive Mode (Machine Learning):** If the user is planning a future trip, or if TomTom is unavailable, the system passes features (Hour of day, Day of week, Peak hour flags, distance) into a trained `HistGradientBoostingRegressor` model. The model outputs a predicted travel time, which is converted into a congestion multiplier.
3. **Heuristic Fallback:** If the ML model is missing, the system uses a hardcoded Bengaluru speed profile (e.g., assuming 12 km/h during the 6:00 PM evening peak).

#### Load Balancing (Anti-Herding)
Standard routing algorithms suffer from "herding"—they send everyone to the fastest route, which instantly clogs it. Your system solves this using proactive load balancing.
- The server maintains an in-memory dictionary called `edge_counters`. This tracks exactly how many active vehicles are currently assigned to drive over every specific edge `(u, v)`.
- During the weight calculation loop, the server checks `edge_counters`. If an edge currently has active vehicles on it, the server applies an exponential penalty: 
  `predicted_time = predicted_time * (1.15 ** fleet_count)`
- **The Result:** If 2 users are already routed onto the main highway, the algorithm artificially inflates the highway's travel time by ~32%. When User #3 requests a route, Dijkstra's algorithm will naturally calculate that a parallel, slightly longer side-street is now "faster", forcing the traffic to spread out evenly across the city grid.

#### Shortest Path Calculation
With all weights adjusted for both traffic and fleet load, `networkx.dijkstra_path` is executed to find the mathematically optimal path.

---

### 3. Session Initialization (`/api/start_trip`)
Once the route is displayed, the user taps "Start Navigation".
1. **Frontend Session ID:** The Javascript generates a cryptographically secure `session_id` using `crypto.randomUUID()`. This ensures zero collision risk between concurrent drivers.
2. **Backend Registration:** The frontend sends the chosen route to `/api/start_trip`. 
3. **Thread-Safe State Mutation:** Flask operates in a multi-threaded environment. To prevent two users from corrupting the dictionaries simultaneously, the backend acquires a `threading.Lock()` (`_fleet_lock`). 
4. **Slot Reservation:** The system iterates over the user's route, adds the `session_id` to `active_sessions`, and strictly increments (`+1`) the `edge_counters` for those specific roads, officially reserving the vehicle's slot for the load balancer.

---

### 4. Live Telemetry & Hardware Integration
The frontend takes over to manage the physical driving experience.

#### Hardware GPS & Wake Lock
- The app requests a Screen Wake Lock (`navigator.wakeLock.request`) to prevent the mobile device from falling asleep during navigation.
- It invokes `navigator.geolocation.watchPosition(..., { enableHighAccuracy: true })` to force the mobile device to use its physical GPS chip rather than relying on coarse cell-tower triangulation.

#### The Firehose Throttle
A physical GPS chip fires location updates incredibly fast (1Hz or more). If 5 users are driving, this translates to hundreds of network requests a minute, which would overwhelm a standard Flask server.
- The frontend implements a time-based debounce (`lastTelemetryTime`). 
- While the GPS cursor updates on the map instantly at 60 frames-per-second, the actual POST request to the backend is strictly throttled to fire **at most once every 2.5 seconds**.

---

### 5. Backend Deviation Math (`/api/telemetry`)
Every 2.5 seconds, the server receives a `{ lat, lng }` ping. It must verify if the user is actually driving on their assigned route.

Because roads are curved and GPS signals bounce off buildings, the server cannot demand that the user be exactly on top of a graph node. Instead, it calculates spatial proximity using the `_point_to_segment_dist()` function:
1. It looks up the mathematical straight line drawn between the origin node and destination node of the edge.
2. It converts all Lat/Lng coordinates into radians.
3. It uses an **equirectangular projection** to project the vehicle's GPS coordinate onto that mathematical line.
4. It calculates the perpendicular, cross-track distance in meters.
5. **The Stop Condition:** If the perpendicular distance from the car to the road is strictly under **100 meters**, the server returns `"on_track"`. If the car strays beyond 100 meters, it is flagged as `"deviated"`.

---

### 6. Graceful Termination (`/api/end_trip`)
The system must free up network capacity when a user is done. A trip ends in three ways:
1. **Success (Proximity):** The frontend constantly checks the distance to the final destination node. If the distance drops below **50 meters**, it auto-completes.
2. **Deviation:** The backend deviation math flags a wrong turn.
3. **Cancellation:** The user taps "Stop Navigation".

In all scenarios, a final call is made to `/api/end_trip`. The backend acquires the thread lock, looks up the user's assigned edges, and decrements (`-1`) the `edge_counters`. The `session_id` is erased, freeing up the slots so future drivers are no longer penalized for roads that are now empty.
