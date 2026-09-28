// ================================================================
// Bengaluru ML-Powered Traffic Route Optimizer — Frontend Controller
// Uses /api/predict_route for per-edge congestion coloring
// OSRM road-following paths + Green/Yellow/Red segment rendering
// Fleet Routing: telemetry loop, car animation, deviation detection
// ================================================================

// ── Session ID (generated once per page load) ───────────────────
const SESSION_ID = 'ses-' + Math.random().toString(36).substring(2, 10) +
                   '-' + Date.now().toString(36);
console.log('[FLEET] Session ID:', SESSION_ID);

// ── Map Initialization ──────────────────────────────────────────
const map = L.map("map", {
    center: [12.9716, 77.5946],
    zoom: 12,
    zoomControl: true,
    attributionControl: false,
});

// OpenStreetMap tiles — free, no API key required
// keepBuffer + updateWhenZooming keep old tiles visible when offline
const tileLayer = L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    {
        attribution: '',
        maxZoom: 19,
        keepBuffer: 5,               // retain tiles well beyond the viewport
        updateWhenZooming: false,     // don't discard tiles mid-zoom animation
        updateWhenIdle: true,         // only request new tiles after zoom ends
    }
).addTo(map);

// When a tile fails to load (e.g. offline), keep the old tiles visible
// by preventing the error-tile placeholder from hiding the cached one.
tileLayer.on('tileerror', function (e) {
    // Set the failed tile's src to a transparent 1×1 PNG so the <img>
    // element stays in the DOM without showing a broken-image icon,
    // while the underlying (still-cached) tiles at other zoom levels
    // remain visible through Leaflet's keepBuffer mechanism.
    e.tile.src =
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';
});

// ── State ───────────────────────────────────────────────────────
let mapLayers = [];
let currentMode = "live";

// ── Fleet Routing / Navigation State ────────────────────────────
let lastRoutePath = null;       // Array of node IDs from last computed route
let lastRouteData = null;       // Full response data from /api/predict_route
let routePolylineCoords = [];   // Flattened array of [lat, lng] along the polyline
let carMarker = null;           // Leaflet marker for the simulated car
let telemetryInterval = null;   // setInterval ID for telemetry loop
let carStepIndex = 0;           // Current position along routePolylineCoords
let isNavigating = false;       // Whether navigation is active
let simulateDeviation = false;  // Flag to trigger wrong-turn on next tick

// ── DOM Elements ────────────────────────────────────────────────
const sourceSelect     = document.getElementById("source");
const destSelect       = document.getElementById("destination");
const btnRoute         = document.getElementById("btn-route");
const modeLive         = document.getElementById("mode-live");
const modePredictive   = document.getElementById("mode-predictive");
const datetimeGroup    = document.getElementById("datetime-group");
const departureInput   = document.getElementById("departure-time");
const errorMsg         = document.getElementById("error-msg");
const resultsDashboard = document.getElementById("results-dashboard");
const metricTime       = document.getElementById("metric-time");
const metricDistance    = document.getElementById("metric-distance");
const metricCost       = document.getElementById("metric-cost");
const routePath        = document.getElementById("route-path");
const sourceBadge      = document.getElementById("source-badge");

// Navigation DOM
const btnStartNav      = document.getElementById("btn-start-nav");
const btnWrongTurn     = document.getElementById("btn-wrong-turn");
const telemetryStatus  = document.getElementById("telemetry-status");
const telemetryDot     = document.getElementById("telemetry-dot");
const telemetryText    = document.getElementById("telemetry-text");

// ── Populate Dropdowns ──────────────────────────────────────────
async function loadNodes() {
    try {
        const res = await fetch("/api/nodes");
        const data = await res.json();
        const nodes = data.nodes || [];
        const optionsHtml = nodes
            .map((n) => `<option value="${n.id}">${n.name}</option>`)
            .join("");
        sourceSelect.innerHTML =
            '<option value="" disabled selected>Select origin\u2026</option>' + optionsHtml;
        destSelect.innerHTML =
            '<option value="" disabled selected>Select destination\u2026</option>' + optionsHtml;
    } catch (err) {
        console.error("Failed to load nodes:", err);
    }
}
loadNodes();

// ── Mode Toggle ─────────────────────────────────────────────────
function setMode(mode) {
    currentMode = mode;
    modeLive.classList.toggle("active", mode === "live");
    modePredictive.classList.toggle("active", mode === "predictive");
    if (mode === "predictive") {
        datetimeGroup.classList.remove("hidden");
        datetimeGroup.classList.add("visible");
        if (!departureInput.value) {
            const d = new Date();
            d.setHours(d.getHours() + 1);
            departureInput.value = d.toISOString().slice(0, 16);
        }
    } else {
        datetimeGroup.classList.remove("visible");
        datetimeGroup.classList.add("hidden");
    }
}
modeLive.addEventListener("click", () => setMode("live"));
modePredictive.addEventListener("click", () => setMode("predictive"));

// ── Sliding Panel Toggle ────────────────────────────────────────
const toggleBtn = document.getElementById("toggle-panel");
const controlBody = document.getElementById("control-body");

toggleBtn.addEventListener("click", () => {
    controlBody.classList.toggle("collapsed");
    const isCollapsed = controlBody.classList.contains("collapsed");
    toggleBtn.style.transform = isCollapsed ? "rotate(180deg)" : "rotate(0deg)";
});

// ── Clear Map ───────────────────────────────────────────────────
function clearMap() {
    mapLayers.forEach((layer) => map.removeLayer(layer));
    mapLayers = [];
    // Also remove car marker if present
    if (carMarker) {
        map.removeLayer(carMarker);
        carMarker = null;
    }
}

// ── Format Travel Time ──────────────────────────────────────────
function formatTime(minutes) {
    const total = Math.round(minutes);
    if (total < 60) return `${total} min`;
    const hrs = Math.floor(total / 60);
    const mins = total % 60;
    return mins === 0 ? `${hrs} hr` : `${hrs} hr ${mins} min`;
}

// ── Show / Hide Error ───────────────────────────────────────────
function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.classList.remove("hidden");
}
function hideError() {
    errorMsg.classList.add("hidden");
}

// ================================================================
// OSRM Road Geometry Fetcher
// ================================================================
const osrmCache = {};

/**
 * Decode a Google-encoded polyline string into [[lat, lng], ...]
 */
function decodePolyline(encoded) {
    const points = [];
    let index = 0, lat = 0, lng = 0;
    while (index < encoded.length) {
        let b, shift = 0, result = 0;
        do {
            b = encoded.charCodeAt(index++) - 63;
            result |= (b & 0x1f) << shift;
            shift += 5;
        } while (b >= 0x20);
        lat += (result & 1) ? ~(result >> 1) : (result >> 1);
        shift = 0; result = 0;
        do {
            b = encoded.charCodeAt(index++) - 63;
            result |= (b & 0x1f) << shift;
            shift += 5;
        } while (b >= 0x20);
        lng += (result & 1) ? ~(result >> 1) : (result >> 1);
        points.push([lat / 1e5, lng / 1e5]);
    }
    return points;
}

/**
 * Fetch road-following geometry from OSRM (public demo server).
 * Returns [[lat, lng], ...] or null on failure.
 */
async function fetchRoadGeometry(lat1, lng1, lat2, lng2) {
    const cacheKey = `${lat1},${lng1}-${lat2},${lng2}`;
    if (osrmCache[cacheKey]) return osrmCache[cacheKey];

    const url =
        `https://router.project-osrm.org/route/v1/driving/` +
        `${lng1},${lat1};${lng2},${lat2}?overview=full&geometries=polyline`;

    try {
        const response = await fetch(url);
        if (!response.ok) return null;
        const data = await response.json();
        if (data.code !== "Ok" || !data.routes || data.routes.length === 0)
            return null;
        const coords = decodePolyline(data.routes[0].geometry);
        osrmCache[cacheKey] = coords;
        return coords;
    } catch (err) {
        console.warn("OSRM fetch failed:", err.message);
        return null;
    }
}

/**
 * Squared distance between two [lat, lng] points (for fast comparison only).
 */
function sqDist(a, b) {
    const dlat = a[0] - b[0], dlng = a[1] - b[1];
    return dlat * dlat + dlng * dlng;
}

/**
 * Ensure a polyline array runs from (lat1, lng1) → (lat2, lng2).
 *
 * The graph is undirected, so pre-baked road_coords may be stored in
 * the reverse direction.  We check whether the polyline's first point
 * is closer to the segment's origin or destination and reverse if needed.
 */
function orientCoords(coords, lat1, lng1, lat2, lng2) {
    const origin = [lat1, lng1];
    const first  = coords[0];
    const last   = coords[coords.length - 1];

    // If the first point of the polyline is closer to the *destination*
    // than to the origin, the array is backwards — reverse it.
    if (sqDist(first, origin) > sqDist(last, origin)) {
        return coords.slice().reverse();
    }
    return coords;
}

/**
 * Get road-following coords for an edge.
 * Priority: 1) pre-baked road_coords  2) live OSRM  3) straight line
 *
 * All returned arrays are guaranteed to run origin → destination.
 */
async function getEdgeCoords(lat1, lng1, lat2, lng2, prebakedCoords) {
    // 1. Use pre-baked road coordinates from the server (works offline)
    if (prebakedCoords && Array.isArray(prebakedCoords) && prebakedCoords.length >= 2) {
        return orientCoords(prebakedCoords, lat1, lng1, lat2, lng2);
    }
    // 2. Try live OSRM fetch
    const roadCoords = await fetchRoadGeometry(lat1, lng1, lat2, lng2);
    if (roadCoords && roadCoords.length >= 2) {
        return orientCoords(roadCoords, lat1, lng1, lat2, lng2);
    }
    // 3. Fallback: straight line
    return [[lat1, lng1], [lat2, lng2]];
}


// ================================================================
// ROUTE REQUEST — uses /api/predict_route
// ================================================================
btnRoute.addEventListener("click", async () => {
    hideError();
    resultsDashboard.classList.add("hidden");

    // Stop any active navigation when computing a new route
    stopNavigation();

    const source = sourceSelect.value;
    const destination = destSelect.value;

    if (!source) return showError("Please select an origin.");
    if (!destination) return showError("Please select a destination.");
    if (source === destination)
        return showError("Origin and destination must differ.");

    // Update button state
    btnRoute.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <line x1="12" y1="2" x2="12" y2="6"></line><line x1="12" y1="18" x2="12" y2="22"></line>
            <line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line>
            <line x1="2" y1="12" x2="6" y2="12"></line><line x1="18" y1="12" x2="22" y2="12"></line>
            <line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line>
        </svg>
        Predicting Traffic\u2026`;
    btnRoute.classList.add("loading");

    const payload = {
        source_id: source,
        destination_id: destination,
        mode: currentMode
    };
    if (currentMode === "predictive" && departureInput.value) {
        payload.departure_time = departureInput.value;
    }

    try {
        const res = await fetch("/api/predict_route", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok) {
            showError(data.error || "An error occurred.");
            return;
        }

        // Store route data for navigation
        lastRouteData = data;
        lastRoutePath = data.path || [];

        // Render colored segments on the map
        await renderSegments(data);

        // Update dashboard metrics
        metricTime.textContent = formatTime(data.travel_time_min);
        metricDistance.textContent = `${data.total_distance_km} km`;
        metricCost.textContent = `\u20B9${data.estimated_cost_inr}`;

        // Route path with colored node pills
        const path = data.path || [];
        const pathHtml = path
            .map((node, i) => {
                let cls = "path-node";
                if (i === 0) cls += " origin";
                else if (i === path.length - 1) cls += " destination";
                const arrow = i < path.length - 1
                    ? '<span class="path-arrow">\u2192</span>'
                    : "";
                return `<span class="${cls}">${node}</span>${arrow}`;
            })
            .join("");
        routePath.innerHTML = pathHtml;

        // Data source badge
        const modelName = data.model_name || "Heuristic";
        const peakLabel = data.is_peak ? "Peak Hours" : "Off-Peak";
        sourceBadge.innerHTML =
            `<span class="badge-dot"></span>${modelName} · ${peakLabel}`;

        resultsDashboard.classList.remove("hidden");

        // Reset navigation UI for new route
        resetNavUI();

    } catch (err) {
        console.error("Route error:", err);
        showError("Failed to connect to the server.");
    } finally {
        btnRoute.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
            Find Optimal Route`;
        btnRoute.classList.remove("loading");
    }
});


// ================================================================
// MAP RENDERING — Per-segment congestion coloring
// ================================================================

async function renderSegments(data) {
    clearMap();
    routePolylineCoords = []; // Reset for navigation

    const segments = data.segments || [];
    const path = data.path || [];
    const allNodes = [];

    // First fetch all node positions for the entire graph (for markers)
    let graphNodes = [];
    try {
        const nodesRes = await fetch("/api/nodes");
        const nodesData = await nodesRes.json();
        graphNodes = nodesData.nodes || [];
    } catch (e) {
        console.warn("Could not fetch nodes for markers:", e);
    }

    // Build a lookup of nodes on the path
    const pathNodeSet = new Set(path);
    const nodeDict = {};
    graphNodes.forEach(n => { nodeDict[n.id] = n; });

    // ── Draw each segment with its congestion color ─────────────
    const segmentPromises = segments.map(async (seg, idx) => {
        const [[lat1, lng1], [lat2, lng2]] = seg.coordinates;
        const latlngs = await getEdgeCoords(lat1, lng1, lat2, lng2, seg.road_coords);
        const color = seg.color;

        // Glow / shadow layer for depth
        const glow = L.polyline(latlngs, {
            color: color,
            weight: 14,
            opacity: 0.12,
            lineCap: "round",
            lineJoin: "round",
        }).addTo(map);
        mapLayers.push(glow);

        // Main colored route line
        const line = L.polyline(latlngs, {
            color: color,
            weight: 7,
            opacity: 0.9,
            lineCap: "round",
            lineJoin: "round",
        }).addTo(map);

        // Rich tooltip
        const tooltipContent =
            `<strong>${seg.from} → ${seg.to}</strong><br>` +
            `${seg.distance_km} km · ${seg.time_min} min<br>` +
            `Speed: ${seg.speed_kmh} km/h<br>` +
            `<span style="color:${color};font-weight:700">\u25CF ${seg.congestion}</span>`;

        line.bindTooltip(tooltipContent, {
            sticky: true,
            className: "segment-tooltip",
        });

        mapLayers.push(line);

        // Collect coordinates for bounds AND for navigation animation
        allNodes.push([lat1, lng1]);
        allNodes.push([lat2, lng2]);

        // Return the resolved coords so we can build the animation path
        return { idx, latlngs };
    });

    const resolvedSegments = await Promise.all(segmentPromises);

    // Build the full polyline coordinate list (in order) for car animation
    resolvedSegments
        .sort((a, b) => a.idx - b.idx)
        .forEach((seg, i) => {
            if (i === 0) {
                routePolylineCoords.push(...seg.latlngs);
            } else {
                // Skip the first point of subsequent segments to avoid duplicates
                routePolylineCoords.push(...seg.latlngs.slice(1));
            }
        });

    console.log(`[FLEET] Route polyline: ${routePolylineCoords.length} coordinates collected`);
    if (routePolylineCoords.length >= 2) {
        const first = routePolylineCoords[0];
        const last  = routePolylineCoords[routePolylineCoords.length - 1];
        console.log(`[FLEET]   Start: [${first[0].toFixed(4)}, ${first[1].toFixed(4)}]`);
        console.log(`[FLEET]   End:   [${last[0].toFixed(4)}, ${last[1].toFixed(4)}]`);
    }

    // ── Draw node markers ───────────────────────────────────────
    graphNodes.forEach((node) => {
        const isOnPath = pathNodeSet.has(node.id);
        const isOrigin = node.id === path[0];
        const isDestination = node.id === path[path.length - 1];

        let fillColor = "rgba(163, 158, 149, 0.35)";  // off-path: warm gray
        let radius = 4;
        let weight = 1;
        let fillOpacity = 0.4;

        if (isOrigin) {
            fillColor = "#4A9B6B";   // muted green
            radius = 9;
            weight = 3;
            fillOpacity = 1;
        } else if (isDestination) {
            fillColor = "#B84C3E";   // muted red
            radius = 9;
            weight = 3;
            fillOpacity = 1;
        } else if (isOnPath) {
            fillColor = "#E86C30";   // orange accent
            radius = 6;
            weight = 2;
            fillOpacity = 0.9;
        }

        const marker = L.circleMarker([node.lat, node.lng], {
            radius: radius,
            fillColor: fillColor,
            color: isOnPath ? "#FFFFFF" : "rgba(255,255,255,0.2)",
            weight: weight,
            opacity: 1,
            fillOpacity: fillOpacity,
        })
            .bindTooltip(node.name, {
                permanent: isOnPath,
                direction: "top",
                className: "node-label",
                offset: [0, -8],
            })
            .addTo(map);

        mapLayers.push(marker);
    });

    // ── Fit bounds to the route ─────────────────────────────────
    if (allNodes.length > 0) {
        map.fitBounds(L.latLngBounds(allNodes), { padding: [100, 100] });
    }
}


// ================================================================
// FLEET ROUTING — Navigation & Telemetry
// ================================================================

/**
 * Reset navigation UI to the "ready" state.
 */
function resetNavUI() {
    btnStartNav.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        Start Navigation`;
    btnStartNav.classList.remove("navigating");
    btnWrongTurn.classList.add("hidden");
    telemetryStatus.classList.add("hidden");
    telemetryStatus.classList.remove("on-track", "deviated");
    isNavigating = false;
    simulateDeviation = false;
    carStepIndex = 0;
}

/**
 * Stop an active navigation session.
 */
function stopNavigation() {
    if (telemetryInterval) {
        clearInterval(telemetryInterval);
        telemetryInterval = null;
    }
    if (carMarker) {
        map.removeLayer(carMarker);
        carMarker = null;
    }
    resetNavUI();
}

/**
 * Start the navigation simulation.
 */
async function startNavigation() {
    if (!lastRoutePath || lastRoutePath.length < 2) {
        console.warn("[FLEET] No route to navigate.");
        return;
    }
    if (routePolylineCoords.length < 2) {
        console.warn("[FLEET] No polyline coordinates for animation.");
        return;
    }

    // 1. Register the trip on the backend
    try {
        const tripRes = await fetch("/api/start_trip", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: SESSION_ID,
                path: lastRoutePath,
            }),
        });
        const tripData = await tripRes.json();
        console.log("[FLEET] Trip registered:", tripData);
    } catch (err) {
        console.error("[FLEET] Failed to register trip:", err);
        return;
    }

    // 2. Place the car marker at the start of the route
    isNavigating = true;
    carStepIndex = 0;
    simulateDeviation = false;

    const startPos = routePolylineCoords[0];
    const carIcon = L.divIcon({
        className: "car-marker-icon",
        iconSize: [16, 16],
        iconAnchor: [8, 8],
    });
    carMarker = L.marker(startPos, { icon: carIcon, zIndex: 9999 }).addTo(map);

    // Update UI
    btnStartNav.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>
        </svg>
        Stop Navigation`;
    btnStartNav.classList.add("navigating");
    btnWrongTurn.classList.remove("hidden");
    telemetryStatus.classList.remove("hidden");
    telemetryStatus.classList.remove("deviated");
    telemetryStatus.classList.add("on-track");
    telemetryText.textContent = "Telemetry active \u00b7 On track";

    // 3. Start the telemetry polling loop (every 3 seconds)
    telemetryInterval = setInterval(() => telemetryTick(), 3000);
}

/**
 * A single tick of the telemetry loop.
 * Moves the car, sends position to backend, updates UI.
 */
async function telemetryTick() {
    if (!isNavigating || !carMarker) return;

    let currentPos;

    if (simulateDeviation) {
        // Move the car 1 km away from the route (perpendicular offset)
        const basePos = routePolylineCoords[Math.min(carStepIndex, routePolylineCoords.length - 1)];
        // Offset ~0.009 degrees ≈ ~1 km at Bengaluru's latitude
        currentPos = [basePos[0] + 0.009, basePos[1] + 0.009];
        simulateDeviation = false; // One-shot: only deviate for this tick

        // Visual feedback on the car marker
        const el = carMarker.getElement();
        if (el) el.classList.add("deviated");
    } else {
        // Advance along the route polyline
        // Move 3-5 points per tick for a smooth simulation
        const stepsPerTick = Math.max(1, Math.floor(routePolylineCoords.length / 30));
        carStepIndex = Math.min(carStepIndex + stepsPerTick, routePolylineCoords.length - 1);
        currentPos = routePolylineCoords[carStepIndex];

        // Reset deviated visual if back on track
        const el = carMarker.getElement();
        if (el) el.classList.remove("deviated");
    }

    // Move the marker
    carMarker.setLatLng(currentPos);

    // Send telemetry to backend
    try {
        const res = await fetch("/api/telemetry", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: SESSION_ID,
                lat: currentPos[0],
                lng: currentPos[1],
            }),
        });
        const data = await res.json();
        console.log(`[FLEET] Telemetry response: ${data.status} | pos: [${currentPos[0].toFixed(4)}, ${currentPos[1].toFixed(4)}]`);

        // Update telemetry status UI
        if (data.status === "on_track") {
            telemetryStatus.classList.remove("deviated");
            telemetryStatus.classList.add("on-track");
            telemetryText.textContent = `Telemetry active \u00b7 On track (step ${carStepIndex}/${routePolylineCoords.length - 1})`;
        } else if (data.status === "deviated") {
            telemetryStatus.classList.remove("on-track");
            telemetryStatus.classList.add("deviated");
            telemetryText.textContent = "\u26a0 Deviation detected! Vehicle off-route";
        }
    } catch (err) {
        console.error("[FLEET] Telemetry ping failed:", err);
    }

    // Check if the car has reached the end of the route
    if (carStepIndex >= routePolylineCoords.length - 1 && !simulateDeviation) {
        console.log("[FLEET] Navigation complete \u2014 reached destination.");
        telemetryText.textContent = "\u2713 Arrived at destination";
        telemetryStatus.classList.remove("deviated");
        telemetryStatus.classList.add("on-track");
        clearInterval(telemetryInterval);
        telemetryInterval = null;

        // Keep the car at destination but disable wrong turn
        btnWrongTurn.classList.add("hidden");
        btnStartNav.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="5 3 19 12 5 21 5 3"/>
            </svg>
            Start Navigation`;
        btnStartNav.classList.remove("navigating");
        isNavigating = false;
    }
}

// ── Button Event Listeners ──────────────────────────────────────
btnStartNav.addEventListener("click", () => {
    if (isNavigating) {
        stopNavigation();
    } else {
        startNavigation();
    }
});

btnWrongTurn.addEventListener("click", () => {
    if (!isNavigating) return;
    simulateDeviation = true;
    console.log("[FLEET] Wrong turn simulation queued \u2014 will deviate on next telemetry tick.");
});
