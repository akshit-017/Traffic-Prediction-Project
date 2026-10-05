// ================================================================
// Bengaluru ML-Powered Traffic Route Optimizer — Frontend Controller
// Uses /api/predict_route for per-edge congestion coloring
// OSRM road-following paths + Green/Yellow/Red segment rendering
// Fleet Routing: telemetry loop, car animation, deviation detection
// ================================================================

// ── Session ID (unique per navigation trip, regenerated on each Start) ──
// Using `let` so we can assign a fresh ID for every new trip.
// This prevents trip-2 from touching trip-1's edge counters on /api/start_trip.
let SESSION_ID = _generateSessionId();
function _generateSessionId() {
    // crypto.randomUUID() is cryptographically unique — zero collision
    // risk across concurrent users.  Fallback covers older browsers.
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
        return 'ses-' + crypto.randomUUID();
    }
    // Fallback: combine high-res timestamp + random segment
    return 'ses-' + Date.now().toString(36) + '-' +
           Math.random().toString(36).substring(2, 10) + '-' +
           Math.random().toString(36).substring(2, 6);
}
console.log('[FLEET] Initial Session ID:', SESSION_ID);

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
let carMarker = null;           // Leaflet marker for the live GPS dot
let telemetryInterval = null;   // setInterval ID for SIMULATION-ONLY fallback
let geoWatchId = null;          // navigator.geolocation.watchPosition() ID
let wakeLockSentinel = null;    // Screen Wake Lock sentinel
let carStepIndex = 0;           // Current position along routePolylineCoords (sim only)
let isNavigating = false;       // Whether navigation is active
let simulateDeviation = false;  // Flag to trigger wrong-turn on next tick
let destinationCoords = null;   // [lat, lng] of the route destination for proximity check
let lastTelemetryTime = 0;      // Timestamp of last telemetry ping

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
    const destination = [lat2, lng2];
    const first  = coords[0];
    const last   = coords[coords.length - 1];

    // Use both origin and destination to determine direction.
    // Compare: is (first→origin + last→dest) shorter than (first→dest + last→origin)?
    // If not, the polyline is reversed.
    const forwardCost  = sqDist(first, origin) + sqDist(last, destination);
    const reverseCost  = sqDist(first, destination) + sqDist(last, origin);

    if (reverseCost < forwardCost) {
        return coords.slice().reverse();
    }
    return coords;
}

/**
 * Compute the great-circle distance in km between two [lat, lng] points.
 */
function haversineDist(a, b) {
    const R = 6371; // km
    const dLat = (b[0] - a[0]) * Math.PI / 180;
    const dLng = (b[1] - a[1]) * Math.PI / 180;
    const lat1r = a[0] * Math.PI / 180;
    const lat2r = b[0] * Math.PI / 180;
    const x = Math.sin(dLat / 2) ** 2 +
              Math.cos(lat1r) * Math.cos(lat2r) * Math.sin(dLng / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x));
}

/**
 * Compute the total length of a polyline in km.
 */
function polylineLength(coords) {
    let total = 0;
    for (let i = 0; i < coords.length - 1; i++) {
        total += haversineDist(coords[i], coords[i + 1]);
    }
    return total;
}

/**
 * Clamp a polyline so it doesn't extend far beyond the origin or
 * destination.  OSRM road_coords often overshoot the graph nodes
 * because OSRM snaps to the nearest road segment, which may be
 * past the intended node.  This trims the overshooting tail/head.
 *
 * Strategy: walk from start→end and stop once the remaining distance
 * to the destination starts increasing (we've passed it).
 * Similarly for the beginning relative to the origin.
 */
function clampToEndpoints(coords, origin, destination) {
    if (coords.length <= 3) return coords;

    // Trim the end: stop where distance to destination starts increasing
    let endIdx = coords.length - 1;
    let minDistToDest = haversineDist(coords[endIdx], destination);
    for (let i = coords.length - 2; i >= Math.max(0, coords.length - 80); i--) {
        const d = haversineDist(coords[i], destination);
        if (d < minDistToDest + 0.01) {
            // Still approaching or at minimum — keep going back
            minDistToDest = Math.min(minDistToDest, d);
        } else if (d > minDistToDest + 0.15) {
            // We've gone past the destination — trim here
            endIdx = i + 1;
            break;
        }
    }

    // Trim the start: stop where distance to origin starts increasing
    let startIdx = 0;
    let minDistToOrigin = haversineDist(coords[0], origin);
    for (let i = 1; i <= Math.min(coords.length - 1, 80); i++) {
        const d = haversineDist(coords[i], origin);
        if (d < minDistToOrigin + 0.01) {
            minDistToOrigin = Math.min(minDistToOrigin, d);
        } else if (d > minDistToOrigin + 0.15) {
            startIdx = Math.max(0, i - 1);
            break;
        }
    }

    if (startIdx > 0 || endIdx < coords.length - 1) {
        const clamped = coords.slice(startIdx, endIdx + 1);
        if (clamped.length >= 2) return clamped;
    }

    return coords;
}

/**
 * Get road-following coords for an edge.
 * Priority: 1) pre-baked road_coords  2) live OSRM  3) straight line
 *
 * All returned arrays are guaranteed to run origin → destination.
 * Coordinates are clamped so they don't overshoot the endpoints,
 * and overly circuitous paths (>3x direct distance) are rejected.
 */
async function getEdgeCoords(lat1, lng1, lat2, lng2, prebakedCoords) {
    const origin = [lat1, lng1];
    const destination = [lat2, lng2];
    const directDist = haversineDist(origin, destination);

    // Helper: validate and clamp a set of road coordinates
    function processCoords(raw) {
        let coords = orientCoords(raw, lat1, lng1, lat2, lng2);

        // Reject paths that are absurdly circuitous (>3x direct distance)
        const routeDist = polylineLength(coords);
        if (directDist > 0.1 && routeDist / directDist > 3.5) {
            console.warn(
                `[ROUTE] Rejecting circuitous path: ` +
                `${routeDist.toFixed(1)}km vs ${directDist.toFixed(1)}km direct ` +
                `(${(routeDist / directDist).toFixed(1)}x)`
            );
            return null; // fall through to straight line
        }

        // Clamp to endpoints (trim overshooting OSRM coords)
        coords = clampToEndpoints(coords, origin, destination);

        // Ensure the polyline starts/ends very close to origin/destination
        // by prepending/appending the node coordinates if needed
        if (coords.length >= 2) {
            if (haversineDist(coords[0], origin) > 0.05) {
                coords = [origin, ...coords];
            }
            if (haversineDist(coords[coords.length - 1], destination) > 0.05) {
                coords = [...coords, destination];
            }
        }

        return coords;
    }

    // 1. Use pre-baked road coordinates from the server (works offline)
    if (prebakedCoords && Array.isArray(prebakedCoords) && prebakedCoords.length >= 2) {
        const result = processCoords(prebakedCoords);
        if (result && result.length >= 2) return result;
    }
    // 2. Try live OSRM fetch
    const roadCoords = await fetchRoadGeometry(lat1, lng1, lat2, lng2);
    if (roadCoords && roadCoords.length >= 2) {
        const result = processCoords(roadCoords);
        if (result && result.length >= 2) return result;
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

    // Stop any active navigation when computing a new route.
    // Pass reason='rerouted' so /api/end_trip releases old edge_counters
    // BEFORE the new route is computed (prevents counter double-counting).
    stopNavigation('rerouted');

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
        alert("Route calculation failed. Please try different points.");
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

/**
 * Find how many trailing points of `prevCoords` overlap with leading
 * points of `nextCoords`.  Two points are "the same" when they are
 * within ~11 m of each other (0.0001° ≈ 11 m at Bengaluru's latitude).
 *
 * Returns the number of leading points to trim from `nextCoords`.
 */
function findOverlapCount(prevCoords, nextCoords) {
    if (!prevCoords || !nextCoords || prevCoords.length < 2 || nextCoords.length < 2) {
        return 0;
    }

    const THRESH = 0.0001; // ~11 m tolerance for matching coordinates

    function ptClose(a, b) {
        return Math.abs(a[0] - b[0]) < THRESH && Math.abs(a[1] - b[1]) < THRESH;
    }

    // Strategy: walk backwards from the end of prevCoords and try to
    // find the longest prefix of nextCoords that matches a suffix of
    // prevCoords.  The overlap happens because OSRM routes both edges
    // through the same roads near their shared node.
    //
    // We look for the first point in nextCoords that matches a point
    // near the end of prevCoords, then verify the sequences align.

    const maxCheck = Math.min(prevCoords.length, nextCoords.length, 200);
    let bestOverlap = 0;

    // For each candidate start in nextCoords (up to maxCheck points),
    // check if nextCoords[0..k] matches prevCoords[end-k..end].
    for (let startInPrev = prevCoords.length - 1;
         startInPrev >= Math.max(0, prevCoords.length - maxCheck);
         startInPrev--) {
        if (ptClose(prevCoords[startInPrev], nextCoords[0])) {
            // Found a candidate match — verify how far it extends
            let matchLen = 1;
            while (
                matchLen < nextCoords.length &&
                startInPrev + matchLen < prevCoords.length &&
                ptClose(prevCoords[startInPrev + matchLen], nextCoords[matchLen])
            ) {
                matchLen++;
            }
            // We want the overlap that reaches closest to the end of prevCoords
            // (i.e. startInPrev + matchLen should be near prevCoords.length)
            if (matchLen >= 2 && startInPrev + matchLen >= prevCoords.length - 2) {
                bestOverlap = Math.max(bestOverlap, matchLen);
            }
        }
    }

    return bestOverlap;
}

/**
 * Trim overlapping coordinates between consecutive segments.
 * When OSRM generates road geometry for edges A→B and B→C, they
 * often share road coordinates near node B.  This causes the same
 * road section to be drawn twice with potentially different colors.
 *
 * This function removes the overlapping leading portion of each
 * segment that was already covered by the previous segment.
 */
function trimConsecutiveOverlaps(resolvedSegments) {
    if (resolvedSegments.length <= 1) return resolvedSegments;

    const trimmed = [resolvedSegments[0]]; // first segment stays as-is

    for (let i = 1; i < resolvedSegments.length; i++) {
        const prev = trimmed[trimmed.length - 1].latlngs;
        const curr = resolvedSegments[i].latlngs;

        const overlapCount = findOverlapCount(prev, curr);

        if (overlapCount > 0) {
            console.log(
                `[ROUTE] Trimmed ${overlapCount} overlapping coords ` +
                `between segment ${i - 1} and ${i}`
            );
            // Keep at least 1 overlap point so the line connects smoothly
            const trimStart = Math.max(1, overlapCount - 1);
            trimmed.push({
                ...resolvedSegments[i],
                latlngs: curr.slice(trimStart),
            });
        } else {
            trimmed.push(resolvedSegments[i]);
        }
    }

    return trimmed;
}

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

    // ── Phase 1: Resolve all segment coordinates (in order) ─────
    const resolvedRaw = [];
    for (let idx = 0; idx < segments.length; idx++) {
        const seg = segments[idx];
        const [[lat1, lng1], [lat2, lng2]] = seg.coordinates;
        const latlngs = await getEdgeCoords(lat1, lng1, lat2, lng2, seg.road_coords);
        resolvedRaw.push({ idx, seg, latlngs });
        allNodes.push([lat1, lng1]);
        allNodes.push([lat2, lng2]);
    }

    // ── Phase 2: Trim overlapping coords between consecutive segments
    const resolvedSegments = trimConsecutiveOverlaps(resolvedRaw);

    // ── Phase 3: Render each (trimmed) segment on the map ───────
    for (const resolved of resolvedSegments) {
        const { seg, latlngs } = resolved;
        const color = seg.color;

        // Skip segments that got trimmed to nothing
        if (latlngs.length < 2) continue;

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
    }

    // ── Build navigation polyline from trimmed segments ──────────
    resolvedSegments.forEach((seg, i) => {
        if (i === 0) {
            routePolylineCoords.push(...seg.latlngs);
        } else {
            // Skip the first point of subsequent segments to avoid duplicates
            routePolylineCoords.push(...seg.latlngs.slice(1));
        }
    });

    // ── Safety net: verify the polyline runs source → destination ─
    // Use actual node positions to check direction; reverse if backwards.
    if (routePolylineCoords.length >= 2 && path.length >= 2) {
        const sourceNode = nodeDict[path[0]];
        const destNode   = nodeDict[path[path.length - 1]];
        if (sourceNode && destNode) {
            const first = routePolylineCoords[0];
            const last  = routePolylineCoords[routePolylineCoords.length - 1];
            const sourcePos = [sourceNode.lat, sourceNode.lng];
            const destPos   = [destNode.lat, destNode.lng];

            const forwardCost = sqDist(first, sourcePos) + sqDist(last, destPos);
            const reverseCost = sqDist(first, destPos) + sqDist(last, sourcePos);

            if (reverseCost < forwardCost) {
                routePolylineCoords.reverse();
                console.log("[FLEET] Polyline was reversed to run source → destination.");
            }
        }
    }

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
    lastTelemetryTime = 0;
}

/**
 * Stop an active navigation session.
 * Releases Wake Lock, clears GPS watch, and notifies backend.
 *
 * @param {string} reason - 'cancelled' | 'rerouted' | 'completed'
 */
function stopNavigation(reason = 'cancelled') {
    // 1. Stop GPS tracking
    if (geoWatchId !== null) {
        navigator.geolocation.clearWatch(geoWatchId);
        geoWatchId = null;
        console.log('[FLEET] GPS watch cleared.');
    }
    // Fallback simulation interval
    if (telemetryInterval) {
        clearInterval(telemetryInterval);
        telemetryInterval = null;
    }

    // 2. Release Wake Lock
    if (wakeLockSentinel) {
        wakeLockSentinel.release().then(() => {
            console.log('[FLEET] Wake Lock released.');
        }).catch(e => console.warn('[FLEET] Wake Lock release error:', e));
        wakeLockSentinel = null;
    }

    // 3. Remove car marker
    if (carMarker) {
        map.removeLayer(carMarker);
        carMarker = null;
    }

    // 4. Tell the backend to release this session's edge slots.
    if (isNavigating || reason === 'rerouted') {
        const sid = SESSION_ID;
        fetch('/api/end_trip', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sid, reason: reason }),
        })
        .then(r => r.json())
        .then(d => console.log(`[FLEET] end_trip (${reason}):`, d))
        .catch(e => console.warn('[FLEET] end_trip call failed (non-critical):', e));
    }

    resetNavUI();
    sessionStorage.removeItem('active_session_id');
}

/**
 * Acquire the Screen Wake Lock so the phone stays awake while driving.
 */
async function acquireWakeLock() {
    if ('wakeLock' in navigator) {
        try {
            wakeLockSentinel = await navigator.wakeLock.request('screen');
            console.log('[FLEET] Wake Lock acquired — screen will stay on.');
            // Re-acquire if visibility changes (e.g., user switches tabs then returns)
            wakeLockSentinel.addEventListener('release', () => {
                console.log('[FLEET] Wake Lock released by system.');
            });
        } catch (err) {
            console.warn('[FLEET] Wake Lock request failed:', err.message);
        }
    } else {
        console.warn('[FLEET] Wake Lock API not supported on this browser.');
    }
}

/**
 * Start live GPS navigation.
 * Uses navigator.geolocation.watchPosition() for real hardware GPS.
 * Falls back to simulation if geolocation is unavailable.
 */
async function startNavigation(e) {
    if (e) e.preventDefault();
    btnStartNav.disabled = true;

    if (!lastRoutePath || lastRoutePath.length < 2) {
        console.warn("[FLEET] No route to navigate.");
        btnStartNav.disabled = false;
        return;
    }
    if (routePolylineCoords.length < 2) {
        console.warn("[FLEET] No polyline coordinates for navigation.");
        btnStartNav.disabled = false;
        return;
    }

    // 0. Acquire Wake Lock immediately to preserve user gesture
    await acquireWakeLock();

    // 1. Request GPS permission explicitly before registering trip
    let hasGps = false;
    if ('geolocation' in navigator) {
        try {
            await new Promise((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(resolve, reject, {
                    enableHighAccuracy: true,
                    maximumAge: 0,
                    timeout: 10000
                });
            });
            hasGps = true;
        } catch (error) {
            console.warn(`[FLEET] Initial GPS request failed (code=${error.code}): ${error.message}`);
            if (error.code === 1 || error.code === error.PERMISSION_DENIED) {
                alert("Location permission denied. Navigation requires GPS access.");
                stopNavigation('cancelled');
                btnStartNav.disabled = false;
                return;
            }
        }
    }

    // 2. Lifecycle & Stale Overwrite
    const currentSessionId = sessionStorage.getItem('active_session_id');
    if (currentSessionId) {
        console.log('[FLEET] Cleaning up stale session before starting new one:', currentSessionId);
        try {
            await fetch("/api/end_trip", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: currentSessionId, reason: 'overwritten' }),
            });
        } catch (e) {
            console.warn('[FLEET] Stale session cleanup failed:', e);
        }
    }

    // We DO NOT generate SESSION_ID on the frontend anymore.
    // The server will generate it and we will store it.
    console.log('[FLEET] Requesting new trip session from backend...');

    // 3. Store destination coordinates for proximity-based stop condition
    destinationCoords = routePolylineCoords[routePolylineCoords.length - 1];
    console.log(`[FLEET] Destination coords: [${destinationCoords[0].toFixed(4)}, ${destinationCoords[1].toFixed(4)}]`);

    // 4. Register the trip on the backend
    try {
        const tripRes = await fetch("/api/start_trip", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                path: lastRoutePath, // Client does not send session_id
            }),
        });
        const tripData = await tripRes.json();
        console.log("[FLEET] Trip registered:", tripData);
        
        // 4b. Store the server-generated session_id
        if (tripData.session_id) {
            SESSION_ID = tripData.session_id;
            sessionStorage.setItem('active_session_id', SESSION_ID);
            console.log('[FLEET] Backend provided session ID:', SESSION_ID);
        }
    } catch (err) {
        console.error("[FLEET] Failed to register trip:", err);
        alert("Could not register your trip. Please check your connection.");
        btnStartNav.disabled = false;
        return;
    }

    // 5. Set navigation state
    isNavigating = true;
    carStepIndex = 0;
    simulateDeviation = false;

    // 5. Place the car marker at the start of the route
    const startPos = routePolylineCoords[0];
    const carIcon = L.divIcon({
        className: "car-marker-icon",
        iconSize: [16, 16],
        iconAnchor: [8, 8],
    });
    carMarker = L.marker(startPos, { icon: carIcon, zIndex: 9999 }).addTo(map);

    // 6. Update UI
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
    telemetryText.textContent = "Acquiring GPS signal\u2026";
    btnStartNav.disabled = false; // Re-enable after setup

    // 8. Start GPS tracking — real hardware or simulation fallback
    if ('geolocation' in navigator) {
        console.log('[FLEET] Starting hardware GPS tracking via watchPosition().');
        geoWatchId = navigator.geolocation.watchPosition(
            onGPSPosition,
            onGPSError,
            {
                enableHighAccuracy: true, // Force physical GPS chip
                maximumAge: 0,            // No cached positions
                timeout: 5000,            // 5s timeout per fix
            }
        );
    } else {
        // Fallback: simulation for desktop testing
        console.warn('[FLEET] Geolocation unavailable — falling back to simulation.');
        telemetryText.textContent = "Telemetry active \u00b7 Simulated";
        telemetryInterval = setInterval(() => simulationTick(), 2000);
    }
}

/**
 * Called by watchPosition() on every GPS fix from the phone hardware.
 * Sends position to backend, checks destination proximity, handles deviation.
 */
async function onGPSPosition(position) {
    if (!isNavigating || !carMarker) return;

    let currentPos = [position.coords.latitude, position.coords.longitude];
    const accuracy = position.coords.accuracy; // meters

    // If the user pressed "Simulate Wrong Turn", offset the real position
    if (simulateDeviation) {
        currentPos = [currentPos[0] + 0.009, currentPos[1] + 0.009];
        simulateDeviation = false;
        const el = carMarker.getElement();
        if (el) el.classList.add("deviated");
    } else {
        const el = carMarker.getElement();
        if (el) el.classList.remove("deviated");
    }

    // Move the marker to the live GPS position
    carMarker.setLatLng(currentPos);
    map.panTo(currentPos, { animate: true, duration: 0.5 });

    // ── Stop Condition 1: Destination Reached (< 50 meters) ─────────
    if (destinationCoords) {
        const distToDest = haversineDist(currentPos, destinationCoords) * 1000; // km → m
        if (distToDest < 50) {
            console.log(`[FLEET] Destination reached! Distance: ${distToDest.toFixed(0)}m`);
            telemetryText.textContent = "\u2713 Arrived at destination";
            telemetryStatus.classList.remove("deviated");
            telemetryStatus.classList.add("on-track");

            // Clean up GPS + wake lock + backend
            if (geoWatchId !== null) {
                navigator.geolocation.clearWatch(geoWatchId);
                geoWatchId = null;
            }
            if (wakeLockSentinel) {
                wakeLockSentinel.release().catch(() => {});
                wakeLockSentinel = null;
            }

            const sid = SESSION_ID;
            fetch('/api/end_trip', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sid, reason: 'completed' }),
            })
            .then(r => r.json())
            .then(d => console.log('[FLEET] end_trip (completed):', d))
            .catch(e => console.warn('[FLEET] end_trip failed on completion:', e));

            btnWrongTurn.classList.add("hidden");
            btnStartNav.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <polygon points="5 3 19 12 5 21 5 3"/>
                </svg>
                Start Navigation`;
            btnStartNav.classList.remove("navigating");
            isNavigating = false;
            return;
        }
    }

    // ── Throttle: The Telemetry Firehose Fix ────────────────────────
    const now = Date.now();
    if (now - lastTelemetryTime < 2500) {
        return; // Ignore excess GPS pings to prevent overwhelming the backend
    }
    lastTelemetryTime = now;

    // ── Send telemetry to backend ───────────────────────────────────
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

        if (!res.ok) {
            console.warn(`[FLEET] Telemetry HTTP ${res.status} for session=${SESSION_ID}.`);
        }

        const data = await res.json();
        console.log(`[FLEET] Telemetry: status=${data.status} | GPS accuracy=${accuracy?.toFixed(0)}m`);

        // Update telemetry status UI
        if (data.status === "on_track") {
            telemetryStatus.classList.remove("deviated");
            telemetryStatus.classList.add("on-track");
            const distToDest = destinationCoords
                ? (haversineDist(currentPos, destinationCoords) * 1000).toFixed(0)
                : '?';
            telemetryText.textContent = `GPS active \u00b7 On track \u00b7 ${distToDest}m to dest`;

        } else if (data.status === "deviated") {
            if (geoWatchId !== null) {
                navigator.geolocation.clearWatch(geoWatchId);
                geoWatchId = null;
            }
            resetNavUI();
            sessionStorage.removeItem('active_session_id');

            telemetryStatus.classList.remove("hidden", "on-track");
            telemetryStatus.classList.add("deviated");
            telemetryText.textContent = "\u26a0 Deviation detected! Vehicle off-route";

            alert("You left the route. Recalculate?");

        } else if (data.status === "no_session") {
            console.error("[FLEET] Server has no session for", SESSION_ID,
                          "\u2014 re-registering trip...");
            if (lastRoutePath && lastRoutePath.length >= 2) {
                fetch("/api/start_trip", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ path: lastRoutePath }), // Do not send session_id
                }).then(r => r.json()).then(d => {
                    console.log("[FLEET] Auto re-register result:", d);
                    if (d.session_id) {
                        SESSION_ID = d.session_id;
                        sessionStorage.setItem('active_session_id', SESSION_ID);
                    }
                }).catch(e => console.error("[FLEET] Re-register failed:", e));
            }
        }
    } catch (err) {
        console.error("[FLEET] Telemetry ping failed:", err);
    }
}

function onGPSError(error) {
    console.warn(`[FLEET] GPS error (code=${error.code}): ${error.message}`);
    
    // 1. Immediately clear the watch
    if (geoWatchId !== null) {
        navigator.geolocation.clearWatch(geoWatchId);
        geoWatchId = null;
    }

    // 2. Asynchronous POST /api/end_trip to delete ghost session
    const sid = sessionStorage.getItem('active_session_id') || SESSION_ID;
    if (sid) {
        fetch('/api/end_trip', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sid, reason: 'gps_error' }),
        }).catch(() => {});
    }

    // 3. Alert the user
    alert("GPS signal lost or unavailable indoors. Route released.");

    // 4. Reset the UI button state
    resetNavUI();
    btnStartNav.disabled = false;
}

/**
 * Simulation-only fallback tick for desktop testing (no real GPS).
 * Walks along the precomputed polyline and sends telemetry.
 */
async function simulationTick() {
    if (!isNavigating || !carMarker) return;

    let currentPos;

    if (simulateDeviation) {
        const basePos = routePolylineCoords[Math.min(carStepIndex, routePolylineCoords.length - 1)];
        currentPos = [basePos[0] + 0.009, basePos[1] + 0.009];
        simulateDeviation = false;
        const el = carMarker.getElement();
        if (el) el.classList.add("deviated");
    } else {
        const stepsPerTick = Math.max(1, Math.floor(routePolylineCoords.length / 30));
        carStepIndex = Math.min(carStepIndex + stepsPerTick, routePolylineCoords.length - 1);
        currentPos = routePolylineCoords[carStepIndex];
        const el = carMarker.getElement();
        if (el) el.classList.remove("deviated");
    }

    carMarker.setLatLng(currentPos);

    // Check destination proximity in simulation too
    if (destinationCoords) {
        const distToDest = haversineDist(currentPos, destinationCoords) * 1000;
        if (distToDest < 50) {
            console.log(`[FLEET] Simulation: Destination reached! ${distToDest.toFixed(0)}m`);
            telemetryText.textContent = "\u2713 Arrived at destination";
            telemetryStatus.classList.remove("deviated");
            telemetryStatus.classList.add("on-track");
            clearInterval(telemetryInterval);
            telemetryInterval = null;

            const sid = SESSION_ID;
            fetch('/api/end_trip', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sid, reason: 'completed' }),
            })
            .then(r => r.json())
            .then(d => console.log('[FLEET] end_trip (completed):', d))
            .catch(e => console.warn('[FLEET] end_trip failed on completion:', e));

            btnWrongTurn.classList.add("hidden");
            btnStartNav.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <polygon points="5 3 19 12 5 21 5 3"/>
                </svg>
                Start Navigation`;
            btnStartNav.classList.remove("navigating");
            isNavigating = false;
            return;
        }
    }

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

        if (!res.ok) {
            console.warn(`[FLEET] Telemetry HTTP ${res.status} for session=${SESSION_ID}.`);
        }

        const data = await res.json();
        console.log(`[FLEET] Sim telemetry: status=${data.status} | step ${carStepIndex}/${routePolylineCoords.length - 1}`);

        if (data.status === "on_track") {
            telemetryStatus.classList.remove("deviated");
            telemetryStatus.classList.add("on-track");
            telemetryText.textContent = `Telemetry active \u00b7 On track (step ${carStepIndex}/${routePolylineCoords.length - 1})`;
        } else if (data.status === "deviated") {
            if (telemetryInterval) {
                clearInterval(telemetryInterval);
                telemetryInterval = null;
            }
            resetNavUI();
            sessionStorage.removeItem('active_session_id');

            telemetryStatus.classList.remove("hidden", "on-track");
            telemetryStatus.classList.add("deviated");
            telemetryText.textContent = "\u26a0 Deviation detected! Vehicle off-route";

            alert("You left the route. Recalculate?");
        } else if (data.status === "no_session") {
            console.error("[FLEET] Server has no session for", SESSION_ID, "\u2014 re-registering...");
            if (lastRoutePath && lastRoutePath.length >= 2) {
                fetch("/api/start_trip", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ path: lastRoutePath }), // Do not send session_id
                }).then(r => r.json()).then(d => {
                    console.log("[FLEET] Auto re-register result:", d);
                    if (d.session_id) {
                        SESSION_ID = d.session_id;
                        sessionStorage.setItem('active_session_id', SESSION_ID);
                    }
                }).catch(e => console.error("[FLEET] Re-register failed:", e));
            }
        }
    } catch (err) {
        console.error("[FLEET] Telemetry ping failed:", err);
    }

    // Check if simulation reached end of polyline
    if (carStepIndex >= routePolylineCoords.length - 1 && !simulateDeviation) {
        console.log("[FLEET] Simulation complete \u2014 reached end of polyline.");
        telemetryText.textContent = "\u2713 Arrived at destination";
        telemetryStatus.classList.remove("deviated");
        telemetryStatus.classList.add("on-track");
        clearInterval(telemetryInterval);
        telemetryInterval = null;

        const sid = SESSION_ID;
        fetch('/api/end_trip', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sid, reason: 'completed' }),
        })
        .then(r => r.json())
        .then(d => console.log('[FLEET] end_trip (completed):', d))
        .catch(e => console.warn('[FLEET] end_trip failed on completion:', e));

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

// Re-acquire Wake Lock when user returns to the tab (browser releases it on tab switch)
document.addEventListener('visibilitychange', async () => {
    if (document.visibilityState === 'visible' && isNavigating && !wakeLockSentinel) {
        await acquireWakeLock();
    }
});

// ── Button Event Listeners ──────────────────────────────────────
btnStartNav.addEventListener("click", async (e) => {
    if (isNavigating) {
        stopNavigation();
    } else {
        await startNavigation(e);
    }
});

btnWrongTurn.addEventListener("click", () => {
    if (!isNavigating) return;
    simulateDeviation = true;
    console.log("[FLEET] Wrong turn simulation queued \u2014 will deviate on next GPS fix.");
});

window.addEventListener('beforeunload', () => {
    const activeSession = sessionStorage.getItem('active_session_id');
    if (activeSession) {
        const payload = JSON.stringify({ session_id: activeSession, reason: 'page_unload' });
        navigator.sendBeacon('/api/end_trip', payload);
    }
});
