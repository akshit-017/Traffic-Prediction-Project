// ================================================================
// Bengaluru ML-Powered Traffic Route Optimizer — Frontend Controller
// Uses /api/predict_route for per-edge congestion coloring
// OSRM road-following paths + Green/Yellow/Red segment rendering
// ================================================================

// ── Map Initialization ──────────────────────────────────────────
const map = L.map("map", {
    center: [12.9716, 77.5946],
    zoom: 12,
    zoomControl: true,
    attributionControl: false,
});

// OpenStreetMap tiles — free, no API key required
L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    {
        attribution: '',
        maxZoom: 19,
    }
).addTo(map);

// ── State ───────────────────────────────────────────────────────
let mapLayers = [];
let currentMode = "live";

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
 * Get road-following coords for an edge.
 * Falls back to straight line if OSRM is unavailable.
 */
async function getEdgeCoords(lat1, lng1, lat2, lng2) {
    const roadCoords = await fetchRoadGeometry(lat1, lng1, lat2, lng2);
    if (roadCoords && roadCoords.length >= 2) return roadCoords;
    // Fallback: straight line
    return [[lat1, lng1], [lat2, lng2]];
}


// ================================================================
// ROUTE REQUEST — uses /api/predict_route
// ================================================================
btnRoute.addEventListener("click", async () => {
    hideError();
    resultsDashboard.classList.add("hidden");

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
        const latlngs = await getEdgeCoords(lat1, lng1, lat2, lng2);
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

        // Collect coordinates for bounds
        allNodes.push([lat1, lng1]);
        allNodes.push([lat2, lng2]);
    });

    await Promise.all(segmentPromises);

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
