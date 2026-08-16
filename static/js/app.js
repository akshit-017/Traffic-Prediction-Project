// Initialize the Map centered on central Bengaluru
const map = L.map('map', {
    dragging: true,
    touchZoom: true,
    scrollWheelZoom: true,
    tap: true
}).setView([12.94, 77.62], 12);

// Explicitly enable dragging in case any CSS or init timing issue disabled it
map.dragging.enable();

// Add a dark theme tile layer (CartoDB Dark Matter)
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 20
}).addTo(map);

// Keep track of layers to remove them on new searches
let mapLayers = [];

function clearMap() {
    mapLayers.forEach(layer => map.removeLayer(layer));
    mapLayers = [];
}

function getCongestionColor(multiplier) {
    if (multiplier < 1.3) return '#2ecc71'; // Green (Free flow)
    if (multiplier < 1.7) return '#f1c40f'; // Yellow (Moderate)
    return '#e74c3c'; // Red (Heavy)
}

// ============================================================
// Travel-time formatting: effective_weight is already in minutes
// ============================================================

/**
 * Format minutes into a human-readable travel time string.
 * Examples: "12 mins", "1 hr 12 mins", "2 hr"
 */
function formatTravelTime(minutes) {
    const totalMinutes = Math.round(minutes);
    if (totalMinutes < 60) {
        return `${totalMinutes} mins`;
    }
    const hours = Math.floor(totalMinutes / 60);
    const mins  = totalMinutes % 60;
    if (mins === 0) {
        return `${hours} hr`;
    }
    return `${hours} hr ${mins} mins`;
}

document.getElementById('route-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const source = document.getElementById('source').value;
    const destination = document.getElementById('destination').value;
    
    if (source === destination) {
        alert("Source and Destination cannot be the same!");
        return;
    }
    
    try {
        const response = await fetch('/api/route', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source, destination })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            alert(data.error || "An error occurred");
            return;
        }
        
        // Render Map Graphics
        renderNetwork(data.graph, data.route);
        
        // Update Results UI
        document.getElementById('optimal-path-text').innerText = data.route.join(' → ');
        document.getElementById('travel-time-text').innerText = formatTravelTime(data.total_weight);
        document.getElementById('results').classList.remove('hidden');

        // Show ML model status
        const mlBadge = document.getElementById('ml-status');
        if (mlBadge) {
            if (data.ml_model_loaded) {
                mlBadge.innerHTML = '🤖 <strong>ML Model Active</strong> — Random Forest';
                mlBadge.className = 'ml-badge ml-active';
            } else {
                mlBadge.innerHTML = '⚠️ <strong>Heuristic Mode</strong> — Run <code>python main.py</code> to enable ML';
                mlBadge.className = 'ml-badge ml-fallback';
            }
            mlBadge.style.display = 'block';
        }

        // On mobile, auto-close sidebar after finding route so user sees the map
        if (window.innerWidth <= 768) {
            closeSidebar();
        }
        
    } catch (error) {
        console.error("Error fetching route:", error);
        alert("Failed to connect to the prediction server.");
    }
});

/**
 * Build the full coordinate array for an edge.
 * Uses the waypoints stored on the edge to trace actual road curves
 * instead of drawing a straight line between source and target nodes.
 */
function buildEdgeLatLngs(sourceNode, targetNode, waypoints) {
    const coords = [[sourceNode.lat, sourceNode.lng]];

    if (waypoints && waypoints.length > 0) {
        waypoints.forEach(wp => coords.push([wp[0], wp[1]]));
    }

    coords.push([targetNode.lat, targetNode.lng]);
    return coords;
}

function renderNetwork(graph, optimalRoute) {
    clearMap();
    
    // Create a dictionary for quick node lookup by ID
    const nodeDict = {};
    graph.nodes.forEach(node => {
        nodeDict[node.id] = node;
        
        // Add Marker
        const marker = L.circleMarker([node.lat, node.lng], {
            radius: 6,
            fillColor: '#3b82f6',
            color: '#fff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.8
        }).bindTooltip(`${node.name} (${node.id})`, {
            permanent: true,
            direction: 'top',
            className: 'node-label'
        }).addTo(map);
        
        mapLayers.push(marker);
    });
    
    // Draw edges
    graph.edges.forEach(edge => {
        const sourceNode = nodeDict[edge.source];
        const targetNode = nodeDict[edge.target];
        
        if (sourceNode && targetNode) {
            // Build polyline through waypoints instead of a straight line
            const latlngs = buildEdgeLatLngs(sourceNode, targetNode, edge.waypoints);
            
            // Check if this edge is part of the optimal route
            let isOptimal = false;
            for (let i = 0; i < optimalRoute.length - 1; i++) {
                if ((optimalRoute[i] === edge.source && optimalRoute[i+1] === edge.target) ||
                    (optimalRoute[i] === edge.target && optimalRoute[i+1] === edge.source)) {
                    isOptimal = true;
                    break;
                }
            }
            
            const color = getCongestionColor(edge.congestion_multiplier);
            
            // Draw background line for congestion visualization
            const polyline = L.polyline(latlngs, {
                color: color,
                weight: isOptimal ? 8 : 4,
                opacity: isOptimal ? 0.9 : 0.4,
                dashArray: isOptimal ? null : '5, 10'
            }).addTo(map);
            
            // Add tooltip with real travel info + ML predicted volume
            const freeFlow = edge.base_weight.toFixed(1);
            const withTraffic = edge.effective_weight.toFixed(1);
            const dist = edge.distance_km.toFixed(1);
            const cong = edge.congestion_multiplier.toFixed(2);
            const predVol = Math.round(edge.predicted_volume || 0).toLocaleString();
            let tooltipHtml = `${dist} km · Free-flow: ${freeFlow} min<br>`
                + `With traffic: ${withTraffic} min (×${cong})`;
            if (edge.predicted_volume > 0) {
                tooltipHtml += `<br>ML Predicted Volume: ${predVol} vehicles`;
            }
            polyline.bindTooltip(tooltipHtml, { sticky: true });
            
            mapLayers.push(polyline);
            
            // Highlight optimal route with a glowing effect
            if (isOptimal) {
                const glowLine = L.polyline(latlngs, {
                    color: '#fff',
                    weight: 2,
                    opacity: 0.8
                }).addTo(map);
                mapLayers.push(glowLine);
            }
        }
    });
    
    // Fit bounds to show all nodes
    const bounds = L.latLngBounds(graph.nodes.map(n => [n.lat, n.lng]));
    map.fitBounds(bounds, { padding: [50, 50] });
}

// ============================================================
// Sidebar Toggle Logic (mobile hamburger + overlay backdrop)
// ============================================================
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebar = document.getElementById('sidebar');
const sidebarOverlay = document.getElementById('sidebar-overlay');

function openSidebar() {
    sidebar.classList.add('active');
    if (sidebarOverlay) sidebarOverlay.classList.add('active');
}

function closeSidebar() {
    sidebar.classList.remove('active');
    if (sidebarOverlay) sidebarOverlay.classList.remove('active');
    // Recalculate map size after CSS transition finishes
    setTimeout(() => map.invalidateSize(), 300);
}

if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener('click', () => {
        if (sidebar.classList.contains('active')) {
            closeSidebar();
        } else {
            openSidebar();
        }
    });
}

// Tap the backdrop overlay to close the sidebar on mobile
if (sidebarOverlay) {
    sidebarOverlay.addEventListener('click', closeSidebar);
}
