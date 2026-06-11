console.log("Arbitrage Model Initialized");

const ENDPOINTS = {
    opportunities: '../data/processed/opportunities.json',
    matches: '../data/processed/match_by_match.json'
};

// Global State
let globalOpps = null;
let globalMatches = null;
let evChartInstance = null;

let filterSearch = "";
let filterActionable = false;
let filterGroup = "All";

// Theme Toggle
const themeBtn = document.getElementById('theme-toggle');
let isDarkMode = false;
themeBtn.addEventListener('click', () => {
    isDarkMode = !isDarkMode;
    if (isDarkMode) {
        document.body.classList.add('dark-mode');
        themeBtn.textContent = '☀ Light Mode';
    } else {
        document.body.classList.remove('dark-mode');
        themeBtn.textContent = '☾ Dark Mode';
    }
    // Update chart colors if chart exists
    if (evChartInstance) {
        evChartInstance.options.scales.x.ticks.color = isDarkMode ? '#94A3B8' : '#64748B';
        evChartInstance.options.scales.y.ticks.color = isDarkMode ? '#94A3B8' : '#64748B';
        evChartInstance.options.plugins.legend.labels.color = isDarkMode ? '#F8FAFC' : '#1E3A8A';
        evChartInstance.update();
    }
});

// Controls
document.getElementById('search-input').addEventListener('input', (e) => {
    filterSearch = e.target.value.toLowerCase().trim();
    renderAll();
});

document.getElementById('ev-toggle').addEventListener('change', (e) => {
    filterActionable = e.target.checked;
    renderAll();
});

document.getElementById('group-filter').addEventListener('change', (e) => {
    filterGroup = e.target.value;
    renderAll();
});

// Clock
setInterval(() => {
    document.getElementById('clock').textContent = new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
}, 1000);

async function fetchPipelineData() {
    try {
        console.log("Fetching pipeline JSONs...");
        const ts = new Date().getTime();
        const [oppsRes, matchRes] = await Promise.all([
            fetch(`${ENDPOINTS.opportunities}?t=${ts}`).catch(err => { throw new Error("Opp: " + err.message); }),
            fetch(`${ENDPOINTS.matches}?t=${ts}`).catch(err => { throw new Error("Match: " + err.message); })
        ]);

        globalOpps = (oppsRes && oppsRes.ok) ? await oppsRes.json() : null;
        globalMatches = (matchRes && matchRes.ok) ? await matchRes.json() : null;

        renderAll();
        
    } catch (error) {
        console.error("Pipeline failure:", error);
        document.getElementById('opportunities-tbody').innerHTML = 
            `<tr><td colspan="5" class="text-negative text-center py-4 font-bold">ERR: ${error.message}</td></tr>`;
    }
}

function renderAll() {
    renderHighlights();
    renderOpportunities();
    renderMatches();
}

function renderHighlights() {
    const container = document.getElementById('highlights-container');
    if (!globalOpps || !globalOpps.tracked_markets) return;

    const markets = globalOpps.tracked_markets;
    const actionable = markets.filter(m => m.ev_percent > 0);
    
    // Top Target
    const topTarget = actionable.length > 0 ? actionable[0] : markets[0];
    const topName = topTarget ? topTarget.team : "N/A";
    const topEv = topTarget && topTarget.ev_percent > 0 ? `+${topTarget.ev_percent.toFixed(2)}%` : "0.00%";
    
    // Average Edge
    const avgEdge = actionable.length > 0 
        ? (actionable.reduce((sum, m) => sum + m.ev_percent, 0) / actionable.length).toFixed(2) 
        : 0;

    container.innerHTML = `
        <div class="highlight-card">
            <span class="highlight-label">Top Arbitrage Target</span>
            <span class="highlight-value">${topName}</span>
            <span class="highlight-sub text-positive font-bold">Edge: ${topEv}</span>
        </div>
        <div class="highlight-card">
            <span class="highlight-label">Actionable Markets</span>
            <span class="highlight-value">${actionable.length}</span>
            <span class="highlight-sub">Out of ${markets.length} tracked entities</span>
        </div>
        <div class="highlight-card">
            <span class="highlight-label">Average Actionable Edge</span>
            <span class="highlight-value text-positive">+${avgEdge}%</span>
            <span class="highlight-sub">Mean EV across positive markets</span>
        </div>
    `;
}

function renderChart(data) {
    const ctx = document.getElementById('evChart');
    if (!ctx) return;

    // Take top 8 for the chart
    const chartData = data.slice(0, 8);
    const labels = chartData.map(m => m.team);
    const evs = chartData.map(m => m.ev_percent);
    
    const bgColors = evs.map(ev => ev > 0 ? 'rgba(22, 163, 74, 0.7)' : 'rgba(148, 163, 184, 0.4)');
    const borderColors = evs.map(ev => ev > 0 ? 'rgba(22, 163, 74, 1)' : 'rgba(148, 163, 184, 1)');

    if (evChartInstance) {
        evChartInstance.destroy();
    }

    const textColor = isDarkMode ? '#F8FAFC' : '#1E3A8A';
    const gridColor = isDarkMode ? '#1E293B' : '#E2E8F0';

    evChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Statistical Edge (EV %)',
                data: evs,
                backgroundColor: bgColors,
                borderColor: borderColors,
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: textColor,
                        font: { family: "'Fira Sans', sans-serif" }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: gridColor },
                    ticks: { color: isDarkMode ? '#94A3B8' : '#64748B', font: { family: "'Fira Code', monospace" } }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: isDarkMode ? '#94A3B8' : '#64748B', font: { family: "'Fira Sans', sans-serif", weight: 'bold' } }
                }
            }
        }
    });
}

function generateSparkline(history) {
    if (!history || history.length < 2) return '';
    const min = Math.min(...history);
    const max = Math.max(...history);
    const range = max - min || 1;
    const width = 40;
    const height = 15;
    
    const points = history.map((val, i) => {
        const x = (i / (history.length - 1)) * width;
        const y = height - ((val - min) / range) * height;
        return `${x},${y}`;
    }).join(' ');
    
    const isUp = history[history.length - 1] >= history[0];
    const color = isUp ? '#16a34a' : '#ef4444';

    return `
        <svg width="${width}" height="${height}" class="inline-block ml-2 opacity-80" viewBox="-2 -2 ${width+4} ${height+4}">
            <polyline fill="none" stroke="${color}" stroke-width="2" points="${points}" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
    `;
}

function renderOpportunities() {
    const tbody = document.getElementById('opportunities-tbody');
    if (!globalOpps || !globalOpps.tracked_markets) {
        tbody.innerHTML = `<tr><td colspan="5" class="text-neutral text-center py-4">NO DATA</td></tr>`;
        return;
    }

    let markets = globalOpps.tracked_markets;

    // Apply Filters
    if (filterSearch) {
        markets = markets.filter(opp => opp.team.toLowerCase().includes(filterSearch));
    }
    if (filterActionable) {
        markets = markets.filter(opp => opp.ev_percent > 0);
    }

    // Render Chart before slicing if possible, or just render top 10 of filtered
    renderChart(markets);

    markets = markets.slice(0, 15); // Show up to 15
    document.getElementById('opp-count').textContent = `Showing ${markets.length} entities`;

    tbody.innerHTML = markets.map(opp => {
        const isPos = opp.ev_percent > 0;
        const evColor = isPos ? 'text-positive bg-positive' : 'text-neutral';
        const evVal = isPos ? `+${opp.ev_percent.toFixed(2)}%` : `${opp.ev_percent.toFixed(2)}%`;
        const sparklineHtml = generateSparkline(opp.ev_history);
        
        // Visual edge bar
        const barWidth = isPos ? Math.min(100, opp.ev_percent / 2) : 0;
        const barHtml = isPos ? `<div class="edge-bar-container"><div class="edge-bar-fill" style="width: ${barWidth}%"></div></div>` : '';

        return `
            <tr>
                <td class="text-left font-bold text-accent">${opp.team} ${sparklineHtml}</td>
                <td>${(opp.p_consensus * 100).toFixed(1)}%</td>
                <td>${(opp.p_market * 100).toFixed(1)}%</td>
                <td class="${evColor} font-bold">${evVal} ${barHtml}</td>
                <td class="text-highlight font-bold">${(opp.kelly_fraction * 100).toFixed(1)}%</td>
            </tr>
        `;
    }).join('');
    
    if (markets.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" class="text-neutral text-center py-4">No markets match filters</td></tr>`;
    }
}

function formatShortDate(isoString) {
    if (!isoString) return "TBD";
    const d = new Date(isoString);
    const month = d.toLocaleString('default', { month: 'short' });
    const day = d.getDate().toString().padStart(2, '0');
    const hr = d.getHours().toString().padStart(2, '0');
    const min = d.getMinutes().toString().padStart(2, '0');
    return `${month} ${day}, ${hr}:${min}`;
}

function renderMatches() {
    const liveBody = document.getElementById('completed-matches-tbody');
    const upBody = document.getElementById('upcoming-matches-tbody');
    
    if (!globalMatches || !Array.isArray(globalMatches)) return;

    let matches = [...globalMatches]; // Clone array to sort

    // Filter
    if (filterSearch) {
        matches = matches.filter(match => 
            match.home_team.toLowerCase().includes(filterSearch) || 
            match.away_team.toLowerCase().includes(filterSearch)
        );
    }
    
    if (filterGroup !== "All") {
        matches = matches.filter(match => match.group === filterGroup);
    }

    // Sort Chronologically
    matches.sort((a, b) => {
        if (!a.date) return 1;
        if (!b.date) return -1;
        return new Date(a.date) - new Date(b.date);
    });

    let liveRows = [];
    let upRows = [];

    matches.forEach(match => {
        const isLive = match.status !== "Not Started";
        const dateStr = formatShortDate(match.date);
        
        if (isLive) {
            // Actual results
            const hg = match.home_goals || 0;
            const ag = match.away_goals || 0;
            const total = hg + ag;
            const btts = (hg > 0 && ag > 0);
            
            const o15 = (match.over_under_1_5.over * 100).toFixed(1);
            
            const o25Badge = total > 2.5 
                ? '<span class="bg-positive text-positive px-2 py-0.5 rounded text-[10px] font-bold">HIT</span>' 
                : '<span class="bg-negative text-negative px-2 py-0.5 rounded text-[10px] font-bold">MISS</span>';
                
            const bttsBadge = btts 
                ? '<span class="bg-positive text-positive px-2 py-0.5 rounded text-[10px] font-bold">HIT</span>' 
                : '<span class="bg-negative text-negative px-2 py-0.5 rounded text-[10px] font-bold">MISS</span>';

            liveRows.push(`
                <tr>
                    <td class="text-left text-neutral text-xs">${dateStr}</td>
                    <td class="text-left font-bold">${match.home_team} - ${match.away_team}</td>
                    <td class="text-accent text-xs tracking-wider font-bold">${match.status}</td>
                    <td class="font-bold text-[14px]">${hg} - ${ag}</td>
                    <td class="text-neutral">${o15}%</td>
                    <td>${o25Badge}</td>
                    <td>${bttsBadge}</td>
                </tr>
            `);
        } else {
            // Upcoming Predictions
            const h = (match["1X2"].home * 100).toFixed(1);
            const d = (match["1X2"].draw * 100).toFixed(1);
            const a = (match["1X2"].away * 100).toFixed(1);
            const o15 = (match.over_under_1_5.over * 100).toFixed(1);
            const o25 = (match.over_under_2_5.over * 100).toFixed(1);
            const btts = (match.btts.yes * 100).toFixed(1);

            upRows.push(`
                <tr>
                    <td class="text-left text-neutral text-xs">${dateStr}</td>
                    <td class="text-left font-bold">${match.home_team} - ${match.away_team}</td>
                    <td class="text-neutral"><span class="text-accent">${h}</span> / ${d} / <span class="text-highlight">${a}</span></td>
                    <td class="text-positive">${o15}%</td>
                    <td class="text-positive">${o25}%</td>
                    <td class="text-positive">${btts}%</td>
                </tr>
            `);
        }
    });

    liveBody.innerHTML = liveRows.length ? liveRows.join('') : `<tr><td colspan="7" class="text-neutral py-4 text-center">No live/completed matches found</td></tr>`;
    upBody.innerHTML = upRows.length ? upRows.slice(0, 15).join('') : `<tr><td colspan="6" class="text-neutral py-4 text-center">No upcoming fixtures found</td></tr>`;
}

fetchPipelineData();
setInterval(fetchPipelineData, 60000);
