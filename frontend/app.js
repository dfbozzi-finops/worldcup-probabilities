console.log("Arbitrage Model Initialized");

const ENDPOINTS = {
    opportunities: '../data/processed/opportunities.json',
    matches: '../data/processed/match_by_match.json',
    props: '../data/processed/props_probabilities.json',
    groupAnalysis: '../data/processed/group_stage_analysis.json',
    h2h: '../data/processed/h2h_matrix.json'
};

// Global State
let globalOpps = null;
let globalMatches = null;
let globalProps = null;
let groupAnalysisData = null;
let h2hMatrix = null;
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
        document.documentElement.classList.add('dark');
        themeBtn.textContent = '☀ Light Mode';
    } else {
        document.body.classList.remove('dark-mode');
        document.documentElement.classList.remove('dark');
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

// Clock (Local Time)
setInterval(() => {
    const d = new Date();
    document.getElementById('clock').textContent = d.toLocaleString() + ' (Local)';
}, 1000);

// Tab Navigation State
const tabDashboard = document.getElementById('tab-dashboard');
const tabCountry = document.getElementById('tab-country');
const tabGroups = document.getElementById('tab-groups');
const tabKnockout = document.getElementById('tab-knockout');

const viewDashboard = document.getElementById('view-dashboard');
const viewCountry = document.getElementById('view-country');
const viewGroups = document.getElementById('view-groups');
const viewKnockout = document.getElementById('view-knockout');

const countrySelect = document.getElementById('country-select');

let currentCountry = "";

function switchTab(activeTab, activeView) {
    [tabDashboard, tabCountry, tabGroups, tabKnockout].forEach(t => {
        t.classList.remove('text-accent', 'border-accent');
        t.classList.add('text-neutral', 'border-transparent');
    });
    activeTab.classList.remove('text-neutral', 'border-transparent');
    activeTab.classList.add('text-accent', 'border-accent');

    [viewDashboard, viewCountry, viewGroups, viewKnockout].forEach(v => {
        v.classList.add('hidden');
    });
    activeView.classList.remove('hidden');
}

tabDashboard.addEventListener('click', () => switchTab(tabDashboard, viewDashboard));
tabCountry.addEventListener('click', () => {
    switchTab(tabCountry, viewCountry);
    populateCountryDropdown();
});
tabGroups.addEventListener('click', () => switchTab(tabGroups, viewGroups));
tabKnockout.addEventListener('click', () => switchTab(tabKnockout, viewKnockout));

countrySelect.addEventListener('change', (e) => {
    currentCountry = e.target.value;
    renderCountryView();
});

async function fetchPipelineData() {
    try {
        console.log("Fetching pipeline JSONs...");
        const ts = new Date().getTime();
        const [oppsRes, matchRes, propsRes, groupRes, h2hRes] = await Promise.all([
            fetch(`${ENDPOINTS.opportunities}?t=${ts}`).catch(err => { throw new Error("Opp: " + err.message); }),
            fetch(`${ENDPOINTS.matches}?t=${ts}`).catch(err => { throw new Error("Match: " + err.message); }),
            fetch(`${ENDPOINTS.props}?t=${ts}`).catch(err => { console.warn("No props data"); return null; }),
            fetch(`${ENDPOINTS.groupAnalysis}?t=${ts}`).catch(err => { console.warn("No group data"); return null; }),
            fetch(`${ENDPOINTS.h2h}?t=${ts}`).catch(err => { console.warn("No h2h data"); return null; })
        ]);

        globalOpps = (oppsRes && oppsRes.ok) ? await oppsRes.json() : null;
        globalMatches = (matchRes && matchRes.ok) ? await matchRes.json() : null;
        globalProps = (propsRes && propsRes.ok) ? await propsRes.json() : null;
        groupAnalysisData = (groupRes && groupRes.ok) ? await groupRes.json() : null;
        h2hMatrix = (h2hRes && h2hRes.ok) ? await h2hRes.json() : null;
        // Element removed from DOM
        // document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();

        if (currentCountry) {
            renderCountryView();
        }
        renderGroupAnalysis();
        renderKnockoutBracket();
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
    
    // Format to user's Local Time automatically
    const month = d.toLocaleString('default', { month: 'short' });
    const day = d.getDate();
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
        const matchup = `${match.home_team} - ${match.away_team}`;

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
            const pHome = (match["1X2"].home * 100).toFixed(0);
            const pDraw = (match["1X2"].draw * 100).toFixed(0);
            const pAway = (match["1X2"].away * 100).toFixed(0);
            const pred1X2 = `H: ${pHome}% | D: ${pDraw}% | A: ${pAway}%`;
            
            const pO15 = (match.over_under_1_5.over * 100).toFixed(0);
            const pO25 = (match.over_under_2_5.over * 100).toFixed(0);
            const pBTTS = (match.btts.yes * 100).toFixed(0);
            
            // Gather Recommendations > 50%
            let recs = [];
            
            if (match["1X2"].home > 0.50) recs.push(`${match.home_team} ML (${pHome}%)`);
            if (match["1X2"].draw > 0.50) recs.push(`Draw (${pDraw}%)`);
            if (match["1X2"].away > 0.50) recs.push(`${match.away_team} ML (${pAway}%)`);
            
            if (match.over_under_1_5.over > 0.50) recs.push(`Over 1.5 Goals (${pO15}%)`);
            if (match.over_under_1_5.under > 0.50) recs.push(`Under 1.5 Goals (${(match.over_under_1_5.under*100).toFixed(0)}%)`);
            
            if (match.over_under_2_5.over > 0.50) recs.push(`Over 2.5 Goals (${pO25}%)`);
            if (match.over_under_2_5.under > 0.50) recs.push(`Under 2.5 Goals (${(match.over_under_2_5.under*100).toFixed(0)}%)`);
            
            if (match.btts.yes > 0.50) recs.push(`BTTS Yes (${pBTTS}%)`);
            if (match.btts.no > 0.50) recs.push(`BTTS No (${(match.btts.no*100).toFixed(0)}%)`);
            
            // Player props
            if (globalProps && Array.isArray(globalProps)) {
                const pData = globalProps.find(p => p.match === matchup);
                if (pData && pData.player_props) {
                    const pp = pData.player_props;
                    if (pp.anytime_goalscorer) {
                        for (const [player, prob] of Object.entries(pp.anytime_goalscorer)) {
                            if (prob > 0.50) recs.push(`${player} to Score (${(prob*100).toFixed(0)}%)`);
                        }
                    }
                    if (pp["over_under_shots_on_target_1.5"]) {
                        for (const [player, odds] of Object.entries(pp["over_under_shots_on_target_1.5"])) {
                            if (odds.over > 0.50) recs.push(`${player} O1.5 SoT (${(odds.over*100).toFixed(0)}%)`);
                        }
                    }
                    if (pp["over_under_assists_0.5"]) {
                        for (const [player, odds] of Object.entries(pp["over_under_assists_0.5"])) {
                            if (odds.over > 0.50) recs.push(`${player} O0.5 Assists (${(odds.over*100).toFixed(0)}%)`);
                        }
                    }
                }
            }
            
            // Render pills
            const pillsHtml = recs.length > 0 
                ? recs.map(r => `<span class="inline-block bg-blue-100 dark:bg-blue-900/40 text-blue-800 dark:text-blue-300 text-[10px] font-bold px-2 py-1 rounded shadow-sm border border-blue-200 dark:border-blue-800 mb-1 mr-1">${r}</span>`).join('')
                : `<span class="text-neutral text-xs italic">No strong edges</span>`;
            
            upRows.push(`
                <tr class="border-b border-gray-100 dark:border-gray-800/50 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                    <td class="text-left text-neutral text-xs py-3">${dateStr}</td>
                    <td class="text-left font-bold py-3">${matchup}</td>
                    <td class="text-xs py-3">${pred1X2}</td>
                    <td class="font-bold py-3">${pO15}%</td>
                    <td class="font-bold py-3">${pO25}%</td>
                    <td class="font-bold py-3">${pBTTS}%</td>
                    <td class="text-left py-2 max-w-[300px] leading-tight">${pillsHtml}</td>
                </tr>
            `);
        }
    });

    liveBody.innerHTML = liveRows.length ? liveRows.join('') : `<tr><td colspan="7" class="text-neutral py-4 text-center">No live/completed matches found</td></tr>`;
    upBody.innerHTML = upRows.length ? upRows.join('') : `<tr><td colspan="7" class="text-neutral py-4 text-center">No upcoming fixtures found</td></tr>`;
}

// ==========================================
// COUNTRY VIEW LOGIC
// ==========================================

function populateCountryDropdown() {
    if (!globalMatches || countrySelect.options.length > 1) return;
    
    const teams = new Set();
    globalMatches.forEach(m => {
        teams.add(m.home_team);
        teams.add(m.away_team);
    });
    
    const sortedTeams = Array.from(teams).sort();
    
    countrySelect.innerHTML = '<option value="">-- Select a Country --</option>' + 
        sortedTeams.map(t => `<option value="${t}">${t}</option>`).join('');
}

function renderCountryView() {
    if (!currentCountry) {
        document.getElementById('country-overview').innerHTML = '<div class="text-neutral italic col-span-3">Select a country to view detailed analytics.</div>';
        document.getElementById('country-fixtures-tbody').innerHTML = '<tr><td colspan="4" class="text-neutral py-4">No data</td></tr>';
        document.getElementById('country-props-tbody').innerHTML = '<tr><td colspan="4" class="text-neutral py-4">No data</td></tr>';
        return;
    }
    
    // Overview Stats
    const opp = globalOpps && globalOpps.tracked_markets ? globalOpps.tracked_markets.find(m => m.team === currentCountry) : null;
    let overviewHtml = '';
    if (opp) {
        overviewHtml = `
            <div>
                <div class="text-xs text-neutral tracking-wider mb-1">MARKET PROB</div>
                <div class="text-2xl font-bold">${(opp.p_market*100).toFixed(1)}%</div>
            </div>
            <div>
                <div class="text-xs text-neutral tracking-wider mb-1">MODEL EV</div>
                <div class="text-2xl font-bold ${opp.ev_percent > 0 ? 'text-positive' : 'text-neutral'}">${opp.ev_percent > 0 ? '+' : ''}${opp.ev_percent.toFixed(2)}%</div>
            </div>
            <div>
                <div class="text-xs text-neutral tracking-wider mb-1">KELLY STAKE</div>
                <div class="text-2xl font-bold text-highlight">${(opp.kelly_fraction*100).toFixed(1)}%</div>
            </div>
        `;
    } else {
        overviewHtml = `<div class="col-span-3 text-neutral">No outright market data available for ${currentCountry}.</div>`;
    }
    document.getElementById('country-overview').innerHTML = overviewHtml;
    
    // Fixtures
    const cMatches = globalMatches.filter(m => m.home_team === currentCountry || m.away_team === currentCountry);
    const fixBody = document.getElementById('country-fixtures-tbody');
    
    if (cMatches.length === 0) {
        fixBody.innerHTML = '<tr><td colspan="4" class="text-neutral py-4">No fixtures found.</td></tr>';
    } else {
        fixBody.innerHTML = cMatches.map(match => {
            const dateStr = formatShortDate(match.date);
            const isLive = match.status !== "Not Started";
            const matchupStr = `${match.home_team} - ${match.away_team}`;
            
            let statusStr = match.status;
            if (isLive && match.home_goals !== null) {
                statusStr = `<span class="font-bold text-[14px]">${match.home_goals} - ${match.away_goals}</span> <span class="text-xs text-neutral">(${match.status})</span>`;
            }
            
            const pHome = (match["1X2"].home * 100).toFixed(0);
            const pDraw = (match["1X2"].draw * 100).toFixed(0);
            const pAway = (match["1X2"].away * 100).toFixed(0);
            const pred1X2 = `H: ${pHome}% | D: ${pDraw}% | A: ${pAway}%`;
            
            return `
                <tr class="border-b border-gray-100 dark:border-gray-800/50 hover:bg-slate-50 dark:hover:bg-slate-800/50">
                    <td class="text-left text-neutral text-xs py-3">${dateStr}</td>
                    <td class="text-left font-bold py-3">${matchupStr}</td>
                    <td class="py-3">${statusStr}</td>
                    <td class="text-xs py-3">${pred1X2}</td>
                </tr>
            `;
        }).join('');
    }
    
    // Player Props
    const propsBody = document.getElementById('country-props-tbody');
    let propsHtml = '';
    
    if (globalProps && Array.isArray(globalProps)) {
        // Find all player props for matches involving this country
        const relevantProps = globalProps.filter(p => p.match.includes(currentCountry));
        
        let playerStats = {};
        
        relevantProps.forEach(pData => {
            if (pData.player_props && pData.player_props[currentCountry]) {
                const pp = pData.player_props[currentCountry];
                if (pp.anytime_goalscorer) {
                    for (const [player, prob] of Object.entries(pp.anytime_goalscorer)) {
                        if (!playerStats[player]) playerStats[player] = {};
                        playerStats[player].goal = (prob * 100).toFixed(0) + '%';
                    }
                }
                if (pp["over_under_shots_on_target_1.5"]) {
                    for (const [player, odds] of Object.entries(pp["over_under_shots_on_target_1.5"])) {
                        if (!playerStats[player]) playerStats[player] = {};
                        playerStats[player].sot = (odds.over * 100).toFixed(0) + '%';
                    }
                }
                if (pp["over_under_assists_0.5"]) {
                    for (const [player, odds] of Object.entries(pp["over_under_assists_0.5"])) {
                        if (!playerStats[player]) playerStats[player] = {};
                        playerStats[player].ast = (odds.over * 100).toFixed(0) + '%';
                    }
                }
            }
        });
        
        const sortedPlayers = Object.keys(playerStats).sort();
        if (sortedPlayers.length > 0) {
            propsHtml = sortedPlayers.map(player => {
                const s = playerStats[player];
                return `
                    <tr class="border-b border-gray-100 dark:border-gray-800/50 hover:bg-slate-50 dark:hover:bg-slate-800/50">
                        <td class="text-left font-bold py-2">${player}</td>
                        <td class="py-2">${s.goal || '-'}</td>
                        <td class="py-2">${s.sot || '-'}</td>
                        <td class="py-2">${s.ast || '-'}</td>
                    </tr>
                `;
            }).join('');
        }
    }
    
    propsBody.innerHTML = propsHtml || '<tr><td colspan="4" class="text-neutral py-4">No player prop data available</td></tr>';
}

const WORLD_CUP_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Tunisia", "Ukraine"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Uzbekistan", "Colombia", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"]
};

function renderGroupAnalysis() {
    const container = document.getElementById('group-analysis-container');
    if (!groupAnalysisData || Object.keys(groupAnalysisData).length === 0) {
        container.innerHTML = '<div class="text-neutral italic">Data not available. Waiting for backend...</div>';
        return;
    }
    
    let html = '';
    for (const [groupName, teams] of Object.entries(WORLD_CUP_GROUPS)) {
        const sortedTeams = [...teams].sort((a, b) => {
            const dataA = groupAnalysisData[a] || { "1st": 0, "2nd": 0, "3rd_adv": 0 };
            const dataB = groupAnalysisData[b] || { "1st": 0, "2nd": 0, "3rd_adv": 0 };
            const pA = dataA["1st"] + dataA["2nd"] + dataA["3rd_adv"];
            const pB = dataB["1st"] + dataB["2nd"] + dataB["3rd_adv"];
            return pB - pA;
        });
        
        let tableRows = sortedTeams.map(t => {
            const data = groupAnalysisData[t] || { "1st": 0, "2nd": 0, "3rd_adv": 0, "3rd_elim": 0, "4th": 0 };
            const pAdv = ((data["1st"] + data["2nd"] + data["3rd_adv"]) * 100).toFixed(1);
            const p1st = (data["1st"] * 100).toFixed(1);
            const p2nd = (data["2nd"] * 100).toFixed(1);
            const p3rdAdv = (data["3rd_adv"] * 100).toFixed(1);
            return `
                <tr class="border-b border-gray-100 dark:border-gray-800/50 hover:bg-slate-50 dark:hover:bg-slate-800/50">
                    <td class="py-2 text-left font-semibold text-sm">${t}</td>
                    <td class="py-2 text-center text-sm font-bold text-green-600 dark:text-green-400">${pAdv}%</td>
                    <td class="py-2 text-center text-xs text-neutral">${p1st}%</td>
                    <td class="py-2 text-center text-xs text-neutral">${p2nd}%</td>
                    <td class="py-2 text-center text-xs text-neutral">${p3rdAdv}%</td>
                </tr>
            `;
        }).join('');
        
        html += `
            <div class="bg-slate-50 dark:bg-slate-800/30 border border-gray-100 dark:border-gray-800 rounded-lg p-4">
                <h3 class="font-bold text-accent mb-3 text-sm">GROUP ${groupName}</h3>
                <table class="w-full">
                    <thead>
                        <tr class="border-b border-gray-200 dark:border-gray-700 text-xs text-neutral uppercase">
                            <th class="py-2 text-left font-medium">Team</th>
                            <th class="py-2 text-center font-bold text-green-600 dark:text-green-400">Adv</th>
                            <th class="py-2 text-center font-medium">1st</th>
                            <th class="py-2 text-center font-medium">2nd</th>
                            <th class="py-2 text-center font-medium">3rd(Q)</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${tableRows}
                    </tbody>
                </table>
            </div>
        `;
    }
    container.innerHTML = html;
}

let bracketState = {
    r16: [
        {id: "r16-1", t1: "Mexico", t2: "Canada", winner: null},
        {id: "r16-2", t1: "Brazil", t2: "United States", winner: null},
        {id: "r16-3", t1: "Germany", t2: "Netherlands", winner: null},
        {id: "r16-4", t1: "Belgium", t2: "Spain", winner: null},
        {id: "r16-5", t1: "France", t2: "Argentina", winner: null},
        {id: "r16-6", t1: "Portugal", t2: "England", winner: null},
        {id: "r16-7", t1: "Uruguay", t2: "Colombia", winner: null},
        {id: "r16-8", t1: "Senegal", t2: "Croatia", winner: null}
    ],
    qf: [
        {id: "qf-1", t1: null, t2: null, winner: null},
        {id: "qf-2", t1: null, t2: null, winner: null},
        {id: "qf-3", t1: null, t2: null, winner: null},
        {id: "qf-4", t1: null, t2: null, winner: null}
    ],
    sf: [
        {id: "sf-1", t1: null, t2: null, winner: null},
        {id: "sf-2", t1: null, t2: null, winner: null}
    ],
    f: [
        {id: "f-1", t1: null, t2: null, winner: null}
    ]
};

function advanceTeam(round, matchIndex, team) {
    bracketState[round][matchIndex].winner = team;
    
    // Propagate to next round
    if (round === 'r16') {
        const nextMatch = Math.floor(matchIndex / 2);
        const slot = matchIndex % 2 === 0 ? 't1' : 't2';
        bracketState.qf[nextMatch][slot] = team;
        // reset downstream
        bracketState.qf[nextMatch].winner = null;
        bracketState.sf[Math.floor(nextMatch/2)][nextMatch%2===0?'t1':'t2'] = null;
        bracketState.sf[Math.floor(nextMatch/2)].winner = null;
        bracketState.f[0][Math.floor(nextMatch/2)===0?'t1':'t2'] = null;
        bracketState.f[0].winner = null;
    } else if (round === 'qf') {
        const nextMatch = Math.floor(matchIndex / 2);
        const slot = matchIndex % 2 === 0 ? 't1' : 't2';
        bracketState.sf[nextMatch][slot] = team;
        bracketState.sf[nextMatch].winner = null;
        bracketState.f[0][Math.floor(nextMatch/2)===0?'t1':'t2'] = null;
        bracketState.f[0].winner = null;
    } else if (round === 'sf') {
        const slot = matchIndex === 0 ? 't1' : 't2';
        bracketState.f[0][slot] = team;
        bracketState.f[0].winner = null;
    }
    
    renderKnockoutBracket();
}

function renderMatchBox(match, round, matchIndex) {
    let html = `<div class="bg-white dark:bg-slate-800 border border-gray-200 dark:border-gray-700 rounded p-3 w-64 shadow-sm mb-4">`;
    
    const t1 = match.t1 || "TBD";
    const t2 = match.t2 || "TBD";
    
    let p1 = "-", p2 = "-";
    if (match.t1 && match.t2 && h2hMatrix && h2hMatrix[match.t1] && h2hMatrix[match.t1][match.t2]) {
        const h2h = h2hMatrix[match.t1][match.t2];
        const win1 = h2h.home + (h2h.draw / 2);
        const win2 = h2h.away + (h2h.draw / 2);
        p1 = (win1 * 100).toFixed(1) + "%";
        p2 = (win2 * 100).toFixed(1) + "%";
    }
    
    const btnClass1 = match.winner === match.t1 ? "bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-400 font-bold" : "hover:bg-gray-50 dark:hover:bg-slate-700 cursor-pointer";
    const btnClass2 = match.winner === match.t2 ? "bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-400 font-bold" : "hover:bg-gray-50 dark:hover:bg-slate-700 cursor-pointer";

    html += `
        <div class="flex justify-between items-center p-1.5 rounded ${match.t1 ? btnClass1 : 'text-gray-400'}" 
             ${match.t1 && match.t2 ? `onclick="advanceTeam('${round}', ${matchIndex}, '${match.t1}')"` : ''}>
            <span class="truncate">${t1}</span>
            <span class="text-xs font-bold">${p1}</span>
        </div>
        <div class="h-px bg-gray-100 dark:bg-gray-700 my-1"></div>
        <div class="flex justify-between items-center p-1.5 rounded ${match.t2 ? btnClass2 : 'text-gray-400'}"
             ${match.t1 && match.t2 ? `onclick="advanceTeam('${round}', ${matchIndex}, '${match.t2}')"` : ''}>
            <span class="truncate">${t2}</span>
            <span class="text-xs font-bold">${p2}</span>
        </div>
    </div>`;
    return html;
}

function renderKnockoutBracket() {
    const container = document.getElementById('knockout-bracket-container');
    if (!container) return;
    
    let html = `
        <div class="flex flex-col justify-around">
            <h3 class="text-center font-bold text-xs text-neutral mb-4 uppercase">Round of 16</h3>
            ${bracketState.r16.map((m, i) => renderMatchBox(m, 'r16', i)).join('')}
        </div>
        <div class="flex flex-col justify-around">
            <h3 class="text-center font-bold text-xs text-neutral mb-4 uppercase">Quarter Finals</h3>
            ${bracketState.qf.map((m, i) => renderMatchBox(m, 'qf', i)).join('')}
        </div>
        <div class="flex flex-col justify-around">
            <h3 class="text-center font-bold text-xs text-neutral mb-4 uppercase">Semi Finals</h3>
            ${bracketState.sf.map((m, i) => renderMatchBox(m, 'sf', i)).join('')}
        </div>
        <div class="flex flex-col justify-around">
            <h3 class="text-center font-bold text-xs text-neutral mb-4 uppercase">Final</h3>
            ${bracketState.f.map((m, i) => renderMatchBox(m, 'f', i)).join('')}
            ${bracketState.f[0].winner ? `<div class="mt-4 p-4 bg-yellow-100 dark:bg-yellow-900/30 border border-yellow-300 dark:border-yellow-700 rounded text-center">
                <span class="text-yellow-800 dark:text-yellow-400 font-bold text-lg text-center w-full block">🏆 ${bracketState.f[0].winner} 🏆</span>
            </div>` : ''}
        </div>
    `;
    container.innerHTML = html;
}

document.getElementById('reset-bracket-btn')?.addEventListener('click', () => {
    bracketState.r16.forEach(m => m.winner = null);
    ['qf', 'sf', 'f'].forEach(round => {
        bracketState[round].forEach(m => { m.t1 = null; m.t2 = null; m.winner = null; });
    });
    renderKnockoutBracket();
});

fetchPipelineData();
setInterval(fetchPipelineData, 60000);
