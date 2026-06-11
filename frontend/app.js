console.log("Terminal App Initialized");

const ENDPOINTS = {
    opportunities: '../data/processed/opportunities.json',
    matches: '../data/processed/match_by_match.json'
};

// Global State
let globalOpps = null;
let globalMatches = null;

let filterSearch = "";
let filterActionable = false;

// Theme Toggle
const themeBtn = document.getElementById('theme-toggle');
let isLightMode = false;
themeBtn.addEventListener('click', () => {
    isLightMode = !isLightMode;
    if (isLightMode) {
        document.body.classList.add('light-mode');
        themeBtn.textContent = '☾ Dark Mode';
    } else {
        document.body.classList.remove('light-mode');
        themeBtn.textContent = '☀ Light Mode';
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
            `<tr><td colspan="5" class="text-negative text-center py-4">ERR: ${error.message}</td></tr>`;
    }
}

function renderAll() {
    renderOpportunities();
    renderMatches();
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

    markets = markets.slice(0, 15); // Show up to 15 now that it's scrollable
    document.getElementById('opp-count').textContent = `Total Match: ${markets.length}`;

    tbody.innerHTML = markets.map(opp => {
        const isPos = opp.ev_percent > 0;
        const evColor = isPos ? 'text-positive bg-positive' : 'text-neutral';
        const evVal = isPos ? `+${opp.ev_percent.toFixed(2)}%` : `${opp.ev_percent.toFixed(2)}%`;
        
        // Visual edge bar
        const barWidth = isPos ? Math.min(100, opp.ev_percent / 2) : 0;
        const barHtml = isPos ? `<div class="edge-bar-container"><div class="edge-bar-fill" style="width: ${barWidth}%"></div></div>` : '';

        return `
            <tr>
                <td class="text-left font-bold text-accent">${opp.team}</td>
                <td>${(opp.p_consensus * 100).toFixed(1)}%</td>
                <td>${(opp.p_market * 100).toFixed(1)}%</td>
                <td class="${evColor} font-bold">${evVal} ${barHtml}</td>
                <td>${(opp.kelly_fraction * 100).toFixed(1)}%</td>
            </tr>
        `;
    }).join('');
    
    if (markets.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" class="text-neutral text-center py-4">No markets match filters</td></tr>`;
    }
}

function renderMatches() {
    const liveBody = document.getElementById('completed-matches-tbody');
    const upBody = document.getElementById('upcoming-matches-tbody');
    
    if (!globalMatches || !Array.isArray(globalMatches)) return;

    let matches = globalMatches;

    // Apply Filter
    if (filterSearch) {
        matches = matches.filter(match => 
            match.home_team.toLowerCase().includes(filterSearch) || 
            match.away_team.toLowerCase().includes(filterSearch)
        );
    }

    let liveRows = [];
    let upRows = [];

    matches.forEach(match => {
        const isLive = match.status !== "Not Started";
        
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
                    <td class="text-left font-bold">${match.home_team} - ${match.away_team}</td>
                    <td class="text-accent text-xs tracking-wider">${match.status}</td>
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
                    <td class="text-left font-bold">${match.home_team} - ${match.away_team}</td>
                    <td class="text-neutral"><span class="text-accent">${h}</span> / ${d} / <span class="text-indigo-400">${a}</span></td>
                    <td class="text-positive">${o15}%</td>
                    <td class="text-positive">${o25}%</td>
                    <td class="text-positive">${btts}%</td>
                </tr>
            `);
        }
    });

    liveBody.innerHTML = liveRows.length ? liveRows.join('') : `<tr><td colspan="6" class="text-neutral py-4 text-center">No live/completed matches found</td></tr>`;
    upBody.innerHTML = upRows.length ? upRows.slice(0, 15).join('') : `<tr><td colspan="5" class="text-neutral py-4 text-center">No upcoming fixtures found</td></tr>`;
}

fetchPipelineData();
setInterval(fetchPipelineData, 60000);
