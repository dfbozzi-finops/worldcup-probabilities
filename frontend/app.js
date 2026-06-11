console.log("Terminal App Initialized");

const ENDPOINTS = {
    opportunities: '../data/processed/opportunities.json',
    matches: '../data/processed/match_by_match.json'
};

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

        const oppsData = (oppsRes && oppsRes.ok) ? await oppsRes.json() : null;
        const matchesData = (matchRes && matchRes.ok) ? await matchRes.json() : null;

        renderOpportunities(oppsData);
        renderMatches(matchesData);
        
    } catch (error) {
        console.error("Pipeline failure:", error);
        document.getElementById('opportunities-tbody').innerHTML = 
            `<tr><td colspan="5" class="text-negative text-center py-4">ERR: ${error.message}</td></tr>`;
    }
}

function renderOpportunities(data) {
    const tbody = document.getElementById('opportunities-tbody');
    if (!data || !data.tracked_markets) {
        tbody.innerHTML = `<tr><td colspan="5" class="text-neutral text-center py-4">NO DATA</td></tr>`;
        return;
    }

    const markets = data.tracked_markets.slice(0, 10); // top 10 already sorted by p_consensus descending
    document.getElementById('opp-count').textContent = `Total Tracked: ${data.tracked_markets.length}`;

    tbody.innerHTML = markets.map(opp => {
        const evColor = opp.ev_percent > 0 ? 'text-positive bg-positive' : 'text-neutral';
        const evVal = opp.ev_percent > 0 ? `+${opp.ev_percent.toFixed(2)}` : opp.ev_percent.toFixed(2);
        
        return `
            <tr>
                <td class="text-left font-bold text-white">${opp.team}</td>
                <td>${(opp.p_consensus * 100).toFixed(2)}</td>
                <td>${(opp.p_market * 100).toFixed(2)}</td>
                <td class="${evColor} font-bold">${evVal}</td>
                <td>${opp.kelly_fraction.toFixed(2)}</td>
            </tr>
        `;
    }).join('');
}

function renderMatches(data) {
    const liveBody = document.getElementById('completed-matches-tbody');
    const upBody = document.getElementById('upcoming-matches-tbody');
    
    if (!data || !Array.isArray(data)) return;

    let liveRows = [];
    let upRows = [];

    data.forEach(match => {
        const isLive = match.status !== "Not Started";
        
        if (isLive) {
            // Actual results
            const hg = match.home_goals || 0;
            const ag = match.away_goals || 0;
            const total = hg + ag;
            const btts = (hg > 0 && ag > 0);
            
            const o25Badge = total > 2.5 
                ? '<span class="bg-positive text-positive px-2 py-0.5 rounded text-[10px]">HIT</span>' 
                : '<span class="bg-negative text-negative px-2 py-0.5 rounded text-[10px]">MISS</span>';
                
            const bttsBadge = btts 
                ? '<span class="bg-positive text-positive px-2 py-0.5 rounded text-[10px]">HIT</span>' 
                : '<span class="bg-negative text-negative px-2 py-0.5 rounded text-[10px]">MISS</span>';

            liveRows.push(`
                <tr>
                    <td class="text-left font-bold text-white">${match.home_team} - ${match.away_team}</td>
                    <td class="text-sky-400">${match.status}</td>
                    <td class="font-bold text-[14px]">${hg} - ${ag}</td>
                    <td>${o25Badge}</td>
                    <td>${bttsBadge}</td>
                </tr>
            `);
        } else {
            // Upcoming Predictions
            const h = (match["1X2"].home * 100).toFixed(1);
            const d = (match["1X2"].draw * 100).toFixed(1);
            const a = (match["1X2"].away * 100).toFixed(1);
            const o25 = (match.over_under_2_5.over * 100).toFixed(1);
            const btts = (match.btts.yes * 100).toFixed(1);

            upRows.push(`
                <tr>
                    <td class="text-left font-bold text-white">${match.home_team} - ${match.away_team}</td>
                    <td class="text-neutral"><span class="text-sky-400">${h}</span> / ${d} / <span class="text-indigo-400">${a}</span></td>
                    <td class="text-emerald-400">${o25}%</td>
                    <td class="text-emerald-400">${btts}%</td>
                </tr>
            `);
        }
    });

    liveBody.innerHTML = liveRows.length ? liveRows.join('') : `<tr><td colspan="5" class="text-neutral py-4">No live/completed fixtures</td></tr>`;
    upBody.innerHTML = upRows.length ? upRows.slice(0, 10).join('') : `<tr><td colspan="4" class="text-neutral py-4">No upcoming fixtures</td></tr>`;
}

fetchPipelineData();
setInterval(fetchPipelineData, 60000);
