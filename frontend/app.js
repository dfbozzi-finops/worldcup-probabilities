const ENDPOINTS = {
    opportunities: '/data/processed/opportunities.json',
    matches: '/data/processed/match_by_match.json',
    props: '/data/processed/props_probabilities.json'
};

/**
 * Main Data Orchestrator
 */
async function fetchPipelineData() {
    try {
        const [oppsRes, matchRes, propsRes] = await Promise.all([
            fetch(ENDPOINTS.opportunities).catch(() => null),
            fetch(ENDPOINTS.matches).catch(() => null),
            fetch(ENDPOINTS.props).catch(() => null)
        ]);

        const opps = oppsRes?.ok ? await oppsRes.json() : null;
        const matches = matchRes?.ok ? await matchRes.json() : null;
        const props = propsRes?.ok ? await propsRes.json() : null;

        renderOpportunities(opps);
        renderMatches(matches);
        renderProps(props);
        
        document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
    } catch (error) {
        console.error("Failed to load pipeline data:", error);
        document.getElementById('opportunities-container').innerHTML = 
            `<div class="text-red-400 p-4 bg-red-900/20 border border-red-500/50 rounded-lg">
                <strong>Error loading data.</strong><br>
                If testing locally, ensure you are running a local server (e.g. <code>python -m http.server</code>).
            </div>`;
    }
}

/**
 * Render Arbitrage Opportunities
 */
function renderOpportunities(data) {
    const container = document.getElementById('opportunities-container');
    if (!data) {
        container.innerHTML = `<div class="text-slate-400 italic p-4">No opportunities data available.</div>`;
        return;
    }

    container.innerHTML = '';
    
    if (Array.isArray(data) && data.length > 0) {
        data.forEach(opp => {
            // Highly profitable EV gets a different styling
            const evColor = opp.ev_percent > 100 ? 'text-emerald-400' : 'text-cyan-400';
            const borderGlow = opp.ev_percent > 150 ? 'border-emerald-500/50 shadow-[0_0_15px_rgba(16,185,129,0.15)]' : 'border-slate-600 hover:border-slate-400';
            
            const card = document.createElement('div');
            card.className = `flex items-center justify-between p-5 bg-slate-800/50 rounded-xl border transition-all ${borderGlow}`;
            
            card.innerHTML = `
                <div>
                    <h3 class="text-lg font-bold text-white tracking-wide">${opp.team}</h3>
                    <div class="text-sm text-slate-400 flex gap-6 mt-1.5">
                        <span class="flex items-center gap-1">
                            <span class="w-2 h-2 rounded-full bg-blue-500"></span>
                            Model: ${(opp.p_consensus * 100).toFixed(1)}%
                        </span>
                        <span class="flex items-center gap-1">
                            <span class="w-2 h-2 rounded-full bg-slate-500"></span>
                            Market: ${(opp.p_market * 100).toFixed(1)}%
                        </span>
                    </div>
                </div>
                <div class="text-right">
                    <div class="text-2xl font-black ${evColor}">+${opp.ev_percent.toFixed(1)}% <span class="text-sm font-semibold opacity-70">EV</span></div>
                    <div class="text-xs text-slate-400 mt-1 uppercase tracking-wider font-semibold">Allocation: <span class="text-white">${opp.kelly_fraction.toFixed(2)}%</span></div>
                </div>
            `;
            container.appendChild(card);
        });
    } else {
        container.innerHTML = `<div class="text-slate-400 italic p-4">No active +EV opportunities found meeting filter criteria.</div>`;
    }
}

/**
 * Render Match-by-Match Probabilities
 */
function renderMatches(data) {
    const container = document.getElementById('matches-container');
    if (!data || Object.keys(data).length === 0) {
        container.innerHTML = `<div class="text-slate-400 italic p-4">No match predictions available.</div>`;
        return;
    }

    container.innerHTML = '';
    
    // Flatten grouped matches to display
    let allMatches = [];
    for (const [group, matches] of Object.entries(data)) {
        allMatches.push(...matches);
    }
    
    // Render top 6 marquee matches
    allMatches.slice(0, 6).forEach(match => {
        const homeWin = (match.probabilities["1X2"].Home * 100).toFixed(1);
        const draw = (match.probabilities["1X2"].Draw * 100).toFixed(1);
        const awayWin = (match.probabilities["1X2"].Away * 100).toFixed(1);
        const o25 = (match.probabilities["Over/Under 2.5"].Over * 100).toFixed(1);
        const btts = (match.probabilities["BTTS"]["Yes"] * 100).toFixed(1);
        
        const teams = match.match.split(' vs ');
        
        const card = document.createElement('div');
        card.className = 'p-5 bg-slate-800/50 rounded-xl border border-slate-600/50 flex flex-col justify-between';
        card.innerHTML = `
            <div class="flex justify-between items-center mb-4">
                <span class="font-bold text-white truncate w-5/12">${teams[0]}</span>
                <span class="text-slate-500 text-xs font-bold uppercase tracking-widest w-2/12 text-center">vs</span>
                <span class="font-bold text-white truncate text-right w-5/12">${teams[1]}</span>
            </div>
            
            <!-- Custom Progress Bar for 1X2 Probabilities -->
            <div class="w-full bg-slate-700/50 rounded-full h-2 mb-2 flex overflow-hidden">
                <div class="bg-cyan-500 h-2 progress-anim" style="width: ${homeWin}%" title="Home: ${homeWin}%"></div>
                <div class="bg-slate-500 h-2 progress-anim" style="width: ${draw}%" title="Draw: ${draw}%"></div>
                <div class="bg-indigo-500 h-2 progress-anim" style="width: ${awayWin}%" title="Away: ${awayWin}%"></div>
            </div>
            
            <div class="flex justify-between text-[11px] text-slate-400 uppercase tracking-wider font-semibold mt-4 pt-4 border-t border-slate-700/50">
                <span>O2.5 Goals: <span class="text-white">${o25}%</span></span>
                <span>BTTS: <span class="text-white">${btts}%</span></span>
            </div>
        `;
        container.appendChild(card);
    });
}

/**
 * Render Advanced Props Engine
 */
function renderProps(data) {
    const container = document.getElementById('props-container');
    if (!data) {
        container.innerHTML = `<div class="text-slate-400 italic p-4">No stochastic prop data available.</div>`;
        return;
    }

    const team1 = Object.keys(data.team_props)[0];
    const team2 = Object.keys(data.team_props)[1];

    container.innerHTML = `
        <div class="mb-6 pb-4 border-b border-slate-700/50">
            <h3 class="text-xs text-slate-400 mb-2 uppercase tracking-widest font-bold">Marquee Matchup</h3>
            <div class="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-white to-slate-400">
                ${data.match}
            </div>
        </div>
        
        <h3 class="text-xs text-slate-400 mb-3 uppercase tracking-widest font-bold">Anytime Goalscorer Probabilities</h3>
        <div class="space-y-2 mb-6">
            ${Object.entries(data.player_props.anytime_goalscorer).map(([player, prob]) => `
                <div class="flex justify-between items-center bg-slate-800/40 border border-slate-700/30 p-3 rounded-lg hover:bg-slate-800 transition-colors">
                    <span class="text-sm font-medium text-slate-200">${player}</span>
                    <span class="text-sm font-black text-emerald-400">${(prob * 100).toFixed(1)}%</span>
                </div>
            `).join('')}
        </div>
        
        <h3 class="text-xs text-slate-400 mb-3 uppercase tracking-widest font-bold">Over/Under Corners (Line 4.5)</h3>
        <div class="grid grid-cols-2 gap-3 mb-2">
            <div class="bg-slate-800/40 border border-slate-700/30 p-3 rounded-lg text-center">
                <div class="text-xs text-slate-400 mb-1">${team1} (Over)</div>
                <div class="text-lg font-bold text-cyan-400">${(data.team_props[team1]["over_under_corners_4.5"].over * 100).toFixed(1)}%</div>
            </div>
            <div class="bg-slate-800/40 border border-slate-700/30 p-3 rounded-lg text-center">
                <div class="text-xs text-slate-400 mb-1">${team2} (Over)</div>
                <div class="text-lg font-bold text-cyan-400">${(data.team_props[team2]["over_under_corners_4.5"].over * 100).toFixed(1)}%</div>
            </div>
        </div>
    `;
}

// Bootstrap
fetchPipelineData();
setInterval(fetchPipelineData, 60000); // refresh every minute
