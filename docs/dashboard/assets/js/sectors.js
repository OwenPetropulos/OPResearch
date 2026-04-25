/* sectors.js — Sector Intelligence Page */

document.addEventListener('DOMContentLoaded', async () => {
  // Load data
  const [sectorsOverview, sectorNews] = await Promise.all([
    fetchJSON('data/sectors_overview.json'),
    fetchJSON('data/sector_news.json')
  ]);

  if (!sectorsOverview?.sectors) {
    document.getElementById('sectorsLayout').innerHTML =
      '<div class="error-state">Could not load sector data.</div>';
    return;
  }

  const sectors = sectorsOverview.sectors;
  const stories = sectorNews?.stories || [];

  // Determine which sector to show initially
  const params = new URLSearchParams(location.search);
  const initialSector = params.get('sector') || sectors[0]?.sector;

  // Build sidebar navigation
  renderSectorNav(sectors, initialSector);

  // Build all sector panels
  renderSectorPanels(sectors, stories);

  // Show initial sector
  activateSector(initialSector);
});

/* --- Sidebar Nav --- */
function renderSectorNav(sectors, activeSector) {
  const container = document.getElementById('sectorNavList');
  if (!container) return;

  const sentimentColors = {
    'Positive': '#22c55e',
    'Negative': '#ef4444',
    'Neutral':  '#f59e0b'
  };

  container.innerHTML = sectors.map(s => `
    <a
      class="sector-nav-item ${s.sector === activeSector ? 'active' : ''}"
      href="#"
      data-sector="${s.sector}"
      onclick="activateSector('${s.sector}'); return false;"
    >
      <span>${s.sector}</span>
      <span class="sector-nav-dot" style="background:${sentimentColors[s.sentiment] || '#435570'}"></span>
    </a>
  `).join('');
}

/* --- Sector Panels --- */
function renderSectorPanels(sectors, stories) {
  const container = document.getElementById('sectorPanelsContainer');
  if (!container) return;

  container.innerHTML = sectors.map(s => {
    const sectorStories = stories.filter(st => st.sector === s.sector || (st.sector_tags && st.sector_tags.includes(s.sector)));
    const mainStories   = sectorStories.filter(st => st.source_type !== 'Reddit').slice(0, 3);
    const redditStories = sectorStories.filter(st => st.source_type === 'Reddit').slice(0, 2);
    const allOther      = sectorStories.filter(st => !mainStories.includes(st) && !redditStories.includes(st)).slice(0, 2);

    return `
      <div class="sector-panel" id="panel-${sanitizeId(s.sector)}">

        <!-- Sector Header -->
        <div class="sector-panel-header">
          <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
            <div class="sector-panel-title">${s.sector}</div>
            <span class="badge ${s.sentiment === 'Positive' ? 'tag-positive' : s.sentiment === 'Negative' ? 'tag-negative' : 'tag-review'}">${s.sentiment}</span>
          </div>
          <div class="sector-panel-tone">${s.tone}</div>
          <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-muted);margin-bottom:8px;">Key Drivers</div>
          <div class="sector-panel-drivers">
            ${(s.key_drivers || []).map(d => `<div class="sector-driver-item">${d}</div>`).join('')}
          </div>
          <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-muted);margin-bottom:8px;margin-top:14px;">Key Tickers</div>
          <div class="sector-tickers-row">
            ${(s.key_tickers || []).map(t => `<span class="tag-ticker">${t}</span>`).join('')}
          </div>
        </div>

        <!-- Top Stories -->
        ${mainStories.length > 0 ? `
          <div class="section-header">
            <span class="section-title">Top Stories</span>
            <span class="section-count">${mainStories.length}</span>
          </div>
          ${mainStories.map(renderStoryRow).join('')}
        ` : ''}

        ${allOther.length > 0 ? allOther.map(renderStoryRow).join('') : ''}

        <!-- Reddit / Alternative Signals Block -->
        <div class="alt-signal-block" style="margin-top:20px;">
          <div class="alt-signal-title">Alternative Signals &amp; Retail Narrative</div>
          ${redditStories.length > 0
            ? redditStories.map(renderStoryRow).join('')
            : `<div style="font-size:0.8rem;color:var(--text-muted);">No Reddit signals for this sector at this time.</div>`
          }
          <div style="margin-top:16px;">
            <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-muted);margin-bottom:10px;">Trending Tickers (Social)</div>
            <div class="trending-tickers-row">
              ${(s.trending_tickers || []).map(t => `<span class="trending-ticker">${t}</span>`).join('')}
            </div>
          </div>
        </div>

        <!-- Supporting Data Placeholder -->
        <div class="alt-signal-block" style="margin-top:16px;">
          <div class="alt-signal-title">Supporting Data — Price &amp; Mention Activity</div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px;">
            <div>
              <div style="font-size:0.68rem;color:var(--text-muted);margin-bottom:8px;letter-spacing:0.06em;text-transform:uppercase;font-weight:600;">Price Movement (30d)</div>
              <div class="placeholder-chart">PRICE CHART — COMING SOON</div>
            </div>
            <div>
              <div style="font-size:0.68rem;color:var(--text-muted);margin-bottom:8px;letter-spacing:0.06em;text-transform:uppercase;font-weight:600;">Mention Frequency</div>
              <div class="placeholder-chart">MENTION CHART — COMING SOON</div>
            </div>
          </div>
        </div>

      </div>
    `;
  }).join('');
}

/* --- Activate / Switch Sector Panel --- */
function activateSector(sectorName) {
  // Update nav items
  document.querySelectorAll('.sector-nav-item').forEach(item => {
    item.classList.toggle('active', item.dataset.sector === sectorName);
  });

  // Show correct panel
  document.querySelectorAll('.sector-panel').forEach(panel => {
    panel.classList.remove('active');
  });

  const target = document.getElementById('panel-' + sanitizeId(sectorName));
  if (target) target.classList.add('active');
}

/* --- Utility: sanitize sector name for element ID --- */
function sanitizeId(str) {
  return str.replace(/\s+/g, '-').replace(/[^a-zA-Z0-9\-]/g, '').toLowerCase();
}
