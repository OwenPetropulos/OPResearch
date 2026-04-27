/* sectors.js — Sector Intelligence Page */

document.addEventListener('DOMContentLoaded', async () => {
  const [sectorsOverview, sectorNews] = await Promise.all([
    fetchJSON('./data/sectors_overview.json'),
    fetchJSON('./data/sector_news.json')
  ]);

  if (!sectorsOverview?.sectors?.length) {
    const layout = document.getElementById('sectorsLayout');
    if (layout) layout.innerHTML = '<div class="error-state">Sector data could not be loaded.</div>';
    return;
  }

  const sectors = sectorsOverview.sectors;
  const stories = Array.isArray(sectorNews?.stories) ? sectorNews.stories : [];

  const params        = new URLSearchParams(location.search);
  const initialSector = params.get('sector') || sectors[0]?.sector || '';

  renderSectorNav(sectors, initialSector);
  renderSectorPanels(sectors, stories);
  activateSector(initialSector);
});

function renderSectorNav(sectors, activeSector) {
  const container = document.getElementById('sectorNavList');
  if (!container) return;

  const sentimentColors = {
    'Positive': '#22c55e',
    'Negative': '#ef4444',
    'Neutral':  '#f59e0b'
  };

  container.innerHTML = sectors.map(s => {
    const dotColor = sentimentColors[s.sentiment] || '#435570';
    const isActive = s.sector === activeSector;
    return `
      <a class="sector-nav-item ${isActive ? 'active' : ''}" href="#"
         data-sector="${s.sector}"
         onclick="activateSector('${escapeSectorName(s.sector)}'); return false;">
        <span>${s.sector}</span>
        <span class="sector-nav-dot" style="background:${dotColor}"></span>
      </a>`;
  }).join('');
}

function renderSectorPanels(sectors, allStories) {
  const container = document.getElementById('sectorPanelsContainer');
  if (!container) return;

  container.innerHTML = sectors.map(s => {
    const relevant = allStories.filter(st =>
      st.sector === s.sector ||
      (Array.isArray(st.sector_tags) && st.sector_tags.includes(s.sector))
    );

    const sorted = sortByTimestamp(relevant);
    // Up to 10 stories per sector, compact layout
    const topStories = sorted.slice(0, 10);
    const redditStories = sorted.filter(st => st.source_type === 'Reddit').slice(0, 3);

    const sentimentBadgeClass = s.sentiment === 'Positive' ? 'tag-positive'
      : s.sentiment === 'Negative' ? 'tag-negative' : 'tag-review';

    return `
      <div class="sector-panel" id="panel-${sanitizeId(s.sector)}">

        <div class="sector-panel-header">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
            <div class="sector-panel-title">${s.sector || ''}</div>
            <span class="badge ${sentimentBadgeClass}">${s.sentiment || ''}</span>
          </div>
          <div class="sector-panel-tone">${s.tone || ''}</div>

          <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-muted);margin-bottom:8px;margin-top:12px;">Key Drivers</div>
          <div class="sector-panel-drivers">
            ${(Array.isArray(s.key_drivers) ? s.key_drivers : []).map(d => `<div class="sector-driver-item">${d}</div>`).join('')}
          </div>

          <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-muted);margin-bottom:8px;margin-top:14px;">Key Tickers</div>
          <div class="sector-tickers-row">
            ${(Array.isArray(s.key_tickers) ? s.key_tickers : []).map(t => `<span class="tag-ticker">${t}</span>`).join('')}
          </div>
        </div>

        ${topStories.length ? `
          <div class="section-header" style="margin-top:16px;">
            <span class="section-title">Stories</span>
            <span class="section-count">${topStories.length}</span>
          </div>
          ${topStories.map(renderStoryRow).join('')}
        ` : '<div style="font-size:0.8rem;color:var(--text-muted);padding:12px 0;">No stories for this sector in the current update cycle.</div>'}

        <div class="alt-signal-block" style="margin-top:16px;">
          <div class="alt-signal-title">Alternative Signals &amp; Retail Narrative</div>
          ${redditStories.length
            ? redditStories.map(renderStoryRow).join('')
            : `<div style="font-size:0.8rem;color:var(--text-muted);">No Reddit signals at this time.</div>`
          }
          ${Array.isArray(s.trending_tickers) && s.trending_tickers.length ? `
            <div style="margin-top:14px;">
              <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-muted);margin-bottom:8px;">Trending Tickers (Social)</div>
              <div class="trending-tickers-row">
                ${s.trending_tickers.map(t => `<span class="trending-ticker">${t}</span>`).join('')}
              </div>
            </div>` : ''}
        </div>

        <div class="alt-signal-block" style="margin-top:12px;">
          <div class="alt-signal-title">Supporting Data — Price &amp; Mention Activity</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px;">
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

      </div>`;
  }).join('');
}

function activateSector(sectorName) {
  document.querySelectorAll('.sector-nav-item').forEach(item => {
    item.classList.toggle('active', item.dataset.sector === sectorName);
  });
  document.querySelectorAll('.sector-panel').forEach(panel => {
    panel.classList.remove('active');
  });
  const target = document.getElementById('panel-' + sanitizeId(sectorName));
  if (target) target.classList.add('active');
}

function sanitizeId(str) {
  if (!str) return '';
  return str.replace(/\s+/g, '-').replace(/[^a-zA-Z0-9\-]/g, '').toLowerCase();
}

function escapeSectorName(str) {
  if (!str) return '';
  return str.replace(/'/g, "\\'");
}