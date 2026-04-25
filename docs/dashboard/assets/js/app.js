/* app.js — Morning Brief Homepage */

document.addEventListener('DOMContentLoaded', async () => {
  // Set active nav link
  document.querySelectorAll('.navbar-nav a').forEach(a => {
    if (a.href === location.href || (a.getAttribute('href') === 'index.html' && location.pathname.endsWith('/'))) {
      a.classList.add('active');
    }
  });

  // Load all data concurrently
  const [snapshot, macroNews, sectorNews, watchlist, sectorsOverview] = await Promise.all([
    fetchJSON('data/market_snapshot.json'),
    fetchJSON('data/macro_news.json'),
    fetchJSON('data/sector_news.json'),
    fetchJSON('data/watchlist.json'),
    fetchJSON('data/sectors_overview.json')
  ]);

  // --- Page Header ---
  renderPageHeader(snapshot);

  // --- Ticker Strip ---
  renderTickerStrip(snapshot);

  // --- Macro Snapshot Cards ---
  if (snapshot) {
    renderSnapshotGroup('equitiesRow', snapshot.equities, 'Equities');
    renderSnapshotGroup('ratesRow', snapshot.rates, 'Rates');
    renderSnapshotGroup('commoditiesRow', snapshot.commodities, 'Commodities');
  } else {
    document.getElementById('snapshotSection').innerHTML = '<div class="error-state">Could not load market snapshot data.</div>';
  }

  // --- Global Markets ---
  if (snapshot?.global_markets) {
    renderGlobalMarkets(snapshot.global_markets);
  }

  // --- Overnight Macro Stories ---
  if (macroNews?.stories) {
    const container = document.getElementById('macroStoriesContainer');
    container.innerHTML = macroNews.stories.map(renderStoryRow).join('');
  } else {
    document.getElementById('macroStoriesContainer').innerHTML =
      '<div class="error-state">Could not load macro stories.</div>';
  }

  // --- Sector Overview Strip ---
  if (sectorsOverview?.sectors) {
    renderSectorStrip(sectorsOverview.sectors);
  }

  // --- Key Developments Feed (from sector_news) ---
  if (sectorNews?.stories) {
    const container = document.getElementById('keyDevsContainer');
    container.innerHTML = sectorNews.stories.slice(0, 8).map(renderStoryRow).join('');
  } else {
    document.getElementById('keyDevsContainer').innerHTML =
      '<div class="error-state">Could not load development stories.</div>';
  }

  // --- Watchlist Preview ---
  if (watchlist?.watchlist) {
    renderWatchlistPreview(watchlist.watchlist.slice(0, 5));
  }
});

/* --- Page Header --- */
function renderPageHeader(snapshot) {
  if (!snapshot) return;

  const el = document.getElementById('macroSummaryLine');
  if (el) el.textContent = snapshot.macro_summary || '';

  const ts = document.getElementById('lastUpdated');
  if (ts) ts.textContent = 'Updated ' + formatTimestamp(snapshot.last_updated);

  const ms = document.getElementById('marketStatus');
  if (ms) {
    const status = snapshot.market_status || 'Unknown';
    ms.textContent = status;
    if (status.toLowerCase().includes('pre')) ms.classList.add('premarket');
    else if (status.toLowerCase().includes('closed')) ms.classList.add('closed');
  }
}

/* --- Ticker Strip (top of page) --- */
function renderTickerStrip(snapshot) {
  if (!snapshot) return;
  const container = document.getElementById('tickerStrip');
  if (!container) return;

  const items = [
    ...(snapshot.equities || []),
    ...(snapshot.rates || []),
    ...(snapshot.commodities || [])
  ];

  container.innerHTML = items.map(item => `
    <div class="ticker-chip">
      <span class="ticker-label">${item.ticker || item.label}</span>
      <span class="ticker-val">${item.price.toLocaleString()}</span>
      <span class="ticker-chg ${dirClass(item.change)}">
        ${dirArrow(item.change)} ${formatPct(item.percent_change)}
      </span>
    </div>
  `).join('');
}

/* --- Snapshot Group Renderer --- */
function renderSnapshotGroup(containerId, items, label) {
  const container = document.getElementById(containerId);
  if (!container || !items) return;

  container.innerHTML = items.map(item => `
    <div class="snapshot-card">
      <div class="card-label">${item.label}</div>
      <div class="card-value">${item.price.toLocaleString()}</div>
      <div class="card-changes">
        <span class="card-abs ${dirClass(item.change)}">
          <span class="dir-arrow">${dirArrow(item.change)}</span>
          ${formatSigned(item.change)}
        </span>
        <span class="card-pct ${dirClass(item.percent_change)}">
          ${formatPct(item.percent_change)}
        </span>
      </div>
    </div>
  `).join('');
}

/* --- Global Markets --- */
function renderGlobalMarkets(globalData) {
  const renderBlock = (containerId, items) => {
    const container = document.getElementById(containerId);
    if (!container || !items) return;
    container.innerHTML = items.map(item => `
      <div class="global-row">
        <span class="global-row-label">${item.label}</span>
        <div class="global-row-right">
          <span class="global-row-price">${item.price.toLocaleString()}</span>
          <span class="global-row-pct ${dirClass(item.percent_change)}">
            ${dirArrow(item.percent_change)} ${formatPct(item.percent_change)}
          </span>
        </div>
      </div>
    `).join('');
  };

  renderBlock('asiaMarkets', globalData.asia);
  renderBlock('europeMarkets', globalData.europe);
}

/* --- Sector Strip --- */
function renderSectorStrip(sectors) {
  const container = document.getElementById('sectorStripContainer');
  if (!container) return;

  container.innerHTML = sectors.map(s => `
    <a class="sector-chip" href="sectors.html?sector=${encodeURIComponent(s.sector)}">
      <div class="sector-chip-name">${s.sector}</div>
      <div class="sector-chip-sentiment ${sentimentClass(s.sentiment)}">${s.sentiment}</div>
      <div class="sector-chip-driver">${s.primary_driver}</div>
      <div class="sector-chip-count">${s.story_count} stories</div>
    </a>
  `).join('');
}

/* --- Watchlist Preview --- */
function renderWatchlistPreview(items) {
  const container = document.getElementById('watchlistPreviewContainer');
  if (!container) return;

  container.innerHTML = items.map(item => `
    <div class="watchlist-card">
      <div>
        <div class="watchlist-card-ticker">${item.ticker}</div>
        <div class="watchlist-card-sector">${item.sector}</div>
        <div style="margin-top:10px;">
          <span class="badge ${alertTagClass(item.alert_tag)}">${item.alert_tag}</span>
        </div>
      </div>
      <div class="watchlist-card-notes">${item.notes}</div>
      <div class="watchlist-card-meta">
        <div class="watchlist-card-price">${formatCurrency(item.price)}</div>
        <div class="watchlist-card-move ${dirClass(item.percent_move)}">
          ${formatPct(item.percent_move)}
        </div>
      </div>
    </div>
  `).join('');
}
