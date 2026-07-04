/* app.js — Morning Brief Homepage (Updated) */

const WATCHLIST_STORAGE_KEY = 'opr_watchlist';

document.addEventListener('DOMContentLoaded', async () => {
  document.querySelectorAll('.navbar-nav a').forEach(a => {
    const href = a.getAttribute('href');
    if (href === 'index.html' || href === './index.html') a.classList.add('active');
  });

  const [snapshot, macroNews, sectorNews, sectorsOverview, earningsCalendar, maDeals] = await Promise.all([
    fetchJSON('./data/market_snapshot.json'),
    fetchJSON('./data/macro_news.json'),
    fetchJSON('./data/sector_news.json'),
    fetchJSON('./data/sectors_overview.json'),
    fetchJSON('./data/earnings_calendar.json'),
    fetchJSON('./data/ma_deals.json'),
  ]);

  const watchlistData = await loadWatchlistData();

  renderPageHeader(snapshot);
  renderTickerStrip(snapshot);

  if (snapshot) {
    renderSnapshotGroup('equitiesRow',    snapshot.equities,    'Equities & Volatility');
    renderSnapshotGroup('ratesRow',       snapshot.rates,       'Rates');
    renderSnapshotGroup('commoditiesRow', snapshot.commodities, 'Commodities');
    renderFXRow('fxRow', snapshot.fx || []);
  } else {
    document.getElementById('snapshotSection').innerHTML =
      '<div class="error-state">Market snapshot data could not be loaded.</div>';
  }

  // Global markets + calendar rendered together
  if (snapshot?.global_markets) {
    renderGlobalMarkets(snapshot.global_markets);
  }
  
  // Load earnings calendar from JSON (replaces hardcoded version)
  if (earningsCalendar) {
    renderEarningsCalendarFromData(earningsCalendar);
  } else {
    renderEarningsCalendarFromData(null);  // Fallback to empty state
  }

  if (macroNews?.stories?.length) {
    const sorted = sortByTimestamp(macroNews.stories);
    setInner('macroStoriesContainer', sorted.slice(0, 15).map(renderStoryRow).join(''));
  } else {
    setInner('macroStoriesContainer', '<div class="error-state">Macro stories could not be loaded.</div>');
  }

  if (sectorsOverview?.sectors?.length) {
    renderSectorStrip(sectorsOverview.sectors);
  } else {
    setInner('sectorStripContainer', '<div class="error-state">Sector data unavailable.</div>');
  }

  if (sectorNews?.stories?.length) {
    const sorted = sortByTimestamp(sectorNews.stories);
    setInner('keyDevsContainer', sorted.slice(0, 15).map(renderStoryRow).join(''));
  } else {
    setInner('keyDevsContainer', '<div class="error-state">Development stories could not be loaded.</div>');
  }

  // Optional: Render M&A deals if container exists
  if (maDeals?.deals?.length && document.getElementById('maDealsContainer')) {
    renderMADeals(maDeals.deals.slice(0, 10));
  }

  if (watchlistData?.length) {
    renderWatchlistPreview(watchlistData.slice(0, 5));
  } else {
    setInner('watchlistPreviewContainer', '<div class="error-state">Watchlist unavailable.</div>');
  }
});

/* ============================================================
   WATCHLIST DATA LOADING
   ============================================================ */

async function loadWatchlistData() {
  try {
    const stored = localStorage.getItem(WATCHLIST_STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      if (Array.isArray(parsed) && parsed.length) return parsed;
    }
  } catch (e) {
    console.warn('localStorage watchlist parse error:', e);
  }
  const data = await fetchJSON('./data/watchlist.json');
  if (data?.watchlist?.length) {
    try { localStorage.setItem(WATCHLIST_STORAGE_KEY, JSON.stringify(data.watchlist)); } catch (e) {}
    return data.watchlist;
  }
  return [];
}

/* ============================================================
   PAGE HEADER
   ============================================================ */

function renderPageHeader(snapshot) {
  if (!snapshot) return;
  const el = document.getElementById('macroSummaryLine');
  if (el) el.textContent = snapshot.macro_summary || '';
  const ts = document.getElementById('lastUpdated');
  if (ts) ts.textContent = snapshot.last_updated ? 'Updated ' + formatTimestamp(snapshot.last_updated) : '';
  const ms = document.getElementById('marketStatus');
  if (!ms) return;
  ms.textContent = snapshot.market_status || 'Unknown';
  ms.classList.remove('premarket', 'closed');
  const lower = (snapshot.market_status || '').toLowerCase();
  if (lower.includes('pre'))    ms.classList.add('premarket');
  else if (lower.includes('closed')) ms.classList.add('closed');
}

/* ============================================================
   TICKER STRIP
   ============================================================ */

function renderTickerStrip(snapshot) {
  const container = document.getElementById('tickerStrip');
  if (!container) return;
  if (!snapshot) { container.innerHTML = '<div class="loading-state">Market data unavailable.</div>'; return; }

  const items = [
    ...(Array.isArray(snapshot.equities)    ? snapshot.equities         : []),
    ...(Array.isArray(snapshot.rates)       ? snapshot.rates.slice(0,2) : []),
    ...(Array.isArray(snapshot.fx)          ? snapshot.fx               : []),
    ...(Array.isArray(snapshot.commodities) ? snapshot.commodities      : [])
  ];

  container.innerHTML = items.map(item => {
    const chg = item.change ?? 0;
    const pct = item.percent_change ?? 0;
    return `
      <div class="ticker-chip">
        <span class="ticker-label">${item.ticker || item.label || ''}</span>
        <span class="ticker-val">${formatPrice(item.price)}</span>
        <span class="ticker-chg ${dirClass(chg)}">${dirArrow(chg)} ${formatPct(pct)}</span>
      </div>`;
  }).join('');
}

/* ============================================================
   SNAPSHOT CARDS
   ============================================================ */

function renderSnapshotGroup(containerId, items, label) {
  const container = document.getElementById(containerId);
  if (!container) return;
  if (!Array.isArray(items) || !items.length) {
    container.innerHTML = '<div class="error-state">Data unavailable.</div>';
    return;
  }
  container.innerHTML = items.map(item => {
    const chg = item.change ?? 0;
    const pct = item.percent_change ?? 0;
    return `
      <div class="snapshot-card">
        <div class="card-label">${item.label || ''}</div>
        <div class="card-value">${formatPrice(item.price)}</div>
        <div class="card-changes">
          <span class="card-abs ${dirClass(chg)}"><span class="dir-arrow">${dirArrow(chg)}</span>${formatSigned(chg)}</span>
          <span class="card-pct ${dirClass(pct)}">${formatPct(pct)}</span>
        </div>
      </div>`;
  }).join('');
}

function renderFXRow(containerId, items) {
  const container = document.getElementById(containerId);
  if (!container) return;
  if (!Array.isArray(items) || !items.length) {
    container.innerHTML = '<div class="error-state">FX data unavailable.</div>';
    return;
  }
  container.innerHTML = items.map(item => {
    const chg = item.change ?? 0;
    const pct = item.percent_change ?? 0;
    return `
      <div class="snapshot-card">
        <div class="card-label">${item.label || ''}</div>
        <div class="card-value">${formatPrice(item.price)}</div>
        <div class="card-changes">
          <span class="card-abs ${dirClass(chg)}"><span class="dir-arrow">${dirArrow(chg)}</span>${formatSigned(chg)}</span>
          <span class="card-pct ${dirClass(pct)}">${formatPct(pct)}</span>
        </div>
      </div>`;
  }).join('');
}

/* ============================================================
   GLOBAL MARKETS
   ============================================================ */

function renderGlobalMarkets(globalData) {
  const renderBlock = (containerId, items) => {
    const container = document.getElementById(containerId);
    if (!container) return;
    if (!Array.isArray(items) || !items.length) {
      container.innerHTML = '<div class="error-state">Data unavailable.</div>';
      return;
    }
    container.innerHTML = items.map(item => {
      const pct = item.percent_change ?? 0;
      return `
        <div class="global-row">
          <span class="global-row-label">${item.label || ''}</span>
          <div class="global-row-right">
            <span class="global-row-price">${formatPrice(item.price)}</span>
            <span class="global-row-pct ${dirClass(pct)}">${dirArrow(pct)} ${formatPct(pct)}</span>
          </div>
        </div>`;
    }).join('');
  };
  renderBlock('asiaMarkets',   globalData.asia);
  renderBlock('europeMarkets', globalData.europe);
}

/* ============================================================
   EARNINGS CALENDAR (JSON-DRIVEN)
   Replaces the hardcoded version. Auto-updates on Mondays.
   ============================================================ */

function renderEarningsCalendarFromData(data) {
  const container = document.getElementById('calendarContainer');
  if (!container) return;

  // If no data, render empty state
  if (!data || !data.calendar || !Array.isArray(data.calendar)) {
    container.innerHTML = '<div style="padding: 20px; text-align: center; color: var(--muted);">Calendar data unavailable. Refresh to reload.</div>';
    return;
  }

  const calendar = data.calendar;
  const todayKey = formatDateKey(new Date());

  container.innerHTML = calendar.map((dayData, i) => {
    const key = dayData.date;
    const isToday = key === todayKey;
    const events = dayData.events || [];

    const eventsHtml = events.length
      ? events.map(ev => `
          <div class="calendar-event ${ev.type}">
            ${ev.url
              ? `<a href="${ev.url}" target="_blank" rel="noopener">${ev.title}</a>`
              : ev.title}
            ${ev.time ? `<span class="calendar-event-time">${ev.time}</span>` : ''}
          </div>`).join('')
      : `<div style="font-size:0.65rem;color:var(--text-muted);padding:4px 0;">—</div>`;

    return `
      <div class="calendar-day ${isToday ? 'today' : ''}">
        <div class="calendar-day-header">
          <span class="calendar-day-name">${dayData.day}</span>
          <span class="calendar-day-date">${new Date(dayData.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}</span>
        </div>
        <div class="calendar-events">${eventsHtml}</div>
      </div>`;
  }).join('');
}

function formatDateKey(d) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

/* ============================================================
   M&A DEALS (NEW)
   ============================================================ */

function renderMADeals(deals) {
  const container = document.getElementById('maDealsContainer');
  if (!container) return;

  container.innerHTML = deals.map(deal => `
    <div class="ma-deal-card">
      <div class="ma-deal-title">${deal.title}</div>
      <div class="ma-deal-meta">
        <span>${deal.acquirer} → ${deal.target}</span>
        ${deal.value_usd_millions ? `<span class="ma-value">$${deal.value_usd_millions}M</span>` : ''}
      </div>
      <div class="ma-deal-date">${deal.announced_date || 'TBD'} • ${deal.status}</div>
      ${deal.url ? `<a href="${deal.url}" target="_blank" rel="noopener">Read more →</a>` : ''}
    </div>
  `).join('');
}

/* ============================================================
   SECTOR STRIP
   ============================================================ */

function renderSectorStrip(sectors) {
  const container = document.getElementById('sectorStripContainer');
  if (!container) return;
  container.innerHTML = sectors.map(s => `
    <a class="sector-chip" href="sectors.html?sector=${encodeURIComponent(s.sector || '')}">
      <div class="sector-chip-name">${s.sector || ''}</div>
      <div class="sector-chip-sentiment ${sentimentClass(s.sentiment)}">${s.sentiment || ''}</div>
      <div class="sector-chip-driver">${s.primary_driver || ''}</div>
      <div class="sector-chip-count">${s.story_count ?? 0} stories</div>
    </a>`).join('');
}

/* ============================================================
   WATCHLIST PREVIEW
   ============================================================ */

function renderWatchlistPreview(items) {
  const container = document.getElementById('watchlistPreviewContainer');
  if (!container) return;
  if (!items.length) { container.innerHTML = '<div class="empty-state">No watchlist items.</div>'; return; }
  container.innerHTML = items.map(item => {
    const move     = item.percent_move ?? 0;
    const alertCls = alertTagClass(item.alert_tag || '');
    return `
      <div class="watchlist-card">
        <div>
          <div class="watchlist-card-ticker">${item.ticker || ''}</div>
          <div class="watchlist-card-sector">${item.sector || ''}</div>
          <div style="margin-top:10px;"><span class="badge ${alertCls}">${item.alert_tag || ''}</span></div>
        </div>
        <div class="watchlist-card-notes">${item.notes || ''}</div>
        <div class="watchlist-card-meta">
          <div class="watchlist-card-price">${formatCurrency(item.price)}</div>
          <div class="watchlist-card-move ${dirClass(move)}">${dirArrow(move)} ${formatPct(move)}</div>
        </div>
      </div>`;
  }).join('');
}

function setInner(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html;
}
