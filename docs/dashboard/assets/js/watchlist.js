/* watchlist.js — Watchlist Page with full add/remove CRUD + export */

const WATCHLIST_STORAGE_KEY = 'opr_watchlist';

let watchlistItems = [];

document.addEventListener('DOMContentLoaded', async () => {
  watchlistItems = await loadWatchlist();
  renderAll();
  wireForm();
});

/* ============================================================
   DATA LOADING
   ============================================================ */

async function loadWatchlist() {
  try {
    const stored = localStorage.getItem(WATCHLIST_STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      if (Array.isArray(parsed) && parsed.length) return parsed;
    }
  } catch (e) {
    console.warn('Watchlist localStorage parse error:', e);
  }

  const data = await fetchJSON('./data/watchlist.json');
  if (data?.watchlist?.length) {
    saveWatchlist(data.watchlist);
    return data.watchlist;
  }
  return [];
}

function saveWatchlist(items) {
  try {
    localStorage.setItem(WATCHLIST_STORAGE_KEY, JSON.stringify(items));
  } catch (e) {
    showToast('Could not save watchlist.', 'error');
  }
}

/* ============================================================
   RENDER
   ============================================================ */

function renderAll() {
  const countEl = document.getElementById('watchlistCount');
  if (countEl) countEl.textContent = watchlistItems.length;
  renderWatchlistFull(watchlistItems);
}

function renderWatchlistFull(items) {
  const container = document.getElementById('watchlistContainer');
  if (!container) return;

  if (!items.length) {
    container.innerHTML = `
      <div class="empty-state">
        No watchlist items. Add a ticker using the form above.
      </div>`;
    return;
  }

  container.innerHTML = `<div class="watchlist-stack">
    ${items.map((item, index) => renderWatchlistCard(item, index)).join('')}
  </div>`;
}

function renderWatchlistCard(item, index) {
  if (!item) return '';

  const ticker   = item.ticker       || '—';
  const sector   = item.sector       || '—';
  const price    = item.price        ?? null;
  const move     = item.percent_move ?? null;
  const alertTag = item.alert_tag    || 'No Major Change';
  const notes    = item.notes        || '';

  const moveClass  = move != null ? dirClass(move)  : 'flat';
  const moveArrow  = move != null ? dirArrow(move)  : '';
  const moveFmt    = move != null ? formatPct(move) : '—';
  const alertCls   = alertTagClass(alertTag);
  const sectorLink = `sectors.html?sector=${encodeURIComponent(sector)}`;

  return `
    <div class="watchlist-card" id="wl-card-${index}">
      <div>
        <div class="watchlist-card-ticker">${ticker}</div>
        <div class="watchlist-card-sector">${sector}</div>
        <div style="margin-top:10px; display:flex; flex-direction:column; gap:6px;">
          <span class="badge ${alertCls}">${alertTag}</span>
        </div>
      </div>

      <div>
        ${notes ? `<div class="watchlist-card-notes">${notes}</div>` : ''}
        <div style="margin-top:12px; display:flex; gap:12px; align-items:center;">
          <a href="${sectorLink}"
             style="font-size:0.72rem; color:var(--text-muted); font-weight:500;">
            → View ${sector} sector
          </a>
          <button
            onclick="removeItem(${index})"
            style="font-size:0.68rem; font-weight:600; color:var(--red);
                   background:var(--red-dim); border:1px solid rgba(239,68,68,0.2);
                   border-radius:3px; padding:3px 10px; cursor:pointer; letter-spacing:0.04em;"
          >Remove</button>
        </div>
      </div>

      <div class="watchlist-card-meta">
        <div class="watchlist-card-price">${price != null ? formatCurrency(price) : '—'}</div>
        <div class="watchlist-card-move ${moveClass}">
          ${moveArrow} ${moveFmt}
        </div>
      </div>
    </div>
  `;
}

/* ============================================================
   ADD / REMOVE
   ============================================================ */

function wireForm() {
  const addBtn    = document.getElementById('addTickerBtn');
  const exportBtn = document.getElementById('exportWatchlistBtn');

  if (addBtn)    addBtn.addEventListener('click', handleAddTicker);
  if (exportBtn) exportBtn.addEventListener('click', handleExportWatchlist);
}

function handleAddTicker() {
  const ticker   = (document.getElementById('wlTicker')?.value    || '').trim().toUpperCase();
  const sector   = (document.getElementById('wlSector')?.value    || '').trim();
  const alertTag = document.getElementById('wlAlertTag')?.value   || 'No Major Change';
  const notes    = (document.getElementById('wlNotes')?.value     || '').trim();
  const price    = parseFloat(document.getElementById('wlPrice')?.value  || '');
  const pctMove  = parseFloat(document.getElementById('wlPctMove')?.value || '');

  if (!ticker) {
    showToast('Ticker is required.', 'error');
    return;
  }

  if (watchlistItems.some(item => item.ticker === ticker)) {
    showToast(`${ticker} is already on your watchlist.`, 'error');
    return;
  }

  const newItem = {
    ticker,
    sector:       sector    || 'Uncategorized',
    alert_tag:    alertTag,
    notes:        notes     || '',
    price:        isNaN(price)   ? null : price,
    percent_move: isNaN(pctMove) ? null : pctMove,
  };

  watchlistItems.unshift(newItem);
  saveWatchlist(watchlistItems);
  renderAll();
  clearAddForm();
  showToast(`${ticker} added to watchlist.`, 'success');
}

function removeItem(index) {
  const ticker = watchlistItems[index]?.ticker || '';
  if (!confirm(`Remove ${ticker} from watchlist?`)) return;
  watchlistItems.splice(index, 1);
  saveWatchlist(watchlistItems);
  renderAll();
  showToast(`${ticker} removed.`, 'default');
}

function clearAddForm() {
  ['wlTicker', 'wlSector', 'wlNotes', 'wlPrice', 'wlPctMove'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = '';
  });
}

/* ============================================================
   EXPORT WATCHLIST TO JSON
   Downloads current localStorage watchlist as watchlist.json.
   Commit this file to docs/dashboard/data/watchlist.json so
   the pipeline fetches prices for all your tracked tickers.
   ============================================================ */

function handleExportWatchlist() {
  if (!watchlistItems.length) {
    showToast('Watchlist is empty — nothing to export.', 'error');
    return;
  }

  // Format matches what the pipeline reads from watchlist.json
  const output = { watchlist: watchlistItems };

  try {
    const blob = new Blob([JSON.stringify(output, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'watchlist.json';
    a.click();
    URL.revokeObjectURL(url);
    showToast('Watchlist exported. Commit this file to docs/dashboard/data/ to update live prices.', 'success');
  } catch (e) {
    showToast('Export failed: ' + e.message, 'error');
  }
}
