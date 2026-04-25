/* watchlist.js — Watchlist Page */

document.addEventListener('DOMContentLoaded', async () => {
  const data = await fetchJSON('data/watchlist.json');

  if (!data?.watchlist) {
    document.getElementById('watchlistContainer').innerHTML =
      '<div class="error-state">Could not load watchlist data.</div>';
    return;
  }

  renderWatchlistFull(data.watchlist);

  // Update count in section header
  const countEl = document.getElementById('watchlistCount');
  if (countEl) countEl.textContent = data.watchlist.length;
});

function renderWatchlistFull(items) {
  const container = document.getElementById('watchlistContainer');
  if (!container) return;

  if (!items.length) {
    container.innerHTML = '<div class="empty-state">No watchlist items found.</div>';
    return;
  }

  container.innerHTML = `<div class="watchlist-stack">
    ${items.map(item => renderWatchlistCard(item)).join('')}
  </div>`;
}

function renderWatchlistCard(item) {
  const moveClass  = dirClass(item.percent_move);
  const moveArrow  = dirArrow(item.percent_move);
  const alertClass = alertTagClass(item.alert_tag);
  const sectorLink = `sectors.html?sector=${encodeURIComponent(item.sector)}`;

  return `
    <div class="watchlist-card">
      <!-- Left: Ticker + sector + alert -->
      <div>
        <div class="watchlist-card-ticker">${item.ticker}</div>
        <div class="watchlist-card-sector">${item.sector}</div>
        <div style="margin-top:10px; display:flex; flex-direction:column; gap:6px;">
          <span class="badge ${alertClass}">${item.alert_tag}</span>
          ${item.related_story_count > 0
            ? `<span style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);">${item.related_story_count} related stories</span>`
            : ''}
        </div>
      </div>

      <!-- Center: Notes + sector link -->
      <div>
        <div class="watchlist-card-notes">${item.notes}</div>
        <div style="margin-top:12px;">
          <a href="${sectorLink}" style="font-size:0.72rem; color:var(--text-muted); font-weight:500; letter-spacing:0.04em;">
            → View ${item.sector} sector
          </a>
        </div>
      </div>

      <!-- Right: Price + move -->
      <div class="watchlist-card-meta">
        <div class="watchlist-card-price">${formatCurrency(item.price)}</div>
        <div class="watchlist-card-move ${moveClass}">
          ${moveArrow} ${formatPct(item.percent_move)}
        </div>
      </div>
    </div>
  `;
}
