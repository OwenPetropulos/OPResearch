/* utils.js — Shared utility functions */

/* ============================================================
   NUMBER FORMATTING
   ============================================================ */

function formatCurrency(val, decimals = 2) {
  if (val == null || isNaN(val)) return '—';
  const abs  = Math.abs(val);
  const sign = val < 0 ? '-' : '';
  return sign + '$' + abs.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
}

function formatPrice(val, decimals = 2) {
  if (val == null || isNaN(val)) return '—';
  return parseFloat(val).toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
}

function formatPct(val, decimals = 2) {
  if (val == null || isNaN(val)) return '—';
  const sign = val >= 0 ? '+' : '';
  return sign + parseFloat(val).toFixed(decimals) + '%';
}

function formatSigned(val, decimals = 2) {
  if (val == null || isNaN(val)) return '—';
  const sign = val >= 0 ? '+' : '';
  return sign + parseFloat(val).toFixed(decimals);
}

/* ============================================================
   DIRECTION HELPERS
   ============================================================ */

function dirClass(val) {
  if (val == null || isNaN(val)) return 'flat';
  if (val > 0) return 'up';
  if (val < 0) return 'down';
  return 'flat';
}

function dirArrow(val) {
  if (val == null || isNaN(val)) return '—';
  if (val > 0) return '▲';
  if (val < 0) return '▼';
  return '—';
}

/* ============================================================
   TIMESTAMP / TIME HELPERS
   ============================================================ */

function formatTimestamp(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return '—';
  return d.toLocaleString('en-US', {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', hour12: true
  });
}

function timeAgo(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return '';
  const diffMs  = Date.now() - d.getTime();
  const diffMin = Math.floor(diffMs / 60000);
  if (diffMin < 1)   return 'just now';
  if (diffMin < 60)  return diffMin + 'm ago';
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24)   return diffHr + 'h ago';
  const diffDay = Math.floor(diffHr / 24);
  return diffDay + 'd ago';
}

function sortByTimestamp(arr) {
  if (!Array.isArray(arr)) return [];
  return [...arr].sort((a, b) => {
    const ta = a.timestamp ? new Date(a.timestamp).getTime() : 0;
    const tb = b.timestamp ? new Date(b.timestamp).getTime() : 0;
    return tb - ta;
  });
}

/* ============================================================
   BADGE / TAG CLASS MAPPINGS
   ============================================================ */

function sourceBadgeClass(sourceType) {
  const map = {
    'Mainstream':  'badge-mainstream',
    'Reddit':      'badge-reddit',
    'Macro Data':  'badge-macro-data',
    'Filing / IR': 'badge-filing',
    'Niche / Blog':'badge-niche'
  };
  return map[sourceType] || 'badge-mainstream';
}

function alertTagClass(tag) {
  const map = {
    'Positive':        'tag-positive',
    'Negative':        'tag-negative',
    'Review':          'tag-review',
    'Narrative Shift': 'tag-narrative-shift',
    'No Major Change': 'tag-no-change'
  };
  return map[tag] || 'tag-no-change';
}

function sentimentClass(s) {
  const map = {
    'Positive': 'sentiment-positive',
    'Negative': 'sentiment-negative',
    'Neutral':  'sentiment-neutral'
  };
  return map[s] || 'sentiment-neutral';
}

/**
 * Map a sector name to its color CSS class.
 */
function sectorTagClass(sector) {
  const map = {
    'Energy':      'tag-energy',
    'Financials':  'tag-financials',
    'Technology':  'tag-technology',
    'Industrials': 'tag-industrials',
    'Consumer':    'tag-consumer',
    'Healthcare':  'tag-healthcare',
    'Macro':       'tag-macro',
  };
  return map[sector] || 'tag-macro';
}

/* ============================================================
   COMPACT STORY ROW RENDERER
   No summary, no why-it-matters.
   Shows: colored sector tags | title (link) | source | time | read →
   ============================================================ */

function renderStoryRow(story) {
  if (!story) return '';

  const title      = story.title       || 'Untitled';
  const sourceName = story.source_name || '';
  const timestamp  = story.timestamp   || '';
  const url        = story.url         || '';
  const sectors    = Array.isArray(story.sector_tags) ? story.sector_tags : [];

  const tsDisplay = timeAgo(timestamp) || formatTimestamp(timestamp);

  // Deduplicate and cap sector tags at 2 for compactness
  const uniqueSectors = [...new Set(sectors)].slice(0, 2);

  const tagsHtml = uniqueSectors.map(s =>
    `<span class="sector-tag-pill ${sectorTagClass(s)}">${s}</span>`
  ).join('');

  const titleHtml = url
    ? `<a class="story-title-link" href="${url}" target="_blank" rel="noopener">${title}</a>`
    : `<span class="story-title-link">${title}</span>`;

  const readHtml = url
    ? `<a class="story-read-link" href="${url}" target="_blank" rel="noopener">Read →</a>`
    : '';

  return `
    <div class="story-row-compact">
      <div class="story-tags">${tagsHtml}</div>
      ${titleHtml}
      <div class="story-meta">
        ${sourceName ? `<span class="story-meta-source">${sourceName}</span>` : ''}
        ${tsDisplay  ? `<span class="story-meta-time">${tsDisplay}</span>`    : ''}
        ${readHtml}
      </div>
    </div>
  `;
}

/* ============================================================
   BADGE RENDERERS (kept for watchlist/sector pages)
   ============================================================ */

function renderSourceBadge(sourceType) {
  const label = sourceType || 'Unknown';
  return `<span class="badge ${sourceBadgeClass(sourceType)}">${label}</span>`;
}

function renderSectorTags(tags) {
  if (!Array.isArray(tags) || !tags.length) return '';
  return tags.map(t => `<span class="badge tag-sector">${t}</span>`).join('');
}

function renderTickerTags(tickers) {
  if (!Array.isArray(tickers) || !tickers.length) return '';
  return tickers.map(t => `<span class="tag-ticker">${t}</span>`).join('');
}

/* ============================================================
   DATA FETCHING
   ============================================================ */

async function fetchJSON(path) {
  try {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn(`fetchJSON failed for "${path}":`, err.message);
    return null;
  }
}

/* ============================================================
   TOAST NOTIFICATIONS
   ============================================================ */

function showToast(msg, type = 'default') {
  let toast = document.getElementById('globalToast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'globalToast';
    toast.className = 'toast';
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.className = `toast ${type}`;
  void toast.offsetWidth;
  toast.classList.add('show');
  clearTimeout(toast._hideTimer);
  toast._hideTimer = setTimeout(() => toast.classList.remove('show'), 3200);
}