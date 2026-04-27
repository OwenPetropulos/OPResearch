/* utils.js — Shared utility functions */

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

function renderStoryRow(story) {
  if (!story) return '';

  const title        = story.title         || 'Untitled';
  const summary      = story.summary       || '';
  const whyItMatters = story.why_it_matters || '';
  const sourceName   = story.source_name   || '';
  const sourceType   = story.source_type   || '';
  const timestamp    = story.timestamp     || '';
  const url          = story.url           || '';
  const sectorTags   = Array.isArray(story.sector_tags)  ? story.sector_tags  : [];
  const tickerTags   = Array.isArray(story.ticker_tags)  ? story.ticker_tags  : [];

  const headlineInner = url
    ? `<a href="${url}" target="_blank" rel="noopener">${title}</a>`
    : title;

  const tsDisplay = timeAgo(timestamp) || formatTimestamp(timestamp);

  return `
    <div class="story-row">
      <div class="story-row-top">
        ${sourceType ? renderSourceBadge(sourceType) : ''}
        ${renderSectorTags(sectorTags)}
        ${renderTickerTags(tickerTags)}
      </div>
      <div class="story-headline">${headlineInner}</div>
      ${summary ? `<div class="story-summary">${summary}</div>` : ''}
      ${whyItMatters ? `
        <div class="story-why-matters">
          <div class="story-why-label">Why it matters</div>
          ${whyItMatters}
        </div>
      ` : ''}
      <div class="story-footer">
        ${sourceName ? `<span class="story-source-name">${sourceName}</span>` : ''}
        ${sourceName && tsDisplay ? `<span class="divider"></span>` : ''}
        ${tsDisplay ? `<span class="story-timestamp">${tsDisplay}</span>` : ''}
        ${url ? `<a class="story-link" href="${url}" target="_blank" rel="noopener">Read →</a>` : ''}
      </div>
    </div>
  `;
}

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