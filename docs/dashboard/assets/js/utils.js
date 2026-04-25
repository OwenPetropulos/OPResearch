/* utils.js — Shared utility functions */

/**
 * Format a number as USD currency
 */
function formatCurrency(val, decimals = 2) {
  if (isNaN(val)) return '—';
  return '$' + Math.abs(val).toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
}

/**
 * Format a percentage with sign
 */
function formatPct(val, decimals = 2) {
  if (isNaN(val)) return '—';
  const sign = val >= 0 ? '+' : '';
  return sign + val.toFixed(decimals) + '%';
}

/**
 * Format a signed number
 */
function formatSigned(val, decimals = 2) {
  if (isNaN(val)) return '—';
  const sign = val >= 0 ? '+' : '';
  return sign + val.toFixed(decimals);
}

/**
 * Get CSS class for directional value
 */
function dirClass(val) {
  if (val > 0) return 'up';
  if (val < 0) return 'down';
  return 'flat';
}

/**
 * Get arrow character for direction
 */
function dirArrow(val) {
  if (val > 0) return '▲';
  if (val < 0) return '▼';
  return '—';
}

/**
 * Map source_type to badge CSS class
 */
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

/**
 * Map alert_tag to CSS class
 */
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

/**
 * Map sentiment to CSS class
 */
function sentimentClass(s) {
  const map = { 'Positive': 'sentiment-positive', 'Negative': 'sentiment-negative', 'Neutral': 'sentiment-neutral' };
  return map[s] || 'sentiment-neutral';
}

/**
 * Format ISO timestamp to readable string
 */
function formatTimestamp(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleString('en-US', {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', hour12: true
  });
}

/**
 * Fetch JSON from a local file with error handling
 */
async function fetchJSON(path) {
  try {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn(`Failed to fetch ${path}:`, err.message);
    return null;
  }
}

/**
 * Show a toast message
 */
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
  requestAnimationFrame(() => {
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3000);
  });
}

/**
 * Render a source badge element
 */
function renderSourceBadge(sourceType) {
  return `<span class="badge ${sourceBadgeClass(sourceType)}">${sourceType}</span>`;
}

/**
 * Render sector tag badges
 */
function renderSectorTags(tags = []) {
  return tags.map(t => `<span class="badge tag-sector">${t}</span>`).join('');
}

/**
 * Render ticker tags
 */
function renderTickerTags(tickers = []) {
  return tickers.map(t => `<span class="tag-ticker">${t}</span>`).join('');
}

/**
 * Build a full story row HTML
 */
function renderStoryRow(story) {
  return `
    <div class="story-row">
      <div class="story-row-top">
        ${renderSourceBadge(story.source_type)}
        ${renderSectorTags(story.sector_tags)}
        ${renderTickerTags(story.ticker_tags)}
      </div>
      <div class="story-headline">
        ${story.url
          ? `<a href="${story.url}" target="_blank" rel="noopener">${story.title}</a>`
          : story.title}
      </div>
      <div class="story-summary">${story.summary}</div>
      ${story.why_it_matters ? `
        <div class="story-why-matters">
          <div class="story-why-label">Why it matters</div>
          ${story.why_it_matters}
        </div>
      ` : ''}
      <div class="story-footer">
        <span class="story-source-name">${story.source_name}</span>
        <span class="divider"></span>
        <span class="story-timestamp">${formatTimestamp(story.timestamp)}</span>
        ${story.url ? `<a class="story-link" href="${story.url}" target="_blank" rel="noopener">Read →</a>` : ''}
      </div>
    </div>
  `;
}
