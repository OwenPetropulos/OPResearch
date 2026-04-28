/* portfolio.js — Paper Portfolio System with localStorage */

/* ============================================================
   CONSTANTS & STATE
   ============================================================ */

const STORAGE_KEY          = 'opr_portfolio';
const WATCHLIST_STORAGE_KEY = 'opr_watchlist';
const STARTING_CASH        = 100000;

let priceMap  = {};
let portfolio = loadPortfolio();

/* ============================================================
   INITIALIZATION
   ============================================================ */

document.addEventListener('DOMContentLoaded', async () => {
  const priceData = await fetchJSON('./data/portfolio_prices.json');
  if (priceData?.prices && typeof priceData.prices === 'object') {
    priceMap = priceData.prices;
  } else {
    showToast('Price data unavailable — valuations may be approximate.', 'error');
  }

  if (!portfolio.bootstrapped) {
    seedSamplePortfolio();
  }

  renderAll();

  document.getElementById('addTradeBtn').addEventListener('click', handleAddTrade);
  document.getElementById('exportBtn').addEventListener('click', handleExport);
  document.getElementById('importBtn').addEventListener('click', handleImport);
  document.getElementById('importFile').addEventListener('change', handleImportFile);
  document.getElementById('resetBtn').addEventListener('click', handleReset);
});

/* ============================================================
   PORTFOLIO STORAGE
   ============================================================ */

function loadPortfolio() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      if (parsed && Array.isArray(parsed.transactions)) return parsed;
    }
  } catch (e) {
    console.warn('Portfolio localStorage parse error:', e);
  }
  return { cash: STARTING_CASH, transactions: [], history: [], bootstrapped: false };
}

function savePortfolio() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(portfolio));
  } catch (e) {
    showToast('Storage write error — portfolio may not be saved.', 'error');
  }
}

/* ============================================================
   WATCHLIST AUTO-SYNC
   When a new ticker is added to the portfolio, automatically
   create a watchlist entry if one does not already exist.
   ============================================================ */

function loadWatchlist() {
  try {
    const stored = localStorage.getItem(WATCHLIST_STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      if (Array.isArray(parsed)) return parsed;
    }
  } catch (e) {
    console.warn('Watchlist localStorage parse error:', e);
  }
  return [];
}

function saveWatchlist(items) {
  try {
    localStorage.setItem(WATCHLIST_STORAGE_KEY, JSON.stringify(items));
  } catch (e) {
    console.warn('Could not save watchlist:', e);
  }
}

/**
 * Add a ticker to the watchlist if it isn't already there.
 * Infers sector from a simple keyword map — user can edit notes
 * on the watchlist page afterward.
 */
function syncTickerToWatchlist(ticker, entryPrice) {
  const watchlist = loadWatchlist();

  // Already on watchlist — do not overwrite user's existing entry
  if (watchlist.some(item => item.ticker === ticker)) {
    return;
  }

  // Infer a rough sector from the ticker for convenience
  const sector = inferSector(ticker);

  const newEntry = {
    ticker,
    sector,
    alert_tag:    'No Major Change',
    notes:        `Added from portfolio on entry at $${entryPrice.toFixed(2)}. Update thesis here.`,
    price:        priceMap[ticker] ?? entryPrice,
    percent_move: null,
  };

  watchlist.unshift(newEntry); // Add to top of watchlist
  saveWatchlist(watchlist);
  log(`Auto-added ${ticker} to watchlist.`);
}

/**
 * Simple sector inference from ticker symbol.
 * Not exhaustive — just catches the most common names.
 * User can always edit the sector on the watchlist page.
 */
function inferSector(ticker) {
  const t = ticker.toUpperCase();
  const map = {
    // Technology
    Technology:  ['NVDA','MSFT','GOOGL','AMZN','META','AAPL','AMD','ORCL','CRM','NOW','TSM','ASML','QCOM','INTC','MU','AMAT','LRCX','KLAC','AVGO','ARM'],
    // Financials
    Financials:  ['JPM','GS','BAC','MS','C','WFC','BRK-B','KRE','ZION','WAL','AXP','V','MA','BX','KKR','APO','SCHW'],
    // Healthcare
    Healthcare:  ['LLY','NVO','AMGN','PFE','VRTX','ABBV','MRK','JNJ','BMY','GILD','REGN','ISRG','UNH','CVS','CI'],
    // Energy
    Energy:      ['XOM','CVX','OXY','LNG','RIG','VAL','ET','SHEL','BP','TTE','COP','EOG','PXD','DVN','FANG','MPC','VLO','PSX','SLB','HAL'],
    // Industrials
    Industrials: ['GE','RTX','LMT','NOC','HON','CAT','DE','BA','GD','TDG','CARR','OTIS','EMR','ETN','ITW','PH'],
    // Consumer
    Consumer:    ['NKE','WMT','COST','TGT','ONON','LULU','HD','MCD','SBUX','TJX','BKNG','MAR','HLT','DIS','NFLX'],
    // Macro (ETFs and macro instruments)
    Macro:       ['GLD','TLT','TIP','SPY','QQQ','IWM','GDX','SHY','HYG','LQD','USO','SLV','COPX','UNG','FXY','FXE','UUP','VIX','DXY'],
  };

  for (const [sector, tickers] of Object.entries(map)) {
    if (tickers.includes(t)) return sector;
  }
  return 'Uncategorized';
}

function log(msg) {
  console.log(`[OPR Portfolio] ${msg}`);
}

/* ============================================================
   SEED DATA
   ============================================================ */

function seedSamplePortfolio() {
  // Start with a clean slate — no seed trades
  // Users start with $100,000 cash and enter their own trades
  portfolio.bootstrapped = true;
  savePortfolio();
}

/* ============================================================
   PORTFOLIO LOGIC
   ============================================================ */

function applyTransaction(tx, saveAfter = true) {
  const trade = {
    id:     Date.now() + Math.random(),
    date:   tx.date,
    ticker: String(tx.ticker).toUpperCase().trim(),
    side:   tx.side,
    shares: parseFloat(tx.shares),
    price:  parseFloat(tx.price),
    note:   tx.note || ''
  };

  if (!trade.ticker || !trade.date || isNaN(trade.shares) || isNaN(trade.price)) return false;

  const cost = trade.shares * trade.price;

  if (trade.side === 'buy') {
    if (portfolio.cash < cost) {
      showToast('Insufficient cash for this trade.', 'error');
      return false;
    }
    portfolio.cash -= cost;

    // Auto-sync new buy to watchlist
    syncTickerToWatchlist(trade.ticker, trade.price);

  } else {
    portfolio.cash += cost;
  }

  portfolio.transactions.push(trade);

  portfolio.history.push({
    date:  trade.date,
    value: computePortfolioValue()
  });

  if (saveAfter) savePortfolio();
  return true;
}

function computeHoldings() {
  const positions = {};

  portfolio.transactions.forEach(tx => {
    if (!positions[tx.ticker]) {
      positions[tx.ticker] = { shares: 0, totalCost: 0 };
    }
    const pos = positions[tx.ticker];

    if (tx.side === 'buy') {
      pos.totalCost += tx.shares * tx.price;
      pos.shares    += tx.shares;
    } else {
      const avgCost  = pos.shares > 0 ? pos.totalCost / pos.shares : 0;
      pos.totalCost  = Math.max(0, pos.totalCost - avgCost * tx.shares);
      pos.shares     = Math.max(0, pos.shares - tx.shares);
    }
  });

  return Object.entries(positions)
    .filter(([, pos]) => pos.shares > 0.0001)
    .map(([ticker, pos]) => ({
      ticker,
      shares:    pos.shares,
      avgCost:   pos.shares > 0 ? pos.totalCost / pos.shares : 0,
      totalCost: pos.totalCost
    }));
}

function computeEquityValue() {
  return computeHoldings().reduce((sum, h) => {
    const price = priceMap[h.ticker] ?? h.avgCost;
    return sum + h.shares * price;
  }, 0);
}

function computePortfolioValue() {
  return computeEquityValue() + portfolio.cash;
}

/* ============================================================
   RENDERING
   ============================================================ */

function renderAll() {
  renderMetrics();
  renderChart();
  renderHoldingsTable();
  renderLedger();
}

/* --- Metrics --- */
function renderMetrics() {
  const holdings   = computeHoldings();
  const equityVal  = computeEquityValue();
  const totalValue = equityVal + portfolio.cash;
  const totalCost  = holdings.reduce((s, h) => s + h.totalCost, 0);
  const unrealPnL  = equityVal - totalCost;
  const returnPct  = ((totalValue - STARTING_CASH) / STARTING_CASH) * 100;

  let dailyPnL = 0;
  const hist = portfolio.history;
  if (hist.length >= 2) {
    dailyPnL = hist[hist.length - 1].value - hist[hist.length - 2].value;
  }

  setMetric('metricTotal',  formatCurrency(totalValue));
  setMetric('metricCash',   formatCurrency(portfolio.cash));
  setMetric('metricDaily',  formatCurrency(dailyPnL),  dailyPnL  >= 0 ? 'up' : 'down');
  setMetric('metricPnL',    formatCurrency(unrealPnL), unrealPnL >= 0 ? 'up' : 'down');
  setMetric('metricReturn', formatPct(returnPct),       returnPct >= 0 ? 'up' : 'down');
}

function setMetric(id, value, colorClass = '') {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = value;
  el.className   = 'metric-value' + (colorClass ? ' ' + colorClass : '');
}

/* --- Chart --- */
function renderChart() {
  const canvas = document.getElementById('portfolioChart');
  if (!canvas) return;

  const history = portfolio.history;
  canvas.width  = canvas.offsetWidth || (canvas.parentElement?.offsetWidth) || 800;
  canvas.height = 240;

  const ctx = canvas.getContext('2d');
  const W   = canvas.width;
  const H   = canvas.height;
  ctx.clearRect(0, 0, W, H);

  if (history.length < 2) {
    ctx.fillStyle = '#435570';
    ctx.font      = '12px IBM Plex Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Add trades to see portfolio performance chart.', W / 2, H / 2);
    return;
  }

  const padL   = 80;
  const padR   = 24;
  const padT   = 24;
  const padB   = 36;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;

  const values = history.map(h => h.value);
  const labels = history.map(h => h.date);
  const minVal = Math.min(...values) * 0.98;
  const maxVal = Math.max(...values) * 1.02;
  const range  = maxVal - minVal || 1;

  const xScale = i => padL + (i / (values.length - 1)) * chartW;
  const yScale = v => padT + chartH - ((v - minVal) / range) * chartH;

  const gridLines = 4;
  for (let i = 0; i <= gridLines; i++) {
    const y   = padT + (i / gridLines) * chartH;
    const val = maxVal - (i / gridLines) * range;
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(W - padR, y);
    ctx.stroke();
    ctx.fillStyle = '#435570';
    ctx.font      = '10px IBM Plex Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText('$' + Math.round(val).toLocaleString(), padL - 8, y + 4);
  }

  const isUp      = values[values.length - 1] >= values[0];
  const lineColor = isUp ? '#22c55e' : '#ef4444';
  const fillColor = isUp ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)';

  ctx.beginPath();
  ctx.moveTo(xScale(0), yScale(values[0]));
  values.forEach((v, i) => ctx.lineTo(xScale(i), yScale(v)));
  ctx.lineTo(xScale(values.length - 1), padT + chartH);
  ctx.lineTo(padL, padT + chartH);
  ctx.closePath();
  ctx.fillStyle = fillColor;
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(xScale(0), yScale(values[0]));
  values.forEach((v, i) => ctx.lineTo(xScale(i), yScale(v)));
  ctx.strokeStyle = lineColor;
  ctx.lineWidth   = 2;
  ctx.lineJoin    = 'round';
  ctx.shadowColor = lineColor;
  ctx.shadowBlur  = 6;
  ctx.stroke();
  ctx.shadowBlur  = 0;

  ctx.fillStyle = '#435570';
  ctx.font      = '10px IBM Plex Mono, monospace';
  ctx.textAlign = 'center';
  const step    = Math.max(1, Math.floor(labels.length / 5));
  labels.forEach((label, i) => {
    if (i % step === 0 || i === labels.length - 1) {
      ctx.fillText(label, xScale(i), H - 8);
    }
  });

  const lastVal = values[values.length - 1];
  ctx.fillStyle  = lineColor;
  ctx.font       = 'bold 11px IBM Plex Mono, monospace';
  ctx.textAlign  = 'right';
  ctx.fillText('$' + Math.round(lastVal).toLocaleString(), W - padR, padT + 14);
}

/* --- Holdings Table --- */
function renderHoldingsTable() {
  const tbody   = document.getElementById('holdingsTbody');
  if (!tbody) return;

  const holdings  = computeHoldings();
  const totalVal  = computePortfolioValue();

  if (!holdings.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-state">No open positions. Add a trade to get started.</td></tr>`;
    return;
  }

  tbody.innerHTML = holdings.map(h => {
    const price     = priceMap[h.ticker] ?? h.avgCost;
    const mktVal    = h.shares * price;
    const unrealPnL = mktVal - h.totalCost;
    const unrealPct = h.totalCost > 0 ? (unrealPnL / h.totalCost) * 100 : 0;
    const weight    = totalVal > 0 ? (mktVal / totalVal) * 100 : 0;
    const pnlClass  = unrealPnL >= 0 ? 'up' : 'down';
    const isPriced  = !!priceMap[h.ticker];

    return `
      <tr>
        <td class="mono strong">
          ${h.ticker}
          ${!isPriced ? `<span style="font-size:0.6rem;color:var(--amber);margin-left:4px;" title="No live price — using avg cost">~</span>` : ''}
        </td>
        <td class="mono">${h.shares.toFixed(4)}</td>
        <td class="mono">${formatCurrency(h.avgCost)}</td>
        <td class="mono">${formatCurrency(price)}${!isPriced ? ' <span style="color:var(--text-muted);font-size:0.65rem;">(est)</span>' : ''}</td>
        <td class="mono">${formatCurrency(mktVal)}</td>
        <td class="mono ${pnlClass}">
          ${formatCurrency(unrealPnL)}
          <span style="opacity:0.7;">(${formatPct(unrealPct)})</span>
        </td>
        <td class="mono">${weight.toFixed(1)}%</td>
      </tr>`;
  }).join('');
}

/* --- Trade Ledger --- */
function renderLedger() {
  const tbody = document.getElementById('ledgerTbody');
  if (!tbody) return;

  const txns = [...portfolio.transactions].reverse();

  if (!txns.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty-state">No trades recorded.</td></tr>`;
    return;
  }

  tbody.innerHTML = txns.map(tx => `
    <tr>
      <td class="mono">${tx.date || '—'}</td>
      <td class="mono strong">${tx.ticker}</td>
      <td><span class="side-${tx.side}">${tx.side.toUpperCase()}</span></td>
      <td class="mono">${tx.shares}</td>
      <td class="mono">${formatCurrency(tx.price)}</td>
      <td style="color:var(--text-muted);font-size:0.78rem;">${tx.note || '—'}</td>
    </tr>`).join('');
}

/* ============================================================
   FORM HANDLERS
   ============================================================ */

function handleAddTrade() {
  const ticker = (document.getElementById('tradeTicker')?.value || '').trim().toUpperCase();
  const side   = document.getElementById('tradeSide')?.value;
  const shares = parseFloat(document.getElementById('tradeShares')?.value);
  const price  = parseFloat(document.getElementById('tradePrice')?.value);
  const date   = document.getElementById('tradeDate')?.value;
  const note   = (document.getElementById('tradeNote')?.value || '').trim();

  if (!ticker || !side || !date || isNaN(shares) || shares <= 0 || isNaN(price) || price <= 0) {
    showToast('Please fill in all required trade fields.', 'error');
    return;
  }

  if (side === 'sell') {
    const holdings = computeHoldings();
    const pos = holdings.find(h => h.ticker === ticker);
    if (!pos || pos.shares < shares) {
      showToast(`Cannot sell ${shares} shares of ${ticker} — insufficient holdings.`, 'error');
      return;
    }
  }

  const ok = applyTransaction({ ticker, side, shares, price, date, note }, true);
  if (ok) {
    const wasAutoAdded = side === 'buy';
    showToast(
      `${side.toUpperCase()} ${shares} ${ticker} @ ${formatCurrency(price)} recorded.` +
      (wasAutoAdded ? ` ${ticker} added to watchlist.` : ''),
      'success'
    );
    clearTradeForm();
    renderAll();
  }
}

function clearTradeForm() {
  ['tradeTicker', 'tradeShares', 'tradePrice', 'tradeNote'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = '';
  });
  const dateEl = document.getElementById('tradeDate');
  if (dateEl) dateEl.value = new Date().toISOString().split('T')[0];
}

function handleExport() {
  try {
    const blob = new Blob([JSON.stringify(portfolio, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `opr_portfolio_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Portfolio exported.', 'success');
  } catch (e) {
    showToast('Export failed: ' + e.message, 'error');
  }
}

function handleImport() {
  document.getElementById('importFile')?.click();
}

function handleImportFile(e) {
  const file = e.target.files?.[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (ev) => {
    try {
      const imported = JSON.parse(ev.target.result);
      if (!imported || typeof imported !== 'object') throw new Error('File is not a valid JSON object.');
      if (!Array.isArray(imported.transactions))       throw new Error('"transactions" must be an array.');
      if (typeof imported.cash !== 'number')            throw new Error('"cash" must be a number.');
      if (!Array.isArray(imported.history))             throw new Error('"history" must be an array.');

      portfolio = imported;
      savePortfolio();
      renderAll();
      showToast('Portfolio imported successfully.', 'success');
    } catch (err) {
      showToast('Import failed: ' + err.message, 'error');
    }
  };
  reader.readAsText(file);
  e.target.value = '';
}

function handleReset() {
  if (!confirm('Reset portfolio? This will erase all trades and cannot be undone.')) return;
  portfolio = {
    cash:         STARTING_CASH,
    transactions: [],
    history:      [],
    bootstrapped: true
  };
  savePortfolio();
  renderAll();
  showToast('Portfolio reset to $100,000 starting cash.', 'default');
}
