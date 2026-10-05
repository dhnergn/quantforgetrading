// Render check for the dashboard page, run under node.
//
// The browser harness is not connected to this session, so this verifies the
// parts that can be verified headlessly and honestly: that the inline script
// parses, that render() emits one card per component, that unavailable
// components render their reason instead of an empty figure, and that an
// undefined value never renders as 0 or as a blank.
//
// It does NOT verify visual layout, colour contrast, or that the charts look
// right. No screenshot was taken.
import { readFileSync } from 'fs';
// Resolve the page relative to this file, so the check works from any cwd.
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
const here = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(resolve(here, '../../quantforge/api/static/dashboard.html'), 'utf8');
const src = html.match(/<script>([\s\S]*?)<\/script>/)[1];

const nodes = {};
const mkNode = (id) => (nodes[id] ||= { id, textContent:'', innerHTML:'', value:'', disabled:false, addEventListener(){} });
globalThis.document = { getElementById: mkNode };
globalThis.fetch = async () => { throw new Error('not stubbed'); };

let failures = 0;
const ok = (label, cond) => { console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${label}`); if (!cond) failures++; };

console.log('=== dashboard.html render check ===');
let api;
try { new Function(src); ok('inline script compiles', true); }
catch (e) { console.log('  FAIL inline script:', e.message); process.exit(1); }

api = new Function(src + '\n; return {R, render, lineChart, barChart, heatmap, num, pct, money, basis};')();

const mk = (available, extra = {}) => ({ name:'x', available, reason: available ? '' : 'because', n_observations: 5, period:'1 month', caveat: available ? 'a caveat' : '', data: available ? {} : null, ...extra });

const report = {
  generated_at:'2026-09-28T00:00:00', base_currency:'USD', n_available:8, n_total:14, disclosure:'d',
  components: {
    equity_curve: mk(true, { data:{ dates:['2024-01-01','2024-01-02','2024-01-03','2024-01-04','2024-01-05'],
      equity:[1,1.02,0.99,1.05,1.08], start_equity:1e6, end_equity:1.08e6,
      benchmark:[1,1.01,1,1.03,1.02] } }),
    daily_pnl: mk(true, { data:{ dates:['2024-01-02','2024-01-03','2024-01-04'], pnl:[2e4,-3e4,6e4] } }),
    unrealized_pnl: mk(true, { data:{ total:5000, realized:1200, fees_paid:40, equity:1.08e6 } }),
    realized_pnl: mk(true, { data:[{ instrument_id:'AAPL', exit_time:'2024-01-04T00:00:00Z', net_pnl:-120.5, return_pct:-0.02, bars_held:3, exit_reason:'target' }] }),
    exposure: mk(true, { data:{ equity:1.08e6, cash:5e5, market_value:5.8e5, gross_exposure:5.8e5, net_exposure:4.2e5, gross_pct_equity:0.537, net_pct_equity:0.389, position_count:2 } }),
    positions: mk(true, { data:[{ instrument_id:'AAPL', direction:'long', quantity:100, average_entry_price:150, last_price:160, unrealized_pnl:1000, unrealized_pct:0.066 }] }),
    sector_exposure: mk(true, { data:[{ label:'Technology', exposure:3e5, pct_of_total:0.6 },{ label:'UNCLASSIFIED', exposure:2e5, pct_of_total:0.4 }] }),
    asset_class_exposure: mk(true, { data:[{ label:'EQUITY', exposure:4e5, pct_of_total:0.8 }] }),
    correlations: mk(true, { data:{ instruments:['AAPL','MSFT'], matrix:[[1,0.72],[0.72,1]] } }),
    portfolio_volatility: mk(true, { data:{ dates:['2024-01-04','2024-01-05'], volatility:[0.21,0.24], window:3, current:0.24 } }),
    var: mk(true, { data:{ var_return:0.018, var_currency:19440, es_return:0.026, es_currency:28080, confidence:0.95, horizon_periods:1, method:'historical', equity_at_measurement:1.08e6 } }),
    drawdown: mk(true, { data:{ dates:['2024-01-01','2024-01-02','2024-01-03'], drawdown:[0,-0.01,-0.03], max_drawdown:0.03, current_drawdown:0.01, peak:'2024-01-01', trough:'2024-01-03' } }),
    summary_statistics: mk(true, { data:{ total_return:0.08, annualised_return:1.2, volatility:0.18, sharpe:0.9, sortino:1.4, calmar:0.5, max_drawdown:0.03, n_trades:12, win_rate:0.58, profit_factor:1.7, years_covered:0.08, annualisation_basis:'measured from the index dates' } }),
    beta: mk(false),
  },
};

for (const id of ['grid','tally','status','pid','load','reload']) mkNode(id);
api.render(report);
const out = mkNode('grid').innerHTML;
ok(`render() emits one card per component (${(out.match(/class=\"card/g)||[]).length})`, (out.match(/class="card/g)||[]).length === 14);
ok('unavailable component shows its reason', out.includes('because'));
ok('unavailable card is visually marked', out.includes('card unavailable'));
ok('available components emit charts or tables', (out.match(/<svg|<table|<dl/g)||[]).length >= 10);
ok('the n= and period basis is on the card', out.includes('n=5') && out.includes('1 month'));
ok('every available card carries its caveat', (out.match(/class="caveat"/g)||[]).length === 13);

ok('num(undefined) is a word, not 0', api.num(undefined).text === 'not defined');
ok('pct(null) is a word, not 0', api.pct(null).text === 'not defined');
ok('num(NaN) is a word, not 0', api.num(NaN).text === 'not defined');
ok('num(Infinity) is a word, not 0', api.num(Infinity).text === 'not finite');
ok('a real 0 still renders as 0', api.num(0).text === '0');

ok('flat series still plots (no divide-by-zero)', api.lineChart(['a','b','c'], [{label:'x',colour:'#fff',values:[1,1,1]}], {}).includes('<svg'));
ok('a single point declines rather than drawing', api.lineChart(['a'], [{label:'x',colour:'#fff',values:[1]}], {}).includes('Not enough'));
ok('all-NaN series declines', api.lineChart(['a','b'], [{label:'x',colour:'#fff',values:[NaN,NaN]}], {}).includes('Not enough'));
ok('barChart emits svg', api.barChart([1,-2,3], {}).includes('<svg'));
// 2x2 matrix: 1 grid container + 1 empty corner + 2 column headers
// + 2 row headers + 4 data cells = 10 divs.
const heat = api.heatmap(['A','B'], [[1,0.5],[0.5,1]]);
ok('heatmap renders a cell per pair (10 divs)', (heat.match(/<div/g)||[]).length === 10);
ok('heatmap diagonal is exactly 1', (heat.match(/title="1.000"/g)||[]).length === 2);
ok('heatmap contains the correlation values', heat.includes('0.50'));

console.log(failures ? `\n  ${failures} CHECK(S) FAILED` : '\n  all render checks passed');
process.exit(failures ? 1 : 0);
