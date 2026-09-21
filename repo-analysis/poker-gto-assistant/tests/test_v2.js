// Comprehensive test suite for poker_assistant.html v2.
// Extracts the <script> block and runs it under Node with stubbed DOM.

const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, '..', 'poker_advanced.html'), 'utf8');
const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/);
if (!scriptMatch) { console.error('No <script> found'); process.exit(1); }
const scriptCode = scriptMatch[1];

// ---- Minimal DOM stubs ----
const noop = () => {};
function stubEl() {
  const el = {
    children: [], dataset: {}, style: {}, classList: {
      _set: new Set(),
      add(c) { this._set.add(c); },
      remove(c) { this._set.delete(c); },
      toggle(c, force) { if (force === true) this._set.add(c); else if (force === false) this._set.delete(c); else (this._set.has(c) ? this._set.delete(c) : this._set.add(c)); },
      contains(c) { return this._set.has(c); }
    },
    addEventListener: noop, removeEventListener: noop,
    appendChild(c) { this.children.push(c); return c; },
    querySelector() { return stubEl(); },
    querySelectorAll() { return []; },
    set innerHTML(v) { this._innerHTML = v; }, get innerHTML() { return this._innerHTML || ''; },
    set textContent(v) { this._textContent = v; }, get textContent() { return this._textContent || ''; },
    onclick: null, onchange: null, onkeydown: null,
    closest() { return null; },
    checked: false,
    value: '',
    get tagName() { return 'DIV'; }
  };
  // self-referential parent so chains don't crash
  el.parentElement = el;
  return el;
}

const localStorage = { _data: {}, getItem(k) { return this._data[k] || null; }, setItem(k, v) { this._data[k] = v; } };
const document = {
  querySelector(sel) { return stubEl(); },
  querySelectorAll() { return []; },
  getElementById() { return stubEl(); },
  addEventListener: noop,
  get activeElement() { return { tagName: 'BODY' }; },
  createElement(tag) { return stubEl(); },
  body: stubEl()
};
const window = {};
const performance = { now: () => Date.now() };
global.requestAnimationFrame = (cb) => setTimeout(cb, 0);
const setTimeout_orig = setTimeout;
// noop debounce helpers — we don't trigger UI in tests
global.document = document;
global.window = window;
global.localStorage = localStorage;
global.performance = performance;

try {
  eval(scriptCode);
} catch (e) {
  console.error('Script eval failed:', e.message);
  console.error(e.stack);
  process.exit(1);
}

const PA = window.__pa;
if (!PA) { console.error('window.__pa not exposed'); process.exit(1); }

// ============================================================
// ====================  TEST FRAMEWORK  ======================
// ============================================================
let passed = 0, failed = 0;
function check(name, cond, info) {
  if (cond) { console.log('  ✓ ' + name); passed++; }
  else { console.log('  ✗ ' + name + (info ? ' — ' + info : '')); failed++; }
}
function approx(a, b, tol) { return Math.abs(a - b) <= tol; }
function section(name) { console.log('\n══ ' + name + ' ══'); }

const c = code => PA.handCode ? null : null; // placeholder
// helper
const C = code => {
  // expose card encoder via internals — copy from script
  const RANKS = '23456789TJQKA';
  const SUITS = ['s','h','d','c'];
  return RANKS.indexOf(code[0].toUpperCase()) * 4 + SUITS.indexOf(code[1].toLowerCase());
};

// ============================================================
// =================== CARD EVAL ==============================
// ============================================================
section('Card evaluator');

check('SF beats quads',
  PA.evaluate5(['As','Ks','Qs','Js','Ts'].map(C)) > PA.evaluate5(['As','Ah','Ad','Ac','2s'].map(C)));
check('quads beat FH',
  PA.evaluate5(['As','Ah','Ad','Ac','2s'].map(C)) > PA.evaluate5(['As','Ah','Ad','Ks','Kh'].map(C)));
check('FH beats flush',
  PA.evaluate5(['As','Ah','Ad','Ks','Kh'].map(C)) > PA.evaluate5(['As','Js','9s','5s','3s'].map(C)));
check('flush beats straight',
  PA.evaluate5(['As','Js','9s','5s','3s'].map(C)) > PA.evaluate5(['9s','8h','7d','6c','5s'].map(C)));
check('wheel detected as straight',
  PA.evaluate5(['As','2h','3d','4c','5s'].map(C)) >= 4e9 && PA.evaluate5(['As','2h','3d','4c','5s'].map(C)) < 5e9);
check('royal flush max',
  PA.evaluate5(['As','Ks','Qs','Js','Ts'].map(C)) === 8e9 + 12);
check('higher kicker wins (KK77Q vs KK77J)',
  PA.evaluate5(['Ks','Kh','7d','7c','Qs'].map(C)) > PA.evaluate5(['Ks','Kh','7d','7c','Js'].map(C)));
check('AA kickers (AA853 > AA742)',
  PA.evaluate5(['As','Ah','8d','5c','3s'].map(C)) > PA.evaluate5(['As','Ah','7d','4c','2s'].map(C)));
check('A-high straight beats 9-high straight',
  PA.evaluate5(['As','Kh','Qd','Jc','Ts'].map(C)) > PA.evaluate5(['9s','8h','7d','6c','5s'].map(C)));
check('AAA22 (FH As over 2s) beats KKK22',
  PA.evaluate5(['As','Ah','Ad','2c','2s'].map(C)) > PA.evaluate5(['Ks','Kh','Kd','2c','2s'].map(C)));
check('paired board not flagged as straight (A-A-K-Q-J)',
  PA.evaluate5(['As','Ah','Kd','Qc','Js'].map(C)) < 4e9); // pair, not straight

// 7-card eval
check('7-card: best 5 of (AsKsQsJsTs+22) = royal',
  PA.evaluate7(['As','Ks','Qs','Js','Ts','2h','2d'].map(C)) === 8e9 + 12);

// ============================================================
// =================== EQUITY VS RANDOM =======================
// ============================================================
section('Equity vs random');

const aa_hu = PA.equityVsRandom([C('As'), C('Ac')], [], 1, 5000);
console.log(`  AA HU vs random: ${(aa_hu.equity*100).toFixed(1)}%`);
check('AA HU equity in [82, 88]', aa_hu.equity > 0.82 && aa_hu.equity < 0.88);

const aa_2 = PA.equityVsRandom([C('As'), C('Ac')], [], 2, 3000);
console.log(`  AA vs 2 random: ${(aa_2.equity*100).toFixed(1)}%`);
check('AA vs 2 in [68, 78]', aa_2.equity > 0.68 && aa_2.equity < 0.78);

const trash = PA.equityVsRandom([C('7c'), C('2d')], [], 1, 5000);
console.log(`  72o HU vs random: ${(trash.equity*100).toFixed(1)}%`);
check('72o HU in [30, 40]', trash.equity > 0.30 && trash.equity < 0.40);

// ============================================================
// =================== EQUITY VS RANGE ========================
// ============================================================
section('Equity vs range (SHOULD differ from vs random)');

// AA vs UTG range — should be ~78% (lower than 85% vs random)
const utgCombos = PA.expandRange(PA.RFI.UTG, [C('As'), C('Ac')]);
console.log(`  UTG range = ${utgCombos.length} valid combos`);
const aa_vs_utg = PA.equityVsRange([C('As'), C('Ac')], [], utgCombos, 4000);
console.log(`  AA vs UTG range: ${(aa_vs_utg.equity*100).toFixed(1)}% (expect ~75-82%)`);
check('AA vs UTG in [73, 84]', aa_vs_utg.equity > 0.73 && aa_vs_utg.equity < 0.84);
check('AA vs UTG < AA vs random (range is stronger)', aa_vs_utg.equity < aa_hu.equity);

// 72o vs random ≈ 35%, vs UTG range ≈ much lower (~20%)
const trash_vs_utg = PA.equityVsRange([C('7c'), C('2d')], [], utgCombos, 4000);
console.log(`  72o vs UTG range: ${(trash_vs_utg.equity*100).toFixed(1)}% (expect <30%)`);
check('72o vs UTG range < 72o vs random', trash_vs_utg.equity < trash.equity);

// AKs on AKQ flop vs UTG range — should still be very strong
const board = ['As', 'Kh', 'Qc'].map(C);
const utg2 = PA.expandRange(PA.RFI.UTG, [C('Ad'), C('Kd'), ...board]);
const akq_vs_utg = PA.equityVsRange([C('Ad'), C('Kd')], board, utg2, 3000);
console.log(`  AdKd on AsKhQc vs UTG range: ${(akq_vs_utg.equity*100).toFixed(1)}%`);
check('top 2-pair vs UTG > 70%', akq_vs_utg.equity > 0.70);

// ============================================================
// =================== RANGE EXPANSION ========================
// ============================================================
section('Range expansion');

const aa_combos = PA.expandRange(new Set(['AA']), []);
check('AA produces 6 combos', aa_combos.length === 6);

const aks_combos = PA.expandRange(new Set(['AKs']), []);
check('AKs produces 4 combos', aks_combos.length === 4);

const ako_combos = PA.expandRange(new Set(['AKo']), []);
check('AKo produces 12 combos', ako_combos.length === 12);

const aa_dead = PA.expandRange(new Set(['AA']), [C('As')]);
check('AA with As dead = 3 combos', aa_dead.length === 3);

// ============================================================
// =================== DRAW DETECTION =========================
// ============================================================
section('Draw detection');

// flush draw on flop
const fd = PA.detectDraws([C('Ah'), C('Kh')], [C('5h'), C('9h'), C('2c')]);
console.log(`  AhKh on 5h9h2c: ${JSON.stringify(fd)}`);
check('FD detected (≥9 outs)', fd.totalOuts >= 9);
check('FD label present', fd.kind.some(k => k.includes('FD')));

// open-ender
const oesd = PA.detectDraws([C('9s'), C('8d')], [C('7c'), C('6h'), C('Ks')]);
console.log(`  98 on 76K: ${JSON.stringify(oesd)}`);
check('OESD ≈8 outs', oesd.totalOuts >= 7 && oesd.totalOuts <= 9);
check('OESD label present', oesd.kind.some(k => k.includes('OESD')));

// gutshot
const gut = PA.detectDraws([C('Ts'), C('9d')], [C('7c'), C('6h'), C('Ks')]);
console.log(`  T9 on 76K: ${JSON.stringify(gut)}`);
check('Gutshot ≈4 outs', gut.totalOuts >= 3 && gut.totalOuts <= 5);

// combo: FD + OESD (real one — 4 hearts on board)
const combo = PA.detectDraws([C('9h'), C('8h')], [C('7h'), C('6h'), C('Kc')]);
console.log(`  98hh on 7h6hKc: ${JSON.stringify(combo)}`);
check('combo draw ≥15 outs (FD+OESD)', combo.totalOuts >= 15);

// pocket pair should NOT trigger "overcards" label
const pp_under = PA.detectDraws([C('8s'), C('8d')], [C('Js'), C('Th'), C('2c')]);
console.log(`  88 under JT2: ${JSON.stringify(pp_under)}`);
check('pocket 88 under JT2: no "overcards" label', !pp_under.kind.some(k => k.includes('overcards')));

const pp_over = PA.detectDraws([C('Ks'), C('Kd')], [C('7h'), C('5c'), C('2s')]);
console.log(`  KK over 752: ${JSON.stringify(pp_over)}`);
check('overpair KK on 752: no "overcards" label', !pp_over.kind.some(k => k.includes('overcards')));

// no draw on dry board
const dry = PA.detectDraws([C('Ah'), C('Ks')], [C('2c'), C('7d'), C('Tc')]);
console.log(`  AKo on 27T rainbow: ${JSON.stringify(dry)}`);
check('AK on dry board: no draws', dry.totalOuts === 0);

// ============================================================
// =================== BOARD TEXTURE ==========================
// ============================================================
section('Board texture');

check('A-K-Q rainbow = wet (broadway+connected)',
  ['wet','semi-wet'].includes(PA.boardTexture(['As','Kh','Qc'].map(C))));
check('K-7-2 rainbow = dry',
  PA.boardTexture(['Ks','7h','2c'].map(C)) === 'dry');
check('all hearts flop = monotone',
  PA.boardTexture(['Ah','Kh','5h'].map(C)) === 'monotone');
check('A-A-K = paired',
  PA.boardTexture(['As','Ah','Kc'].map(C)) === 'paired');
check('9-8-7 two-tone = wet',
  PA.boardTexture(['9s','8s','7c'].map(C)) === 'wet');

// ============================================================
// =================== SPR / COMMITMENT =======================
// ============================================================
section('SPR / commitment');

check('SPR=2 = low (commit territory)',
  PA.sprCommitment(20, 10).level === 'low');
check('SPR=10 = mid',
  PA.sprCommitment(100, 10).level === 'mid');
check('SPR=20 = deep',
  PA.sprCommitment(100, 5).level === 'deep');
check('SPR=0.5 = committed',
  PA.sprCommitment(5, 10).level === 'committed');

// ============================================================
// =================== FACING RAISE LOGIC =====================
// ============================================================
section('Facing-raise advice');

const ako_vs_utg = PA.facingRaiseAdvice('MP', 'UTG', C('Ah'), C('Kc'));
check('AKo vs UTG → 3-BET (premium)', ako_vs_utg.action === '3-BET');

const tt_vs_utg = PA.facingRaiseAdvice('CO', 'UTG', C('Ts'), C('Tc'));
check('TT vs UTG → CALL (set mine)', tt_vs_utg.action === 'CALL');

const trash_vs_utg2 = PA.facingRaiseAdvice('BTN', 'UTG', C('7s'), C('2c'));
check('72o vs UTG → FOLD', trash_vs_utg2.action === 'FOLD');

const ajs_vs_btn = PA.facingRaiseAdvice('BB', 'BTN', C('Ah'), C('Jh'));
check('AJs in BB vs BTN → 3-BET or CALL (defend)', ['3-BET','CALL'].includes(ajs_vs_btn.action));

const _72s_vs_btn = PA.facingRaiseAdvice('BB', 'BTN', C('7h'), C('2h'));
check('72s in BB vs BTN → FOLD (too weak even for BB defense)', _72s_vs_btn.action === 'FOLD');

const t9s_vs_btn = PA.facingRaiseAdvice('BB', 'BTN', C('Th'), C('9h'));
check('T9s in BB vs BTN → CALL (defend)', t9s_vs_btn.action === 'CALL');

// ============================================================
// =================== FULL ADVISE ============================
// ============================================================
section('Full advise() integration');

// Preflop RFI: AKs on BTN unopened
const r1 = PA.advise({
  hole: [C('As'), C('Ks')], board: [], position: 'BTN', format: 6,
  potBb: 1.5, toCallBb: 0, effStackBb: 100,
  isPfa: false, isIp: false, openerPos: '',
  villainRangeName: 'random', iters: 1000, history: []
});
check('AKs BTN unopened → RAISE', r1.verdict === 'RAISE');
check('AKs BTN raise size = 2.5bb', r1.size === 2.5);

// Preflop facing UTG raise: 22 from BTN
const r2 = PA.advise({
  hole: [C('2s'), C('2c')], board: [], position: 'BTN', format: 6,
  potBb: 4.5, toCallBb: 3, effStackBb: 100,
  isPfa: false, isIp: false, openerPos: 'UTG',
  villainRangeName: 'random', iters: 1000, history: []
});
check('22 BTN vs UTG raise → CALL (set mine)', r2.verdict === 'CALL');

// Preflop facing UTG raise: 72o from BTN
const r3 = PA.advise({
  hole: [C('7s'), C('2c')], board: [], position: 'BTN', format: 6,
  potBb: 4.5, toCallBb: 3, effStackBb: 100,
  isPfa: false, isIp: false, openerPos: 'UTG',
  villainRangeName: 'random', iters: 1000, history: []
});
check('72o BTN vs UTG raise → FOLD', r3.verdict === 'FOLD');

// Postflop top-pair facing bet: should call
const r4 = PA.advise({
  hole: [C('As'), C('Kh')], board: [C('Ad'), C('7c'), C('2s')], position: 'BTN', format: 6,
  potBb: 6, toCallBb: 3, effStackBb: 95,
  isPfa: true, isIp: true, openerPos: '',
  villainRangeName: 'random', iters: 2000, history: []
});
console.log(`  TPTK on A72 facing bet: verdict=${r4.verdict}, equity=${(r4.equity*100).toFixed(1)}%`);
check('TPTK facing bet → not FOLD', !r4.verdict.includes('FOLD'));

// Postflop big draw vs bet — implied/draw call
const r5 = PA.advise({
  hole: [C('Ah'), C('Kh')], board: [C('5h'), C('9h'), C('2c')], position: 'BTN', format: 6,
  potBb: 10, toCallBb: 8, effStackBb: 92,
  isPfa: false, isIp: true, openerPos: '',
  villainRangeName: 'random', iters: 2000, history: []
});
console.log(`  AhKh nut FD on 592 facing 8 into 10: verdict=${r5.verdict}, equity=${(r5.equity*100).toFixed(1)}%, outs=${r5.draws.totalOuts}`);
check('Nut FD with 2 overs → not FOLD', !r5.verdict.startsWith('FOLD'));

// BB with no preflop action — should be CHECK/WIN, not FOLD
const r_bb_walk = PA.advise({
  hole: [C('7s'), C('2c')], board: [], position: 'BB', format: 6,
  potBb: 1.5, toCallBb: 0, effStackBb: 100,
  isPfa: false, isIp: false, openerPos: '',
  villainRangeName: 'random', iters: 1000, history: []
});
console.log(`  72o BB unopened: verdict=${r_bb_walk.verdict}`);
check('BB unopened (no history) → CHECK/WIN, not FOLD', r_bb_walk.verdict.includes('WIN') || r_bb_walk.verdict === 'CHECK / WIN');

// Postflop weak hand facing big bet
const r6 = PA.advise({
  hole: [C('5s'), C('5d')], board: [C('Ah'), C('Kc'), C('Qd')], position: 'BB', format: 6,
  potBb: 10, toCallBb: 10, effStackBb: 90,
  isPfa: false, isIp: false, openerPos: '',
  villainRangeName: 'utg_open', iters: 2000, history: []
});
console.log(`  55 on AKQ facing pot bet vs UTG open: verdict=${r6.verdict}, equity=${(r6.equity*100).toFixed(1)}%`);
check('55 vs UTG range on AKQ pot-bet → FOLD', r6.verdict === 'FOLD');

// ============================================================
// =================== SUMMARY ================================
// ============================================================
console.log(`\n══ TOTAL ══\n  passed: ${passed}\n  failed: ${failed}`);
if (failed > 0) process.exit(1);
