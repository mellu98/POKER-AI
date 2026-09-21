// Tests for the simple version (poker_assistant.html with adviseSimple).
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, '..', 'poker_assistant.html'), 'utf8');
const m = html.match(/<script>([\s\S]*?)<\/script>/);
const code = m[1];

const noop = () => {};
function stubEl() {
  const el = {
    children: [], dataset: {}, style: {}, classList: { _s: new Set(),
      add(c){this._s.add(c);}, remove(c){this._s.delete(c);},
      toggle(c, force){ if (force === true) this._s.add(c); else if (force === false) this._s.delete(c); else (this._s.has(c)?this._s.delete(c):this._s.add(c)); },
      contains(c){return this._s.has(c);} },
    addEventListener: noop, appendChild(c){this.children.push(c); return c;},
    querySelector(){ return stubEl(); }, querySelectorAll(){ return []; },
    set innerHTML(v){this._i=v;} , get innerHTML(){return this._i||'';},
    set textContent(v){this._t=v;}, get textContent(){return this._t||'';},
    onclick: null, value: '', checked: false, closest: () => null,
    get tagName(){return 'DIV';}
  };
  el.parentElement = el;
  return el;
}
const localStorage = { _d:{}, getItem(k){return this._d[k]||null;}, setItem(k,v){this._d[k]=v;} };
global.document = {
  querySelector(){return stubEl();}, querySelectorAll(){return [];},
  getElementById(){return stubEl();}, addEventListener: noop,
  get activeElement(){return {tagName:'BODY'};}, createElement(){return stubEl();},
  body: stubEl()
};
global.window = {};
global.localStorage = localStorage;
global.performance = { now: () => Date.now() };
global.requestAnimationFrame = (cb) => setTimeout(cb, 0);

try { eval(code); } catch(e) { console.error('eval failed:', e.message); console.error(e.stack); process.exit(1); }
const PA = window.__pa;
if (!PA) { console.error('window.__pa missing'); process.exit(1); }

const RANKS = '23456789TJQKA', SUITS = ['s','h','d','c'];
const C = code => RANKS.indexOf(code[0].toUpperCase())*4 + SUITS.indexOf(code[1].toLowerCase());

let pass = 0, fail = 0;
function check(name, cond, info) {
  if (cond) { console.log('  ✓ ' + name); pass++; }
  else { console.log('  ✗ ' + name + (info ? ' — ' + info : '')); fail++; }
}
function s(name) { console.log('\n══ ' + name + ' ══'); }

s('Simple advise — BTN preflop');

// BTN with AKs unopened → RAISE
const r1 = PA.adviseSimple({
  hole: [C('As'), C('Ks')], board: [], position: 'BTN', preflopOpp: 'none', postflopOpp: 'check'
});
console.log('  BTN AKs unopened: ' + r1.mainText);
check('BTN AKs → ПОДНИМИ', r1.mainText.startsWith('ПОДНИМИ'));

// BTN with 32o unopened → still RAISE (HU is super wide)
const r2 = PA.adviseSimple({
  hole: [C('3s'), C('2c')], board: [], position: 'BTN', preflopOpp: 'none', postflopOpp: 'check'
});
console.log('  BTN 32o unopened: ' + r2.mainText);
check('BTN 32o → реалистично (RAISE in HU)', r2.mainText.startsWith('ПОДНИМИ') || r2.mainText === 'СБРОСЬ');

// BTN with 22 vs all-in → ?
const r3 = PA.adviseSimple({
  hole: [C('2s'), C('2c')], board: [], position: 'BTN', preflopOpp: 'rallin', postflopOpp: 'check'
});
console.log('  BTN 22 vs all-in: ' + r3.mainText);

// BTN with AA vs opp's 3-bet → 4-BET
const r_btn_4b = PA.adviseSimple({
  hole: [C('As'), C('Ac')], board: [], position: 'BTN', preflopOpp: '3b9', postflopOpp: 'check'
});
console.log('  BTN AA vs opp 3-bet 9bb: ' + r_btn_4b.mainText);
check('BTN AA vs 3-bet → 4-BET with non-zero size', r_btn_4b.mainText.startsWith('4-BET') && !r_btn_4b.mainText.includes(' 0bb') && r_btn_4b.size >= 20);

// BTN with TT vs 3-bet → CALL (set mine)
const r_btn_call = PA.adviseSimple({
  hole: [C('Ts'), C('Td')], board: [], position: 'BTN', preflopOpp: '3b9', postflopOpp: 'check'
});
console.log('  BTN TT vs opp 3-bet 9bb: ' + r_btn_call.mainText);
check('BTN TT vs 3-bet → CALL', r_btn_call.mainText === 'КОЛЛИРУЙ');

// BTN with 72o vs 3-bet → FOLD
const r_btn_fold = PA.adviseSimple({
  hole: [C('7s'), C('2c')], board: [], position: 'BTN', preflopOpp: '3b9', postflopOpp: 'check'
});
console.log('  BTN 72o vs opp 3-bet: ' + r_btn_fold.mainText);
check('BTN 72o vs 3-bet → СБРОСЬ', r_btn_fold.mainText === 'СБРОСЬ');

s('Simple advise — BB preflop');

// BB with no action (opp folded) → WIN
const r4 = PA.adviseSimple({
  hole: [C('7s'), C('2c')], board: [], position: 'BB', preflopOpp: 'none', postflopOpp: 'check'
});
console.log('  BB 72o, opp folded: ' + r4.mainText);
check('BB unopened → ВЫИГРАЛ', r4.mainText.includes('ВЫИГРАЛ'));

// BB vs limp with AK → ISO RAISE
const r5 = PA.adviseSimple({
  hole: [C('As'), C('Kc')], board: [], position: 'BB', preflopOpp: 'limp', postflopOpp: 'check'
});
console.log('  BB AKo vs limp: ' + r5.mainText);
check('BB vs limp AKo → ПОДНИМИ', r5.mainText.startsWith('ПОДНИМИ'));

// BB vs raise 3bb with AA → 3-BET
const r6 = PA.adviseSimple({
  hole: [C('As'), C('Ac')], board: [], position: 'BB', preflopOpp: 'r3', postflopOpp: 'check'
});
console.log('  BB AA vs r3: ' + r6.mainText);
check('BB vs raise AA → 3-BET', r6.mainText.startsWith('3-BET') || r6.mainText.startsWith('СНОВА'));

// BB vs raise 3bb with T9s → CALL
const r7 = PA.adviseSimple({
  hole: [C('Th'), C('9h')], board: [], position: 'BB', preflopOpp: 'r3', postflopOpp: 'check'
});
console.log('  BB T9s vs r3: ' + r7.mainText);
check('BB vs raise T9s → КОЛЛИРУЙ', r7.mainText === 'КОЛЛИРУЙ');

// BB vs raise 3bb with 72o → FOLD
const r8 = PA.adviseSimple({
  hole: [C('7s'), C('2c')], board: [], position: 'BB', preflopOpp: 'r3', postflopOpp: 'check'
});
console.log('  BB 72o vs r3: ' + r8.mainText);
check('BB vs raise 72o → СБРОСЬ', r8.mainText === 'СБРОСЬ');

s('Simple advise — postflop');

// Strong hand on flop, opp checked → BET
const r9 = PA.adviseSimple({
  hole: [C('As'), C('Ks')], board: [C('Ah'), C('7c'), C('2d')],
  position: 'BB', preflopOpp: 'r3', postflopOpp: 'check'
});
console.log('  TPTK on A72, opp checked: ' + r9.mainText + ' / equity=' + (r9.equity*100).toFixed(0) + '%');
check('TPTK opp checked → СТАВЬ', r9.mainText.startsWith('СТАВЬ'));

// Strong vs bet → RAISE
const r10 = PA.adviseSimple({
  hole: [C('As'), C('Ks')], board: [C('Ah'), C('7c'), C('2d')],
  position: 'BB', preflopOpp: 'r3', postflopOpp: 'bhalf'
});
console.log('  TPTK vs half-pot bet: ' + r10.mainText + ' / equity=' + (r10.equity*100).toFixed(0) + '%');
check('TPTK vs bet → ПОДНИМИ', r10.mainText.startsWith('ПОДНИМИ'));

// Nut FD vs bet → CALL or RAISE
const r11 = PA.adviseSimple({
  hole: [C('Ah'), C('Kh')], board: [C('5h'), C('9h'), C('2c')],
  position: 'BB', preflopOpp: 'r3', postflopOpp: 'bhalf'
});
console.log('  Nut FD on 5h9h2c vs half-pot: ' + r11.mainText);
check('Nut FD facing bet → not СБРОСЬ', r11.mainText !== 'СБРОСЬ');

// Trash vs bet → FOLD
const r12 = PA.adviseSimple({
  hole: [C('7s'), C('2c')], board: [C('Ah'), C('Kc'), C('Qd')],
  position: 'BB', preflopOpp: 'r3', postflopOpp: 'bpot'
});
console.log('  72o on AKQ vs pot-bet: ' + r12.mainText);
check('72o vs pot bet on AKQ → СБРОСЬ', r12.mainText === 'СБРОСЬ');

console.log(`\n══ TOTAL: ${pass}/${pass+fail} passed ══`);
if (fail > 0) process.exit(1);
