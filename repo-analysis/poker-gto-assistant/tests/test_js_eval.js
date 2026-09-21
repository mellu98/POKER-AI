// Quick correctness check for the in-page JS evaluator.
// Mirrors evaluate5/evaluate7/equityVsRandom from poker_assistant.html

const RANKS = '23456789TJQKA';
const SUITS = ['s','h','d','c'];
const SUIT_INDEX = {s:0,h:1,d:2,c:3};
function cardCodeToInt(code){
  return RANKS.indexOf(code[0].toUpperCase())*4 + SUIT_INDEX[code[1].toLowerCase()];
}
function r(c){return c>>2;} function s(c){return c&3;}

function evaluate5(cards){
  const ranks=[r(cards[0]),r(cards[1]),r(cards[2]),r(cards[3]),r(cards[4])];
  ranks.sort((a,b)=>b-a);
  const sameSuit=s(cards[0])===s(cards[1])&&s(cards[1])===s(cards[2])&&s(cards[2])===s(cards[3])&&s(cards[3])===s(cards[4]);
  const cnt=new Array(13).fill(0);
  cnt[ranks[0]]++;cnt[ranks[1]]++;cnt[ranks[2]]++;cnt[ranks[3]]++;cnt[ranks[4]]++;
  const cr=[]; for(let i=12;i>=0;i--) if(cnt[i]) cr.push([cnt[i],i]);
  cr.sort((a,b)=>b[0]-a[0]||b[1]-a[1]);
  let st=-1;
  if(cr.length===5 && ranks[0]-ranks[4]===4) st=ranks[0];
  if(ranks[0]===12&&ranks[1]===3&&ranks[2]===2&&ranks[3]===1&&ranks[4]===0) st=3;
  if(sameSuit&&st>=0) return 8e9+st;
  if(cr[0][0]===4) return 7e9+cr[0][1]*13+cr[1][1];
  if(cr[0][0]===3&&cr[1][0]===2) return 6e9+cr[0][1]*13+cr[1][1];
  if(sameSuit) return 5e9+(((ranks[0]*13+ranks[1])*13+ranks[2])*13+ranks[3])*13+ranks[4];
  if(st>=0) return 4e9+st;
  if(cr[0][0]===3) return 3e9+cr[0][1]*169+cr[1][1]*13+cr[2][1];
  if(cr[0][0]===2&&cr[1][0]===2) return 2e9+cr[0][1]*169+cr[1][1]*13+cr[2][1];
  if(cr[0][0]===2) return 1e9+cr[0][1]*2197+cr[1][1]*169+cr[2][1]*13+cr[3][1];
  return (((ranks[0]*13+ranks[1])*13+ranks[2])*13+ranks[3])*13+ranks[4];
}

const COMBOS=(()=>{const o=[];for(let a=0;a<7;a++)for(let b=a+1;b<7;b++)for(let c=b+1;c<7;c++)for(let d=c+1;d<7;d++)for(let e=d+1;e<7;e++)o.push([a,b,c,d,e]);return o;})();
function evaluate7(cards){let best=-1,t=new Array(5);for(let i=0;i<21;i++){const x=COMBOS[i];t[0]=cards[x[0]];t[1]=cards[x[1]];t[2]=cards[x[2]];t[3]=cards[x[3]];t[4]=cards[x[4]];const v=evaluate5(t);if(v>best)best=v;}return best;}

function equityVsRandom(hole,board,nOpp,iters){
  const used=new Set([...hole,...board]);
  const pool=[]; for(let i=0;i<52;i++) if(!used.has(i)) pool.push(i);
  let wins=0,ties=0;
  const fb=new Array(5),h7=new Array(7),v7=new Array(7);
  for(let it=0;it<iters;it++){
    for(let i=pool.length-1;i>0;i--){const j=(Math.random()*(i+1))|0;[pool[i],pool[j]]=[pool[j],pool[i]];}
    let idx=0;
    for(let k=0;k<5;k++) fb[k]=k<board.length?board[k]:pool[idx++];
    h7[0]=hole[0];h7[1]=hole[1];for(let k=0;k<5;k++) h7[2+k]=fb[k];
    const hs=evaluate7(h7);
    let bv=-1;
    for(let v=0;v<nOpp;v++){
      v7[0]=pool[idx++];v7[1]=pool[idx++];for(let k=0;k<5;k++) v7[2+k]=fb[k];
      const sc=evaluate7(v7); if(sc>bv) bv=sc;
    }
    if(hs>bv) wins++; else if(hs===bv) ties++;
  }
  return {win:wins/iters,tie:ties/iters,equity:wins/iters+(ties/iters)/(nOpp+1)};
}

// ---------- Sanity checks ----------
function check(name, cond){console.log((cond?'  ✓ ':'  ✗ ')+name);}
function approx(a,b,tol){return Math.abs(a-b)<=tol;}

// ranking sanity: SF > quads > FH > flush > straight > trips > 2p > pair > high
const sf = ['As','Ks','Qs','Js','Ts'].map(cardCodeToInt);
const quads = ['As','Ah','Ad','Ac','2s'].map(cardCodeToInt);
const fh = ['As','Ah','Ad','Ks','Kh'].map(cardCodeToInt);
const flush = ['As','Js','9s','5s','3s'].map(cardCodeToInt);
const str8 = ['9s','8h','7d','6c','5s'].map(cardCodeToInt);
const trips = ['As','Ah','Ad','Ks','Qh'].map(cardCodeToInt);
const twop = ['As','Ah','Ks','Kh','Qd'].map(cardCodeToInt);
const pair = ['As','Ah','Ks','Qh','Jd'].map(cardCodeToInt);
const hc = ['As','Kh','Qd','Jc','9s'].map(cardCodeToInt);
const wheel = ['As','2h','3d','4c','5s'].map(cardCodeToInt);
const wheelSf = ['As','2s','3s','4s','5s'].map(cardCodeToInt);

check('SF beats quads', evaluate5(sf) > evaluate5(quads));
check('quads beat FH', evaluate5(quads) > evaluate5(fh));
check('FH beats flush', evaluate5(fh) > evaluate5(flush));
check('flush beats straight', evaluate5(flush) > evaluate5(str8));
check('straight beats trips', evaluate5(str8) > evaluate5(trips));
check('trips beat 2-pair', evaluate5(trips) > evaluate5(twop));
check('2-pair beats pair', evaluate5(twop) > evaluate5(pair));
check('pair beats high card', evaluate5(pair) > evaluate5(hc));
check('wheel detected as straight (not high card)', evaluate5(wheel) >= 4e9 && evaluate5(wheel) < 5e9);
check('wheel SF detected as straight flush', evaluate5(wheelSf) >= 8e9);

// equity sanity: AA vs random heads-up ≈ 85%
const t0=Date.now();
const aa = equityVsRandom([cardCodeToInt('As'),cardCodeToInt('Ac')],[],1,5000);
console.log(`  AA vs random HU: equity ${(aa.equity*100).toFixed(1)}% (expect ~85%) — ${Date.now()-t0}ms`);
check('AA equity in [82, 88]', aa.equity > 0.82 && aa.equity < 0.88);

// 72o vs random heads-up ≈ 35%
const t1=Date.now();
const trash = equityVsRandom([cardCodeToInt('7c'),cardCodeToInt('2d')],[],1,5000);
console.log(`  72o vs random HU: equity ${(trash.equity*100).toFixed(1)}% (expect ~35%) — ${Date.now()-t1}ms`);
check('72o equity in [30, 40]', trash.equity > 0.30 && trash.equity < 0.40);

// AKs on AKQ flop vs random ≈ ~85% (top 2 pair)
const t2=Date.now();
const top2pair = equityVsRandom(
  [cardCodeToInt('As'),cardCodeToInt('Ks')],
  [cardCodeToInt('Ah'),cardCodeToInt('Kh'),cardCodeToInt('2c')], 1, 5000);
console.log(`  AKs on AhKh2c HU: equity ${(top2pair.equity*100).toFixed(1)}% (expect ~95%+) — ${Date.now()-t2}ms`);
check('top 2-pair equity > 90%', top2pair.equity > 0.90);

// AA vs 2 random ≈ ~73%
const t3=Date.now();
const aa2 = equityVsRandom([cardCodeToInt('As'),cardCodeToInt('Ac')],[],2,3000);
console.log(`  AA vs 2 random: equity ${(aa2.equity*100).toFixed(1)}% (expect ~73%) — ${Date.now()-t3}ms`);
check('AA vs 2 in [68, 78]', aa2.equity > 0.68 && aa2.equity < 0.78);

console.log('\nDONE');
