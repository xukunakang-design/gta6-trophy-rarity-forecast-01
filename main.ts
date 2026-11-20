// gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn rate
// at the case target, from the history of reads in the case file.
//
// Idea: a trophy's printed rate is earners / players-tracked, and Exophase
// refreshes the page in jumps, so the printed rate can fall when the tracked
// player count jumps even though nobody un-earns a trophy. Fit earners and
// tracked players separately, each as a power law in log time, and divide:
//   ln N(t) = aN + bN * ln(1 + hours since 2026-11-19T00:00Z)
//   ln (E(t) + 0.5) = aE + bE * ln(1 + hours since 2026-11-19T00:00Z)
// with recency weights (36 h half-life) over the distinct page states, slopes
// clamped at zero because neither count can fall. The median rate at the
// target is 100 * E / N. Spread combines a random-walk floor with the fit's
// residuals. Flat history falls back to the last rate, like persistence.
// Story chapter trophies are forced non-increasing in chapter order, and the
// platinum can never exceed the rarest trophy. Node 22, standard library only.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
const EPOCH_MS = Date.parse("2026-11-19T00:00:00Z");
const HALF_LIFE_H = 36;

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clamp = (x: number, lo: number, hi: number): number =>
  Math.min(hi, Math.max(lo, x));
const logit = (pct: number): number => {
  const p = clamp(pct, 0.01, 99.99) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
const hoursSinceEpoch = (iso: string): number =>
  Math.max(0, (Date.parse(iso) - EPOCH_MS) / 3600000);

/** Weighted least squares for y = a + b*x. Returns [a, b, residualSd]. */
function wls(
  xs: number[],
  ys: number[],
  ws: number[]
): [number, number, number] {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < xs.length; i++) {
    sw += ws[i];
    sx += ws[i] * xs[i];
    sy += ws[i] * ys[i];
  }
  const mx = sx / sw;
  const my = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    sxx += ws[i] * (xs[i] - mx) * (xs[i] - mx);
    sxy += ws[i] * (xs[i] - mx) * (ys[i] - my);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  const a = my - b * mx;
  let ss = 0;
  for (let i = 0; i < xs.length; i++) {
    const r = ys[i] - (a + b * xs[i]);
    ss += ws[i] * r * r;
  }
  const sd = Math.sqrt(ss / sw);
  return [a, b, sd];
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith(".in"))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -".in".length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), "utf8")) as Case;
  if (
    c.levels.length !== LEVELS.length ||
    c.levels.some((t, j) => t !== LEVELS[j])
  )
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}`);
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const last = c.history[c.history.length - 1];
  const dDays = c.horizon_minutes / 1440;
  const xTarget = Math.log(1 + hoursSinceEpoch(c.target));
  const xLast = Math.log(1 + hoursSinceEpoch(last.at));

  const medians: number[] = [];
  const spreads: number[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const rLast = last.rates[i];
    // Distinct page states for this trophy: a read counts when the tracked
    // count or this trophy's rate changed from the previous kept read.
    const xs: number[] = [];
    const lnN: number[] = [];
    const lnE: number[] = [];
    const ws: number[] = [];
    let prevN = -1;
    let prevR: number | null = null;
    for (const rd of c.history) {
      const r = rd.rates[i];
      if (r === null || r === undefined) continue;
      if (rd.players_tracked === prevN && r === prevR) continue;
      prevN = rd.players_tracked;
      prevR = r;
      const x = Math.log(1 + hoursSinceEpoch(rd.at));
      const w = Math.pow(
        0.5,
        (hoursSinceEpoch(last.at) - hoursSinceEpoch(rd.at)) / HALF_LIFE_H
      );
      xs.push(x);
      lnN.push(Math.log(Math.max(1, rd.players_tracked)));
      lnE.push(Math.log(Math.max(0.5, (r / 100) * rd.players_tracked + 0.5)));
      ws.push(w);
    }
    let median = rLast === null || rLast === undefined ? 0.01 : rLast;
    let spread = 0.5 * Math.sqrt(dDays);
    if (xs.length >= 3) {
      const [aN, bNraw, sdN] = wls(xs, lnN, ws);
      const [aE, bEraw, sdE] = wls(xs, lnE, ws);
      const bN = Math.max(0, bNraw);
      const bE = Math.max(0, bEraw);
      const nT = Math.exp(aN + bN * xTarget);
      const eT = Math.exp(aE + bE * xTarget) - 0.5;
      const pred = clamp((100 * Math.max(0, eT)) / Math.max(1, nT), 0.01, 99.99);
      // Residual spread on the logit scale, from the implied rate residuals.
      let ss = 0;
      for (let j = 0; j < xs.length; j++) {
        const rateJ = clamp(
          (100 * Math.max(0, Math.exp(aE + bE * xs[j]) - 0.5)) /
            Math.max(1, Math.exp(aN + bN * xs[j])),
          0.01,
          99.99
        );
        const actual = clamp(
          (100 * Math.max(0, Math.exp(lnE[j]) - 0.5)) / Math.max(1, Math.exp(lnN[j])),
          0.01,
          99.99
        );
        const diff = logit(rateJ) - logit(actual);
        ss += ws[j] * diff * diff;
      }
      let sw = 0;
      for (const w of ws) sw += w;
      const residSd = Math.sqrt(ss / Math.max(1e-9, sw));
      median = pred;
      spread = clamp(
        Math.sqrt(0.16 * dDays + residSd * residSd + 0.25 * (sdN * sdN + sdE * sdE)),
        0.35,
        1.15
      );
    }
    medians.push(clamp(median, 0.01, 99.99));
    spreads.push(spread);
  }

  // Story chapters must not rise in chapter order; platinum never exceeds
  // the rarest trophy.
  const chapterOf = (t: Trophy): number => {
    const m = /Chapter (\d)/.exec(t.description);
    if (m) return parseInt(m[1], 10);
    if (/final mission/i.test(t.description)) return 99;
    return -1;
  };
  const chapterIdx = c.trophies
    .map((t, i) => [chapterOf(t), i] as [number, number])
    .filter(([ch]) => ch > 0)
    .sort((a, b) => a[0] - b[0]);
  let cap = 100;
  for (const [, i] of chapterIdx) {
    medians[i] = Math.min(medians[i], cap);
    cap = medians[i];
  }
  let minAll = 100;
  for (const m of medians) minAll = Math.min(minAll, m);
  const platIdx = c.trophies.findIndex((t) => t.tier === "platinum");
  if (platIdx >= 0) medians[platIdx] = Math.min(medians[platIdx], minAll);

  const trophies = c.trophies.map((t, i) => {
    const l = logit(medians[i]);
    const s = spreads[i];
    const q = Z.map((z) => round4(clamp(invLogit(l + z * s), 0, 100)));
    for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
    return { id: t.id, q };
  });
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
