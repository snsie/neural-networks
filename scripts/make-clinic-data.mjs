// Generates public/data/clinic-wait-times.csv for the Clinic Wait-Time Simulator idea.
// Run: node scripts/make-clinic-data.mjs
// The data are synthetic. Wait time rises with queue length, falls with staff on shift,
// and is higher at peak hours and on Mondays, plus random noise.
import { writeFileSync, mkdirSync } from 'node:fs';

let seed = 2026;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
const randInt = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
const normal = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'];

const rows = ['hour,day_of_week,staff_on_shift,patients_in_queue,wait_minutes'];
for (let i = 0; i < 2000; i++) {
  const hour = randInt(8, 17);
  const day = randInt(0, 4);
  const staff = randInt(1, 5);
  const queue = randInt(0, 20);
  const peak = [9, 10, 16, 17].includes(hour) ? 8 : 0;
  const monday = day === 0 ? 4 : 0;
  const wait = Math.max(0, 5 + (queue * 15) / staff + peak + monday + 5 * normal());
  rows.push([hour, days[day], staff, queue, wait.toFixed(1)].join(','));
}

mkdirSync(new URL('../public/data/', import.meta.url), { recursive: true });
writeFileSync(new URL('../public/data/clinic-wait-times.csv', import.meta.url), rows.join('\n') + '\n');
console.log(`Wrote ${rows.length - 1} rows`);
