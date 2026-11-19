/* Calculus intro dialog: lesson chips, Previous/Next, and lazy-loaded lessons.
   Each lesson module (and the shared mathjs code) downloads on first open. */
const lessons = {
  derivative: { title: 'Derivative', load: () => import('./calculus-derivative.js') },
  partial: { title: 'Partial derivative', load: () => import('./calculus-partial.js') },
  chain: { title: 'Chain rule', load: () => import('./calculus-chain.js') },
};

function mountCalculusLesson(host, id) {
  host.innerHTML = '<p class="calculus-example">Loading the lesson…</p>';
  lessons[id].load().then(
    ({ mount }) => { if (host.dataset.lesson === id) mount(host); },
    () => { if (host.dataset.lesson === id) host.innerHTML = '<p class="calculus-example">This lesson couldn’t load. Check your connection and reopen it.</p>'; },
  );
}

export function wireCalculusDialog() {
  const chip = document.querySelector('#openCalculus');
  const dialog = document.querySelector('#calculusDialog');
  const title = dialog.querySelector('#calculusStepTitle');
  const host = dialog.querySelector('#calculusLesson');
  const previous = dialog.querySelector('#calculusPrev');
  const next = dialog.querySelector('#calculusNext');
  const chips = dialog.querySelector('#calculusChips');
  const ids = Object.keys(lessons);
  let index = 0;

  chips.innerHTML = ids.map((id, i) =>
    `<button type="button" class="pill calculus-step-chip" data-index="${i}"><span aria-hidden="true">${i + 1}</span> ${lessons[id].title}</button>`).join('');
  chips.addEventListener('click', event => {
    const chip = event.target.closest('button[data-index]');
    if (!chip || +chip.dataset.index === index) return;
    index = +chip.dataset.index;
    showLesson();
  });

  function showLesson() {
    const id = ids[index];
    title.textContent = lessons[id].title;
    host.dataset.lesson = id;
    chips.querySelectorAll('button').forEach((chip, i) => {
      chip.classList.toggle('on', i === index);
      if (i === index) chip.setAttribute('aria-current', 'step');
      else chip.removeAttribute('aria-current');
    });
    mountCalculusLesson(host, id);
    previous.disabled = index === 0;
    next.textContent = index === ids.length - 1 ? 'Done' : 'Next';
    dialog.scrollTop = 0;
    title.focus();
  }

  function openLesson(start) {
    index = start;
    dialog.showModal();
    showLesson();
  }

  chip.addEventListener('click', () => openLesson(0));
  dialog.addEventListener('close', () => chip.focus());
  previous.addEventListener('click', () => {
    index--;
    showLesson();
  });
  next.addEventListener('click', () => {
    if (index === ids.length - 1) {
      dialog.close();
      return;
    }
    index++;
    showLesson();
  });

  const linkedLesson = ids.indexOf(new URLSearchParams(location.hash.slice(1)).get('step'));
  if (linkedLesson >= 0) openLesson(linkedLesson);
}
