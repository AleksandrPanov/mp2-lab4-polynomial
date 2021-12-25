const drag = document.querySelector('.explore__img-drag'),
      before = document.querySelector('.explore__img-resize_before');

drag.addEventListener('input', () => {
  before.style.width = drag.value + '%'
});


