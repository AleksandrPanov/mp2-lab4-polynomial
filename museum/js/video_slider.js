new Swiper('.video__slider', {
  loop: true,
  spaceBetween: 20,
  slidesPerView: 2,
  navigation: {
    nextEl: '.video__slider-arrow-next',
    prevEl: '.video__slider-arrow-prev',
  },
  pagination: {
    el: '.video__slider-pagination',
    type: 'bullets',
    clickable: true
  },
  breakpoints: {
    1024: {
      slidesPerView: 3,
      spaceBetween: 42,
    }
  }
});

const prevBtn = document.querySelector('.video__slider-arrow-prev'),
      nextBtn = document.querySelector('.video__slider-arrow-next'),
      currentVideo = document.querySelector('.video_current'),
      videoDots = document.querySelectorAll('.video__slider-pagination .swiper-pagination-bullet');

let   counter = 1;

function switchMainVideo() {
  currentVideo.src = `assets/video/${counter}.mp4`;
  currentVideo.poster = `assets/img/video/${counter}.jpg`;
  pauseBtn.src = 'assets/svg/video/play.svg';
  pauseOuter.dataset.paused = 'paused';
  pauseBtnBig.style.display = 'block';
}

for (let i = 0; i < videoDots.length; i++) {
  videoDots[i].addEventListener('click', () => {
    counter = i + 1;
    switchMainVideo();
  });
}

prevBtn.addEventListener('click', () => {
  (counter == 1) ? counter = 5 : counter--;
  switchMainVideo();
})
nextBtn.addEventListener('click', () => {
  (counter == 5) ? counter = 1 : counter++;
  switchMainVideo();
})