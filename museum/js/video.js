let videoWrapper = document.querySelector('.video__wrapper'),
    video = document.querySelector('.video_current'),
    timeBar = document.querySelector('.video__time-bar'),
    timeInProgress = document.querySelector('.video__time-fill'),
    volumeBar = document.querySelector('.video__changevolume-bar'),
    volumeInProgress = document.querySelector('.video__volume-fill'),
    pauseOuter = document.querySelector('.video__pause'),
    pauseBtn = document.querySelector('.video__pause-btn'),
    pauseBtnBig = document.querySelector('.video__pause_big'),
    volumeOuter = document.querySelector('.video__volume'),
    volumeBtn = document.querySelector('.video__volume-btn'),
    fullscreanBtn = document.querySelector('.video__fullscreen-btn');
  
function togglePause() {
  if (pauseOuter.dataset.paused == 'paused') {
    pauseBtn.src = 'assets/svg/video/pause.svg';
    pauseOuter.dataset.paused = 'playing';
    pauseBtnBig.style.display = 'none';
    video.play();
  } else {
    pauseBtn.src = 'assets/svg/video/play.svg';
    pauseOuter.dataset.paused = 'paused';
    pauseBtnBig.style.display = 'block';
    video.pause();
  }
}

function getTime() {
  timeBar.value = (video.currentTime / video.duration) * 100;
  timeInProgress.style.width = `${timeBar.value}%`
}

timeInProgress.addEventListener('click', (e) => {
  let event = e;
  timeInProgress.style.display = 'none';
  starter = document.elementFromPoint(event.clientX, event.clientY);
  console.log(starter);
  starter.click();
  timeInProgress.style.display = 'block';
})

function setTime() {
  timeInProgress.style.display = 'block';
  timeInProgress.style.width = `${timeBar.value}%`
  video.currentTime = timeBar.value / 100 * video.duration;
}

function fullScreen() {
  pauseBtnBig.style.display = 'none';
  if (!document.fullscreenElement) {
    video.requestFullscreen();
    pauseBtnBig.style.display = 'none';
  } else {
    pauseBtnBig.style.display = 'none';
    document.exitFullscreen();
    pauseBtnBig.style.display = 'none';
  }
}

pauseBtnBig.addEventListener('click', togglePause);
pauseOuter.addEventListener('click', togglePause);
video.addEventListener('click', togglePause);
video.addEventListener('ended', togglePause)
video.addEventListener('timeupdate', getTime);

timeBar.addEventListener('input', setTime);
timeBar.addEventListener('keyup', setTime);

volumeOuter.addEventListener('click', () => {
  if (volumeOuter.dataset.vol == 'on') {
    volumeBtn.src = 'assets/svg/video/volume_off.svg';
    volumeOuter.dataset.vol = 'off';
    volumeInProgress.style.width = '0';
    volumeBar.value = 0;
    video.volume = 0;
  } else {
    volumeBtn.src = 'assets/svg/video/volume.svg';
    volumeOuter.dataset.vol = 'on';
    volumeInProgress.style.width = '50%';
    volumeBar.value = '50'
    video.volume = 0.5;
  }
})

volumeBar.addEventListener('input', () => {
  volumeInProgress.style.width = `${volumeBar.value}%`;
  video.volume = volumeBar.value / 100;
  volumeBtn.src = 'assets/svg/video/volume.svg';
  volumeOuter.dataset.vol = 'on';
})

// fullscreanBtn.addEventListener('keyup', () => {
//   if (videoWrapper.classList.contains('video__wrapper_fullscreen'))
//     videoWrapper.classList.remove('video__wrapper_fullscreen');
// })

fullscreanBtn.addEventListener('click', fullScreen)
