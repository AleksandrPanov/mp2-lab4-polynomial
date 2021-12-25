function shuffle(array) {
  for (let i = array.length - 1; i > 0; i--) {
    let j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

function addRandomPictures() {
  let pictures = [];
  let galleryInner = document.querySelector('.gallery-inner');
  for (let i = 0; i < 15; i++) {
    pictures.push(`<img src="assets/img/gallery/${i+1}.jpg" alt="">`)
  }
  shuffle(pictures);
  for (let i = 0; i < 15; i++) {
    // console.log(pictures[i + 1]);
    galleryInner.innerHTML += `<div class="gallery__block gallery__block_${i+1}">
    ${pictures[i]}</div>`
  }
}
addRandomPictures();