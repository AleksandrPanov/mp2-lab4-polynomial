const burger = document.querySelector('.header__burger'),
      mobileMenu = document.querySelector('.navigation-mobile'),
      welcomeInfo = document.querySelector('.welcome__info'),
      mobileMenuItem = document.querySelectorAll('.navigation-mobile a');

function toggleMenu() {
  if (burger.classList.contains('header__burger_active')) {
    burger.classList.remove('header__burger_active');
    mobileMenu.classList.remove('navigation-mobile_active');
    welcomeInfo.classList.remove('welcome__info_hide');
  } else {
    burger.classList.add('header__burger_active')
    mobileMenu.classList.add('navigation-mobile_active');
    welcomeInfo.classList.add('welcome__info_hide');
  }
}

burger.addEventListener('click', toggleMenu);

// document.body.addEventListener('click', (e) => {
//   console.log(e);
//   if (burger.classList.contains('header__burger_active') 
//       && e.target != mobileMenu) {
//         toggleMenu();
//       }
// })

for (let i = 0; i < mobileMenuItem.length; i++) {
  mobileMenuItem[i].addEventListener('click', toggleMenu);
}

