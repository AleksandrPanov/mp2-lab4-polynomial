// open / close form
const btn = document.querySelector('.tickets__amount-button'),
      popup = document.querySelector('.popup'),
      popupBody = document.querySelector('.popup__body'),
      closeBtn = document.querySelector('.popup__close');

function popupClose() {
  popup.classList.remove('popup_active');
  popupBody.classList.remove('popup__body_active');
}
function popupOpen() {
  popup.classList.add('popup_active');
  popupBody.classList.add('popup__body_active');
  updateOnOpening();
} 
 
btn.addEventListener('click', popupOpen);
closeBtn.addEventListener('click', popupClose);

popup.addEventListener('click', (e) => {
  if (popup.classList.contains('popup_active') && popup == e.target)
    popupClose();
})

// price and date
const minuses = document.querySelectorAll('.tickets__amount-minus'),
      pluses = document.querySelectorAll('.tickets__amount-plus'),
      inputBasic = document.querySelector('.count-input-basic'),
      inputSenior = document.querySelector('.count-input-senior'),
      ticketTypes = document.querySelectorAll('.tickets__type input'),
      totalPrice = document.querySelector('.total__counter');

  let type = 'perm',
      typePrice = 20;

const minusesForm = document.querySelectorAll('.popup__entry-minus'),
      plusesForm = document.querySelectorAll('.popup__entry-plus'),
      inputBasicForm = document.querySelector('.count-input-basic-form'),
      inputSeniorForm = document.querySelector('.count-input-senior-form'),
      priceForBasicTicket = document.querySelectorAll('.form-price-basic'),
      priceForSeniorTicket = document.querySelectorAll('.form-price-senior'),
      ticketBasicCounter = document.querySelector('.ticket-basic-counter'),
      ticketSeniorCounter = document.querySelector('.ticket-senior-counter'),
      ticketBasicPrice = document.querySelector('.ticket-basic-price'),
      ticketSeniorPrice = document.querySelector('.ticket-senior-price'),
      totalPriceForm = document.querySelector('.popup__right-total-price'),
      selectForm = document.querySelector('.popup__ticket'),
      timeForm = document.querySelector('.popup__time'),
      dateForm = document.querySelector('.popup__date');
      
function update() {
  totalPrice.innerHTML = typePrice * inputBasic.value + typePrice * inputSenior.value / 2;
}
function updateOnOpening() {
  inputBasicForm.value = inputBasic.value;
  inputSeniorForm.value = inputSenior.value;
  document.querySelector(`.popup__ticket option[value=${type}]`).selected = true;
  priceForBasicTicket.forEach((p) => {
    p.innerHTML = typePrice;
  })
  priceForSeniorTicket.forEach((p) => {
    p.innerHTML = typePrice / 2;
  })
  ticketBasicCounter.innerHTML = inputBasic.value;
  ticketSeniorCounter.innerHTML = inputSenior.value;
  ticketBasicPrice.innerHTML = inputBasic.value * typePrice;
  ticketSeniorPrice.innerHTML = inputSenior.value * typePrice / 2;
  totalPriceForm.innerHTML = totalPrice.innerText + '€';
}
function updateOnForm() {
  priceForBasicTicket.forEach((p) => {
    p.innerHTML = typePrice;
  })
  priceForSeniorTicket.forEach((p) => {
    p.innerHTML = typePrice / 2;
  })
  ticketBasicPrice.innerHTML = inputBasicForm.value * typePrice;
  ticketSeniorPrice.innerHTML = inputSeniorForm.value * typePrice / 2;
  ticketBasicCounter.innerHTML = inputBasicForm.value;
  ticketSeniorCounter.innerHTML = inputSeniorForm.value;
  totalPriceForm.innerHTML = inputBasicForm.value * typePrice + inputSeniorForm.value * typePrice / 2 + '€';
}

minuses.forEach((m) => {
  m.addEventListener('click', (e) => {
    e.target.parentElement.querySelector('input').stepDown();
    update()
  })
})
pluses.forEach((m) => {
  m.addEventListener('click', (e) => {
    e.target.parentElement.querySelector('input').stepUp();
    update()
  })
})
ticketTypes.forEach((rad) => {
  rad.addEventListener('change', () => {
    type = rad.dataset.select;
    typePrice = rad.dataset.price;
    update();
  })
})

minusesForm.forEach((m) => {
  m.addEventListener('click', (e) => {
    e.target.parentElement.querySelector('input').stepDown();
    updateOnForm();
  })
})
plusesForm.forEach((m) => {
  m.addEventListener('click', (e) => {
    e.target.parentElement.querySelector('input').stepUp();
    updateOnForm();
  })
})
selectForm.addEventListener('change', () => {
  selectForm.querySelectorAll('option').forEach((opt) => {
    if (opt.selected) typePrice = +opt.dataset.price;
  })
  updateOnForm();
})
timeForm.addEventListener('change', () => {
  document.querySelector('.popup__right-time-span').innerHTML = timeForm.value;
})
dateForm.addEventListener('change', () => {
  let date = new Date(dateForm.value);
  if (date >= Date.now()) {
    let months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September','October', 'November', 'December'];
    let days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    let res = days[date.getDay()] + ', ' + months[date.getMonth()] + ' ' + date.getDate();
    document.querySelector('.popup__right-date-span').innerHTML = res;
  } else {
    dateForm.value = ''
  }
})

