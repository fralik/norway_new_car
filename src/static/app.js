function sendYearValues() {
  const yearRange = JSON.parse(localStorage.getItem('yearRange'));
  if (yearRange) {
    Shiny.setInputValue('year_range', yearRange);
  }
}

$(function() {
  // Shiny.addCustomMessageHandler('shiny:connected', sendYearValues);
  setTimeout(sendYearValues, 500);
  let plotReady = false;

  // Wait 3 seconds before setting up the listener, so that we initialize the input with the value from localStorage
  setTimeout(function() {
    // When the year_range input changes, save its value in localStorage
    Shiny.addCustomMessageHandler('msg_saveYearRange', function(message) {
      console.log(`Saving yearRange to local storage: ${JSON.stringify(message)}`);
      localStorage.setItem('yearRange', JSON.stringify(message));
    });
  }, 3000);
});
